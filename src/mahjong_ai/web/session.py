from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from mahjong_ai.agents.llm import LLMActionAdapter, LLMDecisionTrace
from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import IllegalActionError, RuleConfigurationError
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.loader import load_rule_config, load_rule_plugin
from mahjong_ai.web.schemas import (
    ActionDescriptor,
    CreateSessionRequest,
    DecisionSnapshot,
    RuleSummary,
    SeatControllerConfig,
    SeatControllerKind,
    SessionSummary,
    serialize_event,
    serialize_full_state,
    serialize_replay,
    serialize_state,
)
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[3]
RULE_CONFIGS = {
    "northern_tuidaohe.v1": ROOT / "configs/rules/tuidaohe_v1.yaml",
}


class WebSession:
    def __init__(
        self,
        *,
        session_id: str,
        table: TableEngine,
        controllers: dict[Seat, SeatControllerConfig],
    ) -> None:
        self.session_id = session_id
        self.table = table
        self.controllers = controllers
        self.pending_responses: dict[Seat, Action] = {}
        self._full_log_frames: list[dict] = []
        self._record_frame(input_action=None, before_event_count=0)

    @property
    def summary(self) -> SessionSummary:
        return SessionSummary(
            session_id=self.session_id,
            rule_id=self.table.state.rule_id,
            rule_config_hash=self.table.state.rule_config_hash,
            wall_provider=self.table.wall_provider.provider_id,
            seed=self.table.seed,
            dealer=int(self.table.state.dealer),
            phase=self.table.state.phase.value,
            current_actor=int(self.table.state.current_actor),
            is_terminal=self.table.state.is_terminal,
            event_count=len(self.table.state.events),
            controllers={
                int(seat): {
                    "kind": controller.kind.value,
                    "model_id": controller.model_id,
                    "provider": controller.provider,
                    "base_url": controller.base_url,
                    "token_configured": bool(controller.token),
                    "model_name": controller.model_name,
                }
                for seat, controller in self.controllers.items()
            },
        )

    def snapshot(self, *, viewer: Seat | None = None) -> dict:
        return {
            "session": asdict(self.summary),
            "state": serialize_state(self.table.state, viewer=viewer),
            "decision": asdict(self.decision_snapshot()),
        }

    def step(
        self,
        descriptor: ActionDescriptor,
        *,
        viewer: Seat | None = None,
        controller_decision: LLMDecisionTrace | None = None,
    ) -> dict:
        before_event_count = len(self.table.state.events)
        self.submit_action(descriptor)
        if controller_decision is not None:
            controller_decision = replace(
                controller_decision,
                validation={
                    **dict(controller_decision.validation),
                    "rule_validated": True,
                },
            )
        if self.table.state.response_window is None and not self.table.state.is_terminal:
            self.table.advance_to_decision()
        return self._record_frame(
            input_action=descriptor,
            before_event_count=before_event_count,
            viewer=viewer,
            controller_decision=controller_decision,
        )

    def controller_step(
        self, actor: Seat, *, viewer: Seat | None = None
    ) -> dict:
        if actor not in self.table.decision_actors():
            raise IllegalActionError(f"seat {int(actor)} is not a decision actor")
        controller = self.controllers[actor]
        if controller.kind is SeatControllerKind.HUMAN:
            raise IllegalActionError(f"seat {int(actor)} is controlled by a human")
        legal_actions = self.table.legal_actions(actor)
        adapter = LLMActionAdapter(
            provider=controller.provider,
            base_url=controller.base_url,
            token=controller.token,
            model_name=controller.model_name,
        )
        trace = adapter.decide(
            state=self.table.state,
            actor=actor,
            rules=self.table.rules,
            legal_actions=legal_actions,
        )
        return self.step(
            ActionDescriptor.from_action(trace.selected_action),
            viewer=viewer,
            controller_decision=trace,
        )

    def decision_snapshot(self) -> DecisionSnapshot:
        actors = self.table.decision_actors()
        legal = {
            int(actor): tuple(
                ActionDescriptor.from_action(action)
                for action in self.table.legal_actions(actor)
            )
            for actor in actors
        }
        window = self.table.state.response_window
        return DecisionSnapshot(
            decision_actors=tuple(int(seat) for seat in actors),
            response_window=(
                {
                    "window_id": window.window_id,
                    "kind": window.kind.value,
                    "source": int(window.source),
                    "tile_type": window.tile.tile_type.code,
                    "eligible_seats": tuple(int(seat) for seat in window.eligible_seats),
                }
                if window is not None
                else None
            ),
            legal_actions=legal,
            pending_responses={
                int(seat): ActionDescriptor.from_action(action)
                for seat, action in self.pending_responses.items()
            },
        )

    def advance_to_decision(self) -> bool:
        self.pending_responses.clear()
        return self.table.advance_to_decision()

    def submit_action(self, descriptor: ActionDescriptor) -> None:
        action = descriptor.to_action()
        if self.table.state.response_window is None:
            self._submit_turn_action(action)
            return
        self._submit_response_action(action)

    def _submit_turn_action(self, action: Action) -> None:
        if action.actor != self.table.state.current_actor:
            raise IllegalActionError("action actor is not the current actor")
        if action not in self.table.legal_actions(action.actor):
            raise IllegalActionError("action is not legal in current state")
        self.pending_responses.clear()
        self.table.submit(action)

    def _submit_response_action(self, action: Action) -> None:
        window = self.table.state.response_window
        assert window is not None
        if action.actor not in window.eligible_seats:
            raise IllegalActionError("response actor is not eligible")
        if action not in self.table.legal_actions(action.actor):
            raise IllegalActionError("response action is not legal in current state")
        self.pending_responses[action.actor] = action
        if set(self.pending_responses) == set(window.eligible_seats):
            responses = dict(self.pending_responses)
            self.pending_responses.clear()
            self.table.submit_responses(responses)

    def submit_pass(self, actor: Seat) -> None:
        pass_action = next(
            (
                action
                for action in self.table.legal_actions(actor)
                if action.kind is ActionKind.PASS
            ),
            None,
        )
        if pass_action is None:
            raise IllegalActionError(f"PASS is not legal for seat {int(actor)}")
        self.submit_action(ActionDescriptor.from_action(pass_action))

    def events(self, *, since: int = 0, viewer: Seat | None = None) -> tuple[dict, ...]:
        return tuple(
            serialize_event(event)
            for event in self.table.state.events
            if event.event_id >= since and (viewer is None or event.is_visible_to(viewer))
        )

    def replay(self) -> dict:
        return serialize_replay(self.table.replay())

    def full_log(self) -> dict:
        return {
            "schema_version": "mahjong_full_log.v1",
            "session": asdict(self.summary),
            "engine_version": self.table.engine_version,
            "rule_id": self.table.state.rule_id,
            "rule_config_hash": self.table.state.rule_config_hash,
            "wall_provider": self.table.wall_provider.provider_id,
            "seed": self.table.seed,
            "dealer": int(self.table.state.dealer),
            "controllers": asdict(self.summary)["controllers"],
            "frames": tuple(self._full_log_frames),
            "replay": self.replay() if self.table.state.is_terminal else None,
        }

    def _record_frame(
        self,
        *,
        input_action: ActionDescriptor | None,
        before_event_count: int,
        viewer: Seat | None = None,
        controller_decision: LLMDecisionTrace | None = None,
    ) -> dict:
        frame = self._build_step_result(
            input_action=input_action,
            before_event_count=before_event_count,
            viewer=viewer,
            controller_decision=controller_decision,
        )
        log_frame = {
            "step_id": len(self._full_log_frames),
            "input": frame["input"],
            "events": frame["events"],
            "full_state": frame["full_state"],
            "legal": frame["legal"],
            "pending": frame["pending"],
            "terminal": frame["terminal"],
        }
        if "controller_decision" in frame:
            log_frame["controller_decision"] = frame["controller_decision"]
        self._full_log_frames.append(log_frame)
        return frame

    def _build_step_result(
        self,
        *,
        input_action: ActionDescriptor | None,
        before_event_count: int,
        viewer: Seat | None,
        controller_decision: LLMDecisionTrace | None,
    ) -> dict:
        decision = asdict(self.decision_snapshot())
        new_events = tuple(
            serialize_event(event)
            for event in self.table.state.events[before_event_count:]
        )
        terminal = self.table.state.terminal_result
        result = {
            "schema_version": "mahjong_step_result.v1",
            "step_id": len(self._full_log_frames),
            "input": (
                {
                    "kind": input_action.kind,
                    "operation": input_action.kind,
                    "actor": input_action.actor,
                    "tile": input_action.tile,
                    "source": input_action.source,
                    "metadata": dict(input_action.metadata),
                }
                if input_action is not None
                else None
            ),
            "session": asdict(self.summary),
            "state": serialize_state(self.table.state, viewer=viewer),
            "full_state": serialize_full_state(self.table.state),
            "legal": {
                "decision_actors": decision["decision_actors"],
                "response_window": decision["response_window"],
                "actions": decision["legal_actions"],
            },
            "events": new_events,
            "pending": {
                "responses": decision["pending_responses"],
                "response_window_open": self.table.state.response_window is not None,
            },
            "terminal": (
                {
                    "reason": terminal.reason,
                    "scores": terminal.scores,
                    "winners": tuple(int(seat) for seat in terminal.winners),
                    "responsible_seat": (
                        int(terminal.responsible_seat)
                        if terminal.responsible_seat is not None
                        else None
                    ),
                }
                if terminal is not None
                else None
            ),
        }
        if controller_decision is not None:
            result["controller_decision"] = {
                "seat": controller_decision.selected_action.actor,
                "controller_kind": self.controllers[
                    controller_decision.selected_action.actor
                ].kind.value,
                "provider": self.controllers[
                    controller_decision.selected_action.actor
                ].provider,
                "model_name": self.controllers[
                    controller_decision.selected_action.actor
                ].model_name,
                **controller_decision.as_log_dict(include_prompt=True),
            }
        return result


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, WebSession] = {}

    def list_rules(self) -> tuple[RuleSummary, ...]:
        summaries: list[RuleSummary] = []
        for rule_id, path in sorted(RULE_CONFIGS.items()):
            config = load_rule_config(path)
            summaries.append(
                RuleSummary(
                    rule_id=rule_id,
                    display_name=config.display_name,
                    config_hash=config.config_hash,
                    status=str(config.values.get("status", "unknown")),
                    implementation_status=(
                        str(config.get("notes.implementation_status"))
                        if config.get("notes.implementation_status") is not None
                        else None
                    ),
                )
            )
        return tuple(summaries)

    def create_session(self, request: CreateSessionRequest) -> WebSession:
        if request.rule_id not in RULE_CONFIGS:
            raise RuleConfigurationError(f"unsupported rule_id: {request.rule_id}")
        controllers = self._normalize_controllers(request.seat_controllers)
        rules = load_rule_plugin(RULE_CONFIGS[request.rule_id])
        wall = self._create_wall_provider(request.rule_id)
        session_id = f"session-{uuid4().hex}"
        table = TableEngine(
            rules,
            wall,
            seed=request.seed,
            dealer=request.dealer,
            table_id=session_id,
        )
        table.start()
        table.advance_to_decision()
        session = WebSession(
            session_id=session_id,
            table=table,
            controllers=controllers,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> WebSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session_id: {session_id}") from exc

    def list_sessions(self) -> tuple[SessionSummary, ...]:
        return tuple(session.summary for session in self._sessions.values())

    @staticmethod
    def _normalize_controllers(
        controllers: dict[Seat, SeatControllerConfig] | object,
    ) -> dict[Seat, SeatControllerConfig]:
        if not controllers:
            return {
                seat: SeatControllerConfig(SeatControllerKind.HUMAN)
                for seat in ALL_SEATS
            }
        normalized = dict(controllers)  # type: ignore[arg-type]
        missing = set(ALL_SEATS) - set(normalized)
        for seat in missing:
            normalized[seat] = SeatControllerConfig(SeatControllerKind.HUMAN)
        extra = set(normalized) - set(ALL_SEATS)
        if extra:
            raise RuleConfigurationError(f"unknown seats in controller config: {extra}")
        return normalized

    @staticmethod
    def _create_wall_provider(rule_id: str) -> DuplicateWallProvider:
        del rule_id
        return DuplicateWallProvider(reserve_dead_wall=False)
