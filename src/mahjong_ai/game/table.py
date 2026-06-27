from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TypeVar
from uuid import uuid4

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import (
    IllegalActionError,
    ReplayCompatibilityError,
    RuleConfigurationError,
)
from mahjong_ai.common.event import EventType, TableEvent
from mahjong_ai.common.seat import Seat
from mahjong_ai.game.invariant import validate_table_state
from mahjong_ai.game.operations import OperationResult
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.physical_table import PhysicalTableOps
from mahjong_ai.game.replay import (
    Replay,
    ReplayCommand,
    ReplayCommandKind,
)
from mahjong_ai.game.response_window import ResponseWindow
from mahjong_ai.game.snapshot import TableSnapshot
from mahjong_ai.game.state import TableState
from mahjong_ai.rules.base import (
    OperationPlan,
    PostActionDirective,
    PostActionKind,
    ResponseWindowSpec,
    RulePlugin,
    TerminalDirective,
)
from mahjong_ai.walls.base import WallProvider


T = TypeVar("T")


class TableEngine:
    engine_version = "table_engine.v3"

    def __init__(
        self,
        rules: RulePlugin,
        wall: WallProvider,
        *,
        seed: int,
        dealer: Seat = Seat.EAST,
        table_id: str | None = None,
        hand_id: str | None = None,
    ) -> None:
        if rules is None:
            raise RuleConfigurationError("TableEngine requires a RulePlugin")
        if wall is None:
            raise RuleConfigurationError("TableEngine requires a WallProvider")
        self._require_methods(
            rules,
            (
                "setup_hand",
                "create_player_rule_state",
                "draw_request",
                "on_draw_unavailable",
                "after_action",
                "after_unclaimed_response",
                "legal_actions",
                "validate_action",
                "legal_responses",
                "resolve_responses",
                "settle_win",
                "validate_rule_state",
                "build_rule_features",
            ),
            "RulePlugin",
        )
        self._require_methods(
            wall,
            ("initialize", "deal_initial", "draw"),
            "WallProvider",
        )
        self.rules = rules
        self.wall_provider = wall
        self.seed = seed
        self.state = TableState(
            table_id=table_id or f"table-{uuid4().hex}",
            hand_id=hand_id or f"hand-{uuid4().hex}",
            rule_id=rules.rule_id,
            rule_config_hash=rules.config.config_hash,
            dealer=dealer,
            current_actor=dealer,
        )
        self.physical_table = PhysicalTableOps(self.state, wall)
        self._commands: list[ReplayCommand] = []

    def start(self) -> None:
        self._atomic(self._start)
        self._commands.append(ReplayCommand(ReplayCommandKind.START))

    def _start(self) -> None:
        if self.state.phase is not GamePhase.INITIAL:
            raise RuntimeError("table has already started")
        self._emit(EventType.TABLE_CREATED)
        self._emit(
            EventType.RULES_LOADED,
            payload={
                "rule_id": self.rules.rule_id,
                "rule_config_hash": self.rules.config.config_hash,
            },
        )
        self._apply_operation_result(self.physical_table.initialize_wall(self.seed))

        setup = self.rules.setup_hand(self.state)
        if set(setup.deal_counts) != set(self.state.players):
            raise RuleConfigurationError("rule deal plan must include every table seat")
        for seat in self.state.players:
            self.state.players[seat].rule_state = self.rules.create_player_rule_state(
                self.state, seat
            )
            count = setup.deal_counts[seat]
            self._apply_operation_result(
                self.physical_table.deal_initial(seat, count)
            )
        self.state.current_actor = setup.initial_actor
        self.state.phase = setup.initial_phase
        self._validate()

    def advance_to_decision(self) -> bool:
        previous_event_count = len(self.state.events)

        def operation() -> bool:
            if self.state.phase is GamePhase.INITIAL:
                self._start()
            if self.state.is_terminal:
                return False
            if self.state.response_window is not None:
                return True
            request = self.rules.draw_request(self.state)
            if request is None:
                return True
            event_type = (
                EventType.KONG_REPLACEMENT_DRAWN
                if request.event_name == "kong_replacement_drawn"
                else EventType.TILE_DRAWN
            )
            draw_result = self.physical_table.draw(
                request.actor,
                request.kind,
                event_type,
            )
            self._apply_operation_result(draw_result)
            if draw_result.produced_tile is None:
                self._apply_terminal(
                    self.rules.on_draw_unavailable(self.state, request)
                )
                return False
            self._validate()
            return True

        result = self._atomic(operation)
        if len(self.state.events) != previous_event_count:
            self._commands.append(ReplayCommand(ReplayCommandKind.ADVANCE))
        return result

    @property
    def is_response_decision(self) -> bool:
        return self.state.response_window is not None

    def decision_actors(self) -> tuple[Seat, ...]:
        if self.state.response_window is not None:
            return self.state.response_window.eligible_seats
        return (self.state.current_actor,)

    def legal_actions(self, actor: Seat | None = None) -> tuple[Action, ...]:
        selected_actor = actor if actor is not None else self.state.current_actor
        if self.state.response_window is not None:
            return self.rules.legal_responses(
                self.state, self.state.response_window, selected_actor
            )
        return self.rules.legal_actions(self.state, selected_actor)

    def submit(self, action: Action) -> None:
        self._atomic(lambda: self._submit(action))
        self._commands.append(ReplayCommand(ReplayCommandKind.SUBMIT, action=action))

    def _submit(self, action: Action) -> None:
        if self.state.response_window is not None:
            raise IllegalActionError("use submit_responses while a response window is open")
        self.rules.validate_action(self.state, action)
        plan: OperationPlan
        if action.kind is ActionKind.DISCARD:
            result = self.physical_table.discard(action)
            self._apply_operation_result(result)
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, action, result.produced_tile)
            )
        elif action.kind is ActionKind.CONCEALED_KONG:
            self._apply_operation_result(
                self.physical_table.commit_concealed_kong(action)
            )
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, action, None)
            )
        elif action.kind is ActionKind.ADDED_KONG:
            result = self.physical_table.propose_added_kong(action)
            self._apply_operation_result(result)
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, action, result.produced_tile)
            )
        elif action.kind is ActionKind.DECLARE:
            result = self.physical_table.declare_tenpai(action)
            self._apply_operation_result(result)
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, action, result.produced_tile)
            )
        elif action.kind is ActionKind.WIN:
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, action, None)
            )
        else:
            raise IllegalActionError(f"unsupported self action: {action.kind}")
        self._apply_plan(plan)
        self._validate()

    def submit_responses(self, responses: Mapping[Seat, Action]) -> None:
        frozen = tuple(sorted(responses.items(), key=lambda item: int(item[0])))
        self._atomic(lambda: self._submit_responses(dict(frozen)))
        self._commands.append(
            ReplayCommand(ReplayCommandKind.SUBMIT_RESPONSES, responses=frozen)
        )

    def _submit_responses(self, responses: Mapping[Seat, Action]) -> None:
        window = self.state.response_window
        if window is None:
            raise IllegalActionError("no response window is open")
        resolution = self.rules.resolve_responses(self.state, window, responses)
        selected = resolution.selected_action
        self._emit(
            EventType.RESPONSES_RESOLVED,
            payload={
                "window_id": window.window_id,
                "submitted": {
                    str(int(seat)): action.kind.value
                    for seat, action in resolution.submitted_actions.items()
                },
                "selected_actor": int(selected.actor) if selected else None,
                "selected_action": selected.kind.value if selected else None,
            },
        )
        self.state.response_window = None

        if selected is None:
            if window.proposed_action is not None:
                self._apply_operation_result(
                    self.physical_table.commit_added_kong(window.proposed_action)
                )
            plan = self._coerce_operation_plan(
                self.rules.after_unclaimed_response(self.state, window)
            )
            self._apply_plan(plan)
        elif selected.kind is ActionKind.WIN:
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, selected, None)
            )
            self._apply_plan(plan)
        elif selected.kind in {
            ActionKind.CHI,
            ActionKind.PENG,
            ActionKind.EXPOSED_KONG,
        }:
            self._apply_operation_result(
                self.physical_table.commit_claim(window, selected)
            )
            plan = self._coerce_operation_plan(
                self.rules.after_action(self.state, selected, window.tile)
            )
            self._apply_plan(plan)
        else:
            raise IllegalActionError(f"unsupported selected response: {selected.kind}")
        self._validate()

    def snapshot(self) -> TableSnapshot:
        return TableSnapshot(
            engine_version=self.engine_version,
            rule_id=self.state.rule_id,
            rule_config_hash=self.state.rule_config_hash,
            state=deepcopy(self.state),
            commands=tuple(self._commands),
        )

    def restore_snapshot(self, snapshot: TableSnapshot) -> None:
        if snapshot.engine_version != self.engine_version:
            raise ReplayCompatibilityError("snapshot engine version mismatch")
        if snapshot.rule_id != self.rules.rule_id:
            raise ReplayCompatibilityError("snapshot rule ID mismatch")
        if snapshot.rule_config_hash != self.rules.config.config_hash:
            raise ReplayCompatibilityError("snapshot rule config hash mismatch")
        previous = deepcopy(self.state)
        previous_commands = list(self._commands)
        try:
            self.state = snapshot.clone_state()
            self.physical_table.bind_state(self.state)
            self._commands = list(snapshot.commands)
            self._validate()
        except Exception:
            self.state = previous
            self.physical_table.bind_state(self.state)
            self._commands = previous_commands
            raise

    def replay(self) -> Replay:
        if not self.state.is_terminal or self.state.terminal_result is None:
            raise RuntimeError("replay is only final after the hand ends")
        return Replay(
            engine_version=self.engine_version,
            rule_id=self.state.rule_id,
            rule_config_hash=self.state.rule_config_hash,
            wall_provider=self.wall_provider.provider_id,
            seed=self.seed,
            dealer=self.state.dealer,
            table_id=self.state.table_id,
            hand_id=self.state.hand_id,
            commands=tuple(self._commands),
            events=tuple(self.state.events),
            final_scores=self.state.terminal_result.scores,
        )

    @classmethod
    def restore_replay(
        cls, rules: RulePlugin, wall: WallProvider, replay: Replay
    ) -> "TableEngine":
        if replay.engine_version != cls.engine_version:
            raise ReplayCompatibilityError("replay engine version mismatch")
        if replay.rule_id != rules.rule_id:
            raise ReplayCompatibilityError("replay rule ID mismatch")
        if replay.rule_config_hash != rules.config.config_hash:
            raise ReplayCompatibilityError("replay rule config hash mismatch")
        if replay.wall_provider != wall.provider_id:
            raise ReplayCompatibilityError("replay wall provider mismatch")
        engine = cls(
            rules,
            wall,
            seed=replay.seed,
            dealer=replay.dealer,
            table_id=replay.table_id,
            hand_id=replay.hand_id,
        )
        for command in replay.commands:
            if command.kind is ReplayCommandKind.START:
                engine.start()
            elif command.kind is ReplayCommandKind.ADVANCE:
                engine.advance_to_decision()
            elif command.kind is ReplayCommandKind.SUBMIT:
                assert command.action is not None
                engine.submit(command.action)
            elif command.kind is ReplayCommandKind.SUBMIT_RESPONSES:
                engine.submit_responses(dict(command.responses))
        restored = engine.replay()
        if restored.events != replay.events or restored.final_scores != replay.final_scores:
            raise ReplayCompatibilityError("replayed state differs from recorded replay")
        return engine

    def _apply_plan(self, plan: OperationPlan) -> None:
        for directive in plan.directives:
            self._apply_directive(directive)

    @staticmethod
    def _coerce_operation_plan(
        plan_or_directive: OperationPlan | PostActionDirective,
    ) -> OperationPlan:
        if isinstance(plan_or_directive, OperationPlan):
            return plan_or_directive
        if isinstance(plan_or_directive, PostActionDirective):
            return OperationPlan.from_directive(plan_or_directive)
        raise RuleConfigurationError("rule hook must return OperationPlan")

    def _apply_directive(self, directive: PostActionDirective) -> None:
        if directive.kind in {
            PostActionKind.OPEN_DISCARD_RESPONSE,
            PostActionKind.OPEN_ADDED_KONG_RESPONSE,
        }:
            if directive.response_spec is None:
                raise RuleConfigurationError("response directive is missing its spec")
            self._open_response_window(directive.response_spec)
        elif directive.kind is PostActionKind.DRAW_KONG_REPLACEMENT:
            self.state.phase = GamePhase.WAITING_FOR_KONG_REPLACEMENT
        elif directive.kind is PostActionKind.WAIT_FOR_DRAW:
            if directive.next_actor is None:
                raise RuleConfigurationError("draw directive requires next_actor")
            self.state.current_actor = directive.next_actor
            self.state.phase = GamePhase.WAITING_FOR_DRAW
        elif directive.kind is PostActionKind.WAIT_FOR_DISCARD:
            self.state.phase = GamePhase.WAITING_FOR_DISCARD
        elif directive.kind is PostActionKind.SCORE_TRANSFERS:
            for transfer in directive.transfers:
                self._apply_operation_result(self.physical_table.transfer_score(transfer))
        elif directive.kind is PostActionKind.TERMINAL:
            if directive.terminal is None:
                raise RuleConfigurationError("terminal directive is missing terminal data")
            self._apply_terminal(directive.terminal)
        else:
            raise RuleConfigurationError(f"unknown post-action directive: {directive.kind}")

    def _open_response_window(self, spec: ResponseWindowSpec) -> None:
        provisional = ResponseWindow(
            window_id=len(self.state.events),
            kind=spec.kind,
            source=spec.source,
            tile=spec.tile,
            eligible_seats=spec.eligible_seats,
            legal_actions={},
            proposed_action=spec.proposed_action,
        )
        legal = {
            seat: self.rules.legal_responses(self.state, provisional, seat)
            for seat in spec.eligible_seats
        }
        self.state.response_window = ResponseWindow(
            window_id=provisional.window_id,
            kind=spec.kind,
            source=spec.source,
            tile=spec.tile,
            eligible_seats=spec.eligible_seats,
            legal_actions=legal,
            proposed_action=spec.proposed_action,
        )
        self.state.phase = (
            GamePhase.WAITING_FOR_ROB_KONG_RESPONSES
            if spec.proposed_action is not None
            else GamePhase.WAITING_FOR_DISCARD_RESPONSES
        )
        self._emit(
            EventType.RESPONSE_WINDOW_OPENED,
            actor=spec.source,
            payload={
                "window_id": provisional.window_id,
                "kind": spec.kind.value,
                "tile_type": spec.tile.tile_type.code,
                "eligible_seats": tuple(int(seat) for seat in spec.eligible_seats),
            },
        )

    def _apply_terminal(self, terminal: TerminalDirective) -> None:
        self._apply_operation_result(self.physical_table.apply_terminal(terminal))

    def _apply_operation_result(self, result: OperationResult) -> None:
        for event in result.events:
            self._emit(
                event.event_type,
                actor=event.actor,
                payload=event.payload,
                visible_to=event.visible_to,
            )

    def _validate(self) -> None:
        validate_table_state(self.state)
        self.rules.validate_rule_state(self.state)

    def _atomic(self, operation: Callable[[], T]) -> T:
        previous_state, previous_events, previous_event_count = (
            self._copy_transaction_state()
        )
        previous_commands = list(self._commands)
        try:
            return operation()
        except Exception:
            previous_state.events = previous_events[:previous_event_count]
            self.state = previous_state
            self.physical_table.bind_state(self.state)
            self._commands = previous_commands
            raise

    def _copy_transaction_state(
        self,
    ) -> tuple[TableState, list[TableEvent], int]:
        events = self.state.events
        event_count = len(events)
        self.state.events = []
        try:
            copied = deepcopy(self.state)
        finally:
            self.state.events = events
        return copied, events, event_count

    def _emit(
        self,
        event_type: EventType,
        *,
        actor: Seat | None = None,
        payload: Mapping[str, object] | None = None,
        visible_to: frozenset[Seat] | None = None,
    ) -> None:
        self.state.events.append(
            TableEvent(
                event_id=len(self.state.events),
                event_type=event_type,
                actor=actor,
                payload=payload or {},
                visible_to=visible_to,
            )
        )

    @staticmethod
    def _require_methods(
        dependency: object, methods: tuple[str, ...], dependency_name: str
    ) -> None:
        missing = [
            method
            for method in methods
            if not callable(getattr(dependency, method, None))
        ]
        if missing:
            raise RuleConfigurationError(
                f"{dependency_name} is missing required methods: {', '.join(missing)}"
            )
