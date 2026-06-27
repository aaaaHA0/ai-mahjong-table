from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from mahjong_ai.agents.base import Agent, AgentContext, AgentDecision
from mahjong_ai.agents.executor import AgentExecutor, DirectAgentExecutor
from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import AgentExecutionError
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.game.replay import Replay
from mahjong_ai.game.table import TableEngine
from mahjong_ai.observation.builder import ObservationBuilder


class AgentFailurePolicy(StrEnum):
    RAISE = "raise"
    PASS_IF_LEGAL = "pass_if_legal"
    FIRST_LEGAL = "first_legal"


@dataclass(frozen=True)
class HandResult:
    replay: Replay
    decisions: int
    agent_failures: int = 0


class MatchController:
    def __init__(
        self,
        table: TableEngine,
        observation_builder: ObservationBuilder | None = None,
        *,
        agent_executor: AgentExecutor | None = None,
        failure_policy: AgentFailurePolicy = AgentFailurePolicy.RAISE,
    ) -> None:
        self.table = table
        self.observation_builder = observation_builder or ObservationBuilder()
        self.agent_executor = agent_executor or DirectAgentExecutor()
        self.failure_policy = failure_policy

    def play_hand(self, agents: Mapping[Seat, Agent]) -> HandResult:
        if set(agents) != set(ALL_SEATS):
            raise ValueError("exactly one agent is required for each of four seats")
        for seat in ALL_SEATS:
            agents[seat].reset(
                AgentContext(
                    seat=seat,
                    rule_id=self.table.rules.rule_id,
                    hand_id=self.table.state.hand_id,
                )
            )

        decisions = 0
        failures = 0
        notified_event_count = 0
        while self.table.advance_to_decision():
            notified_event_count = self._notify_events(agents, notified_event_count)
            if self.table.is_response_decision:
                responses = {}
                for seat in self.table.decision_actors():
                    legal = self.table.legal_actions(seat)
                    responses[seat], failed = self._decide(agents[seat], seat, legal)
                    failures += int(failed)
                    decisions += 1
                self.table.submit_responses(responses)
            else:
                seat = self.table.state.current_actor
                legal = self.table.legal_actions()
                action, failed = self._decide(agents[seat], seat, legal)
                failures += int(failed)
                self.table.submit(action)
                decisions += 1
        self._notify_events(agents, notified_event_count)
        return HandResult(
            replay=self.table.replay(),
            decisions=decisions,
            agent_failures=failures,
        )

    def _decide(
        self, agent: Agent, seat: Seat, legal: tuple[Action, ...]
    ) -> tuple[Action, bool]:
        observation = self.observation_builder.build(
            self.table.state, seat, self.table.rules
        )
        try:
            decision = self.agent_executor.decide(agent, observation, legal)
            if decision.action not in legal:
                raise AgentExecutionError(
                    f"agent {agent.agent_id} returned an illegal action"
                )
            return decision.action, False
        except Exception as error:
            if self.failure_policy is AgentFailurePolicy.RAISE:
                if isinstance(error, AgentExecutionError):
                    raise
                raise AgentExecutionError(
                    f"agent {agent.agent_id} failed during decision"
                ) from error
            if self.failure_policy is AgentFailurePolicy.PASS_IF_LEGAL:
                fallback = next(
                    (action for action in legal if action.kind is ActionKind.PASS),
                    None,
                )
                if fallback is None:
                    raise AgentExecutionError(
                        f"agent {agent.agent_id} failed and PASS is not legal"
                    ) from error
                return fallback, True
            if not legal:
                raise AgentExecutionError("no legal fallback action") from error
            return legal[0], True

    def _notify_events(
        self, agents: Mapping[Seat, Agent], start_index: int
    ) -> int:
        events = self.table.state.events
        for event in events[start_index:]:
            for seat, agent in agents.items():
                if event.is_visible_to(seat):
                    agent.on_event(event)
        return len(events)
