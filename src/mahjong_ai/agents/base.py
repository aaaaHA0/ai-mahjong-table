from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mahjong_ai.common.action import Action
from mahjong_ai.common.event import TableEvent
from mahjong_ai.common.seat import Seat
from mahjong_ai.observation.schema import Observation


@dataclass(frozen=True)
class AgentContext:
    seat: Seat
    rule_id: str
    hand_id: str


@dataclass(frozen=True)
class AgentDecision:
    action: Action
    selected_log_prob: float | None = None
    value_prediction: float | None = None


class Agent(Protocol):
    agent_id: str

    def reset(self, context: AgentContext) -> None: ...

    def act(
        self, observation: Observation, legal_actions: tuple[Action, ...]
    ) -> AgentDecision: ...

    def on_event(self, event: TableEvent) -> None: ...
