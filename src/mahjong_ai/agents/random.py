from __future__ import annotations

import random

from mahjong_ai.agents.base import AgentContext, AgentDecision
from mahjong_ai.common.action import Action
from mahjong_ai.common.event import TableEvent
from mahjong_ai.observation.schema import Observation


class RandomAgent:
    def __init__(self, seed: int, agent_id: str | None = None) -> None:
        self.agent_id = agent_id or f"random-{seed}"
        self._rng = random.Random(seed)
        self._context: AgentContext | None = None

    def reset(self, context: AgentContext) -> None:
        self._context = context

    def act(
        self, observation: Observation, legal_actions: tuple[Action, ...]
    ) -> AgentDecision:
        del observation
        if self._context is None:
            raise RuntimeError("agent must be reset before acting")
        if not legal_actions:
            raise RuntimeError("agent received no legal actions")
        return AgentDecision(self._rng.choice(legal_actions))

    def on_event(self, event: TableEvent) -> None:
        del event
