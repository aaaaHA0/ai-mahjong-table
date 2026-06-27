from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Protocol

from mahjong_ai.agents.base import Agent, AgentDecision
from mahjong_ai.common.action import Action
from mahjong_ai.common.errors import AgentExecutionError
from mahjong_ai.observation.schema import Observation


class AgentExecutor(Protocol):
    def decide(
        self,
        agent: Agent,
        observation: Observation,
        legal_actions: tuple[Action, ...],
    ) -> AgentDecision: ...


@dataclass(frozen=True)
class DirectAgentExecutor:
    def decide(
        self,
        agent: Agent,
        observation: Observation,
        legal_actions: tuple[Action, ...],
    ) -> AgentDecision:
        return agent.act(observation, legal_actions)


@dataclass(frozen=True)
class TimeoutAgentExecutor:
    timeout_seconds: float

    def decide(
        self,
        agent: Agent,
        observation: Observation,
        legal_actions: tuple[Action, ...],
    ) -> AgentDecision:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(agent.act, observation, legal_actions)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeoutError as error:
                future.cancel()
                raise AgentExecutionError(
                    f"agent {agent.agent_id} exceeded {self.timeout_seconds}s"
                ) from error
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
