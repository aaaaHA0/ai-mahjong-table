from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.score import ScoreTransfer
from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import PhysicalTile
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.response_window import (
    ResponseResolution,
    ResponseWindow,
    ResponseWindowKind,
)
from mahjong_ai.game.state import TableState
from mahjong_ai.walls.base import DrawKind

_MISSING = object()


@dataclass(frozen=True)
class RuleConfig:
    rule_id: str
    display_name: str
    config_hash: str
    values: Mapping[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        value = self._get_from(self.values, path, _MISSING)
        if value is not _MISSING:
            return value
        params = self.values.get("params")
        if isinstance(params, Mapping):
            return self._get_from(params, path, default)
        return default

    @property
    def ruleset_impl(self) -> str | None:
        value = self.values.get("ruleset_impl")
        return value if isinstance(value, str) else None

    @property
    def training(self) -> Mapping[str, Any]:
        value = self.values.get("training")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _get_from(values: Mapping[str, Any], path: str, default: Any = None) -> Any:
        current: Any = values
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


@dataclass(frozen=True)
class HandSetup:
    deal_counts: Mapping[Seat, int]
    initial_actor: Seat
    initial_phase: GamePhase


@dataclass(frozen=True)
class DrawRequest:
    actor: Seat
    kind: DrawKind
    event_name: str


@dataclass(frozen=True)
class ResponseWindowSpec:
    kind: ResponseWindowKind
    source: Seat
    tile: PhysicalTile
    eligible_seats: tuple[Seat, ...]
    proposed_action: Action | None = None


@dataclass(frozen=True)
class TerminalDirective:
    reason: str
    winners: tuple[Seat, ...] = ()
    responsible_seat: Seat | None = None
    transfers: tuple[ScoreTransfer, ...] = ()


class PostActionKind(StrEnum):
    OPEN_DISCARD_RESPONSE = "open_discard_response"
    OPEN_ADDED_KONG_RESPONSE = "open_added_kong_response"
    DRAW_KONG_REPLACEMENT = "draw_kong_replacement"
    WAIT_FOR_DRAW = "wait_for_draw"
    WAIT_FOR_DISCARD = "wait_for_discard"
    SCORE_TRANSFERS = "score_transfers"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class PostActionDirective:
    kind: PostActionKind
    response_spec: ResponseWindowSpec | None = None
    terminal: TerminalDirective | None = None
    next_actor: Seat | None = None
    transfers: tuple[ScoreTransfer, ...] = ()


@dataclass(frozen=True)
class OperationPlan:
    """Rule-produced plan that TableEngine executes after a submitted action.

    P2 keeps the existing directive vocabulary as plan steps. This makes the
    rule/engine boundary explicit without changing the already-tested physical
    table operations, replay commands, or response-window semantics in the same
    migration.
    """

    directives: tuple[PostActionDirective, ...] = ()

    @classmethod
    def from_directive(cls, directive: PostActionDirective) -> "OperationPlan":
        return cls((directive,))


class RulePlugin(Protocol):
    rule_id: str
    config: RuleConfig

    def setup_hand(self, state: TableState) -> HandSetup: ...

    def create_player_rule_state(
        self, state: TableState, actor: Seat
    ) -> dict[str, Any]: ...

    def draw_request(self, state: TableState) -> DrawRequest | None: ...

    def on_draw_unavailable(
        self, state: TableState, request: DrawRequest
    ) -> TerminalDirective: ...

    def after_action(
        self, state: TableState, action: Action, tile: PhysicalTile | None
    ) -> OperationPlan: ...

    def after_unclaimed_response(
        self, state: TableState, window: ResponseWindow
    ) -> OperationPlan: ...

    def legal_actions(self, state: TableState, actor: Seat) -> tuple[Action, ...]: ...

    def validate_action(self, state: TableState, action: Action) -> None: ...

    def legal_responses(
        self, state: TableState, window: ResponseWindow, actor: Seat
    ) -> tuple[Action, ...]: ...

    def resolve_responses(
        self, state: TableState, window: ResponseWindow, responses: Mapping[Seat, Action]
    ) -> ResponseResolution: ...

    def settle_win(
        self, state: TableState, action: Action, source: Seat | None
    ) -> tuple[ScoreTransfer, ...]: ...

    def validate_rule_state(self, state: TableState) -> None: ...

    def build_rule_features(self, state: TableState, actor: Seat) -> Mapping[str, Any]: ...
