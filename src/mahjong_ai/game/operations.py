from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from mahjong_ai.common.event import EventType
from mahjong_ai.common.meld import Meld
from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import PhysicalTile
from mahjong_ai.game.state import TerminalResult


class OperationKind(StrEnum):
    INITIALIZE_WALL = "initialize_wall"
    ROLL_DICE = "roll_dice"
    DEAL_INITIAL = "deal_initial"
    DRAW = "draw"
    MOVE_TILE = "move_tile"
    REVEAL_TILE = "reveal_tile"
    DISCARD = "discard"
    DECLARE_TENPAI = "declare_tenpai"
    COMMIT_CLAIM = "commit_claim"
    PROPOSE_ADDED_KONG = "propose_added_kong"
    COMMIT_ADDED_KONG = "commit_added_kong"
    COMMIT_CONCEALED_KONG = "commit_concealed_kong"
    TRANSFER_SCORE = "transfer_score"
    END_HAND = "end_hand"


class TileZoneKind(StrEnum):
    CONCEALED = "concealed"
    DISCARD = "discard"
    WALL = "wall"
    REVEALED = "revealed"


@dataclass(frozen=True)
class TileZone:
    kind: TileZoneKind
    seat: Seat | None = None

    def label(self) -> str:
        return (
            self.kind.value
            if self.seat is None
            else f"{self.kind.value}:{int(self.seat)}"
        )


@dataclass(frozen=True)
class EventDraft:
    event_type: EventType
    actor: Seat | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    visible_to: frozenset[Seat] | None = None


@dataclass(frozen=True)
class OperationResult:
    kind: OperationKind
    actor: Seat | None = None
    events: tuple[EventDraft, ...] = ()
    produced_tile: PhysicalTile | None = None
    produced_tiles: tuple[PhysicalTile, ...] = ()
    produced_meld: Meld | None = None
    terminal_result: TerminalResult | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def event_types(self) -> tuple[EventType, ...]:
        return tuple(event.event_type for event in self.events)
