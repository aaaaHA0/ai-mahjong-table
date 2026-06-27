from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mahjong_ai.common.event import TableEvent
from mahjong_ai.common.meld import Meld
from mahjong_ai.common.score import ScoreTransfer
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.response_window import ResponseWindow
from mahjong_ai.walls.base import WallState


@dataclass
class Discard:
    tile: PhysicalTile
    claimed_by: Seat | None = None
    claim_kind: str | None = None


@dataclass
class PlayerState:
    seat: Seat
    concealed_tiles: list[PhysicalTile] = field(default_factory=list)
    tenpai_marker: list[PhysicalTile] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[Discard] = field(default_factory=list)
    score: int = 0
    rule_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminalResult:
    reason: str
    scores: tuple[int, int, int, int]
    winners: tuple[Seat, ...] = ()
    responsible_seat: Seat | None = None


@dataclass(frozen=True)
class DiceRoll:
    values: tuple[int, ...]
    actor: Seat | None = None
    reason: str = "unspecified"

    @property
    def total(self) -> int:
        return sum(self.values)


@dataclass
class TableState:
    table_id: str
    hand_id: str
    rule_id: str
    rule_config_hash: str
    dealer: Seat
    current_actor: Seat
    physical_tile_count: int = 0
    phase: GamePhase = GamePhase.INITIAL
    players: dict[Seat, PlayerState] = field(
        default_factory=lambda: {seat: PlayerState(seat) for seat in ALL_SEATS}
    )
    wall_state: WallState | None = None
    response_window: ResponseWindow | None = None
    pending_kong_actor: Seat | None = None
    dice_rolls: list[DiceRoll] = field(default_factory=list)
    revealed_tiles: list[PhysicalTile] = field(default_factory=list)
    score_ledger: list[ScoreTransfer] = field(default_factory=list)
    events: list[TableEvent] = field(default_factory=list)
    terminal_result: TerminalResult | None = None

    @property
    def is_terminal(self) -> bool:
        return self.phase is GamePhase.TERMINAL
