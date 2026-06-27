from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import TileType
from mahjong_ai.game.phase import GamePhase


@dataclass(frozen=True)
class Observation:
    schema_version: str
    viewer: Seat
    phase: GamePhase
    current_actor: Seat
    concealed_tiles: tuple[TileType, ...]
    discarded_tiles: Mapping[Seat, tuple[TileType, ...]]
    public_discards: Mapping[Seat, tuple[TileType, ...]]
    public_melds: Mapping[Seat, tuple[tuple[str, tuple[TileType, ...]], ...]]
    known_other_player_tiles: Mapping[Seat, tuple[tuple[str, tuple[TileType, ...]], ...]]
    wall_remaining_by_seat: Mapping[Seat, int]
    response_window_kind: str | None
    rule_features: Mapping[str, Any]
