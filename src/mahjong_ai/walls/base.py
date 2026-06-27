from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import PhysicalTile


class DrawKind(StrEnum):
    NORMAL = "normal"
    KONG_REPLACEMENT = "kong_replacement"


@dataclass
class WallState:
    provider_id: str
    seed: int
    subwalls: dict[Seat, list[PhysicalTile]] = field(default_factory=dict)
    drawn_tiles: list[PhysicalTile] = field(default_factory=list)
    completed_kong_count: int = 0

    @property
    def remaining_count(self) -> int:
        return sum(len(tiles) for tiles in self.subwalls.values())

    def remaining_for(self, seat: Seat) -> int:
        return len(self.subwalls[seat])


class WallProvider(Protocol):
    provider_id: str

    def initialize(self, seed: int) -> WallState: ...

    def deal_initial(
        self, wall: WallState, seat: Seat, count: int
    ) -> tuple[PhysicalTile, ...]: ...

    def draw(
        self, wall: WallState, seat: Seat, draw_kind: DrawKind
    ) -> PhysicalTile | None: ...
