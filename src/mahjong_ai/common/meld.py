from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import PhysicalTile, TileType


class MeldKind(StrEnum):
    CHI = "chi"
    PENG = "peng"
    EXPOSED_KONG = "exposed_kong"
    CONCEALED_KONG = "concealed_kong"
    ADDED_KONG = "added_kong"


@dataclass
class Meld:
    kind: MeldKind
    tiles: tuple[PhysicalTile, ...]
    source: Seat | None = None
    claimed_tile: PhysicalTile | None = None

    @property
    def tile_type(self) -> TileType:
        return self.tiles[0].tile_type

    @property
    def effective_size(self) -> int:
        return 3

