from __future__ import annotations

import random

from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile, standard_136_tiles
from mahjong_ai.walls.base import DrawKind, WallState


class DuplicateWallProvider:
    """Four-seat deterministic subwalls for local table play."""

    provider_id = "duplicate_four_subwalls.v1"

    def __init__(self, *, reserve_dead_wall: bool = False) -> None:
        self.reserve_dead_wall = reserve_dead_wall

    def initialize(self, seed: int) -> WallState:
        tiles = standard_136_tiles()
        random.Random(seed).shuffle(tiles)
        subwalls = {seat: [] for seat in ALL_SEATS}
        for index, tile in enumerate(tiles):
            subwalls[Seat(index % 4)].append(tile)
        return WallState(provider_id=self.provider_id, seed=seed, subwalls=subwalls)

    def deal_initial(
        self, wall: WallState, seat: Seat, count: int
    ) -> tuple[PhysicalTile, ...]:
        if len(wall.subwalls[seat]) < count:
            raise ValueError(f"seat {seat} subwall cannot deal {count} tiles")
        dealt = tuple(wall.subwalls[seat][:count])
        del wall.subwalls[seat][:count]
        return dealt

    def draw(
        self, wall: WallState, seat: Seat, draw_kind: DrawKind
    ) -> PhysicalTile | None:
        subwall = wall.subwalls[seat]
        if draw_kind is DrawKind.KONG_REPLACEMENT:
            if not self.reserve_dead_wall:
                if not subwall:
                    return None
                tile = subwall.pop()
                wall.completed_kong_count += 1
                wall.drawn_tiles.append(tile)
                return tile
            reserve = self._reserved_tile_count(wall.completed_kong_count + 1)
            if len(subwall) <= reserve:
                return None
            tile = subwall.pop(len(subwall) - reserve - 1)
            wall.completed_kong_count += 1
            wall.drawn_tiles.append(tile)
            return tile

        if self.reserve_dead_wall and len(subwall) <= self._reserved_tile_count(
            wall.completed_kong_count
        ):
            return None
        if not subwall:
            return None
        tile = subwall.pop(0)
        wall.drawn_tiles.append(tile)
        return tile

    @staticmethod
    def _reserved_tile_count(completed_kong_count: int) -> int:
        return 14 + 2 * max(0, completed_kong_count - 1)
