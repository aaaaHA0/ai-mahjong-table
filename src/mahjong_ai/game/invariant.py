from __future__ import annotations

from collections import Counter

from mahjong_ai.common.errors import StateInvariantError
from mahjong_ai.common.seat import ALL_SEATS
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.state import TableState


def validate_table_state(state: TableState) -> None:
    if state.wall_state is None:
        raise StateInvariantError("wall must be initialized")

    all_tiles = []
    for seat in ALL_SEATS:
        player = state.players[seat]
        all_tiles.extend(player.concealed_tiles)
        all_tiles.extend(player.tenpai_marker)
        all_tiles.extend(
            discard.tile for discard in player.discards if discard.claimed_by is None
        )
        all_tiles.extend(tile for meld in player.melds for tile in meld.tiles)
    for subwall in state.wall_state.subwalls.values():
        all_tiles.extend(subwall)
    all_tiles.extend(state.revealed_tiles)

    physical_ids = [tile.id for tile in all_tiles]
    if len(physical_ids) != state.physical_tile_count:
        raise StateInvariantError(
            f"tile conservation failed: {len(physical_ids)} != {state.physical_tile_count}"
        )
    if len(set(physical_ids)) != state.physical_tile_count:
        raise StateInvariantError("duplicate physical tile detected")

    counts = Counter(tile.tile_type for tile in all_tiles)
    invalid = {str(tile): count for tile, count in counts.items() if count > 4}
    if invalid:
        raise StateInvariantError(f"tile type counts are invalid: {invalid}")
    if state.phase in {
        GamePhase.WAITING_FOR_DISCARD_RESPONSES,
        GamePhase.WAITING_FOR_ROB_KONG_RESPONSES,
    }:
        if state.response_window is None:
            raise StateInvariantError("response phase requires an open response window")
