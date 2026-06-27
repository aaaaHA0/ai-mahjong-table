from __future__ import annotations

from mahjong_ai.common.meld import MeldKind
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.game.state import TableState
from mahjong_ai.observation.schema import Observation
from mahjong_ai.rules.base import RulePlugin


class ObservationBuilder:
    schema_version = "observation.table.v1"

    def build(
        self, state: TableState, viewer: Seat, rules: RulePlugin
    ) -> Observation:
        if state.wall_state is None:
            raise RuntimeError("wall is not initialized")
        return Observation(
            schema_version=self.schema_version,
            viewer=viewer,
            phase=state.phase,
            current_actor=state.current_actor,
            concealed_tiles=tuple(
                sorted(tile.tile_type for tile in state.players[viewer].concealed_tiles)
            ),
            discarded_tiles={
                seat: tuple(discard.tile.tile_type for discard in state.players[seat].discards)
                for seat in ALL_SEATS
            },
            public_discards={
                seat: tuple(discard.tile.tile_type for discard in state.players[seat].discards)
                for seat in ALL_SEATS
            },
            public_melds={
                seat: tuple(
                    (
                        meld.kind.value,
                        tuple(
                            tile.tile_type
                            for tile in meld.tiles
                            if meld.kind.value != "concealed_kong" or seat == viewer
                        ),
                    )
                    for meld in state.players[seat].melds
                )
                for seat in ALL_SEATS
            },
            known_other_player_tiles={
                seat: tuple(
                    (
                        meld.kind.value,
                        tuple(tile.tile_type for tile in meld.tiles),
                    )
                    for meld in state.players[seat].melds
                    if seat != viewer and meld.kind is not MeldKind.CONCEALED_KONG
                )
                for seat in ALL_SEATS
                if seat != viewer
            },
            wall_remaining_by_seat={
                seat: state.wall_state.remaining_for(seat) for seat in ALL_SEATS
            },
            response_window_kind=(
                state.response_window.kind.value if state.response_window else None
            ),
            rule_features=rules.build_rule_features(state, viewer),
        )
