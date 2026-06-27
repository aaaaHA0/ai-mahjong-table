from __future__ import annotations

import unittest
from pathlib import Path

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.event import EventType
from mahjong_ai.common.meld import Meld, MeldKind
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile, TileType
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.response_window import ResponseWindow, ResponseWindowKind
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def started_table(seed: int = 42) -> TableEngine:
    rules = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")
    table = TableEngine(rules, DuplicateWallProvider(), seed=seed)
    table.advance_to_decision()
    return table


def force_four_copies_in_actor_hand(table: TableEngine, actor: Seat) -> TileType:
    assert table.state.wall_state is not None
    hand = table.state.players[actor].concealed_tiles
    target = hand[0].tile_type

    containers: list[list[PhysicalTile]] = [
        table.state.players[seat].concealed_tiles
        for seat in ALL_SEATS
        if seat != actor
    ]
    containers.extend(table.state.wall_state.subwalls.values())
    while sum(tile.tile_type == target for tile in hand) < 4:
        source = next(
            container
            for container in containers
            if any(tile.tile_type == target for tile in container)
        )
        incoming = next(tile for tile in source if tile.tile_type == target)
        outgoing = next(tile for tile in hand if tile.tile_type != target)
        source[source.index(incoming)] = outgoing
        hand[hand.index(outgoing)] = incoming
    return target


def force_target_distribution(
    table: TableEngine,
    tile_type: TileType,
    desired: dict[Seat, int],
) -> None:
    assert table.state.wall_state is not None
    containers: list[tuple[Seat | None, list[PhysicalTile]]] = [
        (seat, table.state.players[seat].concealed_tiles) for seat in ALL_SEATS
    ]
    containers.extend((None, subwall) for subwall in table.state.wall_state.subwalls.values())

    for seat, wanted in desired.items():
        hand = table.state.players[seat].concealed_tiles
        while sum(tile.tile_type == tile_type for tile in hand) < wanted:
            source = next(
                container
                for owner, container in containers
                if owner != seat
                and (
                    owner not in desired
                    or sum(tile.tile_type == tile_type for tile in container)
                    > desired[owner]
                )
                and any(tile.tile_type == tile_type for tile in container)
            )
            incoming = next(tile for tile in source if tile.tile_type == tile_type)
            outgoing = next(tile for tile in hand if tile.tile_type != tile_type)
            source[source.index(incoming)] = outgoing
            hand[hand.index(outgoing)] = incoming

    for seat, wanted in desired.items():
        hand = table.state.players[seat].concealed_tiles
        while sum(tile.tile_type == tile_type for tile in hand) > wanted:
            destination = next(
                container
                for owner, container in containers
                if owner not in desired
                and any(tile.tile_type != tile_type for tile in container)
            )
            outgoing = next(tile for tile in hand if tile.tile_type == tile_type)
            incoming = next(tile for tile in destination if tile.tile_type != tile_type)
            destination[destination.index(incoming)] = outgoing
            hand[hand.index(outgoing)] = incoming


def pass_responses(table: TableEngine) -> None:
    responses = {
        seat: next(
            action
            for action in table.legal_actions(seat)
            if action.kind is ActionKind.PASS
        )
        for seat in table.decision_actors()
    }
    table.submit_responses(responses)


class TableActionTests(unittest.TestCase):
    def test_exposed_kong_claim_consumes_discard_and_draws_replacement(self) -> None:
        table = started_table()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        force_target_distribution(
            table, tile_type, {Seat.EAST: 1, Seat.NORTH: 3}
        )

        table.submit(Action(ActionKind.DISCARD, Seat.EAST, tile_type))
        responses = {
            seat: next(
                action
                for action in table.legal_actions(seat)
                if (
                    action.kind is ActionKind.EXPOSED_KONG
                    if seat is Seat.NORTH
                    else action.kind is ActionKind.PASS
                )
            )
            for seat in table.decision_actors()
        }
        table.submit_responses(responses)

        discard = table.state.players[Seat.EAST].discards[-1]
        self.assertEqual(discard.claimed_by, Seat.NORTH)
        self.assertEqual(discard.claim_kind, ActionKind.EXPOSED_KONG.value)
        self.assertEqual(
            table.state.players[Seat.NORTH].melds[0].kind,
            MeldKind.EXPOSED_KONG,
        )
        self.assertEqual(table.state.current_actor, Seat.NORTH)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_KONG_REPLACEMENT)
        table.advance_to_decision()
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_DISCARD)

    def test_concealed_kong_draws_replacement_for_same_actor(self) -> None:
        table = started_table()
        tile_type = force_four_copies_in_actor_hand(table, Seat.EAST)

        table.submit(Action(ActionKind.CONCEALED_KONG, Seat.EAST, tile_type))
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_KONG_REPLACEMENT)
        table.advance_to_decision()

        meld = table.state.players[Seat.EAST].melds[0]
        self.assertEqual(meld.kind, MeldKind.CONCEALED_KONG)
        self.assertEqual(len(meld.tiles), 4)
        self.assertEqual(table.state.current_actor, Seat.EAST)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_DISCARD)
        self.assertEqual(
            table.state.events[-1].event_type, EventType.KONG_REPLACEMENT_DRAWN
        )

    def test_response_resolution_prefers_win_then_nearest_equal_priority(self) -> None:
        table = started_table()
        tile = table.state.players[Seat.EAST].concealed_tiles[0]
        pass_north = Action(ActionKind.PASS, Seat.NORTH, tile.tile_type, Seat.EAST)
        pass_west = Action(ActionKind.PASS, Seat.WEST, tile.tile_type, Seat.EAST)
        pass_south = Action(ActionKind.PASS, Seat.SOUTH, tile.tile_type, Seat.EAST)
        peng_north = Action(ActionKind.PENG, Seat.NORTH, tile.tile_type, Seat.EAST)
        peng_west = Action(ActionKind.PENG, Seat.WEST, tile.tile_type, Seat.EAST)
        win_south = Action(
            ActionKind.WIN,
            Seat.SOUTH,
            tile.tile_type,
            Seat.EAST,
            {"fan": 8, "win_type": "discard"},
        )
        window = ResponseWindow(
            window_id=1,
            kind=ResponseWindowKind.DISCARD,
            source=Seat.EAST,
            tile=tile,
            eligible_seats=(Seat.NORTH, Seat.WEST, Seat.SOUTH),
            legal_actions={
                Seat.NORTH: (pass_north, peng_north),
                Seat.WEST: (pass_west, peng_west),
                Seat.SOUTH: (pass_south, win_south),
            },
        )

        resolution = table.rules.resolve_responses(
            table.state,
            window,
            {
                Seat.NORTH: peng_north,
                Seat.WEST: peng_west,
                Seat.SOUTH: win_south,
            },
        )
        self.assertEqual(resolution.selected_action.kind, ActionKind.WIN)
        self.assertEqual(resolution.selected_action.actor, Seat.SOUTH)

        no_win_window = ResponseWindow(
            window_id=2,
            kind=window.kind,
            source=window.source,
            tile=window.tile,
            eligible_seats=window.eligible_seats,
            legal_actions={
                Seat.NORTH: (pass_north, peng_north),
                Seat.WEST: (pass_west, peng_west),
                Seat.SOUTH: (pass_south,),
            },
        )
        resolution = table.rules.resolve_responses(
            table.state,
            no_win_window,
            {
                Seat.NORTH: peng_north,
                Seat.WEST: peng_west,
                Seat.SOUTH: pass_south,
            },
        )
        self.assertEqual(resolution.selected_action, peng_north)


if __name__ == "__main__":
    unittest.main()
