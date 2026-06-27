from __future__ import annotations

import unittest
from pathlib import Path

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.event import EventType
from mahjong_ai.common.errors import IllegalActionError
from mahjong_ai.common.meld import Meld, MeldKind
from mahjong_ai.common.score import ScoreTransfer
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile, TileType, all_tile_types
from mahjong_ai.game.operations import OperationKind, TileZone, TileZoneKind
from mahjong_ai.game.response_window import ResponseWindow, ResponseWindowKind
from mahjong_ai.game.state import Discard
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.base import DrawKind
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def make_table(seed: int = 42) -> TableEngine:
    table = TableEngine(
        load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
        DuplicateWallProvider(),
        seed=seed,
    )
    table.start()
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


def force_actor_copies(
    table: TableEngine,
    actor: Seat,
    tile_type: TileType,
    count: int,
    *,
    exclude_sources: frozenset[Seat] = frozenset(),
) -> None:
    assert table.state.wall_state is not None
    hand = table.state.players[actor].concealed_tiles
    containers: list[list[PhysicalTile]] = [
        table.state.players[seat].concealed_tiles
        for seat in ALL_SEATS
        if seat != actor and seat not in exclude_sources
    ]
    containers.extend(table.state.wall_state.subwalls.values())
    while sum(tile.tile_type == tile_type for tile in hand) < count:
        source = next(
            container
            for container in containers
            if any(tile.tile_type == tile_type for tile in container)
        )
        incoming = next(tile for tile in source if tile.tile_type == tile_type)
        outgoing = next(tile for tile in hand if tile.tile_type != tile_type)
        source[source.index(incoming)] = outgoing
        hand[hand.index(outgoing)] = incoming


class PhysicalTableOpsTests(unittest.TestCase):
    def test_discard_is_a_physical_operation(self) -> None:
        table = make_table()
        hand = table.state.players[Seat.EAST].concealed_tiles
        tile_type = hand[0].tile_type
        before_count = len(hand)
        before_events = len(table.state.events)

        result = table.physical_table.discard(
            Action(ActionKind.DISCARD, Seat.EAST, tile_type)
        )

        discarded = result.produced_tile
        self.assertEqual(result.kind, OperationKind.DISCARD)
        self.assertEqual(result.actor, Seat.EAST)
        self.assertEqual(result.event_types, (EventType.TILE_DISCARDED,))
        self.assertIsNotNone(discarded)
        self.assertEqual(discarded.tile_type, tile_type)
        self.assertEqual(len(hand), before_count - 1)
        self.assertEqual(table.state.players[Seat.EAST].discards[-1].tile, discarded)
        self.assertEqual(len(table.state.events), before_events)

    def test_draw_returns_operation_result(self) -> None:
        table = make_table()
        before_count = len(table.state.players[Seat.EAST].concealed_tiles)

        result = table.physical_table.draw(
            Seat.EAST,
            DrawKind.NORMAL,
            EventType.TILE_DRAWN,
        )

        self.assertEqual(result.kind, OperationKind.DRAW)
        self.assertEqual(result.actor, Seat.EAST)
        self.assertEqual(result.event_types, (EventType.TILE_DRAWN,))
        self.assertIsNotNone(result.produced_tile)
        self.assertEqual(
            len(table.state.players[Seat.EAST].concealed_tiles),
            before_count + 1,
        )

    def test_transfer_score_returns_operation_result(self) -> None:
        table = make_table()

        result = table.physical_table.transfer_score(
            ScoreTransfer(
                payer=Seat.EAST,
                receiver=Seat.SOUTH,
                amount=8,
                reason="unit_test",
            )
        )

        self.assertEqual(result.kind, OperationKind.TRANSFER_SCORE)
        self.assertEqual(result.actor, Seat.SOUTH)
        self.assertEqual(result.event_types, (EventType.SCORE_TRANSFERRED,))
        self.assertEqual(result.payload["amount"], 8)
        self.assertEqual(table.state.players[Seat.EAST].score, -8)
        self.assertEqual(table.state.players[Seat.SOUTH].score, 8)

    def test_roll_dice_returns_operation_result(self) -> None:
        table = make_table()

        result = table.physical_table.roll_dice(
            actor=Seat.EAST,
            dice_count=2,
            sides=6,
            values=(3, 5),
            reason="dealer_break_wall",
        )

        self.assertEqual(result.kind, OperationKind.ROLL_DICE)
        self.assertEqual(result.actor, Seat.EAST)
        self.assertEqual(result.event_types, (EventType.DICE_ROLLED,))
        self.assertEqual(result.payload["values"], (3, 5))
        self.assertEqual(result.payload["total"], 8)
        self.assertEqual(table.state.dice_rolls[-1].values, (3, 5))

    def test_move_tile_between_zones_returns_operation_result(self) -> None:
        table = make_table()
        tile = table.state.players[Seat.EAST].concealed_tiles[0]
        before_hand = len(table.state.players[Seat.EAST].concealed_tiles)
        before_revealed = len(table.state.revealed_tiles)

        result = table.physical_table.move_tile(
            TileZone(TileZoneKind.CONCEALED, Seat.EAST),
            TileZone(TileZoneKind.REVEALED),
            actor=Seat.EAST,
            tile=tile,
            reason="unit_test_reveal_area",
        )

        self.assertEqual(result.kind, OperationKind.MOVE_TILE)
        self.assertEqual(result.event_types, (EventType.TILE_MOVED,))
        self.assertEqual(result.produced_tile, tile)
        self.assertEqual(len(table.state.players[Seat.EAST].concealed_tiles), before_hand - 1)
        self.assertEqual(len(table.state.revealed_tiles), before_revealed + 1)
        self.assertIn(tile, table.state.revealed_tiles)

    def test_reveal_tile_returns_operation_result_without_moving_tile(self) -> None:
        table = make_table()
        tile = table.state.players[Seat.EAST].concealed_tiles[0]
        before_hand = tuple(table.state.players[Seat.EAST].concealed_tiles)

        result = table.physical_table.reveal_tile(
            tile,
            actor=Seat.EAST,
            reason="unit_test_show",
            visible_to=frozenset({Seat.EAST, Seat.SOUTH}),
        )

        self.assertEqual(result.kind, OperationKind.REVEAL_TILE)
        self.assertEqual(result.event_types, (EventType.TILE_REVEALED,))
        self.assertEqual(result.produced_tile, tile)
        self.assertEqual(tuple(table.state.players[Seat.EAST].concealed_tiles), before_hand)
        self.assertEqual(result.events[0].visible_to, frozenset({Seat.EAST, Seat.SOUTH}))

    def test_commit_claim_returns_operation_result(self) -> None:
        table = make_table()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        force_actor_copies(
            table,
            Seat.NORTH,
            tile_type,
            2,
            exclude_sources=frozenset({Seat.EAST}),
        )
        discard_tile = next(
            tile
            for tile in table.state.players[Seat.EAST].concealed_tiles
            if tile.tile_type == tile_type
        )
        table.state.players[Seat.EAST].concealed_tiles.remove(discard_tile)
        table.state.players[Seat.EAST].discards.append(Discard(discard_tile))
        action = Action(ActionKind.PENG, Seat.NORTH, tile_type, Seat.EAST)
        window = ResponseWindow(
            window_id=1,
            kind=ResponseWindowKind.DISCARD,
            source=Seat.EAST,
            tile=discard_tile,
            eligible_seats=(Seat.NORTH,),
            legal_actions={Seat.NORTH: (action,)},
        )

        result = table.physical_table.commit_claim(window, action)

        self.assertEqual(result.kind, OperationKind.COMMIT_CLAIM)
        self.assertEqual(result.actor, Seat.NORTH)
        self.assertEqual(result.event_types, (EventType.MELD_COMMITTED,))
        self.assertIsNotNone(result.produced_meld)
        self.assertEqual(result.produced_meld.kind, MeldKind.PENG)
        self.assertEqual(table.state.players[Seat.EAST].discards[-1].claimed_by, Seat.NORTH)

    def test_commit_added_kong_returns_operation_result(self) -> None:
        table = make_table()
        tile_type = force_four_copies_in_actor_hand(table, Seat.EAST)
        hand = table.state.players[Seat.EAST].concealed_tiles
        peng_tiles = tuple(tile for tile in list(hand) if tile.tile_type == tile_type)[:3]
        for tile in peng_tiles:
            hand.remove(tile)
        table.state.players[Seat.EAST].melds.append(
            Meld(MeldKind.PENG, peng_tiles, source=Seat.SOUTH)
        )

        result = table.physical_table.commit_added_kong(
            Action(ActionKind.ADDED_KONG, Seat.EAST, tile_type)
        )

        self.assertEqual(result.kind, OperationKind.COMMIT_ADDED_KONG)
        self.assertEqual(result.actor, Seat.EAST)
        self.assertEqual(result.event_types, (EventType.KONG_COMMITTED,))
        self.assertIsNotNone(result.produced_tile)
        self.assertIsNotNone(result.produced_meld)
        self.assertEqual(result.produced_meld.kind, MeldKind.ADDED_KONG)

    def test_take_tiles_rejects_missing_tile(self) -> None:
        table = make_table()
        hand = table.state.players[Seat.EAST].concealed_tiles
        missing = next(
            tile_type
            for tile_type in all_tile_types()
            if all(tile.tile_type != tile_type for tile in hand)
        )

        with self.assertRaises(IllegalActionError):
            table.physical_table.take_tiles(Seat.EAST, missing, 1)

    def test_table_engine_emits_operation_result_events(self) -> None:
        table = make_table()
        table.advance_to_decision()
        action = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )
        before_events = len(table.state.events)

        table.submit(action)

        new_event_types = tuple(
            event.event_type for event in table.state.events[before_events:]
        )
        self.assertIn(EventType.TILE_DISCARDED, new_event_types)

    def test_physical_table_rebinds_after_engine_rollback(self) -> None:
        table = make_table()
        table.advance_to_decision()
        action = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )

        original_after_action = table.rules.after_action

        def fail_after_action(state, submitted, tile):
            del state, submitted, tile
            raise RuntimeError("forced failure")

        table.rules.after_action = fail_after_action  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            table.submit(action)
        table.rules.after_action = original_after_action  # type: ignore[method-assign]

        self.assertIs(table.physical_table.state, table.state)
        table.submit(action)
        self.assertIs(table.physical_table.state, table.state)


if __name__ == "__main__":
    unittest.main()
