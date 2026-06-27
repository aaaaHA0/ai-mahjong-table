from __future__ import annotations

import unittest
from pathlib import Path

from mahjong_ai.agents.random import RandomAgent
from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import IllegalActionError
from mahjong_ai.common.event import EventType
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import TileType
from mahjong_ai.game.match import MatchController
from mahjong_ai.game.table import TableEngine
from mahjong_ai.observation.builder import ObservationBuilder
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def play(seed: int):
    rules = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")
    table = TableEngine(
        rules,
        DuplicateWallProvider(),
        seed=seed,
        table_id="test-table",
        hand_id=f"test-hand-{seed}",
    )
    agents = {seat: RandomAgent(seed * 10 + int(seat)) for seat in ALL_SEATS}
    return MatchController(table).play_hand(agents)


class TableEngineTests(unittest.TestCase):
    def test_same_seed_and_agents_produce_identical_public_replay(self) -> None:
        first = play(7)
        second = play(7)

        def public_signature(result):
            return [
                (
                    event.event_type,
                    event.actor,
                    event.payload.get("tile_type"),
                    event.payload.get("reason"),
                )
                for event in result.replay.events
            ]

        self.assertEqual(public_signature(first), public_signature(second))

    def test_private_deal_and_draw_events_are_not_public(self) -> None:
        result = play(3)
        private_events = [
            event
            for event in result.replay.events
            if event.event_type in {EventType.TILES_DEALT, EventType.TILE_DRAWN}
        ]
        self.assertTrue(private_events)
        self.assertTrue(all(event.visible_to == frozenset({event.actor}) for event in private_events))

    def test_rule_plugin_rejects_discard_not_in_actor_hand(self) -> None:
        rules = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")
        table = TableEngine(rules, DuplicateWallProvider(), seed=11)
        table.advance_to_decision()
        held_types = {
            tile.tile_type for tile in table.state.players[Seat.EAST].concealed_tiles
        }
        absent_type = next(
            TileType(code)
            for prefix, maximum in (("W", 9), ("B", 9), ("T", 9), ("F", 4), ("J", 3))
            for code in (f"{prefix}{value}" for value in range(1, maximum + 1))
            if TileType(code) not in held_types
        )

        with self.assertRaises(IllegalActionError):
            table.submit(Action(ActionKind.DISCARD, Seat.EAST, absent_type))

    def test_observation_does_not_expose_other_players_concealed_tiles(self) -> None:
        rules = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")
        table = TableEngine(rules, DuplicateWallProvider(), seed=17)
        table.advance_to_decision()
        builder = ObservationBuilder()
        before = builder.build(table.state, Seat.EAST, rules)

        assert table.state.wall_state is not None
        opponent_hand = table.state.players[Seat.NORTH].concealed_tiles
        opponent_hand[0], table.state.wall_state.subwalls[Seat.NORTH][0] = (
            table.state.wall_state.subwalls[Seat.NORTH][0],
            opponent_hand[0],
        )
        after = builder.build(table.state, Seat.EAST, rules)

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
