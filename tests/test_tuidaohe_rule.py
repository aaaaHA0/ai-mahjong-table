from __future__ import annotations

import unittest
from pathlib import Path

from mahjong_ai.agents.random import RandomAgent
from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.event import EventType
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile, TileType
from mahjong_ai.game.match import MatchController
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.base import DrawKind, WallState
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def make_table(seed: int = 42) -> TableEngine:
    return TableEngine(
        load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
        DuplicateWallProvider(),
        seed=seed,
    )


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


def set_started_actor_hand(
    table: TableEngine,
    actor: Seat,
    codes: tuple[str, ...],
) -> None:
    assert table.state.wall_state is not None
    all_tiles: list[PhysicalTile] = []
    for player in table.state.players.values():
        all_tiles.extend(player.concealed_tiles)
        player.concealed_tiles.clear()
    for subwall in table.state.wall_state.subwalls.values():
        all_tiles.extend(subwall)
        subwall.clear()

    by_type: dict[TileType, list[PhysicalTile]] = {}
    for tile in all_tiles:
        by_type.setdefault(tile.tile_type, []).append(tile)

    selected: list[PhysicalTile] = []
    for code in codes:
        selected.append(by_type[TileType(code)].pop())
    table.state.players[actor].concealed_tiles.extend(selected)

    remaining = [tile for tile_type in sorted(by_type) for tile in by_type[tile_type]]
    cursor = 0
    for seat in ALL_SEATS:
        if seat is actor:
            continue
        target_count = 13
        table.state.players[seat].concealed_tiles.extend(
            remaining[cursor : cursor + target_count]
        )
        cursor += target_count
    for seat in ALL_SEATS:
        target_count = 34 - (14 if seat is table.state.dealer else 13)
        table.state.wall_state.subwalls[seat].extend(
            remaining[cursor : cursor + target_count]
        )
        cursor += target_count


TENPAI_F1_HAND = (
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "B2",
    "B3",
    "B4",
    "T7",
    "T8",
    "T9",
    "F1",
    "F1",
)


LUXURY_SEVEN_PAIRS_TENPAI_HAND = (
    "W1",
    "W1",
    "W1",
    "W1",
    "W2",
    "W2",
    "W3",
    "W3",
    "W4",
    "W4",
    "W5",
    "W5",
    "W6",
    "W7",
)


def submit_all_pass(table: TableEngine) -> None:
    table.submit_responses(
        {
            seat: next(
                action
                for action in table.legal_actions(seat)
                if action.kind is ActionKind.PASS
            )
            for seat in table.decision_actors()
        }
    )


class TuidaoheRuleTests(unittest.TestCase):
    def test_basic_win_evaluators_accept_standard_seven_pairs_and_thirteen_orphans(self) -> None:
        table = make_table()
        plugin = table.rules
        rule_state = {"tenpai_declared": True}

        standard = tuple(
            TileType(code)
            for code in (
                "W1",
                "W2",
                "W3",
                "W4",
                "W5",
                "W6",
                "B2",
                "B3",
                "B4",
                "T7",
                "T8",
                "T9",
                "F1",
                "F1",
            )
        )
        seven_pairs = tuple(
            TileType(code)
            for code in (
                "W1",
                "W1",
                "W2",
                "W2",
                "B3",
                "B3",
                "B4",
                "B4",
                "T5",
                "T5",
                "T6",
                "T6",
                "J1",
                "J1",
            )
        )
        thirteen_orphans = tuple(
            TileType(code)
            for code in (
                "W1",
                "W9",
                "B1",
                "B9",
                "T1",
                "T9",
                "F1",
                "F2",
                "F3",
                "F4",
                "J1",
                "J2",
                "J3",
                "J3",
            )
        )

        self.assertTrue(plugin._can_win_tiles(standard, 0, rule_state))
        self.assertTrue(plugin._can_win_tiles(seven_pairs, 0, rule_state))
        self.assertTrue(plugin._can_win_tiles(thirteen_orphans, 0, rule_state))

    def test_start_deals_dealer_14_and_others_13(self) -> None:
        table = make_table()

        table.start()

        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_DISCARD)
        self.assertEqual(table.state.current_actor, Seat.EAST)
        self.assertEqual(len(table.state.players[Seat.EAST].concealed_tiles), 14)
        self.assertEqual(
            tuple(
                len(table.state.players[seat].concealed_tiles)
                for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH)
            ),
            (13, 13, 13),
        )

    def test_tuidaohe_disallows_chi_in_response_window(self) -> None:
        table = make_table()
        table.start()
        discard = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )

        table.submit(discard)

        self.assertIsNotNone(table.state.response_window)
        for seat in table.decision_actors():
            self.assertNotIn(
                ActionKind.CHI,
                {action.kind for action in table.legal_actions(seat)},
            )

    def test_declare_tenpai_moves_tile_to_marker_and_next_player_draws(self) -> None:
        table = make_table()
        table.start()
        set_started_actor_hand(table, Seat.EAST, TENPAI_F1_HAND)
        declare = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DECLARE and action.tile == TileType("F1")
        )
        before_hand = len(table.state.players[Seat.EAST].concealed_tiles)

        table.submit(declare)

        self.assertTrue(table.state.players[Seat.EAST].rule_state["tenpai_declared"])
        self.assertEqual(len(table.state.players[Seat.EAST].tenpai_marker), 1)
        self.assertEqual(len(table.state.players[Seat.EAST].concealed_tiles), before_hand - 1)
        self.assertEqual(table.state.current_actor, Seat.NORTH)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_DRAW)
        self.assertEqual(table.state.events[-1].event_type, EventType.TENPAI_DECLARED)
        self.assertIn("F1", table.state.players[Seat.EAST].rule_state["tenpai_waits"])

    def test_declared_tenpai_player_must_discard_drawn_tile(self) -> None:
        table = make_table()
        table.start()
        set_started_actor_hand(table, Seat.EAST, TENPAI_F1_HAND)
        declare = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DECLARE and action.tile == TileType("F1")
        )
        table.submit(declare)

        for _ in range(20):
            if table.state.current_actor is Seat.EAST and table.state.phase is GamePhase.WAITING_FOR_DRAW:
                break
            table.advance_to_decision()
            if table.state.response_window is not None:
                submit_all_pass(table)
            elif table.state.phase is GamePhase.WAITING_FOR_DISCARD:
                discard = next(
                    action
                    for action in table.legal_actions()
                    if action.kind is ActionKind.DISCARD
                )
                table.submit(discard)
                submit_all_pass(table)
        self.assertEqual(table.state.current_actor, Seat.EAST)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_DRAW)
        table.advance_to_decision()

        drawn_type = table.state.players[Seat.EAST].concealed_tiles[-1].tile_type
        discards = [
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        ]
        self.assertEqual({action.tile for action in discards}, {drawn_type})

    def test_tenpai_declaration_requires_actual_waits(self) -> None:
        table = make_table()
        table.start()

        self.assertNotIn(
            ActionKind.DECLARE,
            {action.kind for action in table.legal_actions()},
        )

    def test_passed_win_locks_discard_win_until_next_draw(self) -> None:
        table = make_table()
        table.start()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        table.state.players[Seat.NORTH].rule_state["tenpai_declared"] = True
        table.state.players[Seat.NORTH].rule_state["discard_winning_tiles"] = {
            tile_type.code: True
        }

        table.submit(Action(ActionKind.DISCARD, Seat.EAST, tile_type))
        responses = {
            seat: next(
                action
                for action in table.legal_actions(seat)
                if action.kind is ActionKind.PASS
            )
            for seat in table.decision_actors()
        }
        self.assertTrue(
            any(
                action.kind is ActionKind.WIN
                for action in table.legal_actions(Seat.NORTH)
            )
        )
        table.submit_responses(responses)

        self.assertTrue(table.state.players[Seat.NORTH].rule_state["passed_win_locked"])
        self.assertFalse(
            table.rules._can_discard_win(
                table.state.players[Seat.NORTH].concealed_tiles,
                len(table.state.players[Seat.NORTH].melds),
                tile_type,
                table.state.players[Seat.NORTH].rule_state,
            )
        )
        table.advance_to_decision()
        table.legal_actions(Seat.NORTH)
        self.assertFalse(table.state.players[Seat.NORTH].rule_state["passed_win_locked"])

    def test_declared_seven_pairs_wait_disallows_kong(self) -> None:
        table = make_table()
        table.start()
        set_started_actor_hand(table, Seat.EAST, LUXURY_SEVEN_PAIRS_TENPAI_HAND)
        declare = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DECLARE and action.tile == TileType("W7")
        )
        table.submit(declare)
        table.state.current_actor = Seat.EAST
        table.state.phase = GamePhase.WAITING_FOR_DISCARD
        assert table.state.wall_state is not None
        draw_tile = next(
            tile
            for subwall in table.state.wall_state.subwalls.values()
            for tile in subwall
            if tile.tile_type == TileType("W8")
        )
        for subwall in table.state.wall_state.subwalls.values():
            if draw_tile in subwall:
                subwall.remove(draw_tile)
                break
        table.state.players[Seat.EAST].concealed_tiles.append(draw_tile)

        self.assertNotIn(
            ActionKind.CONCEALED_KONG,
            {action.kind for action in table.legal_actions()},
        )

    def test_peng_response_is_available_and_selected(self) -> None:
        table = make_table()
        table.start()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        force_actor_copies(
            table,
            Seat.NORTH,
            tile_type,
            2,
            exclude_sources=frozenset({Seat.EAST}),
        )

        table.submit(Action(ActionKind.DISCARD, Seat.EAST, tile_type))
        responses = {
            seat: next(
                action
                for action in table.legal_actions(seat)
                if (
                    action.kind is ActionKind.PENG
                    if seat is Seat.NORTH
                    else action.kind is ActionKind.PASS
                )
            )
            for seat in table.decision_actors()
        }
        table.submit_responses(responses)

        self.assertEqual(table.state.current_actor, Seat.NORTH)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_DISCARD)

    def test_concealed_kong_scores_each_opponent_immediately(self) -> None:
        table = make_table()
        table.start()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        force_actor_copies(table, Seat.EAST, tile_type, 4)

        table.submit(Action(ActionKind.CONCEALED_KONG, Seat.EAST, tile_type))

        self.assertEqual(table.state.players[Seat.EAST].score, 3)
        self.assertEqual(table.state.players[Seat.NORTH].score, -1)
        self.assertEqual(table.state.players[Seat.WEST].score, -1)
        self.assertEqual(table.state.players[Seat.SOUTH].score, -1)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_KONG_REPLACEMENT)

    def test_exposed_kong_scores_discarder_immediately(self) -> None:
        table = make_table()
        table.start()
        assert table.state.wall_state is not None
        outside_east = [
            *table.state.players[Seat.NORTH].concealed_tiles,
            *table.state.players[Seat.WEST].concealed_tiles,
            *table.state.players[Seat.SOUTH].concealed_tiles,
            *(tile for subwall in table.state.wall_state.subwalls.values() for tile in subwall),
        ]
        outside_counts = {
            tile.tile_type: sum(candidate.tile_type == tile.tile_type for candidate in outside_east)
            for tile in outside_east
        }
        tile_type = next(
            tile.tile_type
            for tile in table.state.players[Seat.EAST].concealed_tiles
            if outside_counts.get(tile.tile_type, 0) >= 3
        )
        force_actor_copies(
            table,
            Seat.NORTH,
            tile_type,
            3,
            exclude_sources=frozenset({Seat.EAST}),
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

        self.assertEqual(table.state.players[Seat.EAST].score, -2)
        self.assertEqual(table.state.players[Seat.NORTH].score, 2)
        self.assertEqual(table.state.phase, GamePhase.WAITING_FOR_KONG_REPLACEMENT)

    def test_forced_discard_win_uses_basic_settlement(self) -> None:
        table = make_table()
        table.start()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        table.state.players[Seat.NORTH].rule_state["tenpai_declared"] = True
        table.state.players[Seat.NORTH].rule_state["discard_winning_tiles"] = {
            tile_type.code: True
        }

        table.submit(Action(ActionKind.DISCARD, Seat.EAST, tile_type))
        responses = {
            seat: next(
                action
                for action in table.legal_actions(seat)
                if (
                    action.kind is ActionKind.WIN
                    if seat is Seat.NORTH
                    else action.kind is ActionKind.PASS
                )
            )
            for seat in table.decision_actors()
        }
        table.submit_responses(responses)

        self.assertTrue(table.state.is_terminal)
        self.assertEqual(table.state.terminal_result.winners, (Seat.NORTH,))
        self.assertEqual(table.state.players[Seat.EAST].score, -1)
        self.assertEqual(table.state.players[Seat.NORTH].score, 1)

    def test_self_draw_multiplier_is_capped_per_winner(self) -> None:
        table = make_table()
        table.start()
        set_started_actor_hand(
            table,
            Seat.EAST,
            (
                "W1",
                "W1",
                "W1",
                "W1",
                "W2",
                "W2",
                "W3",
                "W3",
                "W4",
                "W4",
                "W5",
                "W5",
                "W6",
                "W6",
            ),
        )
        table.state.players[Seat.EAST].rule_state["tenpai_declared"] = True

        transfers = table.rules.settle_win(
            table.state,
            Action(
                ActionKind.WIN,
                Seat.EAST,
                TileType("W6"),
                metadata={"win_type": "self_draw"},
            ),
            None,
        )

        self.assertEqual(tuple(transfer.amount for transfer in transfers), (32, 32, 32))

    def test_discard_win_multiplier_uses_patterns_and_event(self) -> None:
        table = make_table()
        table.start()
        set_started_actor_hand(
            table,
            Seat.NORTH,
            (
                "W1",
                "W1",
                "W1",
                "W2",
                "W2",
                "W2",
                "W3",
                "W3",
                "W3",
                "W4",
                "W4",
                "W4",
                "W5",
            ),
        )
        table.state.players[Seat.NORTH].rule_state["tenpai_declared"] = True

        transfers = table.rules.settle_win(
            table.state,
            Action(
                ActionKind.WIN,
                Seat.NORTH,
                TileType("W5"),
                source=Seat.EAST,
                metadata={"win_type": "kong_discard"},
            ),
            Seat.EAST,
        )

        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].amount, 16)

    def test_reserved_dead_wall_blocks_normal_draw_and_uses_tail_replacement(self) -> None:
        provider = DuplicateWallProvider(reserve_dead_wall=True)
        subwall = [
            PhysicalTile(TileType("W1"), 0),
            *[PhysicalTile(TileType(f"B{value}"), copy_id) for value in range(1, 8) for copy_id in range(2)],
        ]
        wall = WallState(
            provider_id=provider.provider_id,
            seed=1,
            subwalls={
                Seat.EAST: list(subwall),
                Seat.NORTH: [],
                Seat.WEST: [],
                Seat.SOUTH: [],
            },
        )

        self.assertEqual(provider.draw(wall, Seat.EAST, DrawKind.NORMAL), subwall[0])
        self.assertIsNone(provider.draw(wall, Seat.EAST, DrawKind.NORMAL))

        wall.subwalls[Seat.EAST] = list(subwall)
        replacement = provider.draw(wall, Seat.EAST, DrawKind.KONG_REPLACEMENT)
        self.assertEqual(replacement, subwall[0])
        self.assertEqual(wall.completed_kong_count, 1)

    def test_multiple_discard_winners_are_settled_independently(self) -> None:
        table = make_table()
        table.start()
        tile_type = table.state.players[Seat.EAST].concealed_tiles[0].tile_type
        for winner in (Seat.NORTH, Seat.WEST):
            table.state.players[winner].rule_state["tenpai_declared"] = True
            table.state.players[winner].rule_state["discard_winning_tiles"] = {
                tile_type.code: True
            }

        table.submit(Action(ActionKind.DISCARD, Seat.EAST, tile_type))
        responses = {
            seat: next(
                action
                for action in table.legal_actions(seat)
                if (
                    action.kind is ActionKind.WIN
                    if seat in {Seat.NORTH, Seat.WEST}
                    else action.kind is ActionKind.PASS
                )
            )
            for seat in table.decision_actors()
        }
        table.submit_responses(responses)

        self.assertTrue(table.state.is_terminal)
        self.assertEqual(table.state.terminal_result.winners, (Seat.NORTH, Seat.WEST))
        self.assertEqual(table.state.players[Seat.EAST].score, -2)
        self.assertEqual(table.state.players[Seat.NORTH].score, 1)
        self.assertEqual(table.state.players[Seat.WEST].score, 1)

    def test_random_agents_can_finish_minimal_tuidaohe_hand(self) -> None:
        table = make_table(seed=7)
        agents = {seat: RandomAgent(700 + int(seat)) for seat in ALL_SEATS}

        result = MatchController(table).play_hand(agents)

        self.assertEqual(result.replay.rule_id, "northern_tuidaohe.v1")
        self.assertEqual(result.replay.final_scores, (0, 0, 0, 0))
        self.assertEqual(result.replay.events[-1].event_type, EventType.HAND_ENDED)

    def test_tuidaohe_random_stress_and_replay_restore(self) -> None:
        for seed in range(10, 20):
            table = make_table(seed=seed)
            agents = {seat: RandomAgent(seed * 100 + int(seat)) for seat in ALL_SEATS}

            result = MatchController(table).play_hand(agents)

            self.assertTrue(table.state.is_terminal)
            self.assertEqual(sum(result.replay.final_scores), 0)
            self.assertEqual(result.replay.rule_id, "northern_tuidaohe.v1")

            if seed == 10:
                restored = TableEngine.restore_replay(
                    load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
                    DuplicateWallProvider(),
                    result.replay,
                )
                self.assertEqual(restored.state, table.state)


if __name__ == "__main__":
    unittest.main()
