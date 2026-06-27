from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from pathlib import Path

from mahjong_ai.agents.base import AgentContext, AgentDecision
from mahjong_ai.agents.executor import TimeoutAgentExecutor
from mahjong_ai.agents.random import RandomAgent
from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import (
    AgentExecutionError,
    IllegalActionError,
    RuleConfigurationError,
)
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import TileType
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.match import AgentFailurePolicy, MatchController
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.base import (
    DrawRequest,
    HandSetup,
    RuleConfig,
    TerminalDirective,
)
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.base import DrawKind, WallState
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def make_table(seed: int = 42) -> TableEngine:
    return TableEngine(
        load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
        DuplicateWallProvider(),
        seed=seed,
        table_id="contract-table",
        hand_id=f"contract-hand-{seed}",
    )


class FailingAgent:
    agent_id = "failing"

    def reset(self, context: AgentContext) -> None:
        del context

    def act(self, observation, legal_actions) -> AgentDecision:
        del observation, legal_actions
        raise RuntimeError("intentional failure")

    def on_event(self, event) -> None:
        del event


class SlowAgent(FailingAgent):
    agent_id = "slow"

    def act(self, observation, legal_actions) -> AgentDecision:
        del observation
        time.sleep(0.05)
        return AgentDecision(legal_actions[0])


class TinyWallProvider:
    provider_id = "tiny-wall.v1"

    def initialize(self, seed: int) -> WallState:
        del seed
        tiles = DuplicateWallProvider().initialize(1)
        subwalls = {
            seat: [tiles.subwalls[seat][0]]
            for seat in ALL_SEATS
        }
        return WallState(self.provider_id, 0, subwalls)

    def deal_initial(self, wall: WallState, seat: Seat, count: int):
        dealt = tuple(wall.subwalls[seat][:count])
        del wall.subwalls[seat][:count]
        return dealt

    def draw(self, wall: WallState, seat: Seat, draw_kind: DrawKind):
        del draw_kind
        return wall.subwalls[seat].pop(0) if wall.subwalls[seat] else None


class LifecycleRule:
    rule_id = "test.lifecycle.v1"
    config = RuleConfig(rule_id, rule_id, "test-hash", {})

    def setup_hand(self, state):
        return HandSetup(
            deal_counts={seat: 1 for seat in ALL_SEATS},
            initial_actor=Seat.SOUTH,
            initial_phase=GamePhase.WAITING_FOR_DRAW,
        )

    def create_player_rule_state(self, state, actor):
        return {"seat": int(actor)}

    def draw_request(self, state):
        if state.phase is GamePhase.WAITING_FOR_DRAW:
            return DrawRequest(state.current_actor, DrawKind.NORMAL, "tile_drawn")
        return None

    def on_draw_unavailable(self, state, request):
        return TerminalDirective("custom_wall_exhausted")

    def validate_rule_state(self, state):
        return None

    def legal_actions(self, state, actor):
        return ()

    def validate_action(self, state, action):
        raise AssertionError("no actions expected")

    def legal_responses(self, state, window, actor):
        return ()

    def resolve_responses(self, state, window, responses):
        raise AssertionError("no responses expected")

    def settle_win(self, state, action, source):
        return ()

    def after_action(self, state, action, tile):
        raise AssertionError("no actions expected")

    def after_unclaimed_response(self, state, window):
        raise AssertionError("no responses expected")

    def build_rule_features(self, state, actor):
        return {"rule_id": self.rule_id}


class TableCoreContractTests(unittest.TestCase):
    def test_rule_plugin_and_wall_provider_are_required(self) -> None:
        rules = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")
        with self.assertRaises(RuleConfigurationError):
            TableEngine(None, DuplicateWallProvider(), seed=1)  # type: ignore[arg-type]
        with self.assertRaises(RuleConfigurationError):
            TableEngine(rules, None, seed=1)  # type: ignore[arg-type]
        with self.assertRaises(RuleConfigurationError):
            TableEngine(object(), DuplicateWallProvider(), seed=1)  # type: ignore[arg-type]

    def test_failed_rule_hook_rolls_back_entire_action(self) -> None:
        table = make_table()
        table.advance_to_decision()
        before = table.snapshot()
        action = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )
        before = table.snapshot()

        original = table.rules.after_action

        def fail_after_mutation(state, submitted, tile):
            del state, submitted, tile
            raise RuntimeError("rule hook failed")

        table.rules.after_action = fail_after_mutation  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            table.submit(action)
        table.rules.after_action = original  # type: ignore[method-assign]

        self.assertEqual(table.state, before.state)

    def test_snapshot_restores_full_table_state(self) -> None:
        table = make_table()
        table.advance_to_decision()
        snapshot = table.snapshot()
        table.submit(table.legal_actions()[0])
        self.assertNotEqual(table.state, snapshot.state)

        table.restore_snapshot(snapshot)

        self.assertEqual(table.state, snapshot.state)
        self.assertEqual(tuple(table._commands), snapshot.commands)

    def test_rule_controls_deal_plan_initial_actor_and_terminal_reason(self) -> None:
        table = TableEngine(
            LifecycleRule(),
            TinyWallProvider(),
            seed=1,
            dealer=Seat.EAST,
        )

        self.assertFalse(table.advance_to_decision())

        self.assertEqual(table.state.current_actor, Seat.SOUTH)
        self.assertEqual(
            tuple(len(table.state.players[seat].concealed_tiles) for seat in ALL_SEATS),
            (1, 1, 1, 1),
        )
        self.assertEqual(
            table.state.terminal_result.reason,
            "custom_wall_exhausted",
        )

    def test_completed_replay_restores_identical_state(self) -> None:
        table = make_table(seed=9)
        agents = {seat: RandomAgent(900 + int(seat)) for seat in ALL_SEATS}
        result = MatchController(table).play_hand(agents)

        restored = TableEngine.restore_replay(
            load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
            DuplicateWallProvider(),
            result.replay,
        )

        self.assertEqual(restored.state, table.state)
        self.assertEqual(restored.replay(), result.replay)

    def test_replay_restores_when_table_was_started_explicitly(self) -> None:
        table = make_table(seed=15)
        table.start()
        agents = {seat: RandomAgent(1500 + int(seat)) for seat in ALL_SEATS}
        result = MatchController(table).play_hand(agents)

        restored = TableEngine.restore_replay(
            load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
            DuplicateWallProvider(),
            result.replay,
        )

        self.assertEqual(restored.state, table.state)

    def test_agent_failure_can_fallback_to_first_legal_action(self) -> None:
        table = make_table(seed=4)
        agents = {seat: FailingAgent() for seat in ALL_SEATS}

        result = MatchController(
            table,
            failure_policy=AgentFailurePolicy.FIRST_LEGAL,
        ).play_hand(agents)

        self.assertTrue(result.replay.events)
        self.assertGreater(result.agent_failures, 0)

    def test_timeout_executor_reports_agent_timeout(self) -> None:
        table = make_table()
        table.advance_to_decision()
        legal = table.legal_actions()
        observation = MatchController(table).observation_builder.build(
            table.state, Seat.EAST, table.rules
        )
        agent = SlowAgent()
        agent.reset(
            AgentContext(Seat.EAST, table.rules.rule_id, table.state.hand_id)
        )

        with self.assertRaises(AgentExecutionError):
            TimeoutAgentExecutor(0.001).decide(agent, observation, legal)

    def test_tuidaohe_rule_contract_accepts_every_reported_legal_action(self) -> None:
        table = make_table(seed=12)
        table.advance_to_decision()
        legal = table.legal_actions()

        self.assertTrue(legal)
        for action in legal:
            table.rules.validate_action(table.state, action)

        absent = next(
            tile_type
            for tile_type in (TileType(f"W{value}") for value in range(1, 10))
            if all(
                tile.tile_type != tile_type
                for tile in table.state.players[Seat.EAST].concealed_tiles
            )
        )
        with self.assertRaises(IllegalActionError):
            table.rules.validate_action(
                table.state,
                Action(ActionKind.DISCARD, Seat.EAST, absent),
            )


if __name__ == "__main__":
    unittest.main()
