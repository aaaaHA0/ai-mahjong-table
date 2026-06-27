from __future__ import annotations

import unittest
from pathlib import Path

from mahjong_ai.common.action import ActionKind
from mahjong_ai.game.response_window import ResponseWindowKind
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.base import OperationPlan, PostActionKind
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def make_table(seed: int = 42) -> TableEngine:
    table = TableEngine(
        load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml"),
        DuplicateWallProvider(),
        seed=seed,
    )
    table.advance_to_decision()
    return table


class OperationPlanTests(unittest.TestCase):
    def test_tuidaohe_rule_returns_operation_plan_after_discard(self) -> None:
        table = make_table()
        action = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )
        result = table.physical_table.discard(action)
        plan = table.rules.after_action(table.state, action, result.produced_tile)

        self.assertIsInstance(plan, OperationPlan)
        self.assertEqual(len(plan.directives), 1)
        self.assertEqual(
            plan.directives[0].kind,
            PostActionKind.OPEN_DISCARD_RESPONSE,
        )

    def test_table_engine_executes_rule_operation_plan(self) -> None:
        table = make_table()
        action = next(
            action
            for action in table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )

        table.submit(action)

        self.assertIsNotNone(table.state.response_window)
        assert table.state.response_window is not None
        self.assertEqual(
            table.state.response_window.kind,
            ResponseWindowKind.DISCARD,
        )


if __name__ == "__main__":
    unittest.main()
