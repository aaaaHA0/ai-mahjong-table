from __future__ import annotations

import unittest
from pathlib import Path

from mahjong_ai.rules.loader import load_rule_plugin


ROOT = Path(__file__).resolve().parents[1]


class RuleLoadingTests(unittest.TestCase):
    def test_tuidaohe_config_loads_as_draft_rule(self) -> None:
        plugin = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")

        self.assertEqual(plugin.rule_id, "northern_tuidaohe.v1")
        self.assertEqual(plugin.config.ruleset_impl, "tuidaohe")
        self.assertEqual(plugin.config.get("players.count"), 4)
        self.assertFalse(plugin.config.get("actions.allow_chi"))
        self.assertEqual(plugin.config.get("winning.minimum_fan"), 0)
        self.assertTrue(plugin.config.get("response.multiple_winners"))
        self.assertEqual(
            plugin.config.training["rule_feature_version"],
            "tuidaohe_features.v1",
        )


if __name__ == "__main__":
    unittest.main()
