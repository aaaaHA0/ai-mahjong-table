from __future__ import annotations

import hashlib
from pathlib import Path

from mahjong_ai.common.errors import RuleConfigurationError
from mahjong_ai.rules.base import RuleConfig, RulePlugin
from mahjong_ai.rules.tuidaohe.ruleset import TuidaoheRulePlugin
from mahjong_ai.rules.yaml_subset import parse_yaml_subset


SUPPORTED_RULES = {
    "northern_tuidaohe.v1": TuidaoheRulePlugin,
}


def load_rule_config(path: str | Path) -> RuleConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    values = parse_yaml_subset(raw.decode("utf-8"))
    rule_id = values.get("rule_id")
    if not isinstance(rule_id, str):
        raise RuleConfigurationError(f"{config_path} does not define a string rule_id")
    return RuleConfig(
        rule_id=rule_id,
        display_name=str(values.get("display_name", rule_id)),
        config_hash=hashlib.sha256(raw).hexdigest(),
        values=values,
    )


def load_rule_plugin(path: str | Path) -> RulePlugin:
    config = load_rule_config(path)
    plugin_type = SUPPORTED_RULES.get(config.rule_id)
    if plugin_type is None:
        raise RuleConfigurationError(
            f"rule {config.rule_id!r} is not enabled; supported rules: "
            f"{', '.join(sorted(SUPPORTED_RULES))}"
        )
    return plugin_type(config)
