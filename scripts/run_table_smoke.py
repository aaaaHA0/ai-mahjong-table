from __future__ import annotations

import argparse
from pathlib import Path

from mahjong_ai.agents.random import RandomAgent
from mahjong_ai.common.seat import ALL_SEATS
from mahjong_ai.game.match import MatchController
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.loader import load_rule_plugin
from mahjong_ai.walls.duplicate import DuplicateWallProvider


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic Tuidaohe table smoke hand")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rules = load_rule_plugin(ROOT / "configs/rules/tuidaohe_v1.yaml")
    table = TableEngine(
        rules,
        DuplicateWallProvider(),
        seed=args.seed,
        table_id="smoke-table",
        hand_id=f"smoke-{args.seed}",
    )
    agents = {
        seat: RandomAgent(args.seed * 10 + int(seat), f"random-seat-{int(seat)}")
        for seat in ALL_SEATS
    }
    result = MatchController(table).play_hand(agents)

    print(f"rule_id={result.replay.rule_id}")
    print(f"rule_config_hash={result.replay.rule_config_hash}")
    print(f"wall_provider={result.replay.wall_provider}")
    print(f"decisions={result.decisions}")
    print(f"terminal_reason={result.replay.events[-1].payload['reason']}")
    print(f"final_scores={result.replay.final_scores}")


if __name__ == "__main__":
    main()
