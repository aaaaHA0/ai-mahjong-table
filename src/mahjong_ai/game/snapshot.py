from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mahjong_ai.game.replay import ReplayCommand
from mahjong_ai.game.state import TableState


@dataclass(frozen=True)
class TableSnapshot:
    engine_version: str
    rule_id: str
    rule_config_hash: str
    state: TableState
    commands: tuple[ReplayCommand, ...] = ()

    def clone_state(self) -> TableState:
        return deepcopy(self.state)
