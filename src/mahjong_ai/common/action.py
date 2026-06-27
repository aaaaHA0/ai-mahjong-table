from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import TileType


class ActionKind(StrEnum):
    DISCARD = "discard"
    CHI = "chi"
    PENG = "peng"
    EXPOSED_KONG = "exposed_kong"
    CONCEALED_KONG = "concealed_kong"
    ADDED_KONG = "added_kong"
    WIN = "win"
    PASS = "pass"
    DECLARE = "declare"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    actor: Seat
    tile: TileType | None = None
    source: Seat | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

