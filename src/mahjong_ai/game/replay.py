from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mahjong_ai.common.action import Action
from mahjong_ai.common.event import TableEvent
from mahjong_ai.common.seat import Seat


class ReplayCommandKind(StrEnum):
    START = "start"
    ADVANCE = "advance"
    SUBMIT = "submit"
    SUBMIT_RESPONSES = "submit_responses"


@dataclass(frozen=True)
class ReplayCommand:
    kind: ReplayCommandKind
    action: Action | None = None
    responses: tuple[tuple[Seat, Action], ...] = ()


@dataclass(frozen=True)
class Replay:
    engine_version: str
    rule_id: str
    rule_config_hash: str
    wall_provider: str
    seed: int
    dealer: Seat
    table_id: str
    hand_id: str
    commands: tuple[ReplayCommand, ...]
    events: tuple[TableEvent, ...]
    final_scores: tuple[int, int, int, int]
