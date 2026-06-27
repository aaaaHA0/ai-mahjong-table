from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mahjong_ai.common.action import Action
from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import PhysicalTile


class ResponseWindowKind(StrEnum):
    DISCARD = "discard"
    ROB_ADDED_KONG = "rob_added_kong"


@dataclass(frozen=True)
class ResponseWindow:
    window_id: int
    kind: ResponseWindowKind
    source: Seat
    tile: PhysicalTile
    eligible_seats: tuple[Seat, ...]
    legal_actions: dict[Seat, tuple[Action, ...]]
    proposed_action: Action | None = None


@dataclass(frozen=True)
class ResponseResolution:
    selected_action: Action | None
    submitted_actions: dict[Seat, Action]

