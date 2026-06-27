from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.common.seat import Seat


@dataclass(frozen=True)
class ScoreTransfer:
    payer: Seat
    receiver: Seat
    amount: int
    reason: str
    source_event_id: int | None = None
