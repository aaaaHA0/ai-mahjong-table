from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from mahjong_ai.common.seat import Seat


class EventType(StrEnum):
    TABLE_CREATED = "table_created"
    RULES_LOADED = "rules_loaded"
    WALL_INITIALIZED = "wall_initialized"
    TILES_SHUFFLED = "tiles_shuffled"
    DICE_ROLLED = "dice_rolled"
    TILES_DEALT = "tiles_dealt"
    TILE_DRAWN = "tile_drawn"
    TILE_MOVED = "tile_moved"
    TILE_REVEALED = "tile_revealed"
    TILE_DISCARDED = "tile_discarded"
    TENPAI_DECLARED = "tenpai_declared"
    RESPONSE_WINDOW_OPENED = "response_window_opened"
    RESPONSES_RESOLVED = "responses_resolved"
    MELD_COMMITTED = "meld_committed"
    KONG_PROPOSED = "kong_proposed"
    KONG_COMMITTED = "kong_committed"
    KONG_REPLACEMENT_DRAWN = "kong_replacement_drawn"
    WIN_DECLARED = "win_declared"
    SCORE_TRANSFERRED = "score_transferred"
    HAND_ENDED = "hand_ended"


@dataclass(frozen=True)
class TableEvent:
    event_id: int
    event_type: EventType
    actor: Seat | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    visible_to: frozenset[Seat] | None = None

    def is_visible_to(self, seat: Seat) -> bool:
        return self.visible_to is None or seat in self.visible_to
