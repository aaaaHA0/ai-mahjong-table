from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.event import EventType, TableEvent
from mahjong_ai.common.meld import Meld
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import TileType
from mahjong_ai.game.state import Discard, PlayerState, TableState
from mahjong_ai.game.replay import Replay, ReplayCommand


class SeatControllerKind(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    RANDOM = "random"


@dataclass(frozen=True)
class SeatControllerConfig:
    kind: SeatControllerKind
    model_id: str | None = None
    provider: str | None = None
    base_url: str | None = None
    token: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class CreateSessionRequest:
    rule_id: str
    seed: int = 1
    dealer: Seat = Seat.EAST
    seat_controllers: Mapping[Seat, SeatControllerConfig] = field(
        default_factory=lambda: {
            seat: SeatControllerConfig(SeatControllerKind.HUMAN)
            for seat in ALL_SEATS
        }
    )


def normalize_action_metadata(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(normalize_action_metadata(item) for item in value)
    if isinstance(value, dict):
        return {
            key: normalize_action_metadata(item)
            for key, item in value.items()
        }
    return value


@dataclass(frozen=True)
class ActionDescriptor:
    kind: str
    actor: int
    tile: str | None = None
    source: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_action(cls, action: Action) -> "ActionDescriptor":
        return cls(
            kind=action.kind.value,
            actor=int(action.actor),
            tile=action.tile.code if action.tile is not None else None,
            source=int(action.source) if action.source is not None else None,
            metadata=dict(action.metadata),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ActionDescriptor":
        operation = payload.get("operation", payload.get("kind"))
        if not isinstance(operation, str):
            raise ValueError("action payload requires operation or kind")
        return cls(
            kind=operation,
            actor=int(payload["actor"]),
            tile=payload.get("tile"),
            source=payload.get("source"),
            metadata=payload.get("metadata", {}),
        )

    def to_action(self) -> Action:
        return Action(
            kind=ActionKind(self.kind),
            actor=Seat(self.actor),
            tile=TileType(self.tile) if self.tile is not None else None,
            source=Seat(self.source) if self.source is not None else None,
            metadata=normalize_action_metadata(dict(self.metadata)),
        )


@dataclass(frozen=True)
class RuleSummary:
    rule_id: str
    display_name: str
    config_hash: str
    status: str
    implementation_status: str | None


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    rule_id: str
    rule_config_hash: str
    wall_provider: str
    seed: int
    dealer: int
    phase: str
    current_actor: int
    is_terminal: bool
    event_count: int
    controllers: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_actors: tuple[int, ...]
    response_window: Mapping[str, Any] | None
    legal_actions: Mapping[int, tuple[ActionDescriptor, ...]]
    pending_responses: Mapping[int, ActionDescriptor]


def serialize_tile_type(tile_type: TileType | None) -> str | None:
    return tile_type.code if tile_type is not None else None


def serialize_discard(discard: Discard) -> dict[str, Any]:
    return {
        "tile": discard.tile.id,
        "tile_type": discard.tile.tile_type.code,
        "claimed_by": int(discard.claimed_by) if discard.claimed_by is not None else None,
        "claim_kind": discard.claim_kind,
    }


def serialize_meld(meld: Meld) -> dict[str, Any]:
    return {
        "kind": meld.kind.value,
        "tile_types": tuple(tile.tile_type.code for tile in meld.tiles),
        "tiles": tuple(tile.id for tile in meld.tiles),
        "source": int(meld.source) if meld.source is not None else None,
        "claimed_tile": meld.claimed_tile.id if meld.claimed_tile is not None else None,
    }


def serialize_player(player: PlayerState, *, viewer: Seat | None = None) -> dict[str, Any]:
    show_concealed = viewer is None or viewer is player.seat
    return {
        "seat": int(player.seat),
        "concealed_count": len(player.concealed_tiles),
        "concealed_tiles": (
            tuple(tile.tile_type.code for tile in player.concealed_tiles)
            if show_concealed
            else None
        ),
        "tenpai_marker_count": len(player.tenpai_marker),
        "tenpai_declared": bool(player.rule_state.get("tenpai_declared")),
        "melds": tuple(serialize_meld(meld) for meld in player.melds),
        "discards": tuple(serialize_discard(discard) for discard in player.discards),
        "score": player.score,
        "rule_state_public": {
            "tenpai_declared": bool(player.rule_state.get("tenpai_declared")),
            "passed_win_locked": bool(player.rule_state.get("passed_win_locked")),
        },
    }


def serialize_player_full(player: PlayerState) -> dict[str, Any]:
    return {
        "seat": int(player.seat),
        "concealed_count": len(player.concealed_tiles),
        "concealed_tiles": tuple(
            {"id": tile.id, "tile_type": tile.tile_type.code}
            for tile in player.concealed_tiles
        ),
        "tenpai_marker": tuple(
            {"id": tile.id, "tile_type": tile.tile_type.code}
            for tile in player.tenpai_marker
        ),
        "melds": tuple(serialize_meld(meld) for meld in player.melds),
        "discards": tuple(serialize_discard(discard) for discard in player.discards),
        "score": player.score,
        "rule_state": player.rule_state,
    }


def serialize_discard_stack(state: TableState) -> tuple[dict[str, Any], ...]:
    discards_by_tile_id: dict[str, tuple[Seat, int, Discard]] = {}
    for seat in ALL_SEATS:
        for index, discard in enumerate(state.players[seat].discards):
            discards_by_tile_id[discard.tile.id] = (seat, index, discard)

    stack: list[dict[str, Any]] = []
    for event in state.events:
        if event.event_type is not EventType.TILE_DISCARDED:
            continue
        tile_id = event.payload.get("tile")
        if not isinstance(tile_id, str) or tile_id not in discards_by_tile_id:
            continue
        seat, index, discard = discards_by_tile_id[tile_id]
        stack.append(
            {
                "event_id": event.event_id,
                "seat": int(seat),
                "seat_label": seat.name,
                "index": index,
                "tile": discard.tile.id,
                "tile_type": discard.tile.tile_type.code,
                "claimed_by": (
                    int(discard.claimed_by)
                    if discard.claimed_by is not None
                    else None
                ),
                "claim_kind": discard.claim_kind,
            }
        )
    return tuple(stack)


def serialize_event(event: TableEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "actor": int(event.actor) if event.actor is not None else None,
        "payload": event.payload,
        "visible_to": (
            tuple(int(seat) for seat in sorted(event.visible_to))
            if event.visible_to is not None
            else None
        ),
    }


def serialize_replay_command(command: ReplayCommand) -> dict[str, Any]:
    return {
        "kind": command.kind.value,
        "action": (
            as_action_dict(command.action)
            if command.action is not None
            else None
        ),
        "responses": tuple(
            {
                "seat": int(seat),
                "action": as_action_dict(action),
            }
            for seat, action in command.responses
        ),
    }


def as_action_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "operation": action.kind.value,
        "actor": int(action.actor),
        "tile": action.tile.code if action.tile is not None else None,
        "source": int(action.source) if action.source is not None else None,
        "metadata": dict(action.metadata),
    }


def serialize_response_window_full(state: TableState) -> dict[str, Any] | None:
    window = state.response_window
    if window is None:
        return None
    return {
        "window_id": window.window_id,
        "kind": window.kind.value,
        "source": int(window.source),
        "tile": window.tile.id,
        "tile_type": window.tile.tile_type.code,
        "eligible_seats": tuple(int(seat) for seat in window.eligible_seats),
        "legal_actions": {
            int(seat): tuple(as_action_dict(action) for action in actions)
            for seat, actions in window.legal_actions.items()
        },
        "proposed_action": (
            as_action_dict(window.proposed_action)
            if window.proposed_action is not None
            else None
        ),
    }


def serialize_full_state(state: TableState) -> dict[str, Any]:
    wall = state.wall_state
    return {
        "table_id": state.table_id,
        "hand_id": state.hand_id,
        "rule_id": state.rule_id,
        "rule_config_hash": state.rule_config_hash,
        "dealer": int(state.dealer),
        "current_actor": int(state.current_actor),
        "physical_tile_count": state.physical_tile_count,
        "phase": state.phase.value,
        "is_terminal": state.is_terminal,
        "players": {
            int(seat): serialize_player_full(player)
            for seat, player in state.players.items()
        },
        "wall": (
            {
                "provider_id": wall.provider_id,
                "seed": wall.seed,
                "remaining_count": wall.remaining_count,
                "remaining_by_seat": {
                    int(seat): wall.remaining_for(seat) for seat in ALL_SEATS
                },
                "subwalls": {
                    int(seat): tuple(
                        {"id": tile.id, "tile_type": tile.tile_type.code}
                        for tile in tiles
                    )
                    for seat, tiles in wall.subwalls.items()
                },
                "drawn_tiles": tuple(
                    {"id": tile.id, "tile_type": tile.tile_type.code}
                    for tile in wall.drawn_tiles
                ),
                "completed_kong_count": wall.completed_kong_count,
            }
            if wall is not None
            else None
        ),
        "response_window": serialize_response_window_full(state),
        "pending_kong_actor": (
            int(state.pending_kong_actor)
            if state.pending_kong_actor is not None
            else None
        ),
        "dice_rolls": tuple(
            {
                "values": roll.values,
                "actor": int(roll.actor) if roll.actor is not None else None,
                "reason": roll.reason,
                "total": roll.total,
            }
            for roll in state.dice_rolls
        ),
        "revealed_tiles": tuple(
            {"id": tile.id, "tile_type": tile.tile_type.code}
            for tile in state.revealed_tiles
        ),
        "score_ledger": tuple(
            {
                "payer": int(transfer.payer),
                "receiver": int(transfer.receiver),
                "amount": transfer.amount,
                "reason": transfer.reason,
                "source_event_id": transfer.source_event_id,
            }
            for transfer in state.score_ledger
        ),
        "discard_stack": serialize_discard_stack(state),
        "events": tuple(serialize_event(event) for event in state.events),
        "terminal_result": (
            {
                "reason": state.terminal_result.reason,
                "scores": state.terminal_result.scores,
                "winners": tuple(int(seat) for seat in state.terminal_result.winners),
                "responsible_seat": (
                    int(state.terminal_result.responsible_seat)
                    if state.terminal_result.responsible_seat is not None
                    else None
                ),
            }
            if state.terminal_result is not None
            else None
        ),
    }


def serialize_replay(replay: Replay) -> dict[str, Any]:
    return {
        "engine_version": replay.engine_version,
        "rule_id": replay.rule_id,
        "rule_config_hash": replay.rule_config_hash,
        "wall_provider": replay.wall_provider,
        "seed": replay.seed,
        "dealer": int(replay.dealer),
        "table_id": replay.table_id,
        "hand_id": replay.hand_id,
        "commands": tuple(serialize_replay_command(command) for command in replay.commands),
        "events": tuple(serialize_event(event) for event in replay.events),
        "final_scores": replay.final_scores,
    }


def serialize_state(state: TableState, *, viewer: Seat | None = None) -> dict[str, Any]:
    wall = state.wall_state
    return {
        "table_id": state.table_id,
        "hand_id": state.hand_id,
        "rule_id": state.rule_id,
        "rule_config_hash": state.rule_config_hash,
        "dealer": int(state.dealer),
        "current_actor": int(state.current_actor),
        "phase": state.phase.value,
        "is_terminal": state.is_terminal,
        "players": {
            int(seat): serialize_player(player, viewer=viewer)
            for seat, player in state.players.items()
        },
        "wall": (
            {
                "provider_id": wall.provider_id,
                "remaining_count": wall.remaining_count,
                "remaining_by_seat": {
                    int(seat): wall.remaining_for(seat) for seat in ALL_SEATS
                },
                "completed_kong_count": wall.completed_kong_count,
            }
            if wall is not None
            else None
        ),
        "response_window": (
            {
                "window_id": state.response_window.window_id,
                "kind": state.response_window.kind.value,
                "source": int(state.response_window.source),
                "tile_type": state.response_window.tile.tile_type.code,
                "eligible_seats": tuple(int(seat) for seat in state.response_window.eligible_seats),
            }
            if state.response_window is not None
            else None
        ),
        "score_ledger": tuple(
            {
                "payer": int(transfer.payer),
                "receiver": int(transfer.receiver),
                "amount": transfer.amount,
                "reason": transfer.reason,
            }
            for transfer in state.score_ledger
        ),
        "discard_stack": serialize_discard_stack(state),
        "terminal_result": (
            {
                "reason": state.terminal_result.reason,
                "scores": state.terminal_result.scores,
                "winners": tuple(int(seat) for seat in state.terminal_result.winners),
                "responsible_seat": (
                    int(state.terminal_result.responsible_seat)
                    if state.terminal_result.responsible_seat is not None
                    else None
                ),
            }
            if state.terminal_result is not None
            else None
        ),
        "events": tuple(
            serialize_event(event)
            for event in state.events
            if viewer is None or event.is_visible_to(viewer)
        ),
        "event_count": len(state.events),
    }
