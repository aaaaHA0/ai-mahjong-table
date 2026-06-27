from __future__ import annotations

from collections import Counter
import random

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import IllegalActionError
from mahjong_ai.common.event import EventType
from mahjong_ai.common.meld import Meld, MeldKind
from mahjong_ai.common.score import ScoreTransfer
from mahjong_ai.common.seat import Seat
from mahjong_ai.common.tile import PhysicalTile, TileType
from mahjong_ai.game.operations import (
    EventDraft,
    OperationKind,
    OperationResult,
    TileZone,
    TileZoneKind,
)
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.response_window import ResponseWindow
from mahjong_ai.game.state import DiceRoll, Discard, TableState, TerminalResult
from mahjong_ai.rules.base import TerminalDirective
from mahjong_ai.walls.base import DrawKind, WallProvider


class PhysicalTableOps:
    """Rule-agnostic physical operations for a Mahjong table.

    This class moves tiles, mutates physical table state, transfers scores, and
    emits the same events that TableEngine used to emit directly. It does not
    decide whether an operation is legal under a ruleset; TableEngine must call
    RulePlugin before invoking these methods.
    """

    def __init__(
        self,
        state: TableState,
        wall_provider: WallProvider,
    ) -> None:
        self.state = state
        self.wall_provider = wall_provider

    def bind_state(self, state: TableState) -> None:
        self.state = state

    def initialize_wall(self, seed: int) -> OperationResult:
        self.state.wall_state = self.wall_provider.initialize(seed)
        self.state.physical_tile_count = self.state.wall_state.remaining_count
        event = EventDraft(
            EventType.WALL_INITIALIZED,
            payload={"provider": self.wall_provider.provider_id, "seed": seed},
        )
        return OperationResult(
            OperationKind.INITIALIZE_WALL,
            events=(event,),
            payload={"seed": seed},
        )

    def roll_dice(
        self,
        *,
        actor: Seat | None = None,
        dice_count: int = 2,
        sides: int = 6,
        values: tuple[int, ...] | None = None,
        reason: str = "unspecified",
    ) -> OperationResult:
        if dice_count <= 0:
            raise IllegalActionError("dice_count must be positive")
        if sides <= 0:
            raise IllegalActionError("dice sides must be positive")
        if values is None:
            rolled = tuple(random.randint(1, sides) for _ in range(dice_count))
        else:
            if len(values) != dice_count:
                raise IllegalActionError("dice values must match dice_count")
            if any(value < 1 or value > sides for value in values):
                raise IllegalActionError("dice value out of range")
            rolled = values
        dice_roll = DiceRoll(rolled, actor=actor, reason=reason)
        self.state.dice_rolls.append(dice_roll)
        payload = {
            "values": rolled,
            "total": dice_roll.total,
            "sides": sides,
            "reason": reason,
        }
        return OperationResult(
            OperationKind.ROLL_DICE,
            actor=actor,
            events=(EventDraft(EventType.DICE_ROLLED, actor=actor, payload=payload),),
            payload=payload,
        )

    def deal_initial(self, seat: Seat, count: int) -> OperationResult:
        if self.state.wall_state is None:
            raise IllegalActionError("wall is not initialized")
        dealt = self.wall_provider.deal_initial(self.state.wall_state, seat, count)
        self.state.players[seat].concealed_tiles.extend(dealt)
        payload = {"count": count, "tiles": tuple(tile.id for tile in dealt)}
        return OperationResult(
            OperationKind.DEAL_INITIAL,
            actor=seat,
            events=(
                EventDraft(
                    EventType.TILES_DEALT,
                    actor=seat,
                    payload=payload,
                    visible_to=frozenset({seat}),
                ),
            ),
            produced_tiles=tuple(dealt),
            payload={"count": count},
        )

    def draw(
        self,
        actor: Seat,
        draw_kind: DrawKind,
        event_type: EventType,
    ) -> OperationResult:
        if self.state.wall_state is None:
            raise IllegalActionError("wall is not initialized")
        tile = self.wall_provider.draw(self.state.wall_state, actor, draw_kind)
        if tile is None:
            return OperationResult(
                OperationKind.DRAW,
                actor=actor,
                payload={"draw_kind": draw_kind.value, "available": False},
            )
        self.state.players[actor].concealed_tiles.append(tile)
        self.state.current_actor = actor
        self.state.phase = GamePhase.WAITING_FOR_DISCARD
        self.state.pending_kong_actor = None
        payload = {"tile": tile.id}
        return OperationResult(
            OperationKind.DRAW,
            actor=actor,
            events=(
                EventDraft(
                    event_type,
                    actor=actor,
                    payload=payload,
                    visible_to=frozenset({actor}),
                ),
            ),
            produced_tile=tile,
            payload={"draw_kind": draw_kind.value, "available": True},
        )

    def discard(self, action: Action) -> OperationResult:
        tile = self.take_tiles(action.actor, action.tile, 1)[0]
        self.state.players[action.actor].discards.append(Discard(tile))
        payload = {"tile": tile.id, "tile_type": tile.tile_type.code}
        return OperationResult(
            OperationKind.DISCARD,
            actor=action.actor,
            events=(EventDraft(EventType.TILE_DISCARDED, actor=action.actor, payload=payload),),
            produced_tile=tile,
        )

    def declare_tenpai(self, action: Action) -> OperationResult:
        tile = self.take_tiles(action.actor, action.tile, 1)[0]
        self.state.players[action.actor].tenpai_marker.append(tile)
        payload = {"tile": tile.id, "tile_type": tile.tile_type.code}
        return OperationResult(
            OperationKind.DECLARE_TENPAI,
            actor=action.actor,
            events=(
                EventDraft(
                    EventType.TENPAI_DECLARED,
                    actor=action.actor,
                    payload=payload,
                    visible_to=frozenset({action.actor}),
                ),
            ),
            produced_tile=tile,
            payload=payload,
        )

    def move_tile(
        self,
        source: TileZone,
        destination: TileZone,
        *,
        actor: Seat | None = None,
        tile: PhysicalTile | None = None,
        tile_type: TileType | None = None,
        reason: str = "unspecified",
    ) -> OperationResult:
        moved = self._remove_from_zone(source, tile=tile, tile_type=tile_type)
        self._add_to_zone(destination, moved)
        payload = {
            "tile": moved.id,
            "tile_type": moved.tile_type.code,
            "source": source.label(),
            "destination": destination.label(),
            "reason": reason,
        }
        return OperationResult(
            OperationKind.MOVE_TILE,
            actor=actor,
            events=(EventDraft(EventType.TILE_MOVED, actor=actor, payload=payload),),
            produced_tile=moved,
            payload=payload,
        )

    def reveal_tile(
        self,
        tile: PhysicalTile,
        *,
        actor: Seat | None = None,
        reason: str = "unspecified",
        visible_to: frozenset[Seat] | None = None,
    ) -> OperationResult:
        if not self._contains_tile(tile):
            raise IllegalActionError(f"tile is not on this table: {tile.id}")
        payload = {
            "tile": tile.id,
            "tile_type": tile.tile_type.code,
            "reason": reason,
        }
        return OperationResult(
            OperationKind.REVEAL_TILE,
            actor=actor,
            events=(
                EventDraft(
                    EventType.TILE_REVEALED,
                    actor=actor,
                    payload=payload,
                    visible_to=visible_to,
                ),
            ),
            produced_tile=tile,
            payload=payload,
        )

    def commit_concealed_kong(self, action: Action) -> OperationResult:
        tiles = self.take_tiles(action.actor, action.tile, 4)
        meld = Meld(MeldKind.CONCEALED_KONG, tuple(tiles))
        self.state.players[action.actor].melds.append(meld)
        kong_result = self.emit_kong(action.actor, meld)
        return OperationResult(
            OperationKind.COMMIT_CONCEALED_KONG,
            actor=action.actor,
            events=kong_result.events,
            produced_tiles=tuple(tiles),
            produced_meld=meld,
        )

    def propose_added_kong(self, action: Action) -> OperationResult:
        tile = next(
            tile
            for tile in self.state.players[action.actor].concealed_tiles
            if tile.tile_type == action.tile
        )
        payload = {"kind": MeldKind.ADDED_KONG.value, "tile_type": tile.tile_type.code}
        return OperationResult(
            OperationKind.PROPOSE_ADDED_KONG,
            actor=action.actor,
            events=(EventDraft(EventType.KONG_PROPOSED, actor=action.actor, payload=payload),),
            produced_tile=tile,
        )

    def commit_claim(self, window: ResponseWindow, action: Action) -> OperationResult:
        discard = self.find_discard(window.source, window.tile)
        discard.claimed_by = action.actor
        discard.claim_kind = action.kind.value
        claimed = discard.tile
        if action.kind is ActionKind.CHI:
            sequence = tuple(TileType(code) for code in action.metadata["sequence"])
            needed = Counter(sequence)
            needed[claimed.tile_type] -= 1
            consumed = [
                tile
                for tile_type, count in needed.items()
                for tile in self.take_tiles(action.actor, tile_type, count)
            ]
            meld = Meld(
                MeldKind.CHI,
                tuple(sorted((*consumed, claimed))),
                source=window.source,
                claimed_tile=claimed,
            )
        elif action.kind is ActionKind.PENG:
            consumed = self.take_tiles(action.actor, claimed.tile_type, 2)
            meld = Meld(
                MeldKind.PENG,
                tuple((*consumed, claimed)),
                source=window.source,
                claimed_tile=claimed,
            )
        else:
            consumed = self.take_tiles(action.actor, claimed.tile_type, 3)
            meld = Meld(
                MeldKind.EXPOSED_KONG,
                tuple((*consumed, claimed)),
                source=window.source,
                claimed_tile=claimed,
            )
        self.state.players[action.actor].melds.append(meld)
        self.state.current_actor = action.actor
        payload = {
            "kind": meld.kind.value,
            "tile_types": tuple(tile.tile_type.code for tile in meld.tiles),
            "source": int(window.source),
        }
        events = (EventDraft(EventType.MELD_COMMITTED, actor=action.actor, payload=payload),)
        if action.kind is ActionKind.EXPOSED_KONG:
            kong_result = self.emit_kong(action.actor, meld)
            events = (*events, *kong_result.events)
        return OperationResult(
            OperationKind.COMMIT_CLAIM,
            actor=action.actor,
            events=events,
            produced_meld=meld,
        )

    def commit_added_kong(self, action: Action) -> OperationResult:
        added_tile = self.take_tiles(action.actor, action.tile, 1)[0]
        player = self.state.players[action.actor]
        meld = next(
            meld
            for meld in player.melds
            if meld.kind is MeldKind.PENG and meld.tile_type == action.tile
        )
        meld.kind = MeldKind.ADDED_KONG
        meld.tiles = (*meld.tiles, added_tile)
        kong_result = self.emit_kong(action.actor, meld)
        return OperationResult(
            OperationKind.COMMIT_ADDED_KONG,
            actor=action.actor,
            events=kong_result.events,
            produced_tile=added_tile,
            produced_meld=meld,
        )

    def emit_kong(self, actor: Seat, meld: Meld) -> OperationResult:
        self.state.current_actor = actor
        self.state.pending_kong_actor = actor
        payload = {
            "kind": meld.kind.value,
            "tile_type": meld.tile_type.code,
            "source": int(meld.source) if meld.source is not None else None,
        }
        return OperationResult(
            OperationKind.COMMIT_CONCEALED_KONG
            if meld.kind is MeldKind.CONCEALED_KONG
            else (
                OperationKind.COMMIT_ADDED_KONG
                if meld.kind is MeldKind.ADDED_KONG
                else OperationKind.COMMIT_CLAIM
            ),
            actor=actor,
            events=(EventDraft(EventType.KONG_COMMITTED, actor=actor, payload=payload),),
            produced_meld=meld,
        )

    def apply_terminal(self, terminal: TerminalDirective) -> OperationResult:
        events: list[EventDraft] = []
        for transfer in terminal.transfers:
            transfer_result = self.transfer_score(transfer)
            events.extend(transfer_result.events)
        if terminal.winners:
            payload = {
                "source": (
                    int(terminal.responsible_seat)
                    if terminal.responsible_seat is not None
                    else None
                )
            }
            events.append(
                EventDraft(
                    EventType.WIN_DECLARED,
                    actor=terminal.winners[0],
                    payload=payload,
                )
            )
        scores = tuple(
            self.state.players[seat].score for seat in sorted(self.state.players)
        )
        self.state.response_window = None
        self.state.terminal_result = TerminalResult(
            reason=terminal.reason,
            scores=scores,
            winners=terminal.winners,
            responsible_seat=terminal.responsible_seat,
        )
        self.state.phase = GamePhase.TERMINAL
        payload = {
            "reason": terminal.reason,
            "scores": scores,
            "winners": tuple(int(seat) for seat in terminal.winners),
        }
        events.append(EventDraft(EventType.HAND_ENDED, payload=payload))
        return OperationResult(
            OperationKind.END_HAND,
            actor=terminal.winners[0] if terminal.winners else None,
            events=tuple(events),
            terminal_result=self.state.terminal_result,
            payload={"reason": terminal.reason, "scores": scores},
        )

    def transfer_score(self, transfer: ScoreTransfer) -> OperationResult:
        self.state.players[transfer.payer].score -= transfer.amount
        self.state.players[transfer.receiver].score += transfer.amount
        self.state.score_ledger.append(transfer)
        payload = {
            "payer": int(transfer.payer),
            "receiver": int(transfer.receiver),
            "amount": transfer.amount,
            "reason": transfer.reason,
        }
        return OperationResult(
            OperationKind.TRANSFER_SCORE,
            actor=transfer.receiver,
            events=(EventDraft(EventType.SCORE_TRANSFERRED, payload=payload),),
            payload=payload,
        )

    def take_tiles(
        self, actor: Seat, tile_type: TileType | None, count: int
    ) -> list[PhysicalTile]:
        if tile_type is None:
            raise IllegalActionError("action requires a tile")
        hand = self.state.players[actor].concealed_tiles
        selected = [tile for tile in hand if tile.tile_type == tile_type][:count]
        if len(selected) != count:
            raise IllegalActionError(
                f"{actor.name} does not hold {count} copies of {tile_type}"
            )
        for tile in selected:
            hand.remove(tile)
        return selected

    def find_discard(self, source: Seat, tile: PhysicalTile) -> Discard:
        return next(
            discard
            for discard in reversed(self.state.players[source].discards)
            if discard.tile == tile and discard.claimed_by is None
        )

    def _remove_from_zone(
        self,
        zone: TileZone,
        *,
        tile: PhysicalTile | None,
        tile_type: TileType | None,
    ) -> PhysicalTile:
        container = self._zone_container(zone)
        selected = self._select_tile(container, tile=tile, tile_type=tile_type)
        if zone.kind is TileZoneKind.DISCARD:
            assert zone.seat is not None
            discards = self.state.players[zone.seat].discards
            discard = next(
                discard
                for discard in discards
                if discard.tile == selected and discard.claimed_by is None
            )
            discards.remove(discard)
            return selected
        container.remove(selected)
        return selected

    def _add_to_zone(self, zone: TileZone, tile: PhysicalTile) -> None:
        if zone.kind is TileZoneKind.DISCARD:
            if zone.seat is None:
                raise IllegalActionError("discard zone requires a seat")
            self.state.players[zone.seat].discards.append(Discard(tile))
            return
        container = self._zone_container(zone)
        container.append(tile)

    def _zone_container(self, zone: TileZone) -> list[PhysicalTile]:
        if zone.kind is TileZoneKind.CONCEALED:
            if zone.seat is None:
                raise IllegalActionError("concealed zone requires a seat")
            return self.state.players[zone.seat].concealed_tiles
        if zone.kind is TileZoneKind.WALL:
            if self.state.wall_state is None:
                raise IllegalActionError("wall is not initialized")
            if zone.seat is None:
                raise IllegalActionError("wall zone requires a seat")
            return self.state.wall_state.subwalls[zone.seat]
        if zone.kind is TileZoneKind.REVEALED:
            return self.state.revealed_tiles
        if zone.kind is TileZoneKind.DISCARD:
            if zone.seat is None:
                raise IllegalActionError("discard zone requires a seat")
            return [
                discard.tile
                for discard in self.state.players[zone.seat].discards
                if discard.claimed_by is None
            ]
        raise IllegalActionError(f"unsupported tile zone: {zone.kind}")

    @staticmethod
    def _select_tile(
        container: list[PhysicalTile],
        *,
        tile: PhysicalTile | None,
        tile_type: TileType | None,
    ) -> PhysicalTile:
        if tile is not None:
            if tile not in container:
                raise IllegalActionError(f"tile is not in source zone: {tile.id}")
            return tile
        if tile_type is None:
            raise IllegalActionError("move_tile requires tile or tile_type")
        try:
            return next(candidate for candidate in container if candidate.tile_type == tile_type)
        except StopIteration as exc:
            raise IllegalActionError(f"tile type is not in source zone: {tile_type}") from exc

    def _contains_tile(self, tile: PhysicalTile) -> bool:
        if any(tile in player.concealed_tiles for player in self.state.players.values()):
            return True
        if any(tile in player.tenpai_marker for player in self.state.players.values()):
            return True
        if any(
            tile == discard.tile
            for player in self.state.players.values()
            for discard in player.discards
        ):
            return True
        if any(
            tile in meld.tiles
            for player in self.state.players.values()
            for meld in player.melds
        ):
            return True
        if self.state.wall_state is not None and any(
            tile in subwall for subwall in self.state.wall_state.subwalls.values()
        ):
            return True
        return tile in self.state.revealed_tiles
