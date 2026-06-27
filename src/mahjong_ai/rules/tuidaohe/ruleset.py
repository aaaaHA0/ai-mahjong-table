from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import IllegalActionError, RuleConfigurationError, StateInvariantError
from mahjong_ai.common.meld import MeldKind
from mahjong_ai.common.score import ScoreTransfer
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile, TileType, all_tile_types
from mahjong_ai.game.phase import GamePhase
from mahjong_ai.game.response_window import (
    ResponseResolution,
    ResponseWindow,
    ResponseWindowKind,
)
from mahjong_ai.game.state import TableState
from mahjong_ai.rules.base import (
    DrawRequest,
    HandSetup,
    OperationPlan,
    PostActionDirective,
    PostActionKind,
    ResponseWindowSpec,
    RuleConfig,
    TerminalDirective,
)
from mahjong_ai.walls.base import DrawKind


class TuidaoheRulePlugin:
    """Minimal project Tuidaohe validator.

    This is intentionally a code validator, not a config-driven rules DSL. The
    first implementation provides a playable table loop and stable integration
    points. Full tenpai, wait-set, special-pattern, and scoring evaluators can
    be added behind this plugin without changing TableEngine.
    """

    rule_id = "northern_tuidaohe.v1"

    def __init__(self, config: RuleConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        expected = {
            "rule_id": self.rule_id,
            "players.count": 4,
            "tiles.total_count": 136,
            "tiles.include_flowers": False,
            "actions.allow_chi": False,
            "response.multiple_winners": True,
        }
        for path, expected_value in expected.items():
            actual = self.config.rule_id if path == "rule_id" else self.config.get(path)
            if actual != expected_value:
                raise RuleConfigurationError(
                    f"Tuidaohe config {path} must be {expected_value!r}, got {actual!r}"
                )

    def setup_hand(self, state: TableState) -> HandSetup:
        return HandSetup(
            deal_counts={
                seat: (
                    int(self.config.get("deal.dealer_initial_tiles"))
                    if seat is state.dealer
                    else int(self.config.get("deal.non_dealer_initial_tiles"))
                )
                for seat in ALL_SEATS
            },
            initial_actor=state.dealer,
            initial_phase=GamePhase.WAITING_FOR_DISCARD,
        )

    def create_player_rule_state(
        self, state: TableState, actor: Seat
    ) -> dict[str, Any]:
        del state, actor
        return {
            "tenpai_declared": False,
            "tenpai_marker_tile": None,
            "tenpai_waits": (),
            "passed_win_locked": False,
            "pending_kong_replacement_draw": False,
            "current_turn_kong_replacement": False,
            "last_discard_after_kong_replacement": False,
            "self_draw_winning_tiles": {},
            "discard_winning_tiles": {},
        }

    def draw_request(self, state: TableState) -> DrawRequest | None:
        if state.phase is GamePhase.WAITING_FOR_DRAW:
            return DrawRequest(state.current_actor, DrawKind.NORMAL, "tile_drawn")
        if state.phase is GamePhase.WAITING_FOR_KONG_REPLACEMENT:
            state.players[state.current_actor].rule_state[
                "pending_kong_replacement_draw"
            ] = True
            return DrawRequest(
                state.current_actor,
                DrawKind.KONG_REPLACEMENT,
                "kong_replacement_drawn",
            )
        return None

    def on_draw_unavailable(
        self, state: TableState, request: DrawRequest
    ) -> TerminalDirective:
        del state, request
        return TerminalDirective(reason="exhaustive_draw")

    def after_action(
        self, state: TableState, action: Action, tile: Any
    ) -> OperationPlan:
        if action.kind is ActionKind.DISCARD:
            assert tile is not None
            actor_rule_state = state.players[action.actor].rule_state
            actor_rule_state["last_discard_after_kong_replacement"] = bool(
                actor_rule_state.pop("current_turn_kong_replacement", False)
            )
            return OperationPlan.from_directive(
                PostActionDirective(
                    PostActionKind.OPEN_DISCARD_RESPONSE,
                    response_spec=ResponseWindowSpec(
                        ResponseWindowKind.DISCARD,
                        source=action.actor,
                        tile=tile,
                        eligible_seats=tuple(seat for seat in ALL_SEATS if seat != action.actor),
                    ),
                )
            )
        if action.kind is ActionKind.ADDED_KONG:
            assert tile is not None
            return OperationPlan.from_directive(
                PostActionDirective(
                    PostActionKind.OPEN_ADDED_KONG_RESPONSE,
                    response_spec=ResponseWindowSpec(
                        ResponseWindowKind.ROB_ADDED_KONG,
                        source=action.actor,
                        tile=tile,
                        eligible_seats=tuple(seat for seat in ALL_SEATS if seat != action.actor),
                        proposed_action=action,
                    ),
                )
            )
        if action.kind is ActionKind.DECLARE:
            assert tile is not None
            player_state = state.players[action.actor].rule_state
            player_state["tenpai_declared"] = True
            player_state["tenpai_marker_tile"] = tile.tile_type.code
            waits = self._winning_waits(
                tuple(
                    tile.tile_type
                    for tile in state.players[action.actor].concealed_tiles
                ),
                len(state.players[action.actor].melds),
                player_state,
                ignore_tenpai_requirement=True,
            )
            if not waits:
                raise IllegalActionError("tenpai declaration must leave a tenpai hand")
            player_state["tenpai_waits"] = tuple(tile_type.code for tile_type in waits)
            return OperationPlan.from_directive(
                PostActionDirective(
                    PostActionKind.WAIT_FOR_DRAW,
                    next_actor=action.actor.next(),
                )
            )
        if action.kind in {ActionKind.CONCEALED_KONG, ActionKind.EXPOSED_KONG}:
            return OperationPlan(
                (
                    PostActionDirective(
                        PostActionKind.SCORE_TRANSFERS,
                        transfers=self._settle_kong(action),
                    ),
                    PostActionDirective(PostActionKind.DRAW_KONG_REPLACEMENT),
                )
            )
        if action.kind is ActionKind.PENG:
            return OperationPlan.from_directive(
                PostActionDirective(PostActionKind.WAIT_FOR_DISCARD)
            )
        if action.kind is ActionKind.WIN:
            winners = self._action_winners(action)
            return OperationPlan.from_directive(
                PostActionDirective(
                    PostActionKind.TERMINAL,
                    terminal=TerminalDirective(
                        reason="win",
                        winners=winners,
                        responsible_seat=action.source,
                        transfers=self.settle_win(state, action, action.source),
                    ),
                )
            )
        raise IllegalActionError(f"unsupported Tuidaohe action directive: {action.kind}")

    def after_unclaimed_response(
        self, state: TableState, window: ResponseWindow
    ) -> OperationPlan:
        del state
        if window.kind is ResponseWindowKind.ROB_ADDED_KONG:
            transfers = (
                self._settle_kong(window.proposed_action)
                if window.proposed_action is not None
                else ()
            )
            return OperationPlan(
                (
                    PostActionDirective(
                        PostActionKind.SCORE_TRANSFERS,
                        transfers=transfers,
                    ),
                    PostActionDirective(PostActionKind.DRAW_KONG_REPLACEMENT),
                )
            )
        return OperationPlan.from_directive(
            PostActionDirective(
                PostActionKind.WAIT_FOR_DRAW,
                next_actor=window.source.next(),
            )
        )

    def legal_actions(self, state: TableState, actor: Seat) -> tuple[Action, ...]:
        if state.phase is not GamePhase.WAITING_FOR_DISCARD or actor != state.current_actor:
            return ()

        player = state.players[actor]
        if player.rule_state.pop("pending_kong_replacement_draw", False):
            player.rule_state["current_turn_kong_replacement"] = True
        player.rule_state["passed_win_locked"] = False
        counts = Counter(tile.tile_type for tile in player.concealed_tiles)
        actions: list[Action] = [
            Action(ActionKind.DISCARD, actor, tile_type)
            for tile_type in sorted(counts)
        ]
        if self.config.get("tenpai_declaration.enabled") and not player.rule_state.get(
            "tenpai_declared"
        ):
            actions.extend(
                Action(
                    ActionKind.DECLARE,
                    actor,
                    tile_type,
                    metadata={"declaration": "tenpai"},
                )
                for tile_type in sorted(counts)
                if self._can_declare_tenpai(player.concealed_tiles, len(player.melds), tile_type)
            )

        if self.config.get("actions.allow_concealed_kong"):
            actions.extend(
                Action(ActionKind.CONCEALED_KONG, actor, tile_type)
                for tile_type, count in sorted(counts.items())
                if count == 4
                and self._kong_allowed_after_tenpai(
                    player.concealed_tiles,
                    len(player.melds),
                    player.rule_state,
                    tile_type,
                    ActionKind.CONCEALED_KONG,
                )
            )

        if self.config.get("actions.allow_added_kong"):
            peng_types = {
                meld.tile_type
                for meld in player.melds
                if meld.kind is MeldKind.PENG
            }
            actions.extend(
                Action(ActionKind.ADDED_KONG, actor, tile_type)
                for tile_type in sorted(peng_types)
                if counts[tile_type] >= 1
                and self._kong_allowed_after_tenpai(
                    player.concealed_tiles,
                    len(player.melds),
                    player.rule_state,
                    tile_type,
                    ActionKind.ADDED_KONG,
                )
            )

        if player.rule_state.get("tenpai_declared"):
            drawn_tile = player.concealed_tiles[-1].tile_type if player.concealed_tiles else None
            actions = [
                action
                for action in actions
                if (
                    action.kind is not ActionKind.DISCARD
                    or action.tile == drawn_tile
                )
            ]

        drawn_tile = player.concealed_tiles[-1].tile_type if player.concealed_tiles else None
        if (
            self.config.get("winning.allow_self_draw")
            and self._can_self_draw(player.concealed_tiles, len(player.melds), player.rule_state)
        ):
            win_type = (
                "kong_flower"
                if player.rule_state.get("current_turn_kong_replacement")
                else "self_draw"
            )
            actions.append(
                Action(
                    ActionKind.WIN,
                    actor,
                    drawn_tile,
                    metadata={"win_type": win_type},
                )
            )
        return tuple(actions)

    def validate_action(self, state: TableState, action: Action) -> None:
        if action not in self.legal_actions(state, action.actor):
            raise IllegalActionError(
                f"illegal Tuidaohe action for {action.actor.name}: {action.kind} {action.tile}"
            )

    def legal_responses(
        self, state: TableState, window: ResponseWindow, actor: Seat
    ) -> tuple[Action, ...]:
        if actor not in window.eligible_seats:
            return ()
        actions: list[Action] = [
            Action(ActionKind.PASS, actor, window.tile.tile_type, source=window.source)
        ]
        player = state.players[actor]
        if (
            self.config.get("winning.allow_discard_win")
            and self._can_discard_win(
                player.concealed_tiles,
                len(player.melds),
                window.tile.tile_type,
                player.rule_state,
            )
        ):
            source_state = state.players[window.source].rule_state
            actions.append(
                Action(
                    ActionKind.WIN,
                    actor,
                    window.tile.tile_type,
                    source=window.source,
                    metadata={
                        "win_type": (
                            "rob_kong"
                            if window.kind is ResponseWindowKind.ROB_ADDED_KONG
                            else (
                                "kong_discard"
                                if source_state.get("last_discard_after_kong_replacement")
                                else "discard"
                            )
                        )
                    },
                )
            )

        if window.kind is ResponseWindowKind.ROB_ADDED_KONG:
            return tuple(actions)

        if (
            player.rule_state.get("tenpai_declared")
            and not self.config.get("tenpai_declaration.allow_peng_after_declaration")
        ):
            return tuple(actions)

        counts = Counter(tile.tile_type for tile in player.concealed_tiles)
        claimed_type = window.tile.tile_type
        if self.config.get("actions.allow_exposed_kong") and counts[claimed_type] >= 3:
            actions.append(
                Action(
                    ActionKind.EXPOSED_KONG,
                    actor,
                    claimed_type,
                    source=window.source,
                )
            )
        if self.config.get("actions.allow_peng") and counts[claimed_type] >= 2:
            actions.append(
                Action(ActionKind.PENG, actor, claimed_type, source=window.source)
            )
        return tuple(actions)

    def resolve_responses(
        self,
        state: TableState,
        window: ResponseWindow,
        responses: Mapping[Seat, Action],
    ) -> ResponseResolution:
        if set(responses) != set(window.eligible_seats):
            raise IllegalActionError("responses must be submitted for every eligible seat")
        for seat, action in responses.items():
            if action not in window.legal_actions[seat]:
                raise IllegalActionError(f"illegal response from {seat.name}: {action}")
            if (
                action.kind is ActionKind.PASS
                and self.config.get("response.pass_win_lock.enabled")
                and any(
                    legal.kind is ActionKind.WIN
                    for legal in window.legal_actions.get(seat, ())
                )
            ):
                state.players[seat].rule_state["passed_win_locked"] = True

        non_pass = [
            action for action in responses.values() if action.kind is not ActionKind.PASS
        ]
        if not non_pass:
            return ResponseResolution(None, dict(responses))
        win_actions = [
            action for action in non_pass if action.kind is ActionKind.WIN
        ]
        if win_actions and self.config.get("response.multiple_winners"):
            ordered_wins = tuple(
                sorted(win_actions, key=lambda action: self._distance(window.source, action.actor))
            )
            first = ordered_wins[0]
            return ResponseResolution(
                Action(
                    ActionKind.WIN,
                    first.actor,
                    first.tile,
                    source=first.source,
                    metadata={
                        **dict(first.metadata),
                        "winners": tuple(int(action.actor) for action in ordered_wins),
                    },
                ),
                dict(responses),
            )
        selected = min(
            non_pass,
            key=lambda action: (
                -self._priority(action.kind),
                self._distance(window.source, action.actor),
            ),
        )
        return ResponseResolution(selected, dict(responses))

    def settle_win(
        self, state: TableState, action: Action, source: Seat | None
    ) -> tuple[ScoreTransfer, ...]:
        winners = self._action_winners(action)
        if source is None:
            amount = self._win_multiplier(state, winners[0], action, source)
            return tuple(
                ScoreTransfer(seat, winners[0], amount, "tuidaohe_self_draw")
                for seat in ALL_SEATS
                if seat != winners[0]
            )
        transfers: list[ScoreTransfer] = []
        for winner in winners:
            amount = self._win_multiplier(state, winner, action, source)
            transfers.append(
                ScoreTransfer(source, winner, amount, f"tuidaohe_{action.metadata.get('win_type', 'discard')}_win")
            )
        return tuple(transfers)

    def _settle_kong(self, action: Action | None) -> tuple[ScoreTransfer, ...]:
        if action is None or not self.config.get("scoring.gang_score_enabled"):
            return ()
        if action.kind is ActionKind.CONCEALED_KONG:
            amount = int(self.config.get("scoring.gang_scores.concealed_kong_each_opponent"))
            return tuple(
                ScoreTransfer(seat, action.actor, amount, "tuidaohe_concealed_kong")
                for seat in ALL_SEATS
                if seat != action.actor
            )
        if action.kind is ActionKind.ADDED_KONG:
            amount = int(self.config.get("scoring.gang_scores.added_kong_each_opponent"))
            return tuple(
                ScoreTransfer(seat, action.actor, amount, "tuidaohe_added_kong")
                for seat in ALL_SEATS
                if seat != action.actor
            )
        if action.kind is ActionKind.EXPOSED_KONG:
            if action.source is None:
                raise IllegalActionError("exposed kong requires a source")
            amount = int(self.config.get("scoring.gang_scores.exposed_kong_discarder_only"))
            return (
                ScoreTransfer(
                    action.source,
                    action.actor,
                    amount,
                    "tuidaohe_exposed_kong",
                ),
            )
        return ()

    def validate_rule_state(self, state: TableState) -> None:
        if len(state.players) != 4:
            raise StateInvariantError("Tuidaohe requires exactly four players")
        structural_sizes = {
            seat: (
                len(state.players[seat].concealed_tiles)
                + sum(meld.effective_size for meld in state.players[seat].melds)
            )
            for seat in ALL_SEATS
        }
        if state.phase is GamePhase.WAITING_FOR_DISCARD:
            if structural_sizes[state.current_actor] != 14:
                raise StateInvariantError(
                    f"Tuidaohe actor must hold structural size 14: {structural_sizes}"
                )
            if any(
                size != 13
                for seat, size in structural_sizes.items()
                if seat != state.current_actor
            ):
                raise StateInvariantError(
                    f"Tuidaohe non-actors must hold structural size 13: {structural_sizes}"
                )
        elif state.phase in {
            GamePhase.WAITING_FOR_DRAW,
            GamePhase.WAITING_FOR_DISCARD_RESPONSES,
            GamePhase.WAITING_FOR_ROB_KONG_RESPONSES,
        }:
            if any(size != 13 for size in structural_sizes.values()):
                raise StateInvariantError(
                    f"Tuidaohe players must hold structural size 13: {structural_sizes}"
                )
        if state.terminal_result is not None and sum(state.terminal_result.scores) != 0:
            raise StateInvariantError("Tuidaohe terminal score must be zero-sum")

    def build_rule_features(
        self, state: TableState, actor: Seat
    ) -> Mapping[str, Any]:
        del state, actor
        return {
            "rule_id": self.rule_id,
            "allow_chi": False,
            "allow_peng": self.config.get("actions.allow_peng"),
            "allow_exposed_kong": self.config.get("actions.allow_exposed_kong"),
            "allow_concealed_kong": self.config.get("actions.allow_concealed_kong"),
            "allow_added_kong": self.config.get("actions.allow_added_kong"),
            "allow_self_draw": self.config.get("winning.allow_self_draw"),
            "allow_discard_win": self.config.get("winning.allow_discard_win"),
            "require_declared_tenpai": self.config.get("winning.require_declared_tenpai"),
            "minimum_fan": self.config.get("winning.minimum_fan"),
            "multiple_winners": self.config.get("response.multiple_winners"),
            "reserve_dead_wall": self.config.get("wall.reserve_dead_wall"),
            "exhaustive_draw_condition": self.config.get(
                "wall.exhaustive_draw_condition"
            ),
            "win_multiplier_cap": self.config.get("scoring.win_multiplier_cap"),
        }

    def _can_self_draw(
        self,
        concealed_tiles: list[PhysicalTile],
        meld_count: int,
        rule_state: Mapping[str, Any],
    ) -> bool:
        drawn_tile = concealed_tiles[-1].tile_type if concealed_tiles else None
        if self._forced_win(rule_state, "self_draw_winning_tiles", drawn_tile):
            return True
        return self._can_win_tiles(
            tuple(tile.tile_type for tile in concealed_tiles),
            meld_count,
            rule_state,
        )

    def _can_discard_win(
        self,
        concealed_tiles: list[PhysicalTile],
        meld_count: int,
        claimed_tile: TileType,
        rule_state: Mapping[str, Any],
    ) -> bool:
        if (
            rule_state.get("passed_win_locked")
            and self.config.get("response.pass_win_lock.enabled")
        ):
            return False
        if self._forced_win(rule_state, "discard_winning_tiles", claimed_tile):
            return True
        return self._can_win_tiles(
            tuple(tile.tile_type for tile in concealed_tiles) + (claimed_tile,),
            meld_count,
            rule_state,
        )

    def _forced_win(
        self,
        rule_state: Mapping[str, Any],
        field: str,
        tile_type: TileType | None,
    ) -> bool:
        if tile_type is None:
            return False
        if (
            self.config.get("winning.require_declared_tenpai")
            and not rule_state.get("tenpai_declared")
        ):
            return False
        values = rule_state.get(field, {})
        return isinstance(values, Mapping) and bool(values.get(tile_type.code))

    def _can_win_tiles(
        self,
        tile_types: tuple[TileType, ...],
        meld_count: int,
        rule_state: Mapping[str, Any],
    ) -> bool:
        if (
            self.config.get("winning.require_declared_tenpai")
            and not rule_state.get("tenpai_declared")
        ):
            return False
        if (
            rule_state.get("passed_win_locked")
            and self.config.get("response.pass_win_lock.enabled")
            and self.config.get("response.pass_win_lock.restrict_self_draw")
        ):
            return False
        if len(tile_types) + meld_count * 3 != 14:
            return False
        counts = Counter(tile_types)
        if (
            meld_count == 0
            and self.config.get("winning.enabled_patterns")
            and self._is_seven_pairs(counts)
        ):
            return True
        if meld_count == 0 and self._is_thirteen_orphans(counts):
            return True
        return self._is_standard_hand(counts, melds_needed=4 - meld_count)

    def _can_declare_tenpai(
        self,
        concealed_tiles: list[PhysicalTile],
        meld_count: int,
        declared_tile: TileType,
    ) -> bool:
        remaining = list(concealed_tiles)
        for index, tile in enumerate(remaining):
            if tile.tile_type == declared_tile:
                del remaining[index]
                break
        else:
            return False
        return bool(
            self._winning_waits(
                tuple(tile.tile_type for tile in remaining),
                meld_count,
                {"tenpai_declared": True},
                ignore_tenpai_requirement=True,
            )
        )

    def _kong_allowed_after_tenpai(
        self,
        concealed_tiles: list[PhysicalTile],
        meld_count: int,
        rule_state: Mapping[str, Any],
        tile_type: TileType,
        kind: ActionKind,
    ) -> bool:
        if not rule_state.get("tenpai_declared"):
            return True
        if not self.config.get("tenpai_declaration.allow_kong_if_waits_unchanged"):
            return False
        if self._is_seven_pairs_tenpai(tuple(tile.tile_type for tile in concealed_tiles)):
            return False
        remaining = list(concealed_tiles)
        remove_count = 4 if kind is ActionKind.CONCEALED_KONG else 1
        removed = 0
        index = 0
        while index < len(remaining) and removed < remove_count:
            if remaining[index].tile_type == tile_type:
                del remaining[index]
                removed += 1
            else:
                index += 1
        if removed != remove_count:
            return False
        next_meld_count = meld_count + (1 if kind is ActionKind.CONCEALED_KONG else 0)
        waits_after = self._winning_waits(
            tuple(tile.tile_type for tile in remaining),
            next_meld_count,
            rule_state,
            ignore_tenpai_requirement=True,
        )
        waits_before = tuple(TileType(code) for code in rule_state.get("tenpai_waits", ()))
        return tuple(sorted(waits_after)) == tuple(sorted(waits_before))

    def _winning_waits(
        self,
        tile_types: tuple[TileType, ...],
        meld_count: int,
        rule_state: Mapping[str, Any],
        *,
        ignore_tenpai_requirement: bool = False,
    ) -> tuple[TileType, ...]:
        waits: list[TileType] = []
        if len(tile_types) + meld_count * 3 != 13:
            return ()
        for tile_type in all_tile_types():
            if Counter(tile_types)[tile_type] >= 4:
                continue
            if self._can_win_tiles(
                (*tile_types, tile_type),
                meld_count,
                rule_state if not ignore_tenpai_requirement else {"tenpai_declared": True},
            ):
                waits.append(tile_type)
        return tuple(sorted(waits))

    @staticmethod
    def _is_seven_pairs_tenpai(tile_types: tuple[TileType, ...]) -> bool:
        if len(tile_types) != 14:
            return False
        counts = Counter(tile_types)
        return sum(count // 2 for count in counts.values()) >= 6 and any(
            count == 4 for count in counts.values()
        )

    def _win_multiplier(
        self,
        state: TableState,
        winner: Seat,
        action: Action,
        source: Seat | None,
    ) -> int:
        player = state.players[winner]
        tile_types = tuple(tile.tile_type for tile in player.concealed_tiles)
        if action.tile is not None and action.metadata.get("win_type") not in {
            "self_draw",
            "kong_flower",
        }:
            tile_types = (*tile_types, action.tile)
        pattern_names = self._winning_pattern_names(tile_types, player.melds)
        multiplier = 1
        for pattern in pattern_names:
            multiplier *= int(self.config.get(f"scoring.pattern_multipliers.{pattern}", 1))
        win_type = str(action.metadata.get("win_type", "self_draw" if source is None else "discard"))
        if win_type in {"self_draw", "kong_flower", "kong_discard", "rob_kong"}:
            multiplier *= int(self.config.get(f"scoring.event_multipliers.{win_type}", 1))
        return max(1, min(multiplier, int(self.config.get("scoring.win_multiplier_cap"))))

    def _winning_pattern_names(
        self, tile_types: tuple[TileType, ...], melds: list[Any]
    ) -> tuple[str, ...]:
        counts = Counter(tile_types)
        patterns: set[str] = set()
        if self._is_thirteen_orphans(counts):
            return ("thirteen_orphans",)
        if self._is_seven_pairs(counts) and not melds:
            patterns.add("luxury_seven_pairs" if any(count == 4 for count in counts.values()) else "seven_pairs")
        else:
            if self._is_all_triplets(counts, melds):
                patterns.add("all_triplets")
            if self._has_pure_straight(counts, melds):
                patterns.add("pure_straight")
            if self._has_big_three_dragons(counts, melds):
                patterns.add("big_three_dragons")
            if self._has_small_four_winds(counts, melds):
                patterns.add("small_four_winds")
        flush = self._flush_pattern(counts, melds)
        if flush:
            patterns.add(flush)
        return tuple(sorted(patterns)) or ("pinfu",)

    @staticmethod
    def _flush_pattern(counts: Counter[TileType], melds: list[Any]) -> str | None:
        tile_types = set(counts)
        for meld in melds:
            tile_types.update(tile.tile_type for tile in meld.tiles)
        suits = {tile.code[0] for tile in tile_types if tile.code[0] in {"W", "B", "T"}}
        has_honor = any(tile.code[0] in {"F", "J"} for tile in tile_types)
        if len(suits) == 1 and not has_honor:
            return "full_flush"
        if len(suits) == 1 and has_honor:
            return "half_flush"
        return None

    def _is_all_triplets(self, counts: Counter[TileType], melds: list[Any]) -> bool:
        if any(meld.kind is MeldKind.CHI for meld in melds):
            return False
        if sum(counts.values()) != (4 - len(melds)) * 3 + 2:
            return False
        for pair_type, count in counts.items():
            if count < 2:
                continue
            remaining = counts.copy()
            remaining[pair_type] -= 2
            if all(count % 3 == 0 for count in remaining.values()):
                return True
        return False

    @staticmethod
    def _has_pure_straight(counts: Counter[TileType], melds: list[Any]) -> bool:
        all_counts = counts.copy()
        for meld in melds:
            for tile in meld.tiles[:3]:
                all_counts[tile.tile_type] += 1
        for prefix in ("W", "B", "T"):
            if all(all_counts[TileType(f"{prefix}{value}")] > 0 for value in range(1, 10)):
                return True
        return False

    @staticmethod
    def _has_big_three_dragons(counts: Counter[TileType], melds: list[Any]) -> bool:
        all_counts = counts.copy()
        for meld in melds:
            for tile in meld.tiles[:3]:
                all_counts[tile.tile_type] += 1
        return all(all_counts[TileType(code)] >= 3 for code in ("J1", "J2", "J3"))

    @staticmethod
    def _has_small_four_winds(counts: Counter[TileType], melds: list[Any]) -> bool:
        all_counts = counts.copy()
        for meld in melds:
            for tile in meld.tiles[:3]:
                all_counts[tile.tile_type] += 1
        triplets = sum(1 for code in ("F1", "F2", "F3", "F4") if all_counts[TileType(code)] >= 3)
        pairs = sum(1 for code in ("F1", "F2", "F3", "F4") if all_counts[TileType(code)] >= 2)
        return triplets == 3 and pairs >= 4

    @staticmethod
    def _is_seven_pairs(counts: Counter[TileType]) -> bool:
        return sum(counts.values()) == 14 and sum(count // 2 for count in counts.values()) == 7

    @staticmethod
    def _is_thirteen_orphans(counts: Counter[TileType]) -> bool:
        required = {
            TileType(code)
            for code in (
                "W1",
                "W9",
                "B1",
                "B9",
                "T1",
                "T9",
                "F1",
                "F2",
                "F3",
                "F4",
                "J1",
                "J2",
                "J3",
            )
        }
        return (
            sum(counts.values()) == 14
            and all(counts[tile_type] >= 1 for tile_type in required)
            and any(counts[tile_type] >= 2 for tile_type in required)
        )

    def _is_standard_hand(
        self, counts: Counter[TileType], *, melds_needed: int
    ) -> bool:
        if sum(counts.values()) != melds_needed * 3 + 2:
            return False
        for pair_type, count in list(counts.items()):
            if count < 2:
                continue
            remaining = counts.copy()
            remaining[pair_type] -= 2
            if self._can_form_melds(remaining, melds_needed):
                return True
        return False

    def _can_form_melds(
        self, counts: Counter[TileType], melds_needed: int
    ) -> bool:
        if melds_needed == 0:
            return all(count == 0 for count in counts.values())
        tile_type = next((tile for tile in sorted(counts) if counts[tile] > 0), None)
        if tile_type is None:
            return False

        if counts[tile_type] >= 3:
            remaining = counts.copy()
            remaining[tile_type] -= 3
            if self._can_form_melds(remaining, melds_needed - 1):
                return True

        if tile_type.code[0] in {"W", "B", "T"}:
            prefix = tile_type.code[0]
            value = int(tile_type.code[1])
            if value <= 7:
                sequence = (
                    tile_type,
                    TileType(f"{prefix}{value + 1}"),
                    TileType(f"{prefix}{value + 2}"),
                )
                if all(counts[part] > 0 for part in sequence):
                    remaining = counts.copy()
                    for part in sequence:
                        remaining[part] -= 1
                    if self._can_form_melds(remaining, melds_needed - 1):
                        return True
        return False

    @staticmethod
    def _priority(kind: ActionKind) -> int:
        return {
            ActionKind.WIN: 3,
            ActionKind.PENG: 2,
            ActionKind.EXPOSED_KONG: 2,
            ActionKind.PASS: 0,
        }[kind]

    @staticmethod
    def _distance(source: Seat, actor: Seat) -> int:
        return (int(actor) - int(source)) % 4

    @staticmethod
    def _action_winners(action: Action) -> tuple[Seat, ...]:
        winners = action.metadata.get("winners")
        if isinstance(winners, tuple):
            return tuple(Seat(int(winner)) for winner in winners)
        if isinstance(winners, list):
            return tuple(Seat(int(winner)) for winner in winners)
        return (action.actor,)
