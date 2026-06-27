from __future__ import annotations

import json
import unittest

from mahjong_ai.agents.llm import (
    LLMActionAdapter,
    build_llm_prompt,
    compact_prompt_for_model,
    parse_tool_choice,
    provider_preset,
)
from mahjong_ai.common.action import Action, ActionKind
from mahjong_ai.common.errors import IllegalActionError
from mahjong_ai.common.meld import Meld, MeldKind
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import PhysicalTile, TileType
from mahjong_ai.web.app import app
from mahjong_ai.web.schemas import ActionDescriptor, CreateSessionRequest
from mahjong_ai.web.schemas import SeatControllerConfig, SeatControllerKind
from mahjong_ai.web.session import SessionManager


class WebSessionApiTests(unittest.TestCase):
    def test_action_payload_normalizes_json_arrays_for_legal_action_matching(self) -> None:
        chi = Action(
            ActionKind.CHI,
            Seat.NORTH,
            TileType("B2"),
            Seat.EAST,
            metadata={"sequence": ("B1", "B2", "B3"), "claimed": "B2"},
        )
        win = Action(
            ActionKind.WIN,
            Seat.NORTH,
            TileType("B2"),
            Seat.EAST,
            metadata={"winners": (1, 2), "win_type": "discard"},
        )

        chi_roundtrip = ActionDescriptor.from_payload(
            {
                "operation": "chi",
                "actor": 1,
                "tile": "B2",
                "source": 0,
                "metadata": {"sequence": ["B1", "B2", "B3"], "claimed": "B2"},
            }
        ).to_action()
        win_roundtrip = ActionDescriptor.from_payload(
            {
                "operation": "win",
                "actor": 1,
                "tile": "B2",
                "source": 0,
                "metadata": {"winners": [1, 2], "win_type": "discard"},
            }
        ).to_action()

        self.assertEqual(chi_roundtrip, chi)
        self.assertEqual(win_roundtrip, win)

    def test_rule_registry_lists_tuidaohe(self) -> None:
        manager = SessionManager()

        rules = {rule.rule_id: rule for rule in manager.list_rules()}

        self.assertEqual(set(rules), {"northern_tuidaohe.v1"})
        self.assertIn("northern_tuidaohe.v1", rules)
        self.assertEqual(
            rules["northern_tuidaohe.v1"].implementation_status,
            "draft_rule_details_validator",
        )

    def test_create_session_returns_snapshot_and_legal_actions(self) -> None:
        manager = SessionManager()

        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=3)
        )
        snapshot = session.snapshot(viewer=Seat.EAST)
        decision = session.decision_snapshot()

        self.assertEqual(snapshot["session"]["rule_id"], "northern_tuidaohe.v1")
        self.assertEqual(snapshot["state"]["phase"], "waiting_for_discard")
        self.assertEqual(decision.decision_actors, (0,))
        self.assertTrue(decision.legal_actions[0])
        self.assertIsNotNone(snapshot["state"]["players"][0]["concealed_tiles"])
        self.assertIsNone(snapshot["state"]["players"][1]["concealed_tiles"])

        full_snapshot = session.snapshot()
        self.assertIsNotNone(full_snapshot["state"]["players"][1]["concealed_tiles"])

    def test_response_actions_can_be_submitted_one_seat_at_a_time(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=4)
        )
        discard = next(
            action
            for action in session.table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )

        session.submit_action(ActionDescriptor.from_action(discard))

        self.assertIsNotNone(session.table.state.response_window)
        for index, seat in enumerate(session.table.decision_actors(), start=1):
            session.submit_pass(seat)
            if index < 3:
                self.assertIsNotNone(session.table.state.response_window)
                self.assertEqual(len(session.pending_responses), index)
            else:
                self.assertIsNone(session.table.state.response_window)
                self.assertFalse(session.pending_responses)

    def test_step_returns_fixed_result_with_full_state_and_full_log_frame(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=5)
        )
        initial_frame_count = len(session.full_log()["frames"])
        discard = next(
            action
            for action in session.table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )

        result = session.step(ActionDescriptor.from_action(discard), viewer=Seat.EAST)

        self.assertEqual(
            set(result),
            {
                "schema_version",
                "step_id",
                "input",
                "session",
                "state",
                "full_state",
                "legal",
                "events",
                "pending",
                "terminal",
            },
        )
        self.assertEqual(result["input"]["operation"], "discard")
        self.assertIn("players", result["full_state"])
        self.assertIn("wall", result["full_state"])
        self.assertIsNotNone(result["full_state"]["players"][1]["concealed_tiles"])
        self.assertIsNone(result["state"]["players"][1]["concealed_tiles"])
        self.assertTrue(result["events"])
        self.assertIn("event_id", result["state"]["discard_stack"][0])
        self.assertEqual(result["state"]["discard_stack"][0]["seat"], 0)
        self.assertEqual(result["state"]["discard_stack"][0]["tile_type"], discard.tile.code)
        self.assertEqual(len(session.full_log()["frames"]), initial_frame_count + 1)

    def test_controller_placeholders_are_preserved_without_echoing_token(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(
                rule_id="northern_tuidaohe.v1",
                seed=12,
                seat_controllers={
                    Seat.EAST: SeatControllerConfig(SeatControllerKind.HUMAN),
                    Seat.NORTH: SeatControllerConfig(
                        SeatControllerKind.MODEL,
                        provider="openai-compatible",
                        base_url="https://example.invalid/v1",
                        token="secret-token",
                        model_name="debug-model",
                    ),
                    Seat.WEST: SeatControllerConfig(SeatControllerKind.HUMAN),
                    Seat.SOUTH: SeatControllerConfig(SeatControllerKind.HUMAN),
                },
            )
        )

        north = session.summary.controllers[1]
        self.assertEqual(north["kind"], "model")
        self.assertEqual(north["provider"], "openai-compatible")
        self.assertEqual(north["base_url"], "https://example.invalid/v1")
        self.assertEqual(north["model_name"], "debug-model")
        self.assertTrue(north["token_configured"])
        self.assertNotIn("token", north)

    def test_llm_provider_presets_cover_vendor_openai_differences(self) -> None:
        self.assertEqual(provider_preset("openai-compatible").provider_id, "openai")
        self.assertEqual(provider_preset("deepseek").default_base_url, "https://api.deepseek.com")
        self.assertEqual(provider_preset("deepseek").default_model_name, "deepseek-v4-flash")
        self.assertTrue(provider_preset("deepseek-v4-pro").use_tools)
        self.assertEqual(provider_preset("deepseek-reasoner").provider_id, "deepseek-v4-pro")
        self.assertEqual(provider_preset("deepseek-v4-pro").thinking_type, "disabled")
        self.assertEqual(provider_preset("gemini").default_model_name, "gemini-2.5-flash")
        self.assertEqual(
            provider_preset("openrouter").default_base_url,
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(
            provider_preset("openrouter").default_model_name,
            "qwen/qwen3.5-flash-02-23",
        )
        self.assertEqual(
            provider_preset("local-openai").default_model_name,
            "qwen3.5-2b-transformers",
        )

    def test_openai_payload_uses_provider_specific_tool_and_json_modes(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=13)
        )
        prompt, _, tool_schema, _ = build_llm_prompt(
            state=session.table.state,
            actor=Seat.EAST,
            rules=session.table.rules,
            legal_actions=session.table.legal_actions(Seat.EAST),
        )

        deepseek = LLMActionAdapter(
            provider="deepseek",
            base_url=None,
            token="token",
            model_name=None,
        )
        deepseek_payload = deepseek._openai_compatible_payload(prompt, tool_schema)
        self.assertEqual(deepseek_payload["model"], "deepseek-v4-flash")
        self.assertIn("tools", deepseek_payload)
        self.assertEqual(deepseek_payload["tool_choice"], "required")
        self.assertFalse(deepseek_payload["parallel_tool_calls"])
        self.assertEqual(deepseek_payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning", deepseek_payload)

        pro = LLMActionAdapter(
            provider="deepseek-v4-pro",
            base_url=None,
            token="token",
            model_name=None,
        )
        pro_payload = pro._openai_compatible_payload(prompt, tool_schema)
        self.assertEqual(pro_payload["model"], "deepseek-v4-pro")
        self.assertIn("tools", pro_payload)
        self.assertEqual(pro_payload["tool_choice"], "required")
        self.assertFalse(pro_payload["parallel_tool_calls"])
        self.assertEqual(pro_payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", pro_payload)
        self.assertNotIn("temperature", pro_payload)

        openrouter = LLMActionAdapter(
            provider="openrouter",
            base_url=None,
            token="token",
            model_name=None,
        )
        openrouter_payload = openrouter._openai_compatible_payload(prompt, tool_schema)
        self.assertEqual(openrouter_payload["model"], "qwen/qwen3.5-flash-02-23")
        self.assertEqual(openrouter_payload["tool_choice"], "auto")
        self.assertFalse(openrouter_payload["parallel_tool_calls"])
        self.assertEqual(openrouter_payload["reasoning"], {"exclude": True})
        self.assertNotIn("provider", openrouter_payload)
        self.assertNotIn("transforms", openrouter_payload)
        self.assertNotIn("thinking", openrouter_payload)

        legacy_openrouter = LLMActionAdapter(
            provider="openrouter",
            base_url=None,
            token="token",
            model_name="system",
        )
        legacy_payload = legacy_openrouter._openai_compatible_payload(prompt, tool_schema)
        self.assertEqual(legacy_payload["model"], "qwen/qwen3.5-flash-02-23")

    def test_parse_tool_choice_accepts_reasoning_content_and_json_content(self) -> None:
        action_id, reason = parse_tool_choice(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "I should preserve waits.",
                            "content": '{"action_id":"a2"}',
                        }
                    }
                ]
            }
        )

        self.assertEqual(action_id, "a2")
        self.assertEqual(reason, "I should preserve waits.")

    def test_llm_prompt_contains_rules_visible_state_and_legal_action_ids(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=13)
        )
        self.assertFalse(session.table.wall_provider.reserve_dead_wall)

        prompt, options, tool_schema, prompt_hash = build_llm_prompt(
            state=session.table.state,
            actor=Seat.EAST,
            rules=session.table.rules,
            legal_actions=session.table.legal_actions(Seat.EAST),
        )

        self.assertEqual(prompt["schema_version"], "mahjong_llm_prompt.v2")
        self.assertEqual(prompt["rule"]["rule_id"], "northern_tuidaohe.v1")
        self.assertIn("require_declared_tenpai", prompt["rule"]["features"])
        self.assertFalse(prompt["rule"]["features"]["reserve_dead_wall"])
        self.assertEqual(
            prompt["rule"]["features"]["exhaustive_draw_condition"],
            "active_seat_subwall_exhaustion",
        )
        self.assertEqual(prompt["tool_call_required"], "choose_mahjong_action")
        self.assertEqual(tool_schema["function"]["name"], "choose_mahjong_action")
        self.assertTrue(prompt_hash)
        self.assertEqual(len(prompt["legal_actions"]), len(options))
        self.assertEqual(prompt["legal_actions"][0]["action_id"], "a0")
        self.assertEqual(
            tuple(prompt["observation"]["concealed_tiles"]),
            tuple(
                sorted(
                    tile.tile_type.code
                    for tile in session.table.state.players[Seat.EAST].concealed_tiles
                )
            ),
        )
        self.assertEqual(
            tuple(prompt["decision_context"]["current_hand"]),
            tuple(prompt["observation"]["concealed_tiles"]),
        )
        self.assertEqual(
            prompt["decision_context"]["discarded_tiles"],
            prompt["observation"]["discarded_tiles"],
        )
        self.assertEqual(
            prompt["decision_context"]["available_actions"],
            prompt["legal_actions"],
        )
        self.assertNotIn("full_state", prompt)
        self.assertNotIn('"subwalls"', json.dumps(prompt, sort_keys=True))

        compact = compact_prompt_for_model(prompt)
        self.assertEqual(compact["rule"]["rule_id"], "northern_tuidaohe.v1")
        self.assertEqual(compact["current_hand"], prompt["decision_context"]["current_hand"])
        self.assertEqual(
            compact["discarded_tiles"],
            prompt["decision_context"]["discarded_tiles"],
        )
        self.assertEqual(
            compact["known_other_player_tiles"],
            prompt["decision_context"]["known_other_player_tiles"],
        )
        self.assertEqual(len(compact["legal_actions"]), len(options))
        self.assertLess(len(json.dumps(compact, sort_keys=True)), 7000)

    def test_llm_prompt_lists_only_exposed_known_other_player_tiles(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=13)
        )
        north = session.table.state.players[Seat.NORTH]
        north.melds.append(
            Meld(
                MeldKind.PENG,
                (
                    PhysicalTile(TileType("W1"), 0),
                    PhysicalTile(TileType("W1"), 1),
                    PhysicalTile(TileType("W1"), 2),
                ),
                source=Seat.EAST,
            )
        )
        north.melds.append(
            Meld(
                MeldKind.CONCEALED_KONG,
                (
                    PhysicalTile(TileType("J1"), 0),
                    PhysicalTile(TileType("J1"), 1),
                    PhysicalTile(TileType("J1"), 2),
                    PhysicalTile(TileType("J1"), 3),
                ),
            )
        )

        prompt, _, _, _ = build_llm_prompt(
            state=session.table.state,
            actor=Seat.EAST,
            rules=session.table.rules,
            legal_actions=session.table.legal_actions(Seat.EAST),
        )

        known_north = prompt["decision_context"]["known_other_player_tiles"]["1"]
        self.assertEqual(len(known_north), 1)
        self.assertEqual(known_north[0]["kind"], "peng")
        self.assertEqual(known_north[0]["tile_types"], ("W1", "W1", "W1"))
        self.assertNotIn("J1", json.dumps(known_north, sort_keys=True))

    def test_controller_step_records_llm_reason_and_prompt_trace(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(
                rule_id="northern_tuidaohe.v1",
                seed=14,
                seat_controllers={
                    Seat.EAST: SeatControllerConfig(
                        SeatControllerKind.MODEL,
                        provider="debug",
                    ),
                },
            )
        )

        result = session.controller_step(Seat.EAST)

        self.assertIn("controller_decision", result)
        decision = result["controller_decision"]
        self.assertEqual(decision["seat"], 0)
        self.assertEqual(decision["controller_kind"], "model")
        self.assertEqual(decision["provider"], "debug")
        self.assertEqual(decision["prompt_schema"], "mahjong_llm_prompt.v2")
        self.assertTrue(decision["prompt_hash"])
        self.assertTrue(decision["natural_language_reason"])
        self.assertEqual(decision["raw_response"]["tool_call"]["name"], "choose_mahjong_action")
        self.assertTrue(decision["validation"]["action_id_found"])
        self.assertTrue(decision["validation"]["rule_validated"])
        self.assertEqual(
            session.full_log()["frames"][-1]["controller_decision"]["selected_action_id"],
            decision["selected_action_id"],
        )
        self.assertIn("legal_actions", decision["prompt"])
        self.assertNotIn('"subwalls"', json.dumps(decision["prompt"], sort_keys=True))
        for seat in ALL_SEATS:
            self.assertIsNotNone(result["state"]["players"][int(seat)]["concealed_tiles"])

    def test_human_controller_step_is_rejected(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=15)
        )

        with self.assertRaises(IllegalActionError):
            session.controller_step(Seat.EAST)

    def test_controller_loop_keeps_debug_hands_visible_and_discard_actions_match_hand(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(
                rule_id="northern_tuidaohe.v1",
                seed=16,
                seat_controllers={
                    seat: SeatControllerConfig(
                        SeatControllerKind.MODEL,
                        provider="debug",
                    )
                    for seat in ALL_SEATS
                },
            )
        )

        result = session.snapshot()
        for _ in range(24):
            if session.table.state.is_terminal:
                break
            actor = session.table.decision_actors()[0]
            hand_tiles = {
                tile.tile_type
                for tile in session.table.state.players[actor].concealed_tiles
            }
            for action in session.table.legal_actions(actor):
                if action.kind is ActionKind.DISCARD:
                    self.assertIn(action.tile, hand_tiles)

            result = session.controller_step(actor)

            for seat in ALL_SEATS:
                player = result["state"]["players"][int(seat)]
                self.assertIsNotNone(player["concealed_tiles"])
                self.assertEqual(len(player["concealed_tiles"]), player["concealed_count"])

        self.assertGreater(result["session"]["event_count"], 7)

    def test_step_accepts_operation_alias_for_response(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=6)
        )
        discard = next(
            action
            for action in session.table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )
        session.step(ActionDescriptor.from_action(discard))
        actor = session.table.decision_actors()[0]
        tile = session.table.state.response_window.tile.tile_type.code

        result = session.step(
            ActionDescriptor.from_payload(
                {
                    "actor": int(actor),
                    "operation": "pass",
                    "tile": tile,
                    "source": int(session.table.state.response_window.source),
                }
            )
        )

        self.assertIn(int(actor), result["pending"]["responses"])
        self.assertTrue(result["pending"]["response_window_open"])

    def test_final_no_response_step_auto_advances_to_next_discard_decision(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=11)
        )
        discard = next(
            action
            for action in session.table.legal_actions()
            if action.kind is ActionKind.DISCARD
        )
        session.step(ActionDescriptor.from_action(discard))
        actors = tuple(session.table.decision_actors())

        result = None
        for actor in actors:
            pass_action = next(
                action
                for action in session.table.legal_actions(actor)
                if action.kind is ActionKind.PASS
            )
            result = session.step(ActionDescriptor.from_action(pass_action))

        self.assertIsNotNone(result)
        self.assertFalse(result["pending"]["response_window_open"])
        self.assertEqual(result["session"]["phase"], "waiting_for_discard")
        self.assertEqual(result["session"]["current_actor"], 1)
        self.assertEqual(result["state"]["players"][1]["concealed_count"], 14)

    def test_all_human_external_driver_can_finish_a_hand(self) -> None:
        manager = SessionManager()
        session = manager.create_session(
            CreateSessionRequest(rule_id="northern_tuidaohe.v1", seed=7)
        )

        for _ in range(500):
            if session.table.state.is_terminal:
                break
            if session.table.state.response_window is not None:
                for seat in tuple(session.table.decision_actors()):
                    if seat in session.pending_responses:
                        continue
                    session.submit_pass(seat)
                continue
            session.advance_to_decision()
            if session.table.state.is_terminal:
                break
            legal = session.table.legal_actions()
            selected = next(
                (action for action in legal if action.kind is ActionKind.WIN),
                None,
            )
            if selected is None:
                selected = next(
                    (action for action in legal if action.kind is ActionKind.DISCARD),
                    legal[0],
                )
            session.submit_action(ActionDescriptor.from_action(selected))

        self.assertTrue(session.table.state.is_terminal)
        self.assertEqual(sum(session.table.state.terminal_result.scores), 0)

    @unittest.skipIf(app is None, "FastAPI is not installed")
    def test_fastapi_routes_create_session_and_return_legal_actions(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(app)

        rules = client.get("/api/rules")
        self.assertEqual(rules.status_code, 200)
        self.assertIn(
            "northern_tuidaohe.v1",
            {rule["rule_id"] for rule in rules.json()},
        )

        created = client.post(
            "/api/sessions",
            json={
                "rule_id": "northern_tuidaohe.v1",
                "seed": 8,
                "seat_controllers": {
                    str(int(seat)): {"kind": "human"} for seat in ALL_SEATS
                },
            },
        )
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["session_id"]

        legal = client.get(f"/api/sessions/{session_id}/legal-actions")
        self.assertEqual(legal.status_code, 200)
        self.assertEqual(legal.json()["decision_actors"], [0])
        self.assertTrue(legal.json()["legal_actions"]["0"])

        first_action = next(
            action
            for action in legal.json()["legal_actions"]["0"]
            if action["kind"] == "discard"
        )
        stepped = client.post(
            f"/api/sessions/{session_id}/step",
            json={
                "actor": first_action["actor"],
                "operation": first_action["kind"],
                "tile": first_action["tile"],
                "metadata": first_action["metadata"],
                "viewer": 0,
            },
        )
        self.assertEqual(stepped.status_code, 200)
        self.assertEqual(stepped.json()["schema_version"], "mahjong_step_result.v1")
        self.assertIn("full_state", stepped.json())

        full_log = client.get(f"/api/sessions/{session_id}/full-log")
        self.assertEqual(full_log.status_code, 200)
        self.assertGreaterEqual(len(full_log.json()["frames"]), 2)

    @unittest.skipIf(app is None, "FastAPI is not installed")
    def test_fastapi_serves_lightweight_debug_ui(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(app)

        index = client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("Mahjong AI Dev Table", index.text)
        self.assertIn("tileAssetUrl", index.text)
        self.assertIn("perthmahjongsoc/mahjong-tiles-svg", index.text)

        script = client.get("/static/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("/step", script.text)
        self.assertIn("resolveNoResponse", script.text)
        self.assertIn("seatControllerPayload", script.text)
        self.assertIn("/static/vendor/mahjong-tiles-svg/", script.text)
        self.assertNotIn("Pass All Pending", script.text)

        styles = client.get("/static/styles.css")
        self.assertEqual(styles.status_code, 200)
        self.assertIn("tile-card", styles.text)
        self.assertIn("Debug Table", index.text)

        tile = client.get("/static/vendor/mahjong-tiles-svg/W1.svg")
        self.assertEqual(tile.status_code, 200)
        self.assertIn("<svg", tile.text)


if __name__ == "__main__":
    unittest.main()
