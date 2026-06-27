from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from mahjong_ai.common.action import Action
from mahjong_ai.common.errors import AgentExecutionError
from mahjong_ai.common.seat import ALL_SEATS, Seat
from mahjong_ai.common.tile import TileType
from mahjong_ai.game.state import TableState
from mahjong_ai.observation.builder import ObservationBuilder
from mahjong_ai.observation.schema import Observation
from mahjong_ai.rules.base import RulePlugin


PROMPT_SCHEMA_VERSION = "mahjong_llm_prompt.v2"
TOOL_NAME = "choose_mahjong_action"


@dataclass(frozen=True)
class LegalActionOption:
    action_id: str
    action: Action
    label: str

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.action.kind.value,
            "actor": int(self.action.actor),
            "tile": self.action.tile.code if self.action.tile is not None else None,
            "source": int(self.action.source) if self.action.source is not None else None,
            "metadata": _jsonable(self.action.metadata),
            "label": self.label,
        }


@dataclass(frozen=True)
class LLMDecisionTrace:
    prompt_schema: str
    prompt_hash: str
    prompt: Mapping[str, Any]
    tool_schema: Mapping[str, Any]
    raw_response: Mapping[str, Any]
    selected_action_id: str
    reason: str | None
    selected_action: Action
    validation: Mapping[str, Any]

    def as_log_dict(self, *, include_prompt: bool = True) -> dict[str, Any]:
        selected = action_to_dict(self.selected_action)
        result: dict[str, Any] = {
            "prompt_schema": self.prompt_schema,
            "prompt_hash": self.prompt_hash,
            "tool_schema": self.tool_schema,
            "raw_response": self.raw_response,
            "selected_action_id": self.selected_action_id,
            "selected_action": selected,
            "natural_language_reason": self.reason,
            "validation": dict(self.validation),
        }
        if include_prompt:
            result["prompt"] = self.prompt
        return result


@dataclass(frozen=True)
class LLMProviderPreset:
    provider_id: str
    aliases: tuple[str, ...]
    label: str
    default_base_url: str | None = None
    default_model_name: str | None = None
    mode: str = "openai_compatible"
    prompt_mode: str = "full"
    use_tools: bool = True
    require_tool_choice: bool = True
    tool_choice: Any = "required"
    parallel_tool_calls: bool | None = False
    json_response_format: bool = False
    include_temperature: bool = True
    include_reasoning_disabled: bool = False
    thinking_type: str | None = None
    reasoning_effort: str | None = None
    extra_body: Mapping[str, Any] | None = None
    headers: Mapping[str, str] | None = None


PROVIDER_PRESETS: tuple[LLMProviderPreset, ...] = (
    LLMProviderPreset(
        "debug",
        ("debug", "fake"),
        "Debug local",
        mode="debug",
    ),
    LLMProviderPreset(
        "apple-fm",
        ("apple-fm", "fm", "llm"),
        "Apple Foundation Models CLI",
        default_model_name="system",
        mode="apple_fm",
        prompt_mode="compact",
    ),
    LLMProviderPreset(
        "openai",
        ("openai", "openai-compatible"),
        "OpenAI Chat Completions",
        default_base_url="https://api.openai.com/v1",
    ),
    LLMProviderPreset(
        "gemini",
        ("gemini", "google", "google-gemini"),
        "Google Gemini OpenAI-compatible",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model_name="gemini-2.5-flash",
        prompt_mode="compact",
        tool_choice="auto",
    ),
    LLMProviderPreset(
        "deepseek",
        ("deepseek", "deepseek-chat", "deepseek-v4-flash"),
        "DeepSeek V4 Flash",
        default_base_url="https://api.deepseek.com",
        default_model_name="deepseek-v4-flash",
        prompt_mode="compact",
        thinking_type="disabled",
    ),
    LLMProviderPreset(
        "deepseek-v4-pro",
        ("deepseek-v4-pro", "deepseek-pro", "deepseek-reasoner", "deepseek-r1"),
        "DeepSeek V4 Pro",
        default_base_url="https://api.deepseek.com",
        default_model_name="deepseek-v4-pro",
        prompt_mode="compact",
        include_temperature=False,
        thinking_type="disabled",
    ),
    LLMProviderPreset(
        "openrouter",
        ("openrouter",),
        "OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        default_model_name="qwen/qwen3.5-flash-02-23",
        prompt_mode="compact",
        tool_choice="auto",
        extra_body={
            "reasoning": {
                "exclude": True,
            },
        },
        headers={
            "HTTP-Referer": "http://127.0.0.1:8765/",
            "X-Title": "Mahjong AI Dev Table",
        },
    ),
    LLMProviderPreset(
        "mistral",
        ("mistral", "mistral-ai"),
        "Mistral",
        default_base_url="https://api.mistral.ai/v1",
        default_model_name="mistral-large-latest",
        prompt_mode="compact",
        tool_choice="any",
    ),
    LLMProviderPreset(
        "groq",
        ("groq",),
        "Groq",
        default_base_url="https://api.groq.com/openai/v1",
        default_model_name="openai/gpt-oss-120b",
        prompt_mode="compact",
        tool_choice="required",
    ),
    LLMProviderPreset(
        "together",
        ("together", "together-ai"),
        "Together AI",
        default_base_url="https://api.together.xyz/v1",
        prompt_mode="compact",
        tool_choice="required",
    ),
    LLMProviderPreset(
        "xai",
        ("xai", "x-ai", "grok"),
        "xAI",
        default_base_url="https://api.x.ai/v1",
        default_model_name="grok-4-fast-non-reasoning",
        prompt_mode="compact",
        tool_choice="required",
    ),
    LLMProviderPreset(
        "local-openai",
        ("local-openai", "qwen-local", "vllm", "transformers-local"),
        "Local OpenAI-compatible HTTP",
        default_base_url="http://127.0.0.1:8001/v1",
        default_model_name="qwen3.5-2b-transformers",
        prompt_mode="compact",
    ),
)


def provider_preset(provider: str | None) -> LLMProviderPreset:
    normalized = (provider or "debug").strip().lower()
    for preset in PROVIDER_PRESETS:
        if normalized == preset.provider_id or normalized in preset.aliases:
            return preset
    return LLMProviderPreset(
        normalized,
        (normalized,),
        provider or "OpenAI-compatible HTTP",
        prompt_mode="compact",
    )


def _normalize_model_name(
    model_name: str | None, preset: LLMProviderPreset
) -> str | None:
    normalized = (model_name or "").strip()
    if not normalized:
        return preset.default_model_name
    if (
        preset.mode == "openai_compatible"
        and preset.default_model_name
        and normalized in {"system", "debug-model"}
    ):
        return preset.default_model_name
    return normalized


def build_legal_action_options(actions: tuple[Action, ...]) -> tuple[LegalActionOption, ...]:
    return tuple(
        LegalActionOption(f"a{index}", action, _action_label(action))
        for index, action in enumerate(actions)
    )


def build_llm_prompt(
    *,
    state: TableState,
    actor: Seat,
    rules: RulePlugin,
    legal_actions: tuple[Action, ...],
    recent_event_limit: int = 8,
) -> tuple[dict[str, Any], tuple[LegalActionOption, ...], dict[str, Any], str]:
    observation = ObservationBuilder().build(state, actor, rules)
    options = build_legal_action_options(legal_actions)
    prompt = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "role": "mahjong_decision_controller",
        "instruction": (
            "Choose exactly one action_id from legal_actions. Use only the visible "
            "observation, known exposed tiles, legal action list, and public events. "
            "Do not infer hidden hands, concealed kong tile faces, or future wall tiles."
        ),
        "tool_call_required": TOOL_NAME,
        "rule": {
            "rule_id": rules.rule_id,
            "config_hash": rules.config.config_hash,
            "features": _jsonable(rules.build_rule_features(state, actor)),
        },
        "seat": {
            "id": int(actor),
            "name": actor.name,
        },
        "observation": observation_to_dict(observation),
        "decision_context": {
            "rules": _jsonable(rules.build_rule_features(state, actor)),
            "current_hand": tuple(tile.code for tile in observation.concealed_tiles),
            "discarded_tiles": _seat_tile_mapping(observation.discarded_tiles),
            "known_other_player_tiles": _known_tiles_to_dict(
                observation.known_other_player_tiles
            ),
            "available_actions": tuple(option.as_prompt_dict() for option in options),
            "visibility_notes": (
                "known_other_player_tiles contains only exposed chi/peng/exposed_kong/"
                "added_kong tiles from other players. Concealed kong tile faces and all "
                "unexposed concealed hands are hidden."
            ),
        },
        "recent_visible_events": _visible_events(state, actor, recent_event_limit),
        "legal_actions": tuple(option.as_prompt_dict() for option in options),
        "response_format": {
            "tool_name": TOOL_NAME,
            "arguments": {
                "action_id": "one action_id from legal_actions",
                "reason": "short natural-language explanation for logging",
            },
        },
    }
    tool_schema = choose_action_tool_schema()
    prompt_hash = _hash_json(prompt)
    return prompt, options, tool_schema, prompt_hash


def choose_action_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Choose one legal Mahjong action from the supplied legal_actions list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "The action_id of one entry from legal_actions.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short natural-language explanation for debug logs.",
                    },
                },
                "required": ["action_id"],
                "additionalProperties": False,
            },
        },
    }


class LLMActionAdapter:
    def __init__(
        self,
        *,
        provider: str | None,
        base_url: str | None,
        token: str | None,
        model_name: str | None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.preset = provider_preset(provider)
        self.provider = self.preset.provider_id
        self.base_url = base_url or self.preset.default_base_url
        self.token = token
        self.model_name = _normalize_model_name(model_name, self.preset)
        self.timeout_seconds = timeout_seconds

    def decide(
        self,
        *,
        state: TableState,
        actor: Seat,
        rules: RulePlugin,
        legal_actions: tuple[Action, ...],
    ) -> LLMDecisionTrace:
        if not legal_actions:
            raise AgentExecutionError(f"seat {int(actor)} has no legal actions")
        prompt, options, tool_schema, prompt_hash = build_llm_prompt(
            state=state,
            actor=actor,
            rules=rules,
            legal_actions=legal_actions,
        )
        raw_response = self._call_model(prompt, tool_schema, options)
        selected_action_id, reason = parse_tool_choice(raw_response)
        option_by_id = {option.action_id: option for option in options}
        selected = option_by_id.get(selected_action_id)
        if selected is None:
            raise AgentExecutionError(
                f"LLM selected unknown action_id {selected_action_id!r}"
            )
        return LLMDecisionTrace(
            prompt_schema=PROMPT_SCHEMA_VERSION,
            prompt_hash=prompt_hash,
            prompt=prompt,
            tool_schema=tool_schema,
            raw_response=raw_response,
            selected_action_id=selected_action_id,
            reason=reason,
            selected_action=selected.action,
            validation={
                "action_id_found": True,
                "rule_validated": False,
            },
        )

    def _call_model(
        self,
        prompt: Mapping[str, Any],
        tool_schema: Mapping[str, Any],
        options: tuple[LegalActionOption, ...],
    ) -> Mapping[str, Any]:
        if self.preset.mode == "debug" and not self.base_url:
            selected = _prefer_non_pass(options)
            return {
                "provider": self.provider,
                "provider_label": self.preset.label,
                "mode": "debug_local",
                "tool_call": {
                    "name": TOOL_NAME,
                    "arguments": {
                        "action_id": selected.action_id,
                        "reason": f"Debug adapter selected {selected.label}.",
                    },
                },
            }
        if self.preset.mode == "apple_fm" and not self.base_url:
            return self._call_apple_fm_cli(prompt, tool_schema)
        return self._call_openai_compatible(prompt, tool_schema)

    def _call_apple_fm_cli(
        self,
        prompt: Mapping[str, Any],
        tool_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        schema = {
            "title": "MahjongActionChoice",
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "string",
                    "description": "The action_id of one entry from legal_actions.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short natural-language explanation for debug logs.",
                },
            },
            "required": ["action_id", "reason"],
            "additionalProperties": False,
            "x-order": ["action_id", "reason"],
        }
        instructions = (
            "You are a Mahjong decision controller. Return only JSON matching "
            "the schema. The action_id must be copied from legal_actions. "
            "Do not invent actions, do not use hidden information, and keep reason short."
        )
        model_prompt = compact_prompt_for_model(prompt)
        prompt_text = json.dumps(model_prompt, ensure_ascii=False, sort_keys=True)
        model = self.model_name or "system"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as schema_file:
            json.dump(schema, schema_file)
            schema_file.flush()
            command = [
                "fm",
                "respond",
                "--no-stream",
                "--model",
                model,
                "--instructions",
                instructions,
                "--schema",
                schema_file.name,
                prompt_text,
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except FileNotFoundError as error:
                raise AgentExecutionError("Apple fm CLI was not found") from error
            except subprocess.TimeoutExpired as error:
                raise AgentExecutionError("Apple fm CLI request timed out") from error
            except subprocess.CalledProcessError as error:
                stderr = error.stderr.strip() or error.stdout.strip()
                raise AgentExecutionError(
                    f"Apple fm CLI request failed: {stderr}"
                ) from error
        try:
            arguments = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise AgentExecutionError(
                f"Apple fm CLI returned non-JSON output: {completed.stdout.strip()!r}"
            ) from error
        return {
            "provider": self.provider,
            "provider_label": self.preset.label,
            "mode": "apple_fm_cli",
            "model": model,
            "prompt_mode": "compact",
            "tool_schema": tool_schema,
            "tool_call": {
                "name": TOOL_NAME,
                "arguments": arguments,
            },
        }

    def _call_openai_compatible(
        self,
        prompt: Mapping[str, Any],
        tool_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self.base_url:
            raise AgentExecutionError("LLM controller requires base_url")
        if not self.token:
            raise AgentExecutionError("LLM controller requires token")
        if not self.model_name:
            raise AgentExecutionError("LLM controller requires model_name")

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = self._openai_compatible_payload(prompt, tool_schema)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if self.preset.headers:
            headers.update(self.preset.headers)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            message = body[:1000] if body else str(error)
            raise AgentExecutionError(
                f"LLM request failed: HTTP {error.code}: {message}"
            ) from error
        except urllib.error.URLError as error:
            raise AgentExecutionError(f"LLM request failed: {error}") from error
        try:
            response_json = json.loads(body)
        except json.JSONDecodeError as error:
            raise AgentExecutionError("LLM response was not valid JSON") from error
        return _with_provider_metadata(response_json, self.preset, self.model_name, payload)

    def _openai_compatible_payload(
        self,
        prompt: Mapping[str, Any],
        tool_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        prompt_body: Mapping[str, Any]
        if self.preset.prompt_mode == "compact":
            prompt_body = compact_prompt_for_model(prompt)
        else:
            prompt_body = prompt
        system_content = (
            "You are a Mahjong decision controller. "
            f"{_response_instruction(self.preset)} "
            "Never invent actions or use hidden information."
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_body, ensure_ascii=False, sort_keys=True),
                },
            ],
        }
        if self.preset.use_tools:
            payload["tools"] = [tool_schema]
        if self.preset.use_tools and self.preset.require_tool_choice:
            payload["tool_choice"] = self.preset.tool_choice
        if self.preset.use_tools and self.preset.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = self.preset.parallel_tool_calls
        if self.preset.json_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.preset.include_reasoning_disabled:
            payload["reasoning"] = {"enabled": False}
        if self.preset.thinking_type:
            payload["thinking"] = {"type": self.preset.thinking_type}
        if self.preset.reasoning_effort:
            payload["reasoning_effort"] = self.preset.reasoning_effort
        if self.preset.extra_body:
            payload.update(_jsonable(self.preset.extra_body))
        if self.preset.include_temperature:
            payload["temperature"] = 0.2
        return payload


def parse_tool_choice(raw_response: Mapping[str, Any]) -> tuple[str, str | None]:
    direct = raw_response.get("tool_call")
    if isinstance(direct, Mapping):
        arguments = direct.get("arguments")
        if isinstance(arguments, Mapping):
            return _extract_choice(arguments)

    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                reasoning_content = _reasoning_text(message)
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, Mapping):
                            continue
                        function = call.get("function")
                        if not isinstance(function, Mapping):
                            continue
                        if function.get("name") != TOOL_NAME:
                            continue
                        raw_arguments = function.get("arguments")
                        if isinstance(raw_arguments, str):
                            try:
                                arguments = json.loads(raw_arguments)
                            except json.JSONDecodeError as error:
                                raise AgentExecutionError(
                                    "LLM tool arguments were not valid JSON"
                                ) from error
                        else:
                            arguments = raw_arguments
                        if isinstance(arguments, Mapping):
                            action_id, reason = _extract_choice(arguments)
                            return action_id, reason or reasoning_content

                content = message.get("content")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, Mapping):
                        action_id, reason = _extract_choice(parsed)
                        return action_id, reason or reasoning_content

    raise AgentExecutionError(f"LLM response did not call {TOOL_NAME}")


def _response_instruction(preset: LLMProviderPreset) -> str:
    if preset.use_tools:
        return f"You must call {TOOL_NAME} with an action_id from legal_actions."
    return (
        "Return only JSON with keys action_id and reason. "
        "The action_id must be copied exactly from legal_actions."
    )


def _with_provider_metadata(
    response: Mapping[str, Any],
    preset: LLMProviderPreset,
    model_name: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(response)
    result.setdefault("provider", preset.provider_id)
    result.setdefault("provider_label", preset.label)
    result.setdefault("mode", "openai_compatible_http")
    result.setdefault("model", model_name)
    result["_request_adapter"] = {
        "provider": preset.provider_id,
        "prompt_mode": preset.prompt_mode,
        "use_tools": preset.use_tools,
        "tool_choice": preset.tool_choice,
        "parallel_tool_calls": preset.parallel_tool_calls,
        "json_response_format": preset.json_response_format,
        "thinking_type": preset.thinking_type,
        "reasoning_effort": preset.reasoning_effort,
        "sent_tool_choice": "tool_choice" in payload,
        "sent_reasoning": "reasoning" in payload,
        "sent_thinking": "thinking" in payload,
        "sent_parallel_tool_calls": "parallel_tool_calls" in payload,
        "sent_temperature": "temperature" in payload,
    }
    return result


def _reasoning_text(message: Mapping[str, Any]) -> str | None:
    for key in ("reasoning_content", "reasoning", "thinking", "think"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            text = value.get("content") or value.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


def compact_prompt_for_model(prompt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the smallest prompt that still contains the required decision inputs."""
    context = prompt.get("decision_context", {})
    if not isinstance(context, Mapping):
        context = {}
    legal_actions = prompt.get("legal_actions", ())
    recent_events = prompt.get("recent_visible_events", ())
    return {
        "schema_version": prompt.get("schema_version"),
        "task": "choose one legal Mahjong action_id",
        "tool_call_required": prompt.get("tool_call_required"),
        "rule": _compact_rule(prompt.get("rule"), context.get("rules")),
        "seat": prompt.get("seat"),
        "current_hand": context.get("current_hand", ()),
        "discarded_tiles": context.get("discarded_tiles", {}),
        "known_other_player_tiles": context.get("known_other_player_tiles", {}),
        "legal_actions": tuple(_compact_action(action) for action in legal_actions),
        "recent_visible_events": tuple(_compact_event(event) for event in recent_events)[-4:],
        "return": {
            "action_id": "copy exactly one action_id from legal_actions",
            "reason": "short explanation",
        },
        "visibility": (
            "Use only listed current_hand, discarded_tiles, exposed known_other_player_tiles, "
            "rule, and legal_actions. Other concealed hands, concealed kong faces, and wall "
            "order are hidden."
        ),
    }


def _compact_rule(
    rule: object, rule_features: object
) -> dict[str, Any]:
    rule_mapping = rule if isinstance(rule, Mapping) else {}
    features = rule_features if isinstance(rule_features, Mapping) else {}
    return {
        "rule_id": rule_mapping.get("rule_id"),
        "features": {
            "allow_chi": features.get("allow_chi"),
            "allow_peng": features.get("allow_peng"),
            "allow_self_draw": features.get("allow_self_draw"),
            "allow_discard_win": features.get("allow_discard_win"),
            "require_declared_tenpai": features.get("require_declared_tenpai"),
            "reserve_dead_wall": features.get("reserve_dead_wall"),
            "exhaustive_draw_condition": features.get("exhaustive_draw_condition"),
            "minimum_fan": features.get("minimum_fan"),
        },
    }


def _compact_action(action: object) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        return {}
    return {
        "action_id": action.get("action_id"),
        "kind": action.get("kind"),
        "tile": action.get("tile"),
        "source": action.get("source"),
        "label": action.get("label"),
    }


def _compact_event(event: object) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        return {}
    payload = event.get("payload")
    compact_payload: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in ("tile_type", "tile", "kind", "selected_action", "selected_actor"):
            if key in payload:
                compact_payload[key] = payload[key]
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "actor": event.get("actor"),
        "payload": compact_payload,
    }


def _extract_choice(arguments: Mapping[str, Any]) -> tuple[str, str | None]:
    action_id = arguments.get("action_id")
    if not isinstance(action_id, str) or not action_id:
        raise AgentExecutionError("LLM tool call requires a non-empty action_id")
    reason = arguments.get("reason")
    return action_id, reason if isinstance(reason, str) else None


def observation_to_dict(observation: Observation) -> dict[str, Any]:
    return {
        "schema_version": observation.schema_version,
        "viewer": int(observation.viewer),
        "phase": observation.phase.value,
        "current_actor": int(observation.current_actor),
        "concealed_tiles": tuple(tile.code for tile in observation.concealed_tiles),
        "discarded_tiles": _seat_tile_mapping(observation.discarded_tiles),
        "public_discards": {
            str(int(seat)): tuple(tile.code for tile in tiles)
            for seat, tiles in observation.public_discards.items()
        },
        "public_melds": {
            str(int(seat)): tuple(
                {
                    "kind": kind,
                    "tile_types": tuple(tile.code for tile in tile_types),
                }
                for kind, tile_types in melds
            )
            for seat, melds in observation.public_melds.items()
        },
        "known_other_player_tiles": _known_tiles_to_dict(
            observation.known_other_player_tiles
        ),
        "wall_remaining_by_seat": {
            str(int(seat)): count
            for seat, count in observation.wall_remaining_by_seat.items()
        },
        "response_window_kind": observation.response_window_kind,
        "rule_features": _jsonable(observation.rule_features),
    }


def _seat_tile_mapping(values: Mapping[Seat, tuple[TileType, ...]]) -> dict[str, tuple[str, ...]]:
    return {
        str(int(seat)): tuple(tile.code for tile in tiles)
        for seat, tiles in values.items()
    }


def _known_tiles_to_dict(
    values: Mapping[Seat, tuple[tuple[str, tuple[TileType, ...]], ...]]
) -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        str(int(seat)): tuple(
            {
                "source": "exposed_meld",
                "kind": kind,
                "tile_types": tuple(tile.code for tile in tile_types),
            }
            for kind, tile_types in melds
        )
        for seat, melds in values.items()
    }


def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "operation": action.kind.value,
        "actor": int(action.actor),
        "tile": action.tile.code if action.tile is not None else None,
        "source": int(action.source) if action.source is not None else None,
        "metadata": _jsonable(action.metadata),
    }


def _visible_events(
    state: TableState, actor: Seat, limit: int
) -> tuple[dict[str, Any], ...]:
    visible = [
        event
        for event in state.events
        if event.is_visible_to(actor)
    ][-limit:]
    return tuple(
        {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "actor": int(event.actor) if event.actor is not None else None,
            "payload": _redact_event_payload(event.payload),
        }
        for event in visible
    )


def _redact_event_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _jsonable(payload)


def _prefer_non_pass(options: tuple[LegalActionOption, ...]) -> LegalActionOption:
    for option in options:
        if option.action.kind.value != "pass":
            return option
    return options[0]


def _action_label(action: Action) -> str:
    parts = [action.kind.value]
    if action.tile is not None:
        parts.append(action.tile.code)
    if action.source is not None:
        parts.append(f"from seat {int(action.source)}")
    return " ".join(parts)


def _jsonable(value: Any) -> Any:
    if isinstance(value, TileType):
        return value.code
    if isinstance(value, Seat):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_jsonable(item) for item in value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _hash_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
