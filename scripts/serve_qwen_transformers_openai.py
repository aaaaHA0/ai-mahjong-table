from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor

from mahjong_ai.agents.llm import TOOL_NAME, compact_prompt_for_model


class ChatMessage(BaseModel):
    role: str
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float | None = 0.0
    max_tokens: int | None = Field(default=96, alias="max_completion_tokens")

    class Config:
        populate_by_name = True
        extra = "allow"


class QwenTransformersServer:
    def __init__(
        self,
        *,
        model_path: str,
        served_model_name: str,
        max_new_tokens: int,
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def _ensure_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        print(f"Loading Qwen model from {self.model_path}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        print("Qwen model loaded.", flush=True)

    def chat_completions(self, request: ChatCompletionRequest) -> dict[str, Any]:
        prompt = self._extract_user_prompt(request.messages)
        model_prompt = self._compact_if_mahjong_prompt(prompt)
        action_ids = self._action_ids(model_prompt)
        generated = self._generate_tool_json(
            model_prompt,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens or self.max_new_tokens,
        )
        arguments = self._parse_or_fallback(generated, action_ids)
        return self._openai_tool_response(arguments, generated)

    @staticmethod
    def _extract_user_prompt(messages: list[ChatMessage]) -> Any:
        for message in reversed(messages):
            if message.role == "user":
                content = message.content
                if isinstance(content, str):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content
                return content
        return {}

    @staticmethod
    def _compact_if_mahjong_prompt(prompt: Any) -> Any:
        if isinstance(prompt, Mapping) and "decision_context" in prompt:
            return compact_prompt_for_model(prompt)
        return prompt

    @staticmethod
    def _action_ids(prompt: Any) -> list[str]:
        if not isinstance(prompt, Mapping):
            return []
        actions = prompt.get("legal_actions", ())
        if not actions:
            context = prompt.get("decision_context", {})
            if isinstance(context, Mapping):
                actions = context.get("available_actions", ())
        action_ids: list[str] = []
        if isinstance(actions, (list, tuple)):
            for action in actions:
                if isinstance(action, Mapping) and isinstance(action.get("action_id"), str):
                    action_ids.append(action["action_id"].strip())
        return action_ids

    def _generate_tool_json(
        self,
        prompt: Any,
        *,
        temperature: float | None,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Mahjong action selector. Return only a compact JSON "
                    "object with action_id and reason. The action_id must exactly match "
                    "one entry in legal_actions. Do not return markdown."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            },
        ]
        self._ensure_loaded()
        assert self.processor is not None
        assert self.model is not None
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max(64, min(max_new_tokens, self.max_new_tokens)),
                do_sample=bool(temperature and temperature > 0),
                temperature=temperature if temperature and temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[-1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _parse_or_fallback(generated: str, action_ids: list[str]) -> dict[str, str]:
        parsed = _extract_json_object(generated)
        if parsed is None:
            parsed = _extract_json_fields(generated)
        action_id = parsed.get("action_id")
        reason = parsed.get("reason")
        if isinstance(action_id, str):
            action_id = action_id.strip()
        if not isinstance(action_id, str) or action_id not in action_ids:
            action_id = action_ids[0] if action_ids else ""
            reason = "Fallback selected the first legal action after invalid model JSON."
        if not isinstance(reason, str) or not reason:
            reason = "Selected a legal action from the supplied action list."
        return {
            "action_id": action_id,
            "reason": reason[:240],
        }

    def _openai_tool_response(
        self, arguments: dict[str, str], generated: str
    ) -> dict[str, Any]:
        created = int(time.time())
        return {
            "id": f"chatcmpl-qwen-transformers-{created}",
            "object": "chat.completion",
            "created": created,
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_choose_mahjong_action",
                                "type": "function",
                                "function": {
                                    "name": TOOL_NAME,
                                    "arguments": json.dumps(
                                        arguments,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "raw_model_output": generated,
        }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    candidates.extend(match.group(0) for match in re.finditer(r"\{[^{}]*\}", stripped))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_json_fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("action_id", "reason"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if match:
            result[key] = match.group(1)
    return result


def create_app(server: QwenTransformersServer) -> FastAPI:
    app = FastAPI(title="Qwen Transformers OpenAI-compatible API")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        return server.chat_completions(request)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3.5-2B")
    parser.add_argument("--served-model-name", default="qwen3.5-2b-transformers")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    server = QwenTransformersServer(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        max_new_tokens=args.max_new_tokens,
    )
    uvicorn.run(create_app(server), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
