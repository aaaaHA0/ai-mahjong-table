from __future__ import annotations

import re
from typing import Any


def parse_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the mapping/list/scalar subset used by repository rule configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("- "):
            # Rule loading currently consumes scalar/mapping fields only.
            # Preserve dependency-free startup by ignoring descriptive block
            # lists such as unresolved_decisions and exclusion tables.
            continue
        key, separator, raw_value = line.partition(":")
        if not separator:
            raise ValueError(f"unsupported YAML line: {raw_line!r}")

        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        key = _parse_key(key.strip())
        raw_value = raw_value.strip()
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _parse_key(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(item.strip()) for item in _split_csv(inner)]
    if value.startswith("{") and value.endswith("}"):
        raise ValueError("inline mappings are not supported")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _split_csv(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in value:
        if char in "\"'":
            quote = None if quote == char else char if quote is None else quote
        if char == "," and quote is None:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts
