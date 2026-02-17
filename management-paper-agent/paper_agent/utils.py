from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(text: str, max_length: int = 40) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", "-", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"[^\w\-]+", "-", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"-{2,}", "-", cleaned, flags=re.UNICODE).strip("-")
    if not cleaned:
        return "paper"
    return cleaned[:max_length]


def render_template(template: str, values: Dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def extract_json_payload(text: str) -> Any:
    content = text.strip()
    if not content:
        return None

    candidates = [content]
    candidates.extend(_extract_fenced_json_blocks(content))

    obj_candidate = _extract_balanced(content, "{", "}")
    if obj_candidate:
        candidates.append(obj_candidate)
    arr_candidate = _extract_balanced(content, "[", "]")
    if arr_candidate:
        candidates.append(arr_candidate)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_fenced_json_blocks(text: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL):
        blocks.append(match.group(1).strip())
    for match in re.finditer(r"```\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL):
        blocks.append(match.group(1).strip())
    return blocks


def _extract_balanced(text: str, left: str, right: str) -> str | None:
    start = text.find(left)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == left:
            depth += 1
            continue

        if char == right:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None
