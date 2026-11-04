import json
import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None

    # ── Strip markdown fences ──────────────────────────────
    text = re.sub(r"```(?:json)?", "", raw.strip())
    text = text.replace("```", "").strip()

    # ── Extract first { } block ────────────────────────────
    # The first JSON object that names a tool. A greedy "{.*}" ran from the
    # first brace to the last, so any brace in text after the object (a
    # note, an emoticon) made a valid answer unparseable and the user was
    # told their question could not be understood.
    decoder = json.JSONDecoder()
    data = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            data = obj
            break
    if data is None:
        logger.warning(f"No tool call found in: {text[:200]}")
        return None

    data.setdefault("params", {})
    data.setdefault("explanation", "")
    if not isinstance(data["params"], dict) or not isinstance(data["tool"], str):
        logger.warning(f"Malformed tool call: {text[:200]}")
        return None
    if not isinstance(data["explanation"], str):
        data["explanation"] = ""

    return data


def parse_insight_array(raw: str) -> list:
    text  = re.sub(r"```(?:json)?", "", raw.strip())
    text  = text.replace("```", "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
