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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning(f"No JSON found in: {text[:200]}")
        return None

    # ── Parse JSON ─────────────────────────────────────────
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON error: {e}")
        return None

    # ── Validate ───────────────────────────────────────────
    if "tool" not in data:
        return None

    data.setdefault("params", {})
    data.setdefault("explanation", "")

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
