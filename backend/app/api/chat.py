"""
api/chat.py — AI chat over the dataset (mirrors page 9, AI Chat).
Safe tool-calling: the LLM picks a whitelisted tool + params;
no generated code is ever executed.
"""
from __future__ import annotations
import logging

import json
from typing import List, Optional

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.ai.llm_client import get_client
from app.ai.prompt_builder import build_chat_system_prompt
from app.ai.response_parser import parse_tool_call
from app.ai.tool_dispatcher import dispatch
from app.config import config
from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.serialize import df_records

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)


def _complete(system: str, messages: list[dict]) -> str:
    """The model's reply, from the provider this deployment allows.

    The system prompt carries the dataset's schema and sample rows, so this
    is client data leaving the machine. In privacy mode only the local model
    may see it; before, the route called Groq unconditionally and privacy
    mode did not apply to chat at all.
    """
    from app.ai import local_llm

    def _local() -> str | None:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1])
        user = ((f"Conversation so far:\n{transcript}\n\n" if transcript else "")
                + f"Current request: {messages[-1]['content']}")
        return local_llm.generate(system, user, max_tokens=config.llm_max_tokens)

    if local_llm.privacy_mode():
        text = _local()
        if not text:
            raise HTTPException(
                503, "LLM_PRIVACY_MODE is on and no local model answered. Set "
                     "LOCAL_LLM_URL and LOCAL_LLM_MODEL to use chat. No data "
                     "was sent anywhere.")
        return text

    if config.groq_api_key:
        try:
            return get_client(config.groq_api_key).chat(messages, system)
        except Exception as e:
            # A provider outage, bad key or rate limit used to come back as
            # "Couldn't understand", blaming the user's wording.
            logger.error(f"Chat LLM call failed: {e}")
            text = _local()
            if text:
                return text
            raise HTTPException(503, f"The language model did not respond: {e}")

    text = _local()
    if text:
        return text
    raise HTTPException(
        503, "No language model is configured for chat — set GROQ_API_KEY, "
             "or LOCAL_LLM_URL and LOCAL_LLM_MODEL.")


@router.post("/{ds_id}")
def chat(ds_id: str, req: ChatRequest, owner: str = Depends(current_owner)):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    system = build_chat_system_prompt(df)

    # Carry prior turns so follow-ups ("now split that by region") resolve
    # correctly instead of every message being answered in isolation.
    # Bounded window keeps prompt size predictable.
    history_msgs = [
        {"role": m.role, "content": m.content}
        for m in req.history[-8:]
        if m.role in ("user", "assistant") and m.content.strip()
    ]
    messages = history_msgs + [{"role": "user", "content": req.message}]

    raw = _complete(system, messages)
    parsed = parse_tool_call(raw)
    if not parsed:
        return {"text": "Couldn't understand. Try: 'Show sales by region as bar chart'",
                "figure": None, "table": None}

    result = dispatch(df, parsed["tool"], parsed["params"],
                      parsed.get("explanation", ""))

    if not result.success:
        return {"text": f"Error: {result.error}", "figure": None, "table": None,
                "tool": parsed["tool"]}

    resp: dict = {
        "text": "\n\n".join(x for x in
                            [parsed.get("explanation", ""), result.text_output or ""]
                            if x).strip(),
        "figure": None,
        "table": None,
        "tool": parsed["tool"],
    }
    if result.figure is not None:
        resp["figure"] = json.loads(pio.to_json(result.figure))
    if result.dataframe is not None:
        resp["table"] = df_records(result.dataframe, 50)
    return resp
