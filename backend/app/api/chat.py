"""
api/chat.py — AI chat over the dataset (mirrors page 9, AI Chat).
Safe tool-calling: the LLM picks a whitelisted tool + params;
no generated code is ever executed.
"""
from __future__ import annotations

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

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)


@router.post("/{ds_id}")
def chat(ds_id: str, req: ChatRequest, owner: str = Depends(current_owner)):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    if not config.groq_api_key:
        raise HTTPException(503, "GROQ_API_KEY is not configured on the server")

    client = get_client(config.groq_api_key)
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

    raw = client.chat_safe(
        messages=messages,
        system=system,
    )
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
