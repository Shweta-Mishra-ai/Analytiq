"""
api/chat.py — AI chat over the dataset (mirrors page 9, AI Chat).
Safe tool-calling: the LLM picks a whitelisted tool + params;
no generated code is ever executed.
"""
from __future__ import annotations
import logging

import json
from typing import List

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


@router.post("/{ds_id}")
def chat(ds_id: str, req: ChatRequest, owner: str = Depends(current_owner)):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    # An ordinary question is a group-by, a sort or a correlation, and
    # the app already does all three. A model was only ever turning the
    # sentence into {tool, params} — so try that mapping locally first.
    # It costs nothing, works offline, and cannot invent a column.
    #
    # This is also what the page promises: it offers four generated
    # questions as buttons, and before this those buttons produced a 503
    # whenever no API key was set.
    from app.ai import intent_parser

    local = intent_parser.parse(req.message, df)
    if local:
        return _respond(df, local)

    # Was: a hard 503 unless GROQ_API_KEY was set — which was wrong the
    # moment there was more than one provider, and wronger still now
    # that the model is chosen per task. Ask the router whether anything
    # can do this job instead of interrogating one vendor's key.
    from app.ai.routing import resolve_models
    if not resolve_models("tool_call"):
        examples = intent_parser.answerable_examples(df)
        suggestion = ("\n\nWithout one, these still work on this file:\n"
                      + "\n".join("- " + e for e in examples)) if examples else ""
        return {
            "text": "That question needs a language model to interpret, and "
                    "none is configured. Assign one on the System page, or "
                    "set GROQ_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY, "
                    "TOGETHER_API_KEY or GEMINI_API_KEY." + suggestion,
            "figure": None, "table": None, "tool": "none",
        }

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

    return _respond(df, parsed)


def _respond(df, parsed: dict) -> dict:
    """Run the chosen tool and shape the reply. Shared so a question
    answered locally and one answered through a model come back in
    exactly the same form."""
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
