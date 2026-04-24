"""LLM generation via OpenAI/Groq API with local fallback (no LangChain)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from .local_llm import local_generate
from .prompt_builder import PromptBundle


def _provider_choice() -> str:
    if LLM_PROVIDER in {"openai", "groq", "local"}:
        return LLM_PROVIDER
    # auto mode: prefer Groq if set, then OpenAI, else local fallback
    if GROQ_API_KEY:
        return "groq"
    if OPENAI_API_KEY:
        return "openai"
    return "local"


def _request_body(bundle: PromptBundle, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": bundle.system},
            {"role": "user", "content": bundle.user},
        ],
        "temperature": 0.2,
    }


async def _chat_http_async(
    *,
    url: str,
    api_key: str,
    model: str,
    bundle: PromptBundle,
    provider_label: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = _request_body(bundle, model)
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
        data = r.json()
    content = data["choices"][0]["message"]["content"]
    return {"error": None, "model": f"{provider_label}:{model}", "content": content, "raw": data}


def _chat_http_sync(
    *,
    url: str,
    api_key: str,
    model: str,
    bundle: PromptBundle,
    provider_label: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = _request_body(bundle, model)
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=headers, content=json.dumps(body))
        r.raise_for_status()
        data = r.json()
    content = data["choices"][0]["message"]["content"]
    return {"error": None, "model": f"{provider_label}:{model}", "content": content, "raw": data}


async def chat_completion(bundle: PromptBundle) -> dict[str, Any]:
    provider = _provider_choice()
    if provider == "groq" and GROQ_API_KEY:
        return await _chat_http_async(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            bundle=bundle,
            provider_label="groq",
        )
    if provider == "openai" and OPENAI_API_KEY:
        return await _chat_http_async(
            url="https://api.openai.com/v1/chat/completions",
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            bundle=bundle,
            provider_label="openai",
        )
    # local fallback if provider not configured or key missing
    return local_generate(bundle)


def chat_completion_sync(bundle: PromptBundle) -> dict[str, Any]:
    provider = _provider_choice()
    if provider == "groq" and GROQ_API_KEY:
        return _chat_http_sync(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            bundle=bundle,
            provider_label="groq",
        )
    if provider == "openai" and OPENAI_API_KEY:
        return _chat_http_sync(
            url="https://api.openai.com/v1/chat/completions",
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            bundle=bundle,
            provider_label="openai",
        )
    return local_generate(bundle)
