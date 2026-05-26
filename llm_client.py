"""Small chat-completion client for LGBO runners."""

from __future__ import annotations

import json
from typing import Any

import requests

import api_config


def call_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    do_sample: bool | None = None,
    timeout: int | None = None,
) -> str:
    key = api_key or api_config.require_api_key()
    url = (base_url or api_config.BASE_URL).rstrip("/") + (endpoint or api_config.ENDPOINT)
    payload: dict[str, Any] = {
        "model": model or api_config.MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": api_config.TEMP if temperature is None else temperature,
        "max_tokens": api_config.MAX_TOKENS if max_tokens is None else max_tokens,
        "do_sample": api_config.DO_SAMPLE if do_sample is None else do_sample,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout or api_config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
