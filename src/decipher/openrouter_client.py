"""Thin async OpenRouter client for vision chat-completions.

API key resolution (in order):
  1. Argument `api_key` passed to OpenRouterClient()
  2. Environment variable OPENROUTER_API_KEY
  3. .env file in repo root (loaded via python-dotenv if installed)

Endpoint reference: POST https://openrouter.ai/api/v1/chat/completions
Image input format: messages content array with
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "httpx is required for the OpenRouter client. "
        "Install with: pip install httpx"
    ) from exc

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_HEADERS_EXTRA = {
    "HTTP-Referer": "https://github.com/KaushalJainAI/Vesuvius",
    "X-Title": "Vesuvius Decipher",
}


def _load_env() -> None:
    if _DOTENV_AVAILABLE:
        # Search up from CWD for a .env
        root = Path(__file__).resolve().parents[2]
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file)


def _resolve_key(api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key
    _load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it or add to .env "
            "(grab from AIAAS credentials vault, slug 'openrouter')."
        )
    return key


@dataclass
class CompletionResult:
    model: str
    text: str                          # raw text content of choices[0].message
    parsed: Optional[Dict[str, Any]]   # JSON-parsed if model returned valid JSON
    finish_reason: Optional[str]
    usage: Optional[Dict[str, int]]
    error: Optional[str] = None        # set on non-200 / exception


class OpenRouterClient:
    """Async client with concurrency limit and retries."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        concurrency: int = 4,
        timeout_s: float = 90.0,
    ):
        self._key = _resolve_key(api_key)
        self._sem = asyncio.Semaphore(concurrency)
        self._timeout = timeout_s

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            **DEFAULT_HEADERS_EXTRA,
        }

    # ---- public ---------------------------------------------------------

    async def chat_with_image(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        image_bytes: bytes,
        image_mime: str = "image/png",
        temperature: float = 0.1,
        max_tokens: int = 6000,
        force_json: bool = True,
    ) -> CompletionResult:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_mime};base64,{b64}"
        body: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        if force_json:
            body["response_format"] = {"type": "json_object"}

        return await self._post_with_retry(body)

    # ---- internal -------------------------------------------------------

    async def _post_with_retry(self, body: Dict[str, Any]) -> CompletionResult:
        url = f"{BASE_URL}/chat/completions"
        model = body.get("model", "")
        backoffs = [0.0, 2.0, 5.0]
        last_error = ""

        async with self._sem:
            async with httpx.AsyncClient(timeout=self._timeout) as cx:
                for attempt, sleep_s in enumerate(backoffs):
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    try:
                        resp = await cx.post(url, headers=self._headers(), json=body)
                    except (httpx.HTTPError, OSError) as e:
                        last_error = f"network: {e!r}"
                        continue

                    if resp.status_code == 200:
                        return _parse_response(resp.json(), model)

                    # 429 / 5xx → retry; 4xx → bail
                    if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                        last_error = f"http {resp.status_code}: {resp.text[:200]}"
                        continue
                    return CompletionResult(
                        model=model,
                        text="",
                        parsed=None,
                        finish_reason=None,
                        usage=None,
                        error=f"http {resp.status_code}: {resp.text[:500]}",
                    )

        return CompletionResult(
            model=model, text="", parsed=None,
            finish_reason=None, usage=None,
            error=f"all retries failed: {last_error}",
        )


def _parse_response(payload: Dict[str, Any], model: str) -> CompletionResult:
    try:
        choice = payload["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        finish = choice.get("finish_reason")
        usage = payload.get("usage")
    except (KeyError, IndexError, TypeError) as e:
        return CompletionResult(
            model=model, text="", parsed=None,
            finish_reason=None, usage=None,
            error=f"unexpected response shape: {e!r}",
        )

    parsed: Optional[Dict[str, Any]] = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Some models wrap JSON in ```json ... ``` despite response_format
            stripped = text.strip().strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None

    return CompletionResult(
        model=model, text=text, parsed=parsed,
        finish_reason=finish, usage=usage, error=None,
    )
