"""LLM provider abstraction and a concrete Anthropic implementation.

Providers implement :class:`LLMClient`. :func:`get_client` is a tiny registry so
new providers can be dropped in without changing callers. Failures bubble up as
:class:`LLMError` (or its subclasses) which the CLI translates to a clean exit
message.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.models import ErrorExplanation

logger = logging.getLogger(__name__)


# ----- Error hierarchy -----------------------------------------------------


class LLMError(Exception):
    """Base class for any LLM-layer failure surfaced to the caller."""


class LLMConfigError(LLMError):
    """Missing or invalid provider configuration (e.g. no API key)."""


class LLMResponseError(LLMError):
    """LLM returned something we could not parse or validate."""


# ----- Abstraction ---------------------------------------------------------


class LLMClient(ABC):
    """Minimal interface every provider must implement."""

    @abstractmethod
    def explain_error(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ErrorExplanation:
        """Call the provider and return a validated :class:`ErrorExplanation`."""


# ----- Concrete provider ---------------------------------------------------


class AnthropicClient(LLMClient):
    """Concrete client backed by the Anthropic Messages API."""

    _MAX_ATTEMPTS = 2
    _RETRY_BACKOFF_SECONDS = 1.0
    _DEFAULT_MAX_TOKENS = 1024

    def __init__(self, *, model: str, api_key: str, timeout_seconds: float) -> None:
        if not api_key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "add your key, or export ANTHROPIC_API_KEY in your shell."
            )
        if not model:
            raise LLMConfigError(
                "No model specified. Set EXPLAIN_ERROR_MODEL or pass --model."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dep boundary
            raise LLMConfigError(
                "The 'anthropic' package is not installed. "
                "Run: pip install -e '.[dev]' or pip install anthropic"
            ) from exc
        self._client = Anthropic(api_key=api_key, timeout=timeout_seconds)
        self._model = model

    def explain_error(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ErrorExplanation:
        raw = self._call_with_retries(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return self._parse_response(raw)

    def _call_with_retries(self, *, system_prompt: str, user_prompt: str) -> str:
        from anthropic import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )

        last_exc: Exception | None = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=self._DEFAULT_MAX_TOKENS,
                )
                return _extract_text(response)
            except AuthenticationError as exc:
                raise LLMConfigError(
                    f"Anthropic rejected the API key: {exc}"
                ) from exc
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                last_exc = exc
                logger.debug("transient Anthropic error on attempt %d: %s", attempt, exc)
                if attempt < self._MAX_ATTEMPTS:
                    time.sleep(self._RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise LLMError(
                    f"Anthropic call failed after {attempt} attempts: {exc}"
                ) from exc
            except APIStatusError as exc:
                is_server_err = 500 <= getattr(exc, "status_code", 0) < 600
                if is_server_err and attempt < self._MAX_ATTEMPTS:
                    last_exc = exc
                    time.sleep(self._RETRY_BACKOFF_SECONDS * attempt)
                    continue
                status = getattr(exc, "status_code", "?")
                raise LLMError(f"Anthropic API error ({status}): {exc}") from exc

        assert last_exc is not None  # loop guarantees this
        raise LLMError(str(last_exc))

    @staticmethod
    def _parse_response(raw: str) -> ErrorExplanation:
        payload = _extract_json_object(raw)
        if payload is None:
            raise LLMResponseError(
                "Model response did not contain a JSON object. "
                f"Raw output (truncated): {raw[:400]!r}"
            )
        try:
            return ErrorExplanation.model_validate(payload)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Model output did not match the expected schema: {exc}. "
                f"Raw output (truncated): {raw[:400]!r}"
            ) from exc


# ----- Helpers -------------------------------------------------------------


def _extract_text(response: Any) -> str:
    """Return concatenated text from the Anthropic response's content blocks."""
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "") or ""
            if text:
                chunks.append(text)
    return "\n".join(chunks)


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object in ``raw`` or ``None`` if none can be found.

    Three fallbacks, in order of increasing tolerance for noise:

    1. Whole-string ``json.loads``.
    2. Regex for a ```` ```json ... ``` ```` fenced block.
    3. Balanced-brace scan that respects string literals.
    """
    if not raw:
        return None
    stripped = raw.strip()

    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = _FENCED_JSON_RE.search(stripped)
    if match:
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    candidate = _find_balanced_object(stripped)
    if candidate is not None:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            return None
    return None


def _find_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring in ``text``, or ``None``."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ----- Registry ------------------------------------------------------------


_SUPPORTED_PROVIDERS = ("anthropic",)


def get_client(*, provider: str, model: str, settings: Settings) -> LLMClient:
    """Return a concrete :class:`LLMClient` for ``provider``.

    Raises :class:`LLMConfigError` if the provider is unknown or required
    configuration (API key, model) is missing.
    """
    normalized = provider.lower()
    if normalized == "anthropic":
        return AnthropicClient(
            model=model,
            api_key=settings.api_key_for("anthropic") or "",
            timeout_seconds=settings.timeout_seconds,
        )
    raise LLMConfigError(
        f"Unsupported provider: {provider!r}. "
        f"Supported: {list(_SUPPORTED_PROVIDERS)}."
    )
