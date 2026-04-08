"""Pipeline orchestration.

The :func:`run` function is the single entrypoint the CLI calls after it has
resolved the raw input. It wires together parser → detector → prompts → llm →
formatter. Each stage lives in its own module and is independently testable.
"""
from __future__ import annotations

from typing import Optional

from app import config, detector, formatter, llm, parser, prompts
from app.models import ErrorExplanation


class ExplainError(Exception):
    """Top-level error surfaced to the CLI as a non-zero exit."""


class ConfigError(ExplainError):
    """Configuration problem (missing API key, unsupported provider, etc.)."""


def run(
    *,
    raw_input: str,
    json_output: bool = False,
    debug: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[str] = None,
) -> ErrorExplanation:
    """Run the full explanation pipeline and render the result.

    Returns the structured explanation so callers (tests, future integrations)
    can consume it programmatically.
    """
    settings = config.Settings.from_env()
    chosen_provider = (provider or settings.provider).lower()
    chosen_model = config.resolve_model(settings, chosen_provider, model)

    normalized = parser.normalize_log(raw_input, max_chars=settings.max_log_chars)
    detection = detector.detect_ecosystem(normalized)

    system_prompt, user_prompt = prompts.build(
        normalized_log=normalized,
        detection=detection,
        user_context=context,
    )

    try:
        client = llm.get_client(
            provider=chosen_provider,
            model=chosen_model,
            settings=settings,
        )
    except llm.LLMConfigError as exc:
        raise ConfigError(str(exc)) from exc

    try:
        explanation = client.explain_error(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except llm.LLMConfigError as exc:
        raise ConfigError(str(exc)) from exc
    except llm.LLMError as exc:
        raise ExplainError(str(exc)) from exc

    if not explanation.suspected_ecosystem:
        explanation.suspected_ecosystem = detection.get("ecosystem")

    if json_output:
        formatter.render_json(explanation)
    else:
        formatter.render_terminal(
            explanation,
            detection=detection if debug else None,
            normalized_log=normalized if debug else None,
        )

    return explanation
