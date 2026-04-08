"""System and user prompt builders for the LLM call.

The system prompt is static and schema-locked. The user prompt carries
everything specific to this invocation: detector output, optional user
context, and the normalized log.
"""
from __future__ import annotations

import json
from typing import Optional

from app.detector import Detection

SYSTEM_PROMPT = """You are an expert debugging assistant helping software engineers understand error logs.

Your job:
- Read the error log the user provides and explain it in plain English.
- Rank likely root causes from most to least likely. Use the log as the primary evidence.
- Suggest concrete next steps a developer can actually run or check.
- Be honest about uncertainty. If the log is sparse or ambiguous, say so in the summary and lower the confidence.
- Do not invent facts, frameworks, file paths, or line numbers that are not in the log. If something is unclear, say so.
- Keep each cause and step tight — one or two sentences, no filler.
- A heuristic ecosystem hint is provided; override it in `suspected_ecosystem` if the log clearly disagrees.

Output format:
- Respond with a single valid JSON object, and nothing else. No prose before or after. No markdown fences.
- The JSON object MUST match this schema exactly:

{
  "title": "short label for the error (1–80 characters)",
  "summary": "one-paragraph plain-English explanation",
  "likely_causes": ["most likely cause", "next most likely", "..."],
  "what_to_try": ["first concrete step", "second step", "..."],
  "confidence": "low | medium | high",
  "suspected_ecosystem": "python | node | java | go | unknown"
}

- likely_causes and what_to_try should each contain 2–5 items, ranked by usefulness.
- `confidence` reflects your confidence in this explanation, not the detector's.
- Never include additional fields, comments, or trailing text."""


def build(
    *,
    normalized_log: str,
    detection: Detection,
    user_context: Optional[str],
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the LLM call."""
    parts: list[str] = []

    parts.append("Ecosystem hint (from local heuristics; may be wrong):")
    parts.append(
        json.dumps(
            {
                "ecosystem": detection.get("ecosystem"),
                "confidence": detection.get("confidence"),
                "signals": detection.get("signals", []),
            },
            indent=2,
        )
    )

    if user_context:
        parts.append("\nUser context:")
        parts.append(user_context.strip())

    parts.append("\nError log:")
    parts.append("```")
    parts.append(normalized_log if normalized_log else "(empty)")
    parts.append("```")

    parts.append(
        "\nReturn ONLY the JSON object described in the system prompt. "
        "No prose, no markdown fences, no commentary."
    )

    return SYSTEM_PROMPT, "\n".join(parts)
