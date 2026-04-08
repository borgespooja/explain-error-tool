"""Structured response schemas used across the pipeline.

The :class:`ErrorExplanation` is the contract between the LLM layer and the
formatter. The LLM is instructed to emit JSON matching this schema exactly;
pydantic validation is how we enforce it on the way in.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ConfidenceLevel = Literal["low", "medium", "high"]


class ErrorExplanation(BaseModel):
    """LLM-produced structured explanation of an error log."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=120,
                       description="Short label for the error type.")
    summary: str = Field(..., min_length=1,
                         description="One-paragraph plain-English explanation.")
    likely_causes: list[str] = Field(
        default_factory=list,
        description="Ranked list of likely root causes, most likely first.",
    )
    what_to_try: list[str] = Field(
        default_factory=list,
        description="Concrete debugging steps to try, in recommended order.",
    )
    confidence: ConfidenceLevel = Field(
        ..., description="Qualitative confidence: 'low' | 'medium' | 'high'."
    )
    suspected_ecosystem: str | None = Field(
        default=None,
        description="Detected ecosystem (python, node, java, go, or unknown).",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> Any:
        """Accept common variants ('High', 'moderate', 'uncertain') and map them."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        aliases = {
            "unknown": "low",
            "uncertain": "low",
            "moderate": "medium",
            "med": "medium",
            "very high": "high",
        }
        return aliases.get(normalized, normalized)

    @field_validator("likely_causes", "what_to_try", mode="before")
    @classmethod
    def _clean_string_list(cls, value: Any) -> Any:
        """Drop empties and trim whitespace; leave non-list values for pydantic."""
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    cleaned.append(stripped)
        return cleaned
