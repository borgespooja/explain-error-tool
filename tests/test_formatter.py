"""Tests for app.formatter."""
from __future__ import annotations

import io
import json

from rich.console import Console

from app.formatter import render_json, render_terminal
from app.models import ErrorExplanation


def _example() -> ErrorExplanation:
    return ErrorExplanation(
        title="ModuleNotFoundError: requests",
        summary="Python could not import the `requests` package.",
        likely_causes=[
            "requests is not installed in the active interpreter",
            "running under a different Python than pip targeted",
        ],
        what_to_try=[
            "python -m pip install requests",
            "python -c 'import sys; print(sys.executable)'",
        ],
        confidence="high",
        suspected_ecosystem="python",
    )


def _render(explanation, **kwargs) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, record=False)
    render_terminal(explanation, console=console, **kwargs)
    return buffer.getvalue()


# ----- render_terminal -----------------------------------------------------


def test_terminal_contains_title_summary_and_sections():
    out = _render(_example())
    assert "ModuleNotFoundError: requests" in out
    assert "Python could not import" in out
    assert "What happened" in out
    assert "Likely causes" in out
    assert "What to try" in out


def test_terminal_numbers_causes_and_steps():
    out = _render(_example())
    assert "1. requests is not installed" in out
    assert "2. running under a different Python" in out
    assert "1. python -m pip install requests" in out


def test_terminal_shows_confidence_and_ecosystem():
    out = _render(_example())
    assert "Confidence:" in out
    assert "high" in out
    assert "Ecosystem:" in out
    assert "python" in out


def test_terminal_omits_empty_sections():
    minimal = ErrorExplanation(
        title="X",
        summary="no causes or steps here",
        likely_causes=[],
        what_to_try=[],
        confidence="low",
    )
    out = _render(minimal)
    assert "Likely causes" not in out
    assert "What to try" not in out
    assert "no causes or steps here" in out


def test_terminal_debug_extras_shown_when_passed():
    detection = {"ecosystem": "python", "confidence": 0.9, "signals": ["Traceback header"]}
    out = _render(_example(), detection=detection, normalized_log="line1\nline2")
    assert "Debug: detection" in out
    assert "Traceback header" in out
    assert "Debug: normalized log" in out
    assert "line1" in out


def test_terminal_debug_extras_hidden_by_default():
    out = _render(_example())
    assert "Debug:" not in out


# ----- render_json ---------------------------------------------------------


def test_render_json_emits_valid_json_with_all_fields():
    buffer = io.StringIO()
    render_json(_example(), stream=buffer)
    payload = json.loads(buffer.getvalue())

    assert payload["title"] == "ModuleNotFoundError: requests"
    assert payload["summary"].startswith("Python could not import")
    assert payload["likely_causes"][0].startswith("requests is not installed")
    assert payload["what_to_try"][0] == "python -m pip install requests"
    assert payload["confidence"] == "high"
    assert payload["suspected_ecosystem"] == "python"


def test_render_json_handles_optional_ecosystem():
    explanation = ErrorExplanation(
        title="t", summary="s", confidence="low", suspected_ecosystem=None
    )
    buffer = io.StringIO()
    render_json(explanation, stream=buffer)
    payload = json.loads(buffer.getvalue())
    assert payload["suspected_ecosystem"] is None
