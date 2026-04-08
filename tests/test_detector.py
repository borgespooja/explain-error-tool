"""Tests for app.detector."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.detector import CONFIDENCE_SATURATION, detect_ecosystem

FIXTURES = Path(__file__).parent / "fixtures"


# ----- Empty / garbage input ------------------------------------------------


def test_empty_input_is_unknown():
    result = detect_ecosystem("")
    assert result == {"ecosystem": "unknown", "confidence": 0.0, "signals": []}


def test_whitespace_only_is_unknown():
    result = detect_ecosystem("   \n\t\n")
    assert result["ecosystem"] == "unknown"
    assert result["confidence"] == 0.0


def test_no_match_is_unknown():
    result = detect_ecosystem("hello world, nothing error-shaped here")
    assert result["ecosystem"] == "unknown"
    assert result["confidence"] == 0.0
    assert result["signals"] == []


# ----- Fixture-driven classification ---------------------------------------


@pytest.mark.parametrize(
    "fixture_name,expected_ecosystem,expected_signal_substrings",
    [
        (
            "python_module_error.txt",
            "python",
            ["Traceback header", "ModuleNotFoundError"],
        ),
        (
            "python_type_error.txt",
            "python",
            ["Traceback header", "Python built-in exception"],
        ),
        (
            "node_missing_module.txt",
            "node",
            ["Cannot find module", "node:internal frame"],
        ),
        (
            "node_econnrefused.txt",
            "node",
            ["Node errno code", "Node.js version banner"],
        ),
        (
            "java_nullpointer.txt",
            "java",
            ["Exception in thread", "NullPointerException", "Caused by:"],
        ),
        (
            "java_spring_startup.txt",
            "java",
            ["org.springframework", "Exception in thread"],
        ),
        (
            "go_nil_pointer.txt",
            "go",
            ["panic:", "goroutine header", "runtime error"],
        ),
    ],
)
def test_fixture_classification(
    fixture_name: str,
    expected_ecosystem: str,
    expected_signal_substrings: list[str],
):
    raw = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    result = detect_ecosystem(raw)

    assert result["ecosystem"] == expected_ecosystem
    assert 0.0 < result["confidence"] <= 1.0
    for needle in expected_signal_substrings:
        assert needle in result["signals"], (
            f"expected signal {needle!r} for {fixture_name}, got {result['signals']}"
        )


# ----- Inline samples: short, distinctive, single-language -----------------


def test_inline_python_traceback():
    log = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in <module>\n'
        "ValueError: bad value\n"
    )
    result = detect_ecosystem(log)
    assert result["ecosystem"] == "python"


def test_inline_node_cannot_find_module():
    log = (
        "Error: Cannot find module 'express'\n"
        "    at require (node:internal/modules/cjs/helpers:110:18)\n"
        "Node.js v20.11.0\n"
    )
    result = detect_ecosystem(log)
    assert result["ecosystem"] == "node"


def test_inline_java_npe():
    log = (
        'Exception in thread "main" java.lang.NullPointerException\n'
        "\tat com.example.App.main(App.java:8)\n"
    )
    result = detect_ecosystem(log)
    assert result["ecosystem"] == "java"


def test_inline_go_panic():
    log = (
        "panic: runtime error: index out of range [3] with length 3\n"
        "\n"
        "goroutine 1 [running]:\n"
        "main.main()\n"
        "\t/app/main.go:7 +0x1c\n"
    )
    result = detect_ecosystem(log)
    assert result["ecosystem"] == "go"


# ----- Confidence behavior --------------------------------------------------


def test_confidence_in_unit_interval():
    raw = (FIXTURES / "python_module_error.txt").read_text(encoding="utf-8")
    result = detect_ecosystem(raw)
    assert 0.0 < result["confidence"] <= 1.0


def test_confidence_scales_with_signal_count():
    """A log hitting multiple strong signals scores higher than one weak hit."""
    weak = "stack.py line 10 - something went sideways"  # only ".py path"
    strong = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "ModuleNotFoundError: No module named 'foo'\n"
    )
    weak_result = detect_ecosystem(weak)
    strong_result = detect_ecosystem(strong)
    assert weak_result["ecosystem"] == "python"
    assert strong_result["ecosystem"] == "python"
    assert strong_result["confidence"] > weak_result["confidence"]


def test_confidence_saturates_at_one():
    """A log stuffed with strong signals should hit the 1.0 ceiling."""
    raw = (FIXTURES / "java_spring_startup.txt").read_text(encoding="utf-8")
    result = detect_ecosystem(raw)
    assert result["confidence"] <= 1.0
    # This fixture has >= 4 strong Java signals, easily above saturation.
    assert result["confidence"] >= min(1.0, 8.0 / CONFIDENCE_SATURATION) * 0.5


# ----- Ambiguous cases ------------------------------------------------------


def test_ambiguous_log_applies_margin_penalty():
    """When two ecosystems tie on one weak signal each, confidence drops.

    Mentioning both a ``.py`` path (python, weight 0.5) and a ``.js`` path
    (node, weight 0.5) should pick one but with a notable margin penalty.
    """
    ambiguous = "see script.py and helper.js for details"
    result = detect_ecosystem(ambiguous)
    assert result["ecosystem"] in {"python", "node"}
    # Equal scores -> full 50% penalty on an already-small base. Expect <= 0.05.
    assert result["confidence"] <= 0.05


def test_java_wins_over_node_when_java_signals_dominate():
    """A Java stack with an incidental .js mention should still classify as Java."""
    mixed = (
        'Exception in thread "main" java.lang.NullPointerException\n'
        "\tat com.example.App.main(App.java:8)\n"
        "Caused by: java.lang.IllegalStateException\n"
        "# note: originally triggered by a call from frontend.js\n"
    )
    result = detect_ecosystem(mixed)
    assert result["ecosystem"] == "java"
    assert result["confidence"] > 0.3


def test_strong_unique_signal_beats_weak_noise():
    log = (
        "panic: runtime error: invalid memory address\n"
        "goroutine 42 [running]:\n"
        "# random .py mention in a comment\n"
    )
    result = detect_ecosystem(log)
    assert result["ecosystem"] == "go"


# ----- Result shape ---------------------------------------------------------


def test_result_shape_is_stable():
    result = detect_ecosystem("Traceback (most recent call last):")
    assert set(result.keys()) == {"ecosystem", "confidence", "signals"}
    assert isinstance(result["ecosystem"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["signals"], list)
    assert all(isinstance(s, str) for s in result["signals"])
