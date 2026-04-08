"""Tests for app.parser."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.parser import (
    is_salient,
    normalize_log,
    normalize_whitespace,
    strip_ansi,
    truncate_lines,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ----- strip_ansi -----------------------------------------------------------


def test_strip_ansi_removes_color_codes():
    assert strip_ansi("\x1b[31mERROR\x1b[0m: boom") == "ERROR: boom"


def test_strip_ansi_removes_cursor_moves():
    assert strip_ansi("line1\x1b[2Aline2") == "line1line2"


def test_strip_ansi_removes_osc_sequences():
    osc = "\x1b]0;window title\x07prompt$"
    assert strip_ansi(osc) == "prompt$"


def test_strip_ansi_no_op_on_plain_text():
    assert strip_ansi("plain text, no escapes") == "plain text, no escapes"


# ----- normalize_whitespace -------------------------------------------------


def test_normalize_whitespace_expands_tabs_and_trims_right():
    text = "line\twith tab   \nnext"
    assert normalize_whitespace(text) == "line    with tab\nnext"


def test_normalize_whitespace_collapses_blank_runs():
    text = "a\n\n\n\nb\n\n\nc"
    assert normalize_whitespace(text) == "a\n\nb\n\nc"


def test_normalize_whitespace_strips_outer_blank_lines():
    text = "\n\n  \nreal content\n\n\n"
    assert normalize_whitespace(text) == "real content"


def test_normalize_whitespace_preserves_single_blanks():
    text = "a\n\nb"
    assert normalize_whitespace(text) == "a\n\nb"


# ----- is_salient -----------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Error: something failed",
        "Exception in thread main",
        "Traceback (most recent call last):",
        "Caused by: some other thing",
        "Build failed with exit code 1",
        "panic: runtime error",
        "ERROR: caps also match",  # case-insensitive
    ],
)
def test_is_salient_true(line: str):
    assert is_salient(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "",
        "just a normal log line",
        "processing item 42",
        "    at com.example.Foo.bar(Foo.java:12)",
    ],
)
def test_is_salient_false(line: str):
    assert is_salient(line) is False


# ----- truncate_lines -------------------------------------------------------


def test_truncate_noop_when_under_budget():
    text = "a\nb\nc"
    assert truncate_lines(text, head_lines=10, tail_lines=10, max_chars=100) == text


def test_truncate_keeps_head_tail_and_salient():
    lines = [f"line {i}" for i in range(200)]
    lines[100] = "some ERROR happened here"
    lines[150] = "Traceback (most recent call last):"
    text = "\n".join(lines)

    out = truncate_lines(text, head_lines=5, tail_lines=5, max_chars=10_000)

    # Head
    assert "line 0" in out
    assert "line 4" in out
    # Tail
    assert "line 199" in out
    assert "line 195" in out
    # Salient lines preserved even though they're in the middle
    assert "some ERROR happened here" in out
    assert "Traceback (most recent call last):" in out
    # A boring middle line that is neither head, tail, nor salient should go
    assert "line 50" not in out
    # Gap markers are inserted for the dropped runs
    assert "lines omitted" in out


def test_truncate_marker_counts_gaps_correctly():
    lines = ["head"] + [f"middle {i}" for i in range(10)] + ["tail"]
    text = "\n".join(lines)
    out = truncate_lines(text, head_lines=1, tail_lines=1, max_chars=10_000)
    assert "head" in out
    assert "tail" in out
    assert "[... 10 lines omitted ...]" in out


def test_truncate_enforces_max_chars_hard_cap():
    # Head + tail is small, but a huge middle salient line blows the budget.
    big_salient = "ERROR: " + ("x" * 5000)
    lines = ["start", big_salient, "end"]
    text = "\n".join(lines)
    out = truncate_lines(text, head_lines=1, tail_lines=1, max_chars=500)
    assert len(out) <= 500
    assert "truncated for length" in out


def test_truncate_singular_line_in_marker():
    lines = ["h", "x", "t"]
    text = "\n".join(lines)
    out = truncate_lines(text, head_lines=1, tail_lines=1, max_chars=10_000)
    assert "[... 1 line omitted ...]" in out


# ----- normalize_log (integration) ------------------------------------------


def test_normalize_log_strips_ansi_and_collapses_blanks():
    raw = "\x1b[31mERROR\x1b[0m: boom\n\n\n\ntail\n"
    assert normalize_log(raw, max_chars=100) == "ERROR: boom\n\ntail"


def test_normalize_log_applies_truncation():
    lines = [f"line {i}" for i in range(500)]
    lines[250] = "panic: nil pointer"
    raw = "\n".join(lines)
    out = normalize_log(raw, max_chars=10_000, head_lines=5, tail_lines=5)
    assert "line 0" in out
    assert "line 499" in out
    assert "panic: nil pointer" in out
    assert "line 100" not in out


def test_normalize_log_handles_empty_input():
    assert normalize_log("") == ""
    assert normalize_log("   \n\n  ") == ""


# ----- Fixture-driven smoke tests -------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,expected_marker",
    [
        ("python_module_error.txt", "ModuleNotFoundError"),
        ("python_type_error.txt", "TypeError"),
        ("node_missing_module.txt", "Cannot find module"),
        ("node_econnrefused.txt", "ECONNREFUSED"),
        ("java_nullpointer.txt", "NullPointerException"),
        ("java_spring_startup.txt", "APPLICATION FAILED TO START"),
        ("go_nil_pointer.txt", "nil pointer dereference"),
    ],
)
def test_normalize_log_preserves_key_marker(fixture_name: str, expected_marker: str):
    raw = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    out = normalize_log(raw)
    assert expected_marker in out
    # Nothing should blow past the configured char cap.
    assert len(out) <= 16_000
