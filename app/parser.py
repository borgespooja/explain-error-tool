"""Log normalization and preprocessing.

Deterministic cleanup that runs before any LLM call. The goals are:

* Remove ANSI color / cursor escape sequences.
* Normalize whitespace (tabs -> spaces, trim right side, strip outer blank lines).
* Collapse runs of blank lines to a single blank line.
* Keep logs bounded in size without losing the most useful lines.

Truncation strategy (see :func:`truncate_lines`):

1. If the log fits under the line and char budgets, return it as-is.
2. Otherwise keep the first ``head_lines`` and last ``tail_lines`` lines plus
   any line matching one of the salient keywords (``error``, ``exception``,
   ``traceback``, ``caused by``, ``failed``, ``panic``). Original ordering is
   preserved and removed regions are marked with a compact ``[... N lines
   omitted ...]`` placeholder so the LLM can tell the log was trimmed.
3. If the result still exceeds ``max_chars``, hard-truncate the middle with a
   second placeholder to guarantee an upper bound.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_HEAD_LINES = 40
DEFAULT_TAIL_LINES = 80
DEFAULT_MAX_CHARS = 16000

# Covers CSI (ESC[...), OSC (ESC]...BEL or ESC\\), and single-char ESC sequences.
_ANSI_RE = re.compile(
    r"""
    \x1B                          # ESC
    (?:
        \[[0-?]*[ -/]*[@-~]       # CSI: ESC [ ... final-byte
        |\]                       # OSC: ESC ]
         [^\x07\x1B]*             #   ... payload (no BEL / ESC)
         (?:\x07|\x1B\\)          #   terminator: BEL or ESC \
        |[@-Z\\-_]                # other 2-char sequences
    )
    """,
    re.VERBOSE,
)

# Tokens we treat as "salient" — a line containing any of these is preserved
# even if truncation would otherwise drop it. Matched case-insensitively.
SALIENT_KEYWORDS: tuple[str, ...] = (
    "error",
    "exception",
    "traceback",
    "caused by",
    "failed",
    "panic",
)

_SALIENT_RE = re.compile(
    "|".join(re.escape(kw) for kw in SALIENT_KEYWORDS),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseOptions:
    """Tunable knobs for :func:`normalize_log`."""

    max_chars: int = DEFAULT_MAX_CHARS
    head_lines: int = DEFAULT_HEAD_LINES
    tail_lines: int = DEFAULT_TAIL_LINES


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colors, cursor moves, OSC) from ``text``."""
    return _ANSI_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Expand tabs, trim trailing spaces, and collapse blank-line runs.

    Leading and trailing blank lines are removed. Internal runs of two or more
    blank lines collapse to a single blank line so the LLM is not fed walls of
    empty space.
    """
    expanded = text.expandtabs(4)
    lines = [line.rstrip() for line in expanded.splitlines()]

    collapsed: list[str] = []
    blank_run = False
    for line in lines:
        if line == "":
            if blank_run:
                continue
            blank_run = True
            collapsed.append("")
        else:
            blank_run = False
            collapsed.append(line)

    # Strip leading/trailing blank lines.
    start, end = 0, len(collapsed)
    while start < end and collapsed[start] == "":
        start += 1
    while end > start and collapsed[end - 1] == "":
        end -= 1
    return "\n".join(collapsed[start:end])


def is_salient(line: str) -> bool:
    """Return True if ``line`` contains any of the salient error keywords."""
    return bool(_SALIENT_RE.search(line))


def truncate_lines(
    text: str,
    *,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Keep head + tail + salient lines; enforce ``max_chars`` as a hard cap.

    The returned text preserves original ordering. Dropped regions are replaced
    with a ``[... N lines omitted ...]`` marker so the LLM can reason about the
    gap.
    """
    lines = text.splitlines()
    n = len(lines)

    # Fast path: nothing to trim.
    if n <= head_lines + tail_lines and len(text) <= max_chars:
        return text

    kept_indices: set[int] = set()
    if head_lines > 0:
        kept_indices.update(range(min(head_lines, n)))
    if tail_lines > 0:
        kept_indices.update(range(max(n - tail_lines, 0), n))
    for idx, line in enumerate(lines):
        if is_salient(line):
            kept_indices.add(idx)

    assembled = _assemble_with_gaps(lines, sorted(kept_indices))

    # Hard cap: if still too long, cut the middle out to guarantee an upper bound.
    if len(assembled) > max_chars:
        assembled = _hard_truncate(assembled, max_chars)
    return assembled


def _assemble_with_gaps(lines: list[str], kept: list[int]) -> str:
    """Join ``kept`` line indices, inserting a marker for each skipped run."""
    if not kept:
        return ""
    parts: list[str] = []
    prev = -1
    for idx in kept:
        if prev != -1 and idx > prev + 1:
            gap = idx - prev - 1
            parts.append(f"[... {gap} line{'s' if gap != 1 else ''} omitted ...]")
        parts.append(lines[idx])
        prev = idx
    return "\n".join(parts)


def _hard_truncate(text: str, max_chars: int) -> str:
    """Cut the middle of ``text`` to fit inside ``max_chars``.

    Keeps the first ~60% and last ~40% of the budget so the error keywords near
    the end of a stack trace remain intact.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n[... truncated for length ...]\n"
    budget = max(max_chars - len(marker), 0)
    head_budget = int(budget * 0.6)
    tail_budget = budget - head_budget
    if tail_budget <= 0:
        return text[:max_chars]
    return text[:head_budget] + marker + text[-tail_budget:]


def normalize_log(
    raw: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> str:
    """Return a cleaned, bounded version of ``raw`` suitable for an LLM call.

    Applies :func:`strip_ansi`, :func:`normalize_whitespace`, and
    :func:`truncate_lines` in order.
    """
    stripped = strip_ansi(raw)
    normalized = normalize_whitespace(stripped)
    return truncate_lines(
        normalized,
        head_lines=head_lines,
        tail_lines=tail_lines,
        max_chars=max_chars,
    )
