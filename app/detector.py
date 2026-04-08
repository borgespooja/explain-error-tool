"""Ecosystem detection via lightweight pattern heuristics.

Given a normalized log, scores it against a small set of regex signals per
ecosystem and returns the highest-scoring ecosystem with a confidence in
``[0.0, 1.0]`` plus the list of matched signal names.

The goal is modest and deterministic: nudge the LLM prompt in the right
direction without pretending to be a classifier. If nothing matches, we return
``unknown`` with confidence ``0.0`` and let the LLM work without the hint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypedDict

Ecosystem = str  # one of: "python", "node", "java", "go", "unknown"


class Detection(TypedDict):
    ecosystem: Ecosystem
    confidence: float
    signals: list[str]


@dataclass(frozen=True)
class Signal:
    """A single regex heuristic contributing evidence for an ecosystem.

    ``weight`` reflects how discriminating the pattern is. Strong, distinctive
    signals like ``Traceback (most recent call last):`` carry more weight than
    generic file-extension mentions like ``.py``.
    """

    name: str
    pattern: re.Pattern[str]
    weight: float


def _sig(name: str, pattern: str, weight: float, *, flags: int = 0) -> Signal:
    return Signal(name=name, pattern=re.compile(pattern, flags), weight=weight)


SIGNALS: dict[Ecosystem, list[Signal]] = {
    "python": [
        _sig("Traceback header", r"Traceback \(most recent call last\):", 3.0),
        _sig("ModuleNotFoundError", r"\bModuleNotFoundError\b", 3.0),
        _sig("ImportError", r"\bImportError\b", 2.0),
        _sig("Python built-in exception",
             r"\b(?:TypeError|ValueError|KeyError|AttributeError|IndexError|"
             r"NameError|ZeroDivisionError|RuntimeError|RecursionError|"
             r"FileNotFoundError|PermissionError)\b",
             2.0),
        _sig("File \"x.py\" frame", r'File "[^"]+\.py"', 2.5),
        _sig(".py path", r"\S+\.py\b", 0.5),
        _sig("pip mention", r"\bpip(?:\s|:)", 0.5),
    ],
    "node": [
        _sig("npm ERR!", r"npm ERR!", 3.0),
        _sig("Cannot find module", r"Error: Cannot find module", 3.0),
        _sig("node_modules path", r"/node_modules/", 2.0),
        _sig("node:internal frame", r"\bnode:internal/", 2.5),
        _sig("Node.js version banner", r"Node\.js v\d+", 2.0),
        _sig("Node errno code",
             r"code:\s*'(?:MODULE_NOT_FOUND|ECONNREFUSED|ENOENT|EACCES|"
             r"ETIMEDOUT|ERR_[A-Z_]+)'",
             2.5),
        _sig(".js file", r"\S+\.(?:js|mjs|cjs)\b", 0.5),
        _sig(".ts file", r"\S+\.(?:ts|tsx)\b", 0.5),
    ],
    "java": [
        _sig("Exception in thread", r'Exception in thread "', 3.0),
        _sig("NullPointerException", r"\bNullPointerException\b", 2.5),
        _sig("Caused by:", r"^\s*Caused by:", 2.0, flags=re.MULTILINE),
        _sig("Java package frame",
             r"\bat\s+(?:[a-zA-Z_][\w$]*\.)+[a-zA-Z_][\w$]*\(",
             2.0),
        _sig("java.lang.*", r"\bjava\.(?:lang|util|io|net)\.", 2.0),
        _sig("org.springframework", r"\borg\.springframework\.", 2.0),
        _sig(".java frame", r"\([A-Za-z_][\w$]*\.java:\d+\)", 2.0),
    ],
    "go": [
        _sig("panic:", r"^\s*panic:\s", 3.0, flags=re.MULTILINE),
        _sig("goroutine header", r"\bgoroutine\s+\d+\s+\[", 3.0),
        _sig("runtime error", r"\bruntime error:", 2.0),
        _sig("SIGSEGV signal", r"\[signal SIGSEGV", 2.0),
        _sig(".go file", r"\S+\.go:\d+", 1.5),
        _sig("exit status", r"^exit status \d+", 0.5, flags=re.MULTILINE),
    ],
}

# Confidence is total_score / CONFIDENCE_SATURATION, capped at 1.0. At
# saturation, a log exhibits the strongest 2-3 signals we look for.
CONFIDENCE_SATURATION: float = 6.0


def _score(normalized_log: str, signals: list[Signal]) -> tuple[float, list[str]]:
    """Return total weight and matched signal names for one ecosystem."""
    total = 0.0
    matched: list[str] = []
    for signal in signals:
        if signal.pattern.search(normalized_log):
            total += signal.weight
            matched.append(signal.name)
    return total, matched


def detect_ecosystem(normalized_log: str) -> Detection:
    """Classify ``normalized_log`` as python / node / java / go, or unknown.

    Confidence is ``min(top_score / CONFIDENCE_SATURATION, 1.0)``, further
    reduced when the runner-up score is close to the winner (ambiguous log).
    """
    if not normalized_log.strip():
        return {"ecosystem": "unknown", "confidence": 0.0, "signals": []}

    scores: dict[Ecosystem, tuple[float, list[str]]] = {
        ecosystem: _score(normalized_log, signals)
        for ecosystem, signals in SIGNALS.items()
    }

    winner, (top_score, matched) = max(scores.items(), key=lambda item: item[1][0])
    if top_score <= 0.0:
        return {"ecosystem": "unknown", "confidence": 0.0, "signals": []}

    runner_up = max(
        (score for eco, (score, _) in scores.items() if eco != winner),
        default=0.0,
    )

    base = min(top_score / CONFIDENCE_SATURATION, 1.0)
    # Margin penalty: when the runner-up is close, we are less sure. Penalty
    # is proportional to (runner_up / top_score), capped at 50% reduction.
    margin_ratio = runner_up / top_score if top_score > 0 else 0.0
    penalty = min(margin_ratio, 0.5)
    confidence = round(base * (1.0 - penalty), 2)

    return {
        "ecosystem": winner,
        "confidence": confidence,
        "signals": matched,
    }
