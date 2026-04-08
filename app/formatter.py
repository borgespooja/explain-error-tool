"""Terminal and JSON rendering of an :class:`ErrorExplanation`.

Two output modes:

* :func:`render_terminal` — polished ``rich`` output with clear section rules,
  numbered causes/steps, and a colorized confidence footer. ``--debug`` adds
  the raw detection dict and the normalized log.
* :func:`render_json` — stable JSON-serializable dump of the explanation for
  scripting.
"""
from __future__ import annotations

import json
from typing import Optional, TextIO

from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from app.detector import Detection
from app.models import ErrorExplanation

_CONFIDENCE_STYLES: dict[str, str] = {
    "low": "yellow",
    "medium": "cyan",
    "high": "green",
}

_RULE_STYLE = "bright_blue"
_TITLE_STYLE = "bold red"


def render_terminal(
    explanation: ErrorExplanation,
    *,
    detection: Optional[Detection] = None,
    normalized_log: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """Render ``explanation`` as polished terminal sections."""
    out = console or Console()

    out.print(
        Panel(
            Text(explanation.title, style=_TITLE_STYLE),
            title="Error",
            border_style="red",
            padding=(0, 2),
            expand=False,
        )
    )

    out.print(Rule("What happened", style=_RULE_STYLE))
    out.print(Padding(Text(explanation.summary), (1, 2)))

    if explanation.likely_causes:
        out.print(Rule("Likely causes", style=_RULE_STYLE))
        for idx, cause in enumerate(explanation.likely_causes, start=1):
            out.print(f"  [bold]{idx}.[/bold] {cause}")
        out.print()

    if explanation.what_to_try:
        out.print(Rule("What to try", style=_RULE_STYLE))
        for idx, step in enumerate(explanation.what_to_try, start=1):
            out.print(f"  [bold]{idx}.[/bold] {step}")
        out.print()

    out.print(Rule(style="dim"))
    out.print(_build_footer(explanation))

    if detection is not None:
        out.print()
        out.print(Rule("Debug: detection", style="dim"))
        out.print(Padding(Text(json.dumps(detection, indent=2), style="dim"), (0, 2)))

    if normalized_log is not None:
        out.print()
        out.print(
            Panel(
                normalized_log,
                title="Debug: normalized log",
                border_style="dim",
                expand=True,
            )
        )


def _build_footer(explanation: ErrorExplanation) -> Text:
    style = _CONFIDENCE_STYLES.get(explanation.confidence, "white")
    footer = Text()
    footer.append("Confidence: ", style="dim")
    footer.append(explanation.confidence, style=f"bold {style}")
    if explanation.suspected_ecosystem:
        footer.append("   ", style="dim")
        footer.append("Ecosystem: ", style="dim")
        footer.append(explanation.suspected_ecosystem, style="bold")
    return footer


def render_json(
    explanation: ErrorExplanation,
    *,
    stream: Optional[TextIO] = None,
) -> None:
    """Print ``explanation`` as pretty JSON to ``stream`` or stdout."""
    text = json.dumps(explanation.model_dump(), indent=2)
    if stream is None:
        print(text)
    else:
        stream.write(text + "\n")
