"""Command-line entrypoint and option parsing.

This module only handles IO concerns: resolving the input (file arg, --text, or
stdin), parsing flags, and delegating to :mod:`app.main`. All pipeline logic
lives in downstream modules.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from app import main as app_main

app = typer.Typer(
    help="Explain an error log, stack trace, or pasted exception in plain English.",
    no_args_is_help=False,
    add_completion=False,
)


@app.command()
def explain(
    path: Optional[Path] = typer.Argument(
        None,
        help="Path to a file containing the error log or stack trace.",
    ),
    text: Optional[str] = typer.Option(
        None,
        "--text",
        "-t",
        help="Inline error text. Takes precedence over PATH and stdin.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the raw structured explanation as JSON.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show internal state: detection, normalized log, prompt metadata.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="LLM provider to use (defaults to EXPLAIN_ERROR_PROVIDER).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override the model name for the selected provider.",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        help="Extra context, e.g. 'started after upgrading Python 3.11 -> 3.12'.",
    ),
) -> None:
    """Read an error log from a file, stdin, or --text and explain it."""
    raw = _read_input(path, text)
    if not raw.strip():
        typer.echo(
            "No error input provided. Pass a file path, --text, or pipe stdin.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        app_main.run(
            raw_input=raw,
            json_output=json_output,
            debug=debug,
            provider=provider,
            model=model,
            context=context,
        )
    except app_main.ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except app_main.ExplainError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _read_input(path: Optional[Path], text: Optional[str]) -> str:
    """Resolve input from the three supported modes: --text, PATH, or stdin."""
    if text is not None:
        return text
    if path is not None:
        if not path.is_file():
            typer.echo(f"error: file not found: {path}", err=True)
            raise typer.Exit(code=2)
        return path.read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


if __name__ == "__main__":
    app()
