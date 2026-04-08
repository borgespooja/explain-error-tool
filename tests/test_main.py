"""End-to-end pipeline tests with a stubbed LLM client."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from app import cli as cli_module
from app import llm, main as app_main
from app.models import ErrorExplanation

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


class _FakeClient(llm.LLMClient):
    """Deterministic fake client for end-to-end pipeline tests."""

    def __init__(self) -> None:
        self.received_system: str | None = None
        self.received_user: str | None = None

    def explain_error(self, *, system_prompt: str, user_prompt: str) -> ErrorExplanation:
        self.received_system = system_prompt
        self.received_user = user_prompt
        return ErrorExplanation(
            title="Fake title",
            summary="Fake summary.",
            likely_causes=["c1", "c2"],
            what_to_try=["s1", "s2"],
            confidence="medium",
            suspected_ecosystem=None,  # force main.run to backfill from detector
        )


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(
        llm, "get_client",
        lambda *, provider, model, settings: client,
    )
    return client


# ----- main.run ------------------------------------------------------------


def test_run_end_to_end_python_fixture(fake_client, capsys):
    raw = (FIXTURES / "python_module_error.txt").read_text(encoding="utf-8")
    explanation = app_main.run(raw_input=raw)

    assert explanation.title == "Fake title"
    # main.run backfills suspected_ecosystem from the detector when the LLM left it blank
    assert explanation.suspected_ecosystem == "python"

    # Prompt payload contains the hint JSON and the log
    assert "ModuleNotFoundError" in fake_client.received_user
    assert '"ecosystem": "python"' in fake_client.received_user
    assert "Traceback header" in fake_client.received_user  # detector signal name

    stdout = capsys.readouterr().out
    assert "Fake title" in stdout
    assert "What happened" in stdout


def test_run_json_output(fake_client, capsys):
    raw = "panic: nil pointer\ngoroutine 1 [running]:\n"
    app_main.run(raw_input=raw, json_output=True)

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["title"] == "Fake title"
    assert payload["confidence"] == "medium"
    assert payload["suspected_ecosystem"] == "go"  # backfilled from detector


def test_run_debug_surfaces_detection_and_log(fake_client, capsys):
    raw = "Traceback (most recent call last):\nValueError: x\n"
    app_main.run(raw_input=raw, debug=True)
    stdout = capsys.readouterr().out
    assert "Debug: detection" in stdout
    assert "Debug: normalized log" in stdout
    assert "Traceback" in stdout


def test_run_forwards_user_context_to_prompt(fake_client):
    app_main.run(
        raw_input="Traceback (most recent call last):\nValueError: x\n",
        context="started after upgrading Python 3.11 -> 3.12",
    )
    assert "upgrading Python 3.11" in fake_client.received_user


def test_run_wraps_llm_error_as_explain_error(monkeypatch):
    class Boom(llm.LLMClient):
        def explain_error(self, *, system_prompt, user_prompt):
            raise llm.LLMError("upstream exploded")

    monkeypatch.setattr(llm, "get_client", lambda **_: Boom())
    with pytest.raises(app_main.ExplainError, match="upstream exploded"):
        app_main.run(raw_input="Traceback: x")


def test_run_wraps_config_error(monkeypatch):
    def raiser(**_):
        raise llm.LLMConfigError("no key")

    monkeypatch.setattr(llm, "get_client", raiser)
    with pytest.raises(app_main.ConfigError, match="no key"):
        app_main.run(raw_input="Traceback: x")


# ----- CLI integration ----------------------------------------------------


def test_cli_with_fixture_file(fake_client, tmp_path):
    runner = CliRunner()
    fixture = FIXTURES / "python_module_error.txt"
    result = runner.invoke(cli_module.app, [str(fixture)])
    assert result.exit_code == 0, result.output
    assert "Fake title" in result.output


def test_cli_text_flag(fake_client):
    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        ["--text", "panic: nil pointer\ngoroutine 1 [running]:"],
    )
    assert result.exit_code == 0
    assert "Fake title" in result.output


def test_cli_json_flag(fake_client):
    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        ["--text", "Traceback (most recent call last):\nValueError: x", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["title"] == "Fake title"


def test_cli_empty_input_exits_with_code_2(fake_client):
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["--text", "   "])
    assert result.exit_code == 2
    assert "No error input provided" in result.output


def test_cli_missing_file_exits_with_code_2(fake_client, tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli_module.app, [str(tmp_path / "nope.txt")])
    assert result.exit_code == 2
    assert "file not found" in result.output


def test_cli_config_error_exits_with_code_3(monkeypatch):
    def raiser(**_):
        raise llm.LLMConfigError("missing ANTHROPIC_API_KEY")

    monkeypatch.setattr(llm, "get_client", raiser)
    runner = CliRunner()
    result = runner.invoke(
        cli_module.app, ["--text", "Traceback: x"]
    )
    assert result.exit_code == 3
    assert "config error" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
