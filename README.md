# explain-error

A small, polished developer CLI that takes an error log, stack trace, or
pasted exception and returns a plain-English explanation, ranked likely
causes, and concrete next steps — with optional JSON output for scripting.

## Why this exists

Debugging starts with reading. Logs and stack traces carry the signal, but
it is often buried in noise, framework internals, or an ecosystem you do
not touch every day. `explain-error` exists to give a fast, grounded
first-pass explanation: "here is what this almost certainly means, here is
what usually causes it, and here is what I would try first."

It is intentionally small: one command, one job, no daemons, no IDE plugin,
no database. Pipe a log in, get an answer out.

## Installation

Requires Python 3.10+.

```bash
# From a clone of this repo
pip install -e .

# Or with dev extras (tests, ruff)
pip install -e '.[dev]'
```

This installs the `explain-error` command on your `PATH`.

## Environment setup

Copy `.env.example` to `.env` and fill in the provider API key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

Supported variables:

| Variable                        | Purpose                                  | Default              |
| ------------------------------- | ---------------------------------------- | -------------------- |
| `EXPLAIN_ERROR_PROVIDER`        | Which LLM provider to call               | `anthropic`          |
| `EXPLAIN_ERROR_MODEL`           | Override model name for the provider     | provider's default   |
| `ANTHROPIC_API_KEY`             | API key for the Anthropic provider       | _(required)_         |
| `EXPLAIN_ERROR_TIMEOUT_SECONDS` | Per-request timeout                      | `30`                 |
| `EXPLAIN_ERROR_MAX_LOG_CHARS`   | Cap on characters sent to the LLM        | `16000`              |

CLI flags always win over environment values.

## CLI usage

```bash
# From a file
explain-error path/to/error.txt

# From stdin
cat error.log | explain-error

# Inline
explain-error --text "ModuleNotFoundError: No module named 'requests'"

# JSON for scripting
explain-error error.log --json

# Show internal state: detection + normalized log
explain-error error.log --debug

# Select provider/model or attach extra context
explain-error error.log \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --context "started after upgrading Python 3.11 -> 3.12"
```

### Exit codes

| Code | Meaning                                                              |
| ---- | -------------------------------------------------------------------- |
| `0`  | Success                                                              |
| `1`  | Pipeline error (model failure, malformed response, network, …)       |
| `2`  | Bad input (file not found, empty input)                              |
| `3`  | Config error (missing API key, unsupported provider)                 |

### Sample terminal output

```
╭────────────────────── Error ──────────────────────╮
│  ModuleNotFoundError: No module named 'requests'  │
╰───────────────────────────────────────────────────╯
──────────────────────────────── What happened ─────────────────────────────────

  Python could not import the requests package at startup, which means it is
  not installed in the interpreter your script is running under.

──────────────────────────────── Likely causes ─────────────────────────────────
  1. requests is not installed in the active virtualenv
  2. The script is running under a different Python than pip install targeted
  3. Package installed only for another user or Python version

───────────────────────────────── What to try ──────────────────────────────────
  1. python -m pip install requests
  2. Confirm the interpreter with: python -c "import sys; print(sys.executable)"
  3. Inside a venv? Check: which python && pip list | grep requests

────────────────────────────────────────────────────────────────────────────────
Confidence: high   Ecosystem: python
```

### Sample `--json` output

```json
{
  "title": "ModuleNotFoundError: No module named 'requests'",
  "summary": "Python could not import the requests package at startup...",
  "likely_causes": [
    "requests is not installed in the active virtualenv",
    "The script is running under a different Python than pip install targeted"
  ],
  "what_to_try": [
    "python -m pip install requests",
    "python -c 'import sys; print(sys.executable)'"
  ],
  "confidence": "high",
  "suspected_ecosystem": "python"
}
```

## Architecture overview

```
app/
  cli.py         command entrypoint + option parsing (typer)
  main.py        pipeline orchestration
  config.py      .env / environment loading
  parser.py      log normalization, ANSI stripping, salience-preserving truncation
  detector.py    heuristic ecosystem classifier (python / node / java / go)
  prompts.py     system + user prompt builders (strict JSON schema)
  llm.py         provider abstraction + Anthropic implementation
  formatter.py   rich terminal + JSON rendering
  models.py      pydantic response schema
tests/
  fixtures/      realistic error snippets per ecosystem
```

The pipeline is linear and each stage is independently testable:

```
raw input
  -> parser.normalize_log          (strip ANSI, collapse blanks, head+tail+salient truncate)
  -> detector.detect_ecosystem     (regex heuristics -> {ecosystem, confidence, signals})
  -> prompts.build                 (system prompt + structured user prompt)
  -> llm.LLMClient.explain_error   (Anthropic; retry on transient errors)
  -> formatter.render_terminal | render_json
```

### Design notes

- **Deterministic preprocessing before the LLM.** ANSI codes, blank-line runs,
  and bulk noise are trimmed locally so the model never sees them. Truncation
  keeps the first 40 + last 80 lines plus any line matching
  `error|exception|traceback|caused by|failed|panic`.
- **Heuristic detector, not a classifier.** The ecosystem hint is a nudge for
  the prompt, not a verdict. Confidence is scored from signal weight and
  penalized when the runner-up is close.
- **Schema-locked output.** The model is instructed to emit a single JSON
  object; extraction has three fallbacks (direct, fenced block,
  balanced-brace scan) and validation is pydantic with lenient normalization.
- **Provider abstraction.** `LLMClient` is an ABC; `get_client` is a registry.
  Adding an OpenAI or Ollama provider is one class + one registry line.

## Testing

```bash
pytest -q
```

Tests cover parser cleanup and truncation, detector classification across all
four ecosystems plus ambiguous cases, formatter output in both modes, and
end-to-end pipeline wiring with a stubbed LLM client (no network required).

## Limitations

- LLM answers are best-effort. Treat them as a starting point, not a verdict.
- Heuristic detection covers Python, Node, Java, and Go. Other ecosystems
  fall back to `unknown` and the LLM works without that hint.
- No network calls other than to the selected provider. No telemetry.
- Multiline inputs typed directly into a terminal are awkward; prefer a file
  argument or a piped stdin.
- Anthropic is the only provider shipped in v1.

## Next improvements

- Additional providers (OpenAI, local via Ollama).
- Language-specific follow-up hints (e.g. `pip show` / `npm ls` suggestions
  generated automatically from the detection signals).
- Shell integration: `explain-error --last` to pull from the most recent
  command in the user's shell history.
- Richer `--json` schema with per-cause confidence and suggested commands.
- Optional prompt caching when a user runs the tool many times in a session.
