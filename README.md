# Claude Code Line-Swap Flicker Repro

Automated detection and analysis of a visual bug in Claude Code's terminal UI: during
context compaction, lines ~3 rows below the compaction spinner visually reorder
("flicker") at high frame rates.

**→ [Full bug report](flicker-report.md)**

---

## What's in here

| File | Description |
|---|---|
| `analyze_flicker.py` | Streaming ttyrec parser + SQLite flicker detector |
| `flicker-report.md` | Full analysis report with corpus stats and fix suggestions |
| `test_analyze_flicker.py` | Unit + integration tests (232 tests) |
| `test_properties.py` | Property-based tests (21 Hypothesis tests) |
| `conftest.py` | Hypothesis profile config |
| `pyproject.py` | Project deps and mutmut config |

## Quick start

```bash
uv sync --extra dev

# Analyze a directory of ttyrec recordings
uv run python analyze_flicker.py ~/.ttyrec/

# Analyze a single file verbosely
uv run python analyze_flicker.py --verbose path/to/recording.ttyrec

# Re-run queries on an existing database
uv run python analyze_flicker.py --query-only --db flicker.db

# Run tests
uv run pytest
```

## How it works

`analyze_flicker.py` does a single streaming pass over each ttyrec file with constant
memory (30-frame sliding window). It looks for windows where **both** small cursor-up
values (≤5 rows, the input area) and large cursor-up values (≥6 rows, the compaction
spinner) appear in rapid alternation inside DEC synchronized output blocks — the
signature of two render paths interleaving faster than the terminal can composite them.

Results go into a SQLite database (`flicker.db`) for querying.

## Findings

- **82/1,617 recordings (5%)** contain detectable flicker; **298 Claude Code sessions** detected via structural welcome-screen analysis
- **Rare in 2.0.x** (2–4% of sessions in 2.0.36, 2.0.76), **absent in 2.0.13–2.0.70**, **dramatically worse in 2.1.x** (55–100% of sessions, 2.1.4 → 2.1.62)
- **~17 flicker bursts per compaction event** in 2.1.34
- Flicker row peaks at **3 rows** below the spinner (44.6% of events)
- **30% of flicker** occurs during thinking/tool-use spinners, not just compaction


See [flicker-report.md](flicker-report.md) for the full breakdown.

## Supports compressed recordings

Handles `.lz` (lzip), `.gz` (gzip), and uncompressed ttyrec files.
Requires `lzip` on PATH for `.lz` files (`brew install lzip`).
