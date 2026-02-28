# Claude Code Line-Swap Flicker Analysis Report

## Summary

This report documents automated detection of a visual bug in Claude Code's terminal UI:
during context compaction, lines visually reorder ("flicker") approximately 3–4 rows below
the compaction spinner.

**Key finding**: The bug is present in Claude Code version **2.1.34** and manifests as
two interleaved sync blocks targeting different rows of the screen during compaction.

---

## Bug Description

### What the user sees

During context compaction, the terminal shows a spinner (e.g., "Compacting conversation…").
Lines approximately 3–4 rows *below* the spinner visually reorder or flicker during this time.

### Root cause (from recording analysis)

Claude Code uses DEC private mode 2026 (synchronized output) to batch screen updates.
During compaction, two separate sync blocks are sent in rapid succession:

1. **Spinner update** — moves cursor up N rows (typically 6–8) to update the compaction
   spinner line, then returns cursor to original position
2. **Input rendering** — moves cursor up M rows (typically 3–4, where M < N) to update
   the user input area

These two sync blocks interleave at high frequency (12–85+ fps), causing different areas
of the screen to be repainted alternately without a full-screen redraw. The visual effect
is lines appearing to swap or flicker.

---

## Detection Method

### Tool chain
- Direct binary ttyrec file parsing (no third-party tools required)
- Streaming one-pass algorithm with constant memory (30-frame sliding window)
- SQLite output for structured querying

### Escape sequence encoding
ttyrec files in this corpus use tmux's `%extended-output` protocol format, which encodes
the ESC character as the literal 4-byte text string `\033` (backslash + "033"), not the
actual ESC byte (0x1B). This is critical for correct pattern matching.

### Heuristic

A **flicker event** is detected when a 30-frame window (≤2.5 seconds) satisfies ALL of:

1. At least one sync block start (`\033[?2026h`) in the window
2. At least 5 cursor-up escape sequences total
3. At least 2 cursor-up values ≤5 ("small" — input row distance)
4. At least 2 cursor-up values ≥6 ("large" — spinner row distance)
5. At least one compaction text frame ("Compacting", "PreCompact", "Auto-compact")
   within the preceding 600 frames

---

## Results (Partial — batch in progress)

> **Note**: These are interim results. The full corpus of ~1,611 files takes ~25 minutes to
> process. Run `python3 analyze-flicker.py --query-only --db flicker.db` for current counts.

### Files analyzed
- Total: 1,612 files in `~/.ttyrec/` (1,603 compressed `.lz`, 9 uncompressed)
- Total size: ~2.6 GB

### Reference file verification
**File**: `df3cbab2-1479-11f1-bbd7-32962d695674`
- Claude version: **2.1.34**
- Total frames: 82,000+
- Compaction events: 109
- Flicker events detected: **396**
- All 396 events have `compaction_above=1` ✓
- Flicker cursor-up distribution: 1 (50 events), 3 (285 events), 4 (61 events)
- Spinner cursor-up distribution: 6–8 typical (up to 16 observed)

---

## Database Schema

```sql
-- recordings: one row per ttyrec file
CREATE TABLE recordings (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    file_size_bytes INTEGER,
    frame_count INTEGER,
    duration_us INTEGER,
    start_ts_us INTEGER,
    end_ts_us INTEGER,
    claude_version TEXT,        -- e.g. "2.1.34"
    has_compaction INTEGER,     -- any compaction text found?
    compaction_count INTEGER,   -- compaction text frame count (may over-count)
    flicker_count INTEGER,      -- detected flicker events
    processed_at TEXT           -- ISO8601 timestamp
);

-- flicker_events: one row per detected flicker window
CREATE TABLE flicker_events (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER REFERENCES recordings(id),
    start_frame INTEGER,
    end_frame INTEGER,
    start_ts_us INTEGER,
    end_ts_us INTEGER,
    duration_us INTEGER,
    frame_count INTEGER,
    frames_per_second REAL,     -- fps during flicker (12–85+ fps observed)
    compaction_above INTEGER,   -- 1 if compaction found in lookback window
    compaction_frame_offset INTEGER, -- frames since nearest compaction
    cursor_up_rows INTEGER,     -- dominant small cursor-up value (3–4 typical)
    cursor_up_spinner INTEGER,  -- dominant large cursor-up value (6–8 typical)
    sync_block_count INTEGER,   -- sync blocks in window
    line_clear_count INTEGER    -- line clear (2K) count in window
);
```

---

## Analysis Queries

```sql
-- Is compaction always before the flicker?
SELECT compaction_above, COUNT(*) FROM flicker_events GROUP BY compaction_above;

-- Which versions show the bug?
SELECT r.claude_version, COUNT(DISTINCT r.id) files, COUNT(f.id) events
FROM recordings r JOIN flicker_events f ON f.recording_id = r.id
GROUP BY r.claude_version ORDER BY events DESC;

-- Cursor-up row distribution (flicker rows)
SELECT cursor_up_rows, COUNT(*) FROM flicker_events
GROUP BY cursor_up_rows ORDER BY cursor_up_rows;

-- How close to compaction does flicker occur?
SELECT AVG(compaction_frame_offset), MIN(compaction_frame_offset), MAX(compaction_frame_offset)
FROM flicker_events WHERE compaction_above = 1;

-- Frame rate during flicker
SELECT AVG(frames_per_second), MAX(frames_per_second), MIN(frames_per_second)
FROM flicker_events;
```

---

## Usage

### Run full analysis
```bash
python3 analyze-flicker.py [--db flicker.db] [~/.ttyrec/]
```

### Options
- `--limit N` — process only N files (for testing)
- `--skip-existing` — skip files already in the database (for resuming)
- `--verbose` — print each detected event
- `--query-only` — skip processing, show current database stats
- `--db PATH` — SQLite database output path (default: `flicker.db`)

### Example: reference file only
```bash
python3 analyze-flicker.py --db flicker.db --verbose \
    ~/.ttyrec/df3cbab2-1479-11f1-bbd7-32962d695674
```

---

## Key Technical Notes

### Why literal `\033` not ESC byte?
The ttyrec corpus was captured from a tmux session using iris-replay. tmux's
`%extended-output` protocol encodes ESC as literal `\033` text. When processing
payloads, always search for the 4-byte sequence `0x5C 0x30 0x33 0x33` (backslash + "033"),
not the ESC byte `0x1B`.

### Compaction count caveat
The `compaction_count` field uses text matching ("Compacting", "PreCompact", "Auto-compact").
Text mentioning these words in normal conversation/output will be counted. The count is an
upper bound on actual compaction events, not an exact count. Flicker detection is unaffected
because it requires BOTH compaction text AND the cursor-up signal pattern.

### False positive risk
Multi-pane tmux sessions can legitimately have mixed cursor-up values from different panes
updating concurrently (e.g., spinner pane at cursor-up 6, thinking animation at cursor-up 7).
The current heuristic may detect some of these as false positives. The requirement for BOTH
≥2 small and ≥2 large cursor-up values in the same window reduces but does not eliminate
this risk.
