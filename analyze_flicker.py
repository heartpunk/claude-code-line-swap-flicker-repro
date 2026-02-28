#!/usr/bin/env python3
"""
analyze-flicker.py — Detect line-swap flicker events in Claude Code ttyrec recordings.

The flicker manifests during context compaction: two sync blocks from the same pane
alternate between a "large" cursor-up (the compaction spinner row) and a "small"
cursor-up (user input row, ~3-4 rows below the spinner).  Each sync block writes to
a different area of the screen in rapid succession, causing visual tearing.

Key discovery:
  • Payloads use tmux %extended-output format where ESC is encoded as literal \\033
  • Compaction spinner frames: \\033[8A (or other large N) + compaction text
  • Flicker frames: \\033[4A (or other small M < N) interleaved with spinner frames
  • No \\033[2K signals in the actual flicker (original plan heuristic was wrong)
"""

import argparse
import collections
import os
import re
import sqlite3
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Escape sequence patterns ────────────────────────────────────────────────
# tmux %extended-output frames encode ESC as the literal 4-byte text "\033"
# (0x5C 0x30 0x33 0x33) NOT as the actual ESC byte (0x1B).  Raw terminal
# frames may have actual ESC bytes.  We handle both.

# Literal \033 marker — backslash + "033" (4 bytes: 0x5C 0x30 0x33 0x33)
# b'\x5c033' = hex-escape for 0x5C (backslash) then literal "033"
_LIT_ESC = b'\x5c033'          # the 4-byte sequence representing ESC in tmux output

# Sync block markers
SYNC_START_LIT = _LIT_ESC + b'[?2026h'   # literal \033[?2026h in tmux output
SYNC_START_RAW = b'\x1b[?2026h'          # actual ESC byte variant

# Cursor-up patterns
# rb'\\033\[(\d+)A': raw string with \\ → regex literal backslash, matches \033[NA
CUP_LIT = re.compile(rb'\\033\[(\d+)A')         # literal \033[NA in tmux output
CUP_RAW = re.compile(b'\x1b' + rb'\[(\d+)A')     # actual ESC[NA variant (rare in these files)

# Line-clear (tracked but not required by heuristic)
CLEAR_LIT = _LIT_ESC + b'[2K'
CLEAR_RAW = b'\x1b[2K'

# Compaction text markers
COMPACT_WORDS = [b'Compacting', b'PreCompact', b'Auto-compact']

# Claude version pattern: e.g. /2.1.34/ or version: 2.1.34
VERSION_RE = re.compile(
    rb'claude[- ]code[/\\ ]+(\d+\.\d+\.\d+)'
    rb'|claude/(\d+\.\d+\.\d+)',
    re.IGNORECASE,
)

# pane id from %extended-output header
PANE_RE = re.compile(rb'^%extended-output %(\d+)')

# Terminal geometry: ESC[8;H;Wt (literal form: \033[8;H;Wt)
GEOM_LIT = re.compile(rb'\\033\[8;(\d+);(\d+)t')        # literal \033 form
GEOM_RAW = re.compile(b'\x1b' + rb'\[8;(\d+);(\d+)t')   # actual ESC byte form

# Session boundary patterns
WELCOME_RE = re.compile(rb'Welcome\s+to\s+Claude\s+Code', re.IGNORECASE)
GOODBYE_RE = re.compile(rb'Goodbye\s+from\s+Claude\s+Code', re.IGNORECASE)


# ── Streaming parser ─────────────────────────────────────────────────────────

def stream_frames(path: str):
    """Yield dicts {sec, usec, payload} one at a time.  Handles .lz and .gz."""
    proc = None
    if path.endswith('.lz'):
        proc = subprocess.Popen(
            ['lzip', '-d', '-c', path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        src = proc.stdout
    elif path.endswith('.gz'):
        proc = subprocess.Popen(
            ['gzip', '-d', '-c', path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        src = proc.stdout
    else:
        src = open(path, 'rb')

    try:
        while True:
            header = src.read(12)
            if len(header) < 12:
                break
            sec, usec, length = struct.unpack('<III', header)
            payload = src.read(length)
            if len(payload) < length:
                break
            yield {'sec': sec, 'usec': usec, 'payload': payload}
    finally:
        src.close()
        if proc:
            proc.wait()


# ── Frame feature extraction ─────────────────────────────────────────────────

def extract_features(frame: dict) -> dict:
    """Extract heuristic signals from a single frame payload."""
    p = frame['payload']
    ts = frame['sec'] + frame['usec'] / 1_000_000

    # Sync blocks
    sync_count = p.count(SYNC_START_LIT) + p.count(SYNC_START_RAW)

    # Cursor-up values (both literal and raw)
    cup_values = [int(m.group(1)) for m in CUP_LIT.finditer(p)]
    cup_values += [int(m.group(1)) for m in CUP_RAW.finditer(p)]

    # Line clears
    clear_count = p.count(CLEAR_LIT) + p.count(CLEAR_RAW)

    # Compaction text
    is_compact = any(w in p for w in COMPACT_WORDS)

    # Pane ID
    pm = PANE_RE.match(p)
    pane_id = int(pm.group(1)) if pm else -1

    # Claude version
    vm = VERSION_RE.search(p)
    version = None
    if vm:
        for g in vm.groups():
            if g:
                version = g.decode('ascii', errors='replace')
                break

    # Terminal geometry
    gm = GEOM_LIT.search(p) or GEOM_RAW.search(p)
    geom = (int(gm.group(1)), int(gm.group(2))) if gm else None

    return {
        'ts': ts,
        'sync_count': sync_count,
        'cup_values': cup_values,
        'clear_count': clear_count,
        'is_compact': is_compact,
        'pane_id': pane_id,
        'version': version,
        'geom': geom,
    }


def is_thinking_spinner_frame(feat: dict) -> bool:
    """Return True if this frame looks like a thinking/tool-use spinner frame.

    A thinking spinner frame has sync blocks and large cursor-up (like a
    compaction spinner) but does NOT contain compaction text.  This covers
    Claude Code's thinking/tool-use animation that triggers the same visual bug.
    """
    return (
        feat['sync_count'] > 0
        and any(v >= LARGE_CUP_MIN for v in feat['cup_values'])
        and not feat['is_compact']
    )


# ── Flicker heuristic ────────────────────────────────────────────────────────

WINDOW_SIZE = 30          # frames in sliding window
MAX_WINDOW_SECS = 2.5     # max wall-clock span of window
COMPACT_LOOKBACK = 600    # frames to look back for compaction context
THINKING_LOOKBACK = COMPACT_LOOKBACK  # same lookback for thinking/tool-use spinners

# Thresholds
MIN_SYNC_BLOCKS   = 3     # minimum sync starts in window
MIN_CUP_TOTAL     = 5     # minimum total cursor-up escapes in window
SMALL_CUP_MAX     = 5     # "small" cursor-up (flicker distance)
LARGE_CUP_MIN     = 6     # "large" cursor-up (spinner distance)
MIN_SMALL_CUP     = 2     # need at least this many small cursor-ups
MIN_LARGE_CUP     = 2     # need at least this many large cursor-ups


def meets_flicker_heuristics(
    window: collections.deque,
    compact_history: collections.deque,
    frame_idx: int,
) -> bool:
    """Return True if the current window looks like a flicker event."""
    frames = list(window)

    # Minimum window size
    if len(frames) < WINDOW_SIZE // 2:
        return False

    # Wall-clock span check
    t0 = frames[0][1]['ts']
    t1 = frames[-1][1]['ts']
    if t1 - t0 > MAX_WINDOW_SECS:
        return False

    # Aggregate signals across all frames in window
    # NOTE: compaction context is tracked for metadata only (compaction_above field),
    # but is NOT required — the flicker pattern can occur in any context where
    # Claude Code is rendering queued messages, not just during compaction.
    total_sync = 0
    small_cup = 0
    large_cup = 0
    total_cup = 0

    for _idx, feat in frames:
        total_sync += feat['sync_count']
        for v in feat['cup_values']:
            total_cup += 1
            if v <= SMALL_CUP_MAX:
                small_cup += 1
            if v >= LARGE_CUP_MIN:
                large_cup += 1

    # Must have enough sync blocks
    if total_sync < MIN_SYNC_BLOCKS:
        return False

    # Must have enough total cursor-up
    if total_cup < MIN_CUP_TOTAL:
        return False

    # Core flicker signal: BOTH small AND large cursor-ups in same window
    if small_cup < MIN_SMALL_CUP:
        return False
    if large_cup < MIN_LARGE_CUP:
        return False

    return True


def extract_event_metadata(
    window: collections.deque,
    compact_history: collections.deque,
    file_start_idx: int,
    thinking_history: collections.deque = None,
) -> dict:
    """Build metadata dict for a confirmed flicker event."""
    frames = list(window)
    start_idx, start_feat = frames[0]
    end_idx, end_feat = frames[-1]

    t0 = start_feat['ts']
    t1 = end_feat['ts']
    dur = t1 - t0
    n_frames = end_idx - start_idx + 1
    fps = n_frames / dur if dur > 0 else 0

    total_sync = sum(f['sync_count'] for _, f in frames)
    clear_total = sum(f['clear_count'] for _, f in frames)

    small_ups = []
    large_ups = []
    for _, f in frames:
        for v in f['cup_values']:
            if v <= SMALL_CUP_MAX:
                small_ups.append(v)
            else:
                large_ups.append(v)

    dominant_small = max(set(small_ups), key=small_ups.count) if small_ups else 0
    dominant_large = max(set(large_ups), key=large_ups.count) if large_ups else 0

    # Nearest compaction frame
    nearest_compact_idx = None
    nearest_compact_dist = None
    nearest_compact_secs = None
    for ci in compact_history:
        dist = start_idx - ci
        if nearest_compact_dist is None or dist < nearest_compact_dist:
            nearest_compact_dist = dist
            nearest_compact_idx = ci

    compaction_above = nearest_compact_dist is not None

    # Nearest thinking/tool-use spinner frame
    nearest_thinking_dist = None
    if thinking_history is not None:
        for ti in thinking_history:
            dist = start_idx - ti
            if nearest_thinking_dist is None or abs(dist) < abs(nearest_thinking_dist):
                nearest_thinking_dist = dist

    thinking_above = nearest_thinking_dist is not None

    return {
        'start_frame': start_idx,
        'end_frame': end_idx,
        'start_ts_us': int(t0 * 1_000_000),
        'end_ts_us': int(t1 * 1_000_000),
        'duration_us': int(dur * 1_000_000),
        'frame_count': n_frames,
        'frames_per_second': fps,
        'compaction_above': 1 if compaction_above else 0,
        'compaction_frame_offset': nearest_compact_dist,
        'cursor_up_rows': dominant_small,
        'cursor_up_spinner': dominant_large,
        'sync_block_count': total_sync,
        'line_clear_count': clear_total,
        'thinking_above': 1 if thinking_above else 0,
        'thinking_frame_offset': nearest_thinking_dist,
    }


# ── Database ─────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    file_size_bytes INTEGER,
    frame_count INTEGER,
    duration_us INTEGER,
    start_ts_us INTEGER,
    end_ts_us INTEGER,
    claude_version TEXT,
    has_compaction INTEGER,
    compaction_count INTEGER,
    flicker_count INTEGER,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS flicker_events (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER REFERENCES recordings(id),
    start_frame INTEGER,
    end_frame INTEGER,
    start_ts_us INTEGER,
    end_ts_us INTEGER,
    duration_us INTEGER,
    frame_count INTEGER,
    frames_per_second REAL,
    compaction_above INTEGER,
    compaction_frame_offset INTEGER,
    cursor_up_rows INTEGER,
    cursor_up_spinner INTEGER,
    sync_block_count INTEGER,
    line_clear_count INTEGER,
    thinking_above INTEGER,
    thinking_frame_offset INTEGER
);

CREATE INDEX IF NOT EXISTS idx_flicker_recording ON flicker_events(recording_id);
CREATE INDEX IF NOT EXISTS idx_recording_version ON recordings(claude_version);
"""


SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_sessions (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    session_start_frame INTEGER NOT NULL,
    session_end_frame INTEGER NOT NULL,
    session_start_ts_us INTEGER,
    session_end_ts_us INTEGER,
    claude_version TEXT,
    confidence INTEGER NOT NULL DEFAULT 0,
    start_signal TEXT,
    detected_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_recording ON claude_sessions(recording_id);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    ensure_thinking_schema(conn)  # migration for existing DBs
    conn.commit()
    return conn


def ensure_session_schema(conn: sqlite3.Connection):
    """Add claude_sessions table and session columns to flicker_events if not present."""
    conn.executescript(SESSION_SCHEMA)
    conn.commit()
    for col_def in [
        "ALTER TABLE flicker_events ADD COLUMN session_id INTEGER REFERENCES claude_sessions(id)",
        "ALTER TABLE flicker_events ADD COLUMN is_in_session INTEGER DEFAULT 0",
    ]:
        try:
            conn.execute(col_def)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


def ensure_thinking_schema(conn: sqlite3.Connection):
    """Add thinking_above and thinking_frame_offset to flicker_events if not present."""
    for col_def in [
        "ALTER TABLE flicker_events ADD COLUMN thinking_above INTEGER",
        "ALTER TABLE flicker_events ADD COLUMN thinking_frame_offset INTEGER",
    ]:
        try:
            conn.execute(col_def)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


def insert_recording(conn, rec: dict) -> int:
    cur = conn.execute(
        """
        INSERT OR REPLACE INTO recordings
            (path, file_size_bytes, frame_count, duration_us,
             start_ts_us, end_ts_us, claude_version,
             has_compaction, compaction_count, flicker_count, processed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            rec['path'], rec['file_size_bytes'], rec['frame_count'],
            rec['duration_us'], rec['start_ts_us'], rec['end_ts_us'],
            rec['claude_version'],
            rec['has_compaction'], rec['compaction_count'],
            rec['flicker_count'], rec['processed_at'],
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_flicker(conn, recording_id: int, event: dict):
    conn.execute(
        """
        INSERT INTO flicker_events
            (recording_id, start_frame, end_frame, start_ts_us, end_ts_us,
             duration_us, frame_count, frames_per_second,
             compaction_above, compaction_frame_offset,
             cursor_up_rows, cursor_up_spinner,
             sync_block_count, line_clear_count,
             thinking_above, thinking_frame_offset)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            recording_id,
            event['start_frame'], event['end_frame'],
            event['start_ts_us'], event['end_ts_us'],
            event['duration_us'], event['frame_count'],
            event['frames_per_second'],
            event['compaction_above'], event['compaction_frame_offset'],
            event['cursor_up_rows'], event['cursor_up_spinner'],
            event['sync_block_count'], event['line_clear_count'],
            event.get('thinking_above'), event.get('thinking_frame_offset'),
        ),
    )
    conn.commit()


def insert_session(conn: sqlite3.Connection, recording_id: int, session: dict) -> int:
    """Insert a detected session into claude_sessions. Returns the session id."""
    cur = conn.execute(
        """
        INSERT INTO claude_sessions
            (recording_id, session_start_frame, session_end_frame,
             session_start_ts_us, session_end_ts_us,
             claude_version, confidence, start_signal, detected_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            recording_id,
            session['start_frame'], session['end_frame'],
            session.get('start_ts_us', 0), session.get('end_ts_us', 0),
            session.get('claude_version'),
            session['confidence'],
            session.get('start_signal'),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_flicker_session_flags(
    conn: sqlite3.Connection,
    recording_id: int,
    sessions: list,
    session_ids: list,
):
    """Mark flicker_events as is_in_session=1 for events within any session boundary."""
    for session, sid in zip(sessions, session_ids):
        conn.execute(
            """
            UPDATE flicker_events
            SET is_in_session = 1, session_id = ?
            WHERE recording_id = ?
              AND start_frame >= ?
              AND start_frame <= ?
            """,
            (sid, recording_id, session['start_frame'], session['end_frame']),
        )
    conn.commit()


# ── Session detection ─────────────────────────────────────────────────────────

def detect_sessions_in_recording(path: str) -> list:
    """
    Detect Claude Code session boundaries in a ttyrec recording.

    Returns a list of session dicts (usually 0 or 1 per file). Each has:
        start_frame, end_frame, start_ts_us, end_ts_us,
        claude_version, confidence, start_signal

    Confidence scale:
        1 = sync block activity only
        2 = compaction text (likely Claude Code)
        3 = version string found (definitely Claude Code)
        5 = welcome banner found
        8 = version + welcome banner
        +5 if goodbye message found (capped at 10)
    """
    frame_count = 0

    first_claude_frame = None
    first_claude_ts = 0.0
    first_signal = None
    best_confidence = 0
    first_version = None
    saw_welcome = False

    last_sync_frame = None
    last_sync_ts = 0.0
    last_frame_ts = 0.0

    goodbye_frame = None
    goodbye_ts = 0.0

    try:
        for frame in stream_frames(path):
            feat = extract_features(frame)
            p = frame['payload']
            ts = feat['ts']

            frame_confidence = 0
            frame_signal = None

            if feat['sync_count'] > 0:
                frame_signal = 'sync_activity'
                frame_confidence = 1
                last_sync_frame = frame_count
                last_sync_ts = ts

            if feat['is_compact']:
                frame_confidence = max(frame_confidence, 2)
                frame_signal = 'compact'

            if feat['version']:
                frame_confidence = max(frame_confidence, 3)
                frame_signal = 'version'
                if first_version is None:
                    first_version = feat['version']

            if WELCOME_RE.search(p):
                frame_confidence = max(frame_confidence, 5)
                frame_signal = 'welcome'
                saw_welcome = True

            if GOODBYE_RE.search(p):
                goodbye_frame = frame_count
                goodbye_ts = ts

            if frame_signal is not None:
                if first_claude_frame is None:
                    first_claude_frame = frame_count
                    first_claude_ts = ts
                    first_signal = frame_signal
                if frame_confidence > best_confidence:
                    best_confidence = frame_confidence

            last_frame_ts = ts
            frame_count += 1
    except Exception:  # noqa: BLE001
        pass

    if first_claude_frame is None:
        return []

    # Combine signals for final confidence
    confidence = best_confidence
    if first_version and saw_welcome:
        confidence = max(confidence, 8)

    if goodbye_frame is not None:
        end_frame = goodbye_frame
        end_ts = goodbye_ts
        confidence = min(confidence + 5, 10)
    elif last_sync_frame is not None:
        end_frame = last_sync_frame
        end_ts = last_sync_ts
    else:
        end_frame = max(frame_count - 1, 0)
        end_ts = last_frame_ts

    return [{
        'start_frame': first_claude_frame,
        'end_frame': end_frame,
        'start_ts_us': int(first_claude_ts * 1_000_000),
        'end_ts_us': int(end_ts * 1_000_000),
        'claude_version': first_version,
        'confidence': confidence,
        'start_signal': first_signal,
    }]


def detect_and_store_sessions(
    conn: sqlite3.Connection,
    verbose: bool = False,
    skip_existing: bool = True,
    flicker_only: bool = False,
) -> dict:
    """
    Detect Claude Code sessions for all recordings in the DB and store results.

    Args:
        conn: DB connection (session schema must already be initialized).
        verbose: Print per-file progress.
        skip_existing: Skip recordings that already have sessions.
        flicker_only: Only process recordings that have flicker events.

    Returns dict with summary stats.
    """
    ensure_session_schema(conn)

    base_query = """
        SELECT r.id, r.path, r.claude_version
        FROM recordings r
    """
    filters = []
    if skip_existing:
        filters.append(
            "NOT EXISTS (SELECT 1 FROM claude_sessions cs WHERE cs.recording_id = r.id)"
        )
    if flicker_only:
        filters.append("r.flicker_count > 0")

    if filters:
        base_query += " WHERE " + " AND ".join(filters)

    rows = conn.execute(base_query).fetchall()
    total = len(rows)

    sessions_found = 0
    no_signals = 0

    print(f"Detecting sessions in {total} recordings...")

    t_start = time.monotonic()
    for i, (rec_id, path, known_version) in enumerate(rows, 1):
        if verbose:
            print(f"  [{i}/{total}] {os.path.basename(path)}")
        else:
            elapsed = time.monotonic() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"\r  [{i:5d}/{total}] {100*i/total:5.1f}%  "
                f"sessions={sessions_found}  "
                f"rate={rate:.1f}/s  eta={eta:.0f}s    ",
                end='', flush=True,
            )

        if not os.path.exists(path):
            no_signals += 1
            continue

        sessions = detect_sessions_in_recording(path)

        if sessions:
            session_ids = []
            for session in sessions:
                if not session['claude_version'] and known_version:
                    session['claude_version'] = known_version
                sid = insert_session(conn, rec_id, session)
                session_ids.append(sid)
            update_flicker_session_flags(conn, rec_id, sessions, session_ids)
            sessions_found += len(sessions)
        else:
            no_signals += 1
            if verbose:
                print(f"    No Claude Code signals detected")

    if not verbose:
        print()

    return {
        'total_processed': total,
        'sessions_found': sessions_found,
        'no_signals': no_signals,
    }


def session_stats_report(conn: sqlite3.Connection):
    """Show session-filtered statistics for flicker events."""
    print("\n" + "=" * 60)
    print("SESSION ANALYSIS")
    print("=" * 60)

    try:
        conn.execute("SELECT session_id FROM flicker_events LIMIT 1")
    except sqlite3.OperationalError:
        print("Session schema not initialized. Run --detect-sessions first.")
        return

    row = conn.execute("SELECT COUNT(*) FROM claude_sessions").fetchone()
    print(f"\nTotal sessions detected: {row[0]}")

    rows = conn.execute(
        """
        SELECT COUNT(*) as recordings_with_sessions,
               COUNT(DISTINCT recording_id) as unique_recordings
        FROM claude_sessions
        """
    ).fetchone()
    print(f"Recordings with sessions: {rows[1]}")

    print("\n-- Confidence distribution --")
    for row in conn.execute(
        "SELECT confidence, COUNT(*) FROM claude_sessions GROUP BY confidence ORDER BY confidence DESC"
    ):
        print(f"  confidence={row[0]:2d}  sessions={row[1]}")

    print("\n-- Start signal distribution --")
    for row in conn.execute(
        "SELECT start_signal, COUNT(*) FROM claude_sessions GROUP BY start_signal ORDER BY COUNT(*) DESC"
    ):
        print(f"  signal={row[0] or 'none':16s}  sessions={row[1]}")

    print("\n-- Flicker events: in-session vs outside --")
    for row in conn.execute(
        "SELECT COALESCE(is_in_session,0), COUNT(*) FROM flicker_events GROUP BY COALESCE(is_in_session,0)"
    ):
        label = "in_session" if row[0] == 1 else "outside_session"
        print(f"  {label}: {row[1]}")

    print("\n-- Non-compaction events (compaction_above=0): in-session vs outside --")
    for row in conn.execute(
        """
        SELECT COALESCE(is_in_session,0), COUNT(*)
        FROM flicker_events WHERE compaction_above = 0
        GROUP BY COALESCE(is_in_session,0)
        """
    ):
        label = "in_session" if row[0] == 1 else "outside_session"
        print(f"  {label}: {row[1]}")

    print("\n-- In-session flicker events by Claude version --")
    for row in conn.execute(
        """
        SELECT r.claude_version, COUNT(f.id)
        FROM flicker_events f
        JOIN recordings r ON r.id = f.recording_id
        WHERE COALESCE(f.is_in_session, 0) = 1
        GROUP BY r.claude_version
        ORDER BY COUNT(f.id) DESC
        """
    ):
        print(f"  version={row[0] or 'unknown':20s}  events={row[1]}")

    print("\n-- Recordings with flicker events but no detected session --")
    rows = conn.execute(
        """
        SELECT r.path, r.claude_version, r.flicker_count
        FROM recordings r
        WHERE r.flicker_count > 0
          AND NOT EXISTS (SELECT 1 FROM claude_sessions cs WHERE cs.recording_id = r.id)
        ORDER BY r.flicker_count DESC
        LIMIT 20
        """
    ).fetchall()
    if rows:
        for row in rows:
            print(f"  {row[2]:4d}x  {row[1] or '?':10s}  {os.path.basename(row[0])}")
    else:
        print("  (none)")

    print("\n-- Top files by non-compaction in-session events --")
    for row in conn.execute(
        """
        SELECT r.path, r.claude_version, COUNT(f.id) as nc_events
        FROM flicker_events f
        JOIN recordings r ON r.id = f.recording_id
        WHERE f.compaction_above = 0 AND COALESCE(f.is_in_session, 0) = 1
        GROUP BY r.id
        ORDER BY nc_events DESC
        LIMIT 10
        """
    ):
        print(f"  {row[2]:5d}x  {row[1] or '?':10s}  {os.path.basename(row[0])}")


# ── Thinking backfill + report ───────────────────────────────────────────────

def backfill_thinking_above(conn: sqlite3.Connection, verbose: bool = False) -> dict:
    """
    Re-read all recordings that have flicker events with thinking_above IS NULL
    and populate thinking_above / thinking_frame_offset for each.

    Returns summary stats dict with keys: total, updated, errors.
    """
    ensure_thinking_schema(conn)

    rows = conn.execute(
        """
        SELECT DISTINCT r.id, r.path
        FROM recordings r
        JOIN flicker_events f ON f.recording_id = r.id
        WHERE f.thinking_above IS NULL
        ORDER BY r.id
        """
    ).fetchall()

    total = len(rows)
    print(f"Backfilling thinking_above for {total} recordings...")

    updated = 0
    errors = 0
    t_start = time.monotonic()

    for i, (rec_id, path) in enumerate(rows, 1):
        if not verbose:
            elapsed = time.monotonic() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"\r  [{i:4d}/{total}] {100*i/total:5.1f}%  "
                f"updated={updated}  errors={errors}  "
                f"rate={rate:.1f}/s  eta={eta:.0f}s    ",
                end='', flush=True,
            )

        if not os.path.exists(path):
            if verbose:
                print(f"  SKIP (missing): {os.path.basename(path)}")
            errors += 1
            continue

        events = conn.execute(
            """
            SELECT id, start_frame, end_frame
            FROM flicker_events
            WHERE recording_id = ? AND thinking_above IS NULL
            ORDER BY end_frame
            """,
            (rec_id,),
        ).fetchall()

        if not events:
            continue

        # Map end_frame → list of (eid, start_frame) for events at that frame
        end_map: dict = {}
        for eid, start_frame, end_frame in events:
            end_map.setdefault(end_frame, []).append((eid, start_frame))
        max_end = max(end_map)

        thinking_history: collections.deque = collections.deque(maxlen=THINKING_LOOKBACK)

        try:
            for frame_idx, frame in enumerate(stream_frames(path)):
                if frame_idx > max_end + THINKING_LOOKBACK:
                    break

                feat = extract_features(frame)
                if is_thinking_spinner_frame(feat):
                    thinking_history.append(frame_idx)

                if frame_idx in end_map:
                    for eid, start_frame in end_map[frame_idx]:
                        nearest_dist = None
                        for ti in thinking_history:
                            dist = start_frame - ti
                            if nearest_dist is None or abs(dist) < abs(nearest_dist):
                                nearest_dist = dist
                        thinking_above = 1 if nearest_dist is not None else 0
                        conn.execute(
                            """
                            UPDATE flicker_events
                            SET thinking_above = ?, thinking_frame_offset = ?
                            WHERE id = ?
                            """,
                            (thinking_above, nearest_dist, eid),
                        )
                        updated += 1
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ERROR {os.path.basename(path)}: {exc}", file=sys.stderr)
            errors += 1

    if not verbose:
        print()

    return {'total': total, 'updated': updated, 'errors': errors}


def thinking_stats_report(conn: sqlite3.Connection):
    """Show thinking_above breakdown for in-session flicker events."""
    print("\n" + "=" * 60)
    print("THINKING SPINNER ANALYSIS")
    print("=" * 60)

    try:
        conn.execute("SELECT thinking_above FROM flicker_events LIMIT 1")
    except sqlite3.OperationalError:
        print("Thinking schema not initialized. Run --backfill-thinking first.")
        return

    null_count = conn.execute(
        "SELECT COUNT(*) FROM flicker_events WHERE thinking_above IS NULL"
    ).fetchone()[0]
    if null_count:
        print(f"\n  WARNING: {null_count} events still have thinking_above=NULL")
        print("  Run --backfill-thinking to populate them.")

    print("\n-- In-session events: compaction_above × thinking_above --")
    for row in conn.execute(
        """
        SELECT compaction_above, COALESCE(thinking_above, -1), COUNT(*)
        FROM flicker_events
        WHERE COALESCE(is_in_session, 0) = 1
        GROUP BY compaction_above, COALESCE(thinking_above, -1)
        ORDER BY compaction_above DESC, COALESCE(thinking_above, -1) DESC
        """
    ):
        thinking_label = {1: 'yes', 0: 'no', -1: 'NULL'}[row[1]]
        print(
            f"  compaction_above={row[0]}  thinking_above={thinking_label:4s}  "
            f"events={row[2]:6d}"
        )

    print("\n-- Non-compaction in-session events: thinking_above breakdown --")
    total_nc = conn.execute(
        "SELECT COUNT(*) FROM flicker_events WHERE COALESCE(is_in_session,0)=1 AND compaction_above=0"
    ).fetchone()[0]
    for row in conn.execute(
        """
        SELECT COALESCE(thinking_above, -1), COUNT(*)
        FROM flicker_events
        WHERE COALESCE(is_in_session, 0) = 1 AND compaction_above = 0
        GROUP BY COALESCE(thinking_above, -1)
        ORDER BY COALESCE(thinking_above, -1) DESC
        """
    ):
        thinking_label = {1: 'yes', 0: 'no', -1: 'NULL'}[row[0]]
        pct = 100 * row[1] / total_nc if total_nc else 0
        print(
            f"  thinking_above={thinking_label:4s}  events={row[1]:6d}  ({pct:.1f}%)"
        )

    print(f"\n  Total non-compaction in-session events: {total_nc}")


# ── Per-file analysis ─────────────────────────────────────────────────────────

def analyze_file(path: str, conn: sqlite3.Connection, verbose: bool = False):
    """Process one ttyrec file.  Constant memory — sliding window + bounded deques."""
    file_size = os.path.getsize(path)

    window: collections.deque = collections.deque(maxlen=WINDOW_SIZE)
    compact_history: collections.deque = collections.deque(maxlen=COMPACT_LOOKBACK)
    thinking_history: collections.deque = collections.deque(maxlen=THINKING_LOOKBACK)

    claude_version = None
    frame_count = 0
    compaction_count = 0
    flicker_events_buf = []
    start_ts_us = None
    end_ts_us = None
    last_flicker_end = -1  # prevent overlapping events

    try:
        for frame in stream_frames(path):
            feat = extract_features(frame)
            ts_us = int(feat['ts'] * 1_000_000)

            if start_ts_us is None:
                start_ts_us = ts_us
            end_ts_us = ts_us

            # Opportunistically capture version
            if claude_version is None and feat['version']:
                claude_version = feat['version']

            # Track compaction events
            if feat['is_compact']:
                compact_history.append(frame_count)
                # Only increment count at the START of a compaction burst
                if not window or not any(f['is_compact'] for _, f in window):
                    compaction_count += 1

            # Track thinking/tool-use spinner frames
            if is_thinking_spinner_frame(feat):
                thinking_history.append(frame_count)

            window.append((frame_count, feat))

            # Evaluate flicker heuristic once window is full
            if (
                len(window) >= WINDOW_SIZE
                and frame_count > last_flicker_end + WINDOW_SIZE // 2
            ):
                if meets_flicker_heuristics(window, compact_history, frame_count):
                    event = extract_event_metadata(
                        window, compact_history, frame_count, thinking_history
                    )
                    flicker_events_buf.append(event)
                    last_flicker_end = frame_count
                    if verbose:
                        print(
                            f"  FLICKER @ frame {event['start_frame']}-{event['end_frame']}"
                            f" cups_small={event['cursor_up_rows']}"
                            f" cups_spin={event['cursor_up_spinner']}"
                            f" fps={event['frames_per_second']:.1f}"
                        )

            frame_count += 1

    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR reading {path}: {exc}", file=sys.stderr)
        return

    duration_us = (end_ts_us or 0) - (start_ts_us or 0)

    rec = {
        'path': path,
        'file_size_bytes': file_size,
        'frame_count': frame_count,
        'duration_us': duration_us,
        'start_ts_us': start_ts_us or 0,
        'end_ts_us': end_ts_us or 0,
        'claude_version': claude_version,
        'has_compaction': 1 if compaction_count > 0 else 0,
        'compaction_count': compaction_count,
        'flicker_count': len(flicker_events_buf),
        'processed_at': datetime.now(timezone.utc).isoformat(),
    }

    rec_id = insert_recording(conn, rec)

    for event in flicker_events_buf:
        insert_flicker(conn, rec_id, event)

    return rec


# ── Main ──────────────────────────────────────────────────────────────────────

def find_ttyrec_files(dirs: list[str]) -> list[str]:
    files = []
    for d in dirs:
        p = Path(d)
        if p.is_file():
            files.append(str(p))
        elif p.is_dir():
            for f in sorted(p.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
                if f.is_file() and (
                    f.suffix in ('.lz', '.gz')
                    or re.match(r'^[0-9a-f-]{36}$', f.name)
                ):
                    files.append(str(f))
    return files


def run_analysis_queries(conn: sqlite3.Connection):
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    rows = conn.execute(
        "SELECT COUNT(*), SUM(flicker_count), COUNT(CASE WHEN flicker_count>0 THEN 1 END) "
        "FROM recordings"
    ).fetchone()
    print(f"\nRecordings processed: {rows[0]}")
    print(f"Files with flicker:   {rows[2]}")
    print(f"Total flicker events: {rows[1] or 0}")

    print("\n-- Flicker events by Claude version --")
    for row in conn.execute(
        """
        SELECT r.claude_version, COUNT(f.id), COUNT(DISTINCT f.recording_id)
        FROM flicker_events f
        JOIN recordings r ON r.id = f.recording_id
        GROUP BY r.claude_version
        ORDER BY COUNT(f.id) DESC
        """
    ):
        print(f"  {row[0] or 'unknown':20s}  events={row[1]:4d}  files={row[2]}")

    print("\n-- Cursor-up distance distribution (flicker row) --")
    for row in conn.execute(
        "SELECT cursor_up_rows, COUNT(*) FROM flicker_events GROUP BY cursor_up_rows ORDER BY cursor_up_rows"
    ):
        print(f"  cursor_up={row[0]}  count={row[1]}")

    print("\n-- Cursor-up distance distribution (spinner row) --")
    for row in conn.execute(
        "SELECT cursor_up_spinner, COUNT(*) FROM flicker_events GROUP BY cursor_up_spinner ORDER BY cursor_up_spinner"
    ):
        print(f"  cursor_up={row[0]}  count={row[1]}")

    print("\n-- Compaction proximity --")
    row = conn.execute(
        "SELECT compaction_above, COUNT(*) FROM flicker_events GROUP BY compaction_above"
    ).fetchall()
    for r in row:
        print(f"  compaction_above={r[0]}  count={r[1]}")

    row = conn.execute(
        "SELECT AVG(compaction_frame_offset), MIN(compaction_frame_offset), MAX(compaction_frame_offset) "
        "FROM flicker_events WHERE compaction_frame_offset IS NOT NULL"
    ).fetchone()
    if row[0] is not None:
        print(f"  frame offset from compaction: avg={row[0]:.1f}  min={row[1]}  max={row[2]}")

    print("\n-- Frame rate during flicker --")
    row = conn.execute(
        "SELECT AVG(frames_per_second), MAX(frames_per_second), MIN(frames_per_second) FROM flicker_events"
    ).fetchone()
    if row[0] is not None:
        print(f"  avg={row[0]:.1f} fps  max={row[1]:.1f}  min={row[2]:.1f}")

    print("\n-- Files with most flicker events --")
    for row in conn.execute(
        """
        SELECT r.path, r.claude_version, r.flicker_count
        FROM recordings r
        WHERE r.flicker_count > 0
        ORDER BY r.flicker_count DESC
        LIMIT 15
        """
    ):
        print(f"  {row[2]:3d}x  {row[1] or '?':8s}  {os.path.basename(row[0])}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect line-swap flicker events in Claude Code ttyrec files'
    )
    parser.add_argument(
        'paths', nargs='*',
        default=[str(Path.home() / '.ttyrec')],
        help='Files or directories to analyze (default: ~/.ttyrec/)',
    )
    parser.add_argument('--db', default='flicker.db', help='SQLite output path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose per-event output')
    parser.add_argument('--limit', type=int, default=0, help='Process at most N files')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files already in the database')
    parser.add_argument('--query-only', action='store_true',
                        help='Skip processing, just run analysis queries')
    parser.add_argument('--detect-sessions', action='store_true',
                        help='Detect Claude Code session boundaries for all recordings in DB')
    parser.add_argument('--session-report', action='store_true',
                        help='Show session-filtered statistics')
    parser.add_argument('--flicker-only', action='store_true',
                        help='With --detect-sessions: only process recordings with flicker events')
    parser.add_argument('--backfill-thinking', action='store_true',
                        help='Backfill thinking_above for existing flicker events')
    parser.add_argument('--thinking-report', action='store_true',
                        help='Show thinking spinner breakdown for in-session events')
    args = parser.parse_args()

    conn = init_db(args.db)

    if args.detect_sessions:
        ensure_session_schema(conn)
        stats = detect_and_store_sessions(
            conn,
            verbose=args.verbose,
            flicker_only=args.flicker_only,
        )
        print(
            f"\nSession detection complete: "
            f"processed={stats['total_processed']}  "
            f"sessions_found={stats['sessions_found']}  "
            f"no_signals={stats['no_signals']}"
        )
        if args.session_report:
            session_stats_report(conn)
        conn.close()
        return

    if args.backfill_thinking:
        stats = backfill_thinking_above(conn, verbose=args.verbose)
        print(
            f"\nBackfill complete: "
            f"recordings={stats['total']}  updated={stats['updated']}  errors={stats['errors']}"
        )
        if args.thinking_report:
            thinking_stats_report(conn)
        conn.close()
        return

    if args.thinking_report:
        try:
            thinking_stats_report(conn)
        finally:
            conn.close()
        return

    if args.session_report:
        try:
            session_stats_report(conn)
        finally:
            conn.close()
        return

    if args.query_only:
        try:
            run_analysis_queries(conn)
        finally:
            conn.close()
        return

    files = find_ttyrec_files(args.paths)
    if args.limit:
        files = files[:args.limit]

    if args.skip_existing:
        existing = {row[0] for row in conn.execute("SELECT path FROM recordings")}
        files = [f for f in files if f not in existing]

    total = len(files)
    print(f"Processing {total} files → {args.db}")

    t_start = time.monotonic()
    flicker_total = 0
    compact_total = 0
    error_count = 0

    for i, path in enumerate(files, 1):
        if args.verbose:
            print(f"[{i}/{total}] {os.path.basename(path)}")
        else:
            # Progress line
            pct = 100 * i / total
            elapsed = time.monotonic() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"\r[{i:5d}/{total}] {pct:5.1f}%  "
                f"flicker={flicker_total}  compact={compact_total}  "
                f"rate={rate:.1f}/s  eta={eta:.0f}s    ",
                end='', flush=True,
            )

        rec = analyze_file(path, conn, verbose=args.verbose)
        if rec:
            flicker_total += rec['flicker_count']
            compact_total += rec['compaction_count']
        else:
            error_count += 1

    elapsed = time.monotonic() - t_start
    print(f"\nDone in {elapsed:.1f}s.  "
          f"flicker_events={flicker_total}  compaction_events={compact_total}  "
          f"errors={error_count}")

    try:
        run_analysis_queries(conn)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
