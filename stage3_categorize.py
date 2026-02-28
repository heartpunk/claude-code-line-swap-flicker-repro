#!/usr/bin/env python3
"""
stage3_categorize.py — Extract non-compaction flicker event frames for categorization.

Samples ~50 events from non-compaction in-session flicker events:
  - ~25 from confirmed_claude tier (confidence >= 3, has version string)
  - ~15 from mystery 08923efa file (confidence=2, large event count)
  - ~10 from uncertain tier (confidence 1-2, spot-check for false positives)

For each event, extracts a window of frames from the ttyrec recording and writes
formatted output to a JSON file for downstream categorization.

Usage:
    uv run python stage3_categorize.py [--limit N] [--out PATH]
    uv run python stage3_categorize.py --db /path/to/flicker.db
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_flicker import extract_features, stream_frames

DEFAULT_DB_DEST = '/private/tmp/claude/flicker_read.db'
DEFAULT_OUT = '/private/tmp/claude/stage3_events.json'
SOURCE_DB = Path(__file__).parent / 'flicker.db'
FRAME_MARGIN = 15    # frames before/after event to include
PAYLOAD_DISPLAY = 500  # max payload bytes to show per frame

_EVENT_COLS = [
    'event_id', 'recording_id', 'file_path', 'recording_version',
    'start_frame', 'end_frame', 'start_ts_us', 'duration_us',
    'cursor_up_rows', 'cursor_up_spinner', 'compaction_frame_offset',
    'confidence', 'start_signal', 'session_version',
]

_EVENT_SELECT = """
    SELECT f.id, f.recording_id, r.path, r.claude_version,
           f.start_frame, f.end_frame, f.start_ts_us, f.duration_us,
           f.cursor_up_rows, f.cursor_up_spinner, f.compaction_frame_offset,
           cs.confidence, cs.start_signal, cs.claude_version
    FROM flicker_events f
    JOIN recordings r ON r.id = f.recording_id
    JOIN claude_sessions cs ON cs.recording_id = f.recording_id
    WHERE f.compaction_above = 0
      AND COALESCE(f.is_in_session, 0) = 1
"""


def ensure_read_db(db_dest: str, source_db: Path) -> str:
    dest = Path(db_dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or source_db.stat().st_mtime > dest.stat().st_mtime:
        print(f"Copying {source_db} → {db_dest}")
        shutil.copy2(str(source_db), db_dest)
    return db_dest


def _rows_to_events(rows, tier: str) -> list[dict]:
    return [dict(zip(_EVENT_COLS, row), tier=tier) for row in rows]


def query_sample_events(conn: sqlite3.Connection, total: int = 50) -> list[dict]:
    n_confirmed = total // 2
    n_mystery = total * 3 // 10
    n_uncertain = total - n_confirmed - n_mystery

    confirmed = _rows_to_events(
        conn.execute(
            _EVENT_SELECT + """
              AND cs.confidence >= 3
              AND cs.claude_version IS NOT NULL
            ORDER BY RANDOM() LIMIT ?""",
            (n_confirmed,),
        ).fetchall(),
        'confirmed_claude',
    )
    mystery = _rows_to_events(
        conn.execute(
            _EVENT_SELECT + """
              AND r.path LIKE '%08923efa%'
            ORDER BY RANDOM() LIMIT ?""",
            (n_mystery,),
        ).fetchall(),
        'mystery_08923efa',
    )
    uncertain = _rows_to_events(
        conn.execute(
            _EVENT_SELECT + """
              AND cs.confidence < 3
              AND r.path NOT LIKE '%08923efa%'
            ORDER BY RANDOM() LIMIT ?""",
            (n_uncertain,),
        ).fetchall(),
        'uncertain',
    )
    return confirmed + mystery + uncertain


def extract_frame_window(path: str, start_frame: int, end_frame: int) -> list[dict]:
    lo = max(0, start_frame - FRAME_MARGIN)
    hi = end_frame + FRAME_MARGIN
    result = []
    for idx, frame in enumerate(stream_frames(path)):
        if idx > hi:
            break
        if idx >= lo:
            result.append({'idx': idx, 'frame': frame})
    return result


def format_frames_for_analysis(frame_window: list[dict], event: dict) -> str:
    """Format frame window as structured text for analysis."""
    lines = [
        "Event metadata:",
        f"  start_frame={event['start_frame']}, end_frame={event['end_frame']}",
        f"  cursor_up_rows={event['cursor_up_rows']} (small, <=5)",
        f"  cursor_up_spinner={event['cursor_up_spinner']} (large, >=6)",
        f"  compaction_frame_offset={event['compaction_frame_offset']} "
        f"(None = no compaction in 600-frame lookback)",
        f"  session_confidence={event['confidence']}, signal={event['start_signal']}",
        f"  version={event.get('session_version') or event.get('recording_version') or 'unknown'}",
        "",
        "Frame window (*** = event frames):",
    ]

    for item in frame_window:
        idx = item['idx']
        frame = item['frame']
        feat = extract_features(frame)
        in_event = event['start_frame'] <= idx <= event['end_frame']
        marker = ' ***' if in_event else '    '

        feats = []
        if feat['sync_count']:
            feats.append(f"sync x{feat['sync_count']}")
        if feat['cup_values']:
            feats.append(f"CUP={feat['cup_values']}")
        if feat['is_compact']:
            feats.append('COMPACT')
        if feat['clear_count']:
            feats.append(f"CLR x{feat['clear_count']}")
        feat_str = ' '.join(feats) if feats else '(no signals)'

        raw = frame['payload'][:PAYLOAD_DISPLAY].decode('latin-1', errors='replace')
        snippet = ''.join(c if c.isprintable() or c in '\n\r\t' else '.' for c in raw)
        snippet = snippet.replace('\n', '|').replace('\r', '|')[:300]
        ts = feat['ts']

        lines.append(f"[{idx:6d}]{marker} t={ts:.3f}s  {feat_str}")
        lines.append(f"         payload: {snippet!r}")

    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage 3: Extract non-compaction flicker event frames'
    )
    parser.add_argument('--db', default=DEFAULT_DB_DEST)
    parser.add_argument('--source-db', default=str(SOURCE_DB))
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--out', default=DEFAULT_OUT, help='Output JSON path')
    args = parser.parse_args()

    db_path = ensure_read_db(args.db, Path(args.source_db))
    conn = sqlite3.connect(db_path)
    print(f"DB: {db_path}")

    events = query_sample_events(conn, total=args.limit)
    n_conf = sum(1 for e in events if e['tier'] == 'confirmed_claude')
    n_myst = sum(1 for e in events if e['tier'] == 'mystery_08923efa')
    n_unc = sum(1 for e in events if e['tier'] == 'uncertain')
    print(f"Sampled {len(events)}: {n_conf} confirmed, {n_myst} mystery, {n_unc} uncertain")

    output = []
    errors = 0

    for i, event in enumerate(events, 1):
        path = event['file_path']
        print(f"[{i:2d}/{len(events)}] {event['tier']:22s} conf={event['confidence']} "
              f"frames {event['start_frame']}-{event['end_frame']}  {os.path.basename(path)}")

        if not os.path.exists(path):
            print(f"  SKIP: file not found")
            errors += 1
            continue

        try:
            frame_window = extract_frame_window(path, event['start_frame'], event['end_frame'])
        except Exception as exc:
            print(f"  SKIP: {exc}")
            errors += 1
            continue

        if not frame_window:
            print(f"  SKIP: empty window")
            errors += 1
            continue

        frames_text = format_frames_for_analysis(frame_window, event)
        output.append({
            'index': i,
            'tier': event['tier'],
            'event': {k: v for k, v in event.items() if k != 'frame'},
            'frames_text': frames_text,
            'window_size': len(frame_window),
        })
        print(f"  OK: {len(frame_window)} frames extracted")

    conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(output)} events to {out_path}  ({errors} errors/skips)")


if __name__ == '__main__':
    main()
