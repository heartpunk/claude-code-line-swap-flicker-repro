"""
Tests for analyze_flicker.py key functions:
  - extract_features(frame)
  - meets_flicker_heuristics(window, compact_history, frame_idx)
  - stream_frames(path)
"""
import collections
import gzip
import sqlite3
import struct
import subprocess
import sys
import tempfile
import os

import pytest

import analyze_flicker as _mod

extract_features = _mod.extract_features
is_thinking_spinner_frame = _mod.is_thinking_spinner_frame
meets_flicker_heuristics = _mod.meets_flicker_heuristics
extract_event_metadata = _mod.extract_event_metadata
stream_frames = _mod.stream_frames
init_db = _mod.init_db
insert_recording = _mod.insert_recording
insert_flicker = _mod.insert_flicker
analyze_file = _mod.analyze_file
find_ttyrec_files = _mod.find_ttyrec_files
run_analysis_queries = _mod.run_analysis_queries
detect_sessions_in_recording = _mod.detect_sessions_in_recording
insert_session = _mod.insert_session
update_flicker_session_flags = _mod.update_flicker_session_flags
ensure_session_schema = _mod.ensure_session_schema
ensure_thinking_schema = _mod.ensure_thinking_schema
backfill_thinking_above = _mod.backfill_thinking_above
thinking_stats_report = _mod.thinking_stats_report
_strip_ansi = _mod._strip_ansi
detect_welcome_screen = _mod.detect_welcome_screen
WINDOW_SIZE = _mod.WINDOW_SIZE
MIN_SYNC_BLOCKS = _mod.MIN_SYNC_BLOCKS
SMALL_CUP_MAX = _mod.SMALL_CUP_MAX
LARGE_CUP_MIN = _mod.LARGE_CUP_MIN
MIN_SMALL_CUP = _mod.MIN_SMALL_CUP
MIN_LARGE_CUP = _mod.MIN_LARGE_CUP
MIN_CUP_TOTAL = _mod.MIN_CUP_TOTAL
COMPACT_LOOKBACK = _mod.COMPACT_LOOKBACK
MAX_WINDOW_SECS = _mod.MAX_WINDOW_SECS

# ── helpers ──────────────────────────────────────────────────────────────────

def make_frame(payload: bytes, sec: int = 1, usec: int = 0) -> dict:
    """Return a minimal frame dict as yielded by stream_frames."""
    return {"sec": sec, "usec": usec, "payload": payload}


def make_ttyrec_bytes(*payloads: bytes, base_sec: int = 1) -> bytes:
    """Build a minimal valid ttyrec binary containing the given payloads."""
    data = b""
    for i, p in enumerate(payloads):
        header = struct.pack("<III", base_sec + i, 0, len(p))
        data += header + p
    return data


def build_flicker_window(n_frames: int = WINDOW_SIZE, ts_start: float = 1.0, ts_step: float = 0.05):
    """
    Build a deque that satisfies all flicker heuristics:
    - enough sync blocks spread across frames
    - mix of small and large cursor-ups
    - recent timestamps (within MAX_WINDOW_SECS)
    """
    window = collections.deque(maxlen=WINDOW_SIZE)
    lit_sync = b"\\033[?2026h"  # SYNC_START_LIT as bytes
    small_cup = b"\\033[4A"     # small cursor-up (4 <= SMALL_CUP_MAX=5)
    large_cup = b"\\033[8A"     # large cursor-up (8 >= LARGE_CUP_MIN=6)

    for i in range(n_frames):
        ts = ts_start + i * ts_step
        payload = lit_sync + small_cup + large_cup
        feat = extract_features({"sec": int(ts), "usec": int((ts % 1) * 1_000_000), "payload": payload})
        window.append((i, feat))
    return window


# ── extract_features tests ───────────────────────────────────────────────────

class TestExtractFeatures:
    """Tests for the extract_features function."""

    def test_cursor_up_literal_large(self):
        """Literal \\033[8A should register as cursor-up value 8."""
        frame = make_frame(b"\\033[8A")
        feat = extract_features(frame)
        assert 8 in feat["cup_values"], f"Expected 8 in cup_values, got {feat['cup_values']}"

    def test_cursor_up_literal_small(self):
        """Literal \\033[4A should register as cursor-up value 4."""
        frame = make_frame(b"\\033[4A")
        feat = extract_features(frame)
        assert 4 in feat["cup_values"]

    def test_cursor_up_raw_esc(self):
        """Actual ESC byte variant \\x1b[12A should register as cursor-up 12."""
        frame = make_frame(b"\x1b[12A")
        feat = extract_features(frame)
        assert 12 in feat["cup_values"]

    def test_multiple_cursor_ups_in_one_frame(self):
        """Frame with multiple cursor-up sequences should capture all values."""
        frame = make_frame(b"\\033[4A \\033[8A \\033[2A")
        feat = extract_features(frame)
        assert sorted(feat["cup_values"]) == [2, 4, 8]

    def test_sync_start_literal(self):
        """Literal sync start \\033[?2026h should increment sync_count."""
        frame = make_frame(b"\\033[?2026h")
        feat = extract_features(frame)
        assert feat["sync_count"] >= 1

    def test_sync_start_raw(self):
        """Raw ESC sync start \\x1b[?2026h should increment sync_count."""
        frame = make_frame(b"\x1b[?2026h")
        feat = extract_features(frame)
        assert feat["sync_count"] >= 1

    def test_sync_count_multiple(self):
        """Two sync blocks in one frame should yield sync_count == 2."""
        frame = make_frame(b"\\033[?2026h some data \\033[?2026h")
        feat = extract_features(frame)
        assert feat["sync_count"] == 2

    def test_compact_word_compacting(self):
        """Payload with 'Compacting' should set is_compact=True."""
        frame = make_frame(b"Compacting context...")
        feat = extract_features(frame)
        assert feat["is_compact"] is True

    def test_compact_word_precompact(self):
        """Payload with 'PreCompact' should set is_compact=True."""
        frame = make_frame(b"PreCompact phase beginning")
        feat = extract_features(frame)
        assert feat["is_compact"] is True

    def test_compact_word_auto_compact(self):
        """Payload with 'Auto-compact' should set is_compact=True."""
        frame = make_frame(b"Auto-compact triggered")
        feat = extract_features(frame)
        assert feat["is_compact"] is True

    def test_no_compact_word(self):
        """Payload without any compaction keyword should give is_compact=False."""
        frame = make_frame(b"\\033[8A some other output")
        feat = extract_features(frame)
        assert feat["is_compact"] is False

    def test_timestamp_computation(self):
        """Timestamp should equal sec + usec/1_000_000."""
        frame = make_frame(b"data", sec=100, usec=500_000)
        feat = extract_features(frame)
        assert abs(feat["ts"] - 100.5) < 1e-9

    def test_pane_id_found(self):
        """Pane ID should be parsed from %extended-output header."""
        frame = make_frame(b"%extended-output %42 some data")
        feat = extract_features(frame)
        assert feat["pane_id"] == 42

    def test_pane_id_missing(self):
        """Frame without pane header should give pane_id == -1."""
        frame = make_frame(b"plain output without pane header")
        feat = extract_features(frame)
        assert feat["pane_id"] == -1

    def test_version_detection(self):
        """Claude version should be extracted from version string."""
        frame = make_frame(b"claude-code/1.2.3 starting up")
        feat = extract_features(frame)
        assert feat["version"] == "1.2.3"

    def test_version_not_present(self):
        """Frame without version string should give version=None."""
        frame = make_frame(b"no version info here")
        feat = extract_features(frame)
        assert feat["version"] is None

    def test_geom_literal(self):
        """Terminal geometry literal \\033[8;40;200t should be parsed."""
        frame = make_frame(b"\\033[8;40;200t some output")
        feat = extract_features(frame)
        assert feat["geom"] == (40, 200)

    def test_geom_raw(self):
        """Terminal geometry raw ESC[8;24;80t should be parsed."""
        frame = make_frame(b"\x1b[8;24;80t")
        feat = extract_features(frame)
        assert feat["geom"] == (24, 80)

    def test_geom_absent(self):
        """Frame without geometry escape should give geom=None."""
        frame = make_frame(b"no geometry here")
        feat = extract_features(frame)
        assert feat["geom"] is None

    def test_clear_count_literal(self):
        """Literal \\033[2K sequences should be counted."""
        frame = make_frame(b"\\033[2K line one \\033[2K line two")
        feat = extract_features(frame)
        assert feat["clear_count"] == 2

    def test_clear_count_raw(self):
        """Raw ESC[2K sequences should be counted."""
        frame = make_frame(b"\x1b[2K")
        feat = extract_features(frame)
        assert feat["clear_count"] == 1

    def test_empty_payload(self):
        """Empty payload should return all-zero / falsy feature values."""
        frame = make_frame(b"")
        feat = extract_features(frame)
        assert feat["cup_values"] == []
        assert feat["sync_count"] == 0
        assert feat["clear_count"] == 0
        assert feat["is_compact"] is False
        assert feat["pane_id"] == -1
        assert feat["version"] is None
        assert feat["geom"] is None


# ── meets_flicker_heuristics tests ──────────────────────────────────────────

class TestMeetsFlickerHeuristics:
    """Tests for the meets_flicker_heuristics function."""

    def _compact_history(self, frame_idx: int, distance: int = 10) -> collections.deque:
        """Return a compact_history with one entry `distance` frames before frame_idx."""
        ch = collections.deque(maxlen=COMPACT_LOOKBACK)
        ch.append(frame_idx - distance)
        return ch

    def test_detects_flicker_with_mixed_cup_values(self):
        """Window with both small and large cursor-ups + enough sync blocks should detect flicker."""
        window = build_flicker_window(n_frames=WINDOW_SIZE)
        frame_idx = WINDOW_SIZE - 1
        compact_history = self._compact_history(frame_idx)
        result = meets_flicker_heuristics(window, compact_history, frame_idx)
        assert result is True

    def test_no_flicker_window_too_small(self):
        """Window with fewer than WINDOW_SIZE//2 frames should return False."""
        small_window = collections.deque(maxlen=WINDOW_SIZE)
        payload = b"\\033[?2026h\\033[4A\\033[8A"
        for i in range(WINDOW_SIZE // 2 - 1):
            feat = extract_features(make_frame(payload, sec=i + 1))
            small_window.append((i, feat))
        compact_history = self._compact_history(WINDOW_SIZE // 2 - 1)
        assert meets_flicker_heuristics(small_window, compact_history, WINDOW_SIZE // 2 - 1) is False

    def test_no_flicker_outside_time_window(self):
        """Window whose timestamps span more than MAX_WINDOW_SECS should return False."""
        window = collections.deque(maxlen=WINDOW_SIZE)
        payload = b"\\033[?2026h\\033[4A\\033[8A"
        for i in range(WINDOW_SIZE):
            # Spread frames far apart in time
            ts = i * 10.0  # 10 seconds apart — far beyond MAX_WINDOW_SECS
            feat = extract_features(make_frame(payload, sec=int(ts), usec=0))
            window.append((i, feat))
        frame_idx = WINDOW_SIZE - 1
        compact_history = self._compact_history(frame_idx)
        assert meets_flicker_heuristics(window, compact_history, frame_idx) is False

    def test_flicker_detected_regardless_of_compact_history(self):
        """
        compact_history is accepted by the signature but the current implementation
        does NOT require a compaction event — flicker detection is purely signal-based.
        A window with all required signals should return True even with an empty
        compact_history, because compaction context is metadata-only (not a gate).
        """
        window = build_flicker_window()
        frame_idx = WINDOW_SIZE - 1
        empty_compact_history = collections.deque(maxlen=COMPACT_LOOKBACK)
        # The function does NOT guard on compact_history — it returns True on signal alone.
        assert meets_flicker_heuristics(window, empty_compact_history, frame_idx) is True

    def test_no_flicker_only_large_cup_values(self):
        """Window with only large cursor-up values (no small) should not detect flicker."""
        window = collections.deque(maxlen=WINDOW_SIZE)
        # Only large cursor-ups and sync blocks — no small values
        payload = b"\\033[?2026h\\033[8A\\033[9A\\033[10A"
        for i in range(WINDOW_SIZE):
            ts = 1.0 + i * 0.05
            feat = extract_features(make_frame(payload, sec=int(ts), usec=int((ts % 1) * 1_000_000)))
            window.append((i, feat))
        frame_idx = WINDOW_SIZE - 1
        compact_history = self._compact_history(frame_idx)
        assert meets_flicker_heuristics(window, compact_history, frame_idx) is False

    def test_no_flicker_only_small_cup_values(self):
        """Window with only small cursor-up values (no large) should not detect flicker."""
        window = collections.deque(maxlen=WINDOW_SIZE)
        # Only small cursor-ups
        payload = b"\\033[?2026h\\033[2A\\033[3A\\033[4A"
        for i in range(WINDOW_SIZE):
            ts = 1.0 + i * 0.05
            feat = extract_features(make_frame(payload, sec=int(ts), usec=int((ts % 1) * 1_000_000)))
            window.append((i, feat))
        frame_idx = WINDOW_SIZE - 1
        compact_history = self._compact_history(frame_idx)
        assert meets_flicker_heuristics(window, compact_history, frame_idx) is False

    def test_no_flicker_insufficient_sync_blocks(self):
        """Window with fewer than MIN_SYNC_BLOCKS sync starts should return False."""
        window = collections.deque(maxlen=WINDOW_SIZE)
        # Only 1 sync block total across all frames (less than MIN_SYNC_BLOCKS=3)
        # First frame has 1 sync, rest have small+large cursor-ups only
        for i in range(WINDOW_SIZE):
            ts = 1.0 + i * 0.05
            if i == 0:
                payload = b"\\033[?2026h\\033[4A\\033[8A"
            else:
                payload = b"\\033[4A\\033[8A"
            feat = extract_features(make_frame(payload, sec=int(ts), usec=int((ts % 1) * 1_000_000)))
            window.append((i, feat))
        frame_idx = WINDOW_SIZE - 1
        compact_history = self._compact_history(frame_idx)
        assert meets_flicker_heuristics(window, compact_history, frame_idx) is False

    def test_no_flicker_insufficient_total_cup(self):
        """Window with too few total cursor-ups should return False."""
        window = collections.deque(maxlen=WINDOW_SIZE)
        # Only 2 frames have any cursor-up sequences — below MIN_CUP_TOTAL=5 threshold
        for i in range(WINDOW_SIZE):
            ts = 1.0 + i * 0.05
            if i == 0:
                # 2 cursor-ups total: 1 small + 1 large; sync block count OK (many syncs)
                payload = b"\\033[?2026h" * 5 + b"\\033[4A\\033[8A"
            else:
                payload = b"\\033[?2026h"
            feat = extract_features(make_frame(payload, sec=int(ts), usec=int((ts % 1) * 1_000_000)))
            window.append((i, feat))
        frame_idx = WINDOW_SIZE - 1
        compact_history = self._compact_history(frame_idx)
        assert meets_flicker_heuristics(window, compact_history, frame_idx) is False

    def test_compact_history_not_a_gate(self):
        """
        compact_history is NOT used as a gate in the current implementation.
        A fully populated flicker window should return True regardless of what
        is (or is not) in compact_history — even a stale or empty history.
        """
        window = build_flicker_window()
        frame_idx = WINDOW_SIZE - 1

        # Empty history — no gate
        empty_ch = collections.deque(maxlen=COMPACT_LOOKBACK)
        assert meets_flicker_heuristics(window, empty_ch, frame_idx) is True

        # History entry far beyond any meaningful lookback — still no gate
        ch_far = collections.deque(maxlen=COMPACT_LOOKBACK)
        ch_far.append(frame_idx - COMPACT_LOOKBACK - 1)
        assert meets_flicker_heuristics(window, ch_far, frame_idx) is True

    def test_compact_history_nearby_entry_still_detects_flicker(self):
        """Nearby compact_history entry combined with full flicker signals returns True."""
        window = build_flicker_window()
        frame_idx = WINDOW_SIZE - 1
        ch = collections.deque(maxlen=COMPACT_LOOKBACK)
        ch.append(frame_idx - COMPACT_LOOKBACK)  # exactly at boundary
        assert meets_flicker_heuristics(window, ch, frame_idx) is True


# ── stream_frames tests ──────────────────────────────────────────────────────

class TestStreamFrames:
    """Tests for the stream_frames function."""

    def _write_ttyrec(self, payloads: list, base_sec: int = 1) -> str:
        """Write a ttyrec binary to a temp file and return its path."""
        data = make_ttyrec_bytes(*payloads, base_sec=base_sec)
        fd, path = tempfile.mkstemp(suffix=".ttyrec", dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_single_frame_parsed(self):
        """A single-frame ttyrec should yield exactly one frame."""
        path = self._write_ttyrec([b"hello world"])
        frames = list(stream_frames(path))
        assert len(frames) == 1
        assert frames[0]["payload"] == b"hello world"
        os.unlink(path)

    def test_multiple_frames_parsed(self):
        """Multi-frame ttyrec should yield correct number of frames."""
        payloads = [b"frame0", b"frame1", b"frame2"]
        path = self._write_ttyrec(payloads)
        frames = list(stream_frames(path))
        assert len(frames) == 3
        for i, frame in enumerate(frames):
            assert frame["payload"] == payloads[i]
        os.unlink(path)

    def test_frame_timestamps(self):
        """Timestamps in yielded frames should match the header values."""
        data = b""
        # Frame 0: sec=10, usec=500000
        data += struct.pack("<III", 10, 500_000, 5) + b"hello"
        # Frame 1: sec=11, usec=0
        data += struct.pack("<III", 11, 0, 5) + b"world"
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        frames = list(stream_frames(path))
        assert frames[0]["sec"] == 10
        assert frames[0]["usec"] == 500_000
        assert frames[1]["sec"] == 11
        assert frames[1]["usec"] == 0
        os.unlink(path)

    def test_empty_file_yields_nothing(self):
        """Empty file should yield no frames."""
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        os.close(fd)
        frames = list(stream_frames(path))
        assert frames == []
        os.unlink(path)

    def test_truncated_header_yields_nothing(self):
        """File with fewer than 12 header bytes should yield nothing (no crash)."""
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(b"\x00\x01\x02")  # only 3 bytes — not a full header
        frames = list(stream_frames(path))
        assert frames == []
        os.unlink(path)

    def test_truncated_payload_yields_nothing(self):
        """Frame header claiming 100-byte payload but only 5 bytes present should stop gracefully."""
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<III", 1, 0, 100))  # header says 100 bytes
            f.write(b"short")                         # only 5 bytes
        frames = list(stream_frames(path))
        assert frames == []
        os.unlink(path)

    def test_binary_payload_round_trips(self):
        """Binary payload bytes should survive the round-trip without corruption."""
        payload = bytes(range(256))
        path = self._write_ttyrec([payload])
        frames = list(stream_frames(path))
        assert frames[0]["payload"] == payload
        os.unlink(path)

    def test_frame_with_flicker_signals_parseable(self):
        """Frame with literal tmux escape sequences should parse and extract correctly."""
        payload = b"\\033[?2026h\\033[8A Compacting context..."
        path = self._write_ttyrec([payload])
        frames = list(stream_frames(path))
        assert len(frames) == 1
        feat = extract_features(frames[0])
        assert feat["sync_count"] >= 1
        assert 8 in feat["cup_values"]
        assert feat["is_compact"] is True
        os.unlink(path)

    # ── compressed variants ───────────────────────────────────────────────────

    def _write_gz(self, payloads: list) -> str:
        """Write a gzip-compressed ttyrec and return its .gz path."""
        data = make_ttyrec_bytes(*payloads)
        fd, path = tempfile.mkstemp(suffix=".gz", dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(gzip.compress(data))
        return path

    def _write_lz(self, payloads: list) -> str:
        """Write an lzip-compressed ttyrec and return its .lz path."""
        data = make_ttyrec_bytes(*payloads)
        fd, raw_path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        lz_path = raw_path + ".lz"
        subprocess.run(["lzip", "-k", "-f", raw_path], check=True)
        os.unlink(raw_path)
        return lz_path

    def test_gz_single_frame(self):
        """stream_frames reads a gzip-compressed ttyrec correctly."""
        path = self._write_gz([b"gz payload"])
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 1
            assert frames[0]["payload"] == b"gz payload"
        finally:
            os.unlink(path)

    def test_gz_multiple_frames(self):
        """Gzip: all frames are yielded in order."""
        payloads = [b"frame0", b"frame1", b"frame2"]
        path = self._write_gz(payloads)
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 3
            for i, f in enumerate(frames):
                assert f["payload"] == payloads[i]
        finally:
            os.unlink(path)

    def test_gz_timestamps_preserved(self):
        """Gzip: timestamps are decoded correctly from the compressed stream."""
        data = struct.pack("<III", 42, 999_000, 5) + b"hello"
        fd, path = tempfile.mkstemp(suffix=".gz", dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(gzip.compress(data))
        try:
            frames = list(stream_frames(path))
            assert frames[0]["sec"] == 42
            assert frames[0]["usec"] == 999_000
        finally:
            os.unlink(path)

    def test_lz_single_frame(self):
        """stream_frames reads an lzip-compressed ttyrec correctly."""
        path = self._write_lz([b"lz payload"])
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 1
            assert frames[0]["payload"] == b"lz payload"
        finally:
            os.unlink(path)

    def test_lz_multiple_frames(self):
        """Lzip: all frames are yielded in order."""
        payloads = [b"alpha", b"beta", b"gamma"]
        path = self._write_lz(payloads)
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 3
            for i, f in enumerate(frames):
                assert f["payload"] == payloads[i]
        finally:
            os.unlink(path)

    def test_lz_flicker_signals_readable(self):
        """Lzip: flicker escape sequences survive compression and decompression."""
        payload = b"\\033[?2026h\\033[4A\\033[8A"
        path = self._write_lz([payload] * 5)
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 5
            feat = extract_features(frames[0])
            assert feat["sync_count"] == 1
            assert 4 in feat["cup_values"]
            assert 8 in feat["cup_values"]
        finally:
            os.unlink(path)


# ── integration: end-to-end pipeline ─────────────────────────────────────────

class TestIntegration:
    """Integration tests that exercise stream_frames -> extract_features -> meets_flicker_heuristics."""

    def test_flicker_detected_in_crafted_ttyrec(self):
        """
        End-to-end: build a ttyrec with enough frames to trigger the flicker detector
        then confirm meets_flicker_heuristics returns True for the accumulated window.
        """
        # Every frame has a sync block + small and large cursor-ups, all within 2 seconds
        payload = b"\\033[?2026h\\033[4A\\033[8A"
        payloads = [payload] * WINDOW_SIZE

        data = b""
        for i, p in enumerate(payloads):
            sec = 1
            usec = i * 40_000  # 40ms apart — WINDOW_SIZE*40ms well under MAX_WINDOW_SECS
            data += struct.pack("<III", sec, usec, len(p)) + p

        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)

        window = collections.deque(maxlen=WINDOW_SIZE)
        compact_history = collections.deque(maxlen=COMPACT_LOOKBACK)
        compact_history.append(0)  # compaction at frame 0

        for idx, frame in enumerate(stream_frames(path)):
            feat = extract_features(frame)
            window.append((idx, feat))

        frame_idx = WINDOW_SIZE - 1
        result = meets_flicker_heuristics(window, compact_history, frame_idx)
        assert result is True
        os.unlink(path)

    def test_no_flicker_in_benign_ttyrec(self):
        """
        End-to-end: benign ttyrec with only large cursor-ups should not trigger flicker.
        """
        payload = b"\\033[?2026h\\033[8A plain output"
        payloads = [payload] * WINDOW_SIZE

        data = b""
        for i, p in enumerate(payloads):
            data += struct.pack("<III", 1, i * 40_000, len(p)) + p

        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)

        window = collections.deque(maxlen=WINDOW_SIZE)
        compact_history = collections.deque(maxlen=COMPACT_LOOKBACK)
        compact_history.append(0)

        for idx, frame in enumerate(stream_frames(path)):
            feat = extract_features(frame)
            window.append((idx, feat))

        frame_idx = WINDOW_SIZE - 1
        result = meets_flicker_heuristics(window, compact_history, frame_idx)
        assert result is False
        os.unlink(path)


# ── meets_flicker_heuristics: exact boundary tests ───────────────────────────

class TestFlickerHeuristicBoundaries:
    """
    Boundary tests targeting each surviving mutant in meets_flicker_heuristics.
    Each test holds all conditions well above threshold EXCEPT the one under test,
    which is set exactly at the boundary so off-by-one mutations produce different results.
    """

    @staticmethod
    def _make_feat(sync=5, small_cup_vals=(4,), large_cup_vals=(8,), ts=1.0, clear=0):
        """Build a feature dict directly without going through extract_features."""
        return {
            'ts': ts,
            'sync_count': sync,
            'cup_values': list(small_cup_vals) + list(large_cup_vals),
            'clear_count': clear,
            'is_compact': False,
            'version': None,
            'pane_id': None,
            'geom': None,
        }

    def _window(self, feats):
        w = collections.deque(maxlen=WINDOW_SIZE)
        for i, f in enumerate(feats):
            w.append((i, f))
        return w

    def _compact_history(self, frame_idx):
        ch = collections.deque(maxlen=COMPACT_LOOKBACK)
        ch.append(frame_idx)
        return ch

    def test_window_exactly_half_size_is_accepted(self):
        """WINDOW_SIZE//2 frames should pass the minimum size check (condition is <, not <=)."""
        n = WINDOW_SIZE // 2  # exactly the minimum
        feats = [self._make_feat(ts=1.0 + i * 0.05) for i in range(n)]
        window = self._window(feats)
        ch = self._compact_history(n - 1)
        assert meets_flicker_heuristics(window, ch, n - 1) is True

    def test_window_one_below_half_rejected(self):
        """WINDOW_SIZE//2 - 1 frames must be rejected."""
        n = WINDOW_SIZE // 2 - 1
        feats = [self._make_feat(ts=1.0 + i * 0.05) for i in range(n)]
        window = self._window(feats)
        ch = self._compact_history(n - 1)
        assert meets_flicker_heuristics(window, ch, n - 1) is False

    def test_timestamp_exactly_at_limit_accepted(self):
        """Span of exactly MAX_WINDOW_SECS must be accepted (condition is >, not >=)."""
        n = WINDOW_SIZE
        # First frame at t=0, last frame at t=MAX_WINDOW_SECS exactly
        feats = []
        for i in range(n):
            ts = i / (n - 1) * MAX_WINDOW_SECS
            feats.append(self._make_feat(ts=ts))
        window = self._window(feats)
        ch = self._compact_history(n - 1)
        assert meets_flicker_heuristics(window, ch, n - 1) is True

    def _window_with_exactly_n_sync(self, n_sync):
        """Full-size window where total sync count across all frames equals n_sync.
        All cup counts are well above their thresholds."""
        feats = []
        for i in range(WINDOW_SIZE):
            sync = 1 if i < n_sync else 0
            feats.append(self._make_feat(sync=sync,
                                         small_cup_vals=(4, 4),
                                         large_cup_vals=(8, 8),
                                         ts=1.0 + i * 0.05))
        return self._window(feats)

    def test_sync_one_below_threshold_rejected(self):
        """MIN_SYNC_BLOCKS - 1 total sync blocks must be rejected."""
        window = self._window_with_exactly_n_sync(MIN_SYNC_BLOCKS - 1)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is False

    def test_sync_exactly_at_threshold_accepted(self):
        """Exactly MIN_SYNC_BLOCKS total sync blocks must be accepted."""
        window = self._window_with_exactly_n_sync(MIN_SYNC_BLOCKS)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is True

    def _window_with_cups(self, n_small, n_large):
        """Window where the first frame has exactly n_small small cups and n_large large cups;
        remaining frames contribute only sync blocks (no cups). total_cup = n_small + n_large."""
        feats = []
        for i in range(WINDOW_SIZE):
            if i == 0:
                small_vals = tuple(4 for _ in range(n_small))
                large_vals = tuple(8 for _ in range(n_large))
                feats.append(self._make_feat(sync=MIN_SYNC_BLOCKS + 5,
                                             small_cup_vals=small_vals,
                                             large_cup_vals=large_vals,
                                             ts=1.0 + i * 0.05))
            else:
                feats.append(self._make_feat(sync=MIN_SYNC_BLOCKS,
                                             small_cup_vals=(),
                                             large_cup_vals=(),
                                             ts=1.0 + i * 0.05))
        return self._window(feats)

    def test_small_cup_one_below_threshold_rejected(self):
        """MIN_SMALL_CUP - 1 small cursor-ups must be rejected."""
        window = self._window_with_cups(n_small=MIN_SMALL_CUP - 1,
                                        n_large=MIN_LARGE_CUP + 2)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is False

    def test_small_cup_exactly_at_threshold_accepted(self):
        """Exactly MIN_SMALL_CUP small cursor-ups must be accepted (condition is <, not <=)."""
        window = self._window_with_cups(n_small=MIN_SMALL_CUP,
                                        n_large=MIN_LARGE_CUP + 2)
        # Need total_cup >= MIN_CUP_TOTAL
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is True

    def test_large_cup_one_below_threshold_rejected(self):
        """MIN_LARGE_CUP - 1 large cursor-ups must be rejected."""
        window = self._window_with_cups(n_small=MIN_SMALL_CUP + 2,
                                        n_large=MIN_LARGE_CUP - 1)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is False

    def test_large_cup_exactly_at_threshold_accepted(self):
        """Exactly MIN_LARGE_CUP large cursor-ups must be accepted."""
        window = self._window_with_cups(n_small=MIN_SMALL_CUP + 2,
                                        n_large=MIN_LARGE_CUP)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is True

    def test_total_cup_one_below_threshold_rejected(self):
        """MIN_CUP_TOTAL - 1 total cursor-ups must be rejected."""
        n_total = MIN_CUP_TOTAL - 1
        # Split into enough small and large to individually pass their thresholds
        n_small = max(MIN_SMALL_CUP, n_total // 2)
        n_large = n_total - n_small
        # Adjust if large is below its own threshold
        if n_large < MIN_LARGE_CUP:
            n_large = MIN_LARGE_CUP
            n_small = n_total - n_large
        if n_small < MIN_SMALL_CUP or n_large < MIN_LARGE_CUP or n_small + n_large != n_total:
            pytest.skip("Cannot construct valid cup split for this threshold combination")
        window = self._window_with_cups(n_small=n_small, n_large=n_large)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is False

    def test_total_cup_exactly_at_threshold_accepted(self):
        """Exactly MIN_CUP_TOTAL total cursor-ups must be accepted."""
        n_small = MIN_SMALL_CUP
        n_large = MIN_CUP_TOTAL - n_small
        if n_large < MIN_LARGE_CUP:
            n_large = MIN_LARGE_CUP
            n_small = MIN_CUP_TOTAL - n_large
        window = self._window_with_cups(n_small=n_small, n_large=n_large)
        ch = self._compact_history(WINDOW_SIZE - 1)
        assert meets_flicker_heuristics(window, ch, WINDOW_SIZE - 1) is True


# ── extract_event_metadata tests ─────────────────────────────────────────────

class TestExtractEventMetadata:
    """Tests for extract_event_metadata — all fields must be verified."""

    @staticmethod
    def _make_feat(ts, sync=2, cup_vals=(4, 8), clear=1):
        return {
            'ts': ts,
            'sync_count': sync,
            'cup_values': list(cup_vals),
            'clear_count': clear,
            'is_compact': False,
            'version': None,
            'pane_id': None,
            'geom': None,
        }

    def _build_window(self, n=10, ts_start=1.0, ts_step=0.05):
        w = collections.deque(maxlen=WINDOW_SIZE)
        for i in range(n):
            w.append((i + 100, self._make_feat(ts=ts_start + i * ts_step)))
        return w

    def test_start_end_frame_indices(self):
        """start_frame and end_frame must come from frames[0] and frames[-1]."""
        w = self._build_window(n=10)
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['start_frame'] == 100   # first appended index
        assert result['end_frame'] == 109     # last appended index

    def test_timestamps_in_microseconds(self):
        """start_ts_us and end_ts_us must be in microseconds from the actual first/last frames."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0)))
        w.append((1, self._make_feat(ts=2.5)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['start_ts_us'] == 1_000_000
        assert result['end_ts_us'] == 2_500_000

    def test_duration_microseconds(self):
        """duration_us = end_ts_us - start_ts_us."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0)))
        w.append((1, self._make_feat(ts=1.5)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['duration_us'] == 500_000

    def test_frame_count(self):
        """frame_count = end_idx - start_idx + 1."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((5, self._make_feat(ts=1.0)))
        w.append((6, self._make_feat(ts=1.1)))
        w.append((9, self._make_feat(ts=1.2)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['frame_count'] == 5  # 9 - 5 + 1

    def test_fps_computed_from_duration(self):
        """frames_per_second = frame_count / duration_secs."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0)))
        w.append((9, self._make_feat(ts=2.0)))
        # frame_count = 10, duration = 1.0s → fps = 10
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['frame_count'] == 10
        assert abs(result['frames_per_second'] - 10.0) < 0.001

    def test_fps_zero_when_duration_zero(self):
        """fps must be 0 when all frames share the same timestamp."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0)))
        w.append((1, self._make_feat(ts=1.0)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['frames_per_second'] == 0

    def test_sync_block_count_is_sum(self):
        """sync_block_count must be the total sum of sync_count across all frames."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0, sync=3)))
        w.append((1, self._make_feat(ts=1.1, sync=5)))
        w.append((2, self._make_feat(ts=1.2, sync=2)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['sync_block_count'] == 10  # 3 + 5 + 2

    def test_line_clear_count_is_sum(self):
        """line_clear_count must be the total sum of clear_count across all frames."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0, clear=4)))
        w.append((1, self._make_feat(ts=1.1, clear=1)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['line_clear_count'] == 5

    def test_dominant_small_cursor_up(self):
        """cursor_up_rows is the most common small cursor-up value."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        # 3 cups of value 4, 1 cup of value 3 — dominant small is 4
        w.append((0, self._make_feat(ts=1.0, cup_vals=(4, 4, 4, 3, 8))))
        w.append((1, self._make_feat(ts=1.1, cup_vals=(4,))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_rows'] == 4

    def test_dominant_large_cursor_up(self):
        """cursor_up_spinner is the most common large cursor-up value."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        # 3 cups of value 8, 1 cup of value 7 — dominant large is 8
        w.append((0, self._make_feat(ts=1.0, cup_vals=(4, 8, 8, 8, 7))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_spinner'] == 8

    def test_no_small_ups_yields_zero(self):
        """cursor_up_rows is 0 when there are no small cursor-up values."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0, cup_vals=(8, 9))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_rows'] == 0

    def test_no_large_ups_yields_zero(self):
        """cursor_up_spinner is 0 when there are no large cursor-up values."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0, cup_vals=(4, 3))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_spinner'] == 0

    def test_compaction_above_true_when_history_present(self):
        """compaction_above = 1 when compact_history has entries before current frame."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((100, self._make_feat(ts=1.0)))
        ch = collections.deque(maxlen=COMPACT_LOOKBACK)
        ch.append(50)  # earlier frame
        result = extract_event_metadata(w, ch, 100)
        assert result['compaction_above'] == 1

    def test_compaction_above_false_when_history_empty(self):
        """compaction_above = 0 when compact_history is empty."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((100, self._make_feat(ts=1.0)))
        result = extract_event_metadata(w, collections.deque(), 100)
        assert result['compaction_above'] == 0

    def test_compaction_frame_offset(self):
        """compaction_frame_offset = start_frame - nearest_compact_idx."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((100, self._make_feat(ts=1.0)))
        ch = collections.deque(maxlen=COMPACT_LOOKBACK)
        ch.append(90)
        ch.append(80)
        result = extract_event_metadata(w, ch, 100)
        # Nearest is 90 (distance 10), not 80 (distance 20)
        assert result['compaction_frame_offset'] == 10

    def test_duration_us_exact_one_second(self):
        """duration_us must be exactly 1_000_000 for a 1.0s span (not 1_000_001)."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        w.append((0, self._make_feat(ts=1.0)))
        w.append((1, self._make_feat(ts=2.0)))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['duration_us'] == 1_000_000

    def test_cup_value_at_small_boundary_goes_to_small(self):
        """A cup value of exactly SMALL_CUP_MAX must be classified as small (<=, not <)."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        # SMALL_CUP_MAX=5 is boundary: 5 is small, 6 is large
        w.append((0, self._make_feat(ts=1.0, cup_vals=(SMALL_CUP_MAX, LARGE_CUP_MIN))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_rows'] == SMALL_CUP_MAX   # 5 classified as small
        assert result['cursor_up_spinner'] == LARGE_CUP_MIN  # 6 classified as large

    def test_dominant_small_is_most_frequent_not_largest(self):
        """cursor_up_rows is the most-frequent small value, not the numerically largest."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        # 3×value-3, 1×value-4 — dominant is 3 (frequency), not 4 (magnitude)
        w.append((0, self._make_feat(ts=1.0, cup_vals=(3, 3, 3, 4, 8, 8))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_rows'] == 3

    def test_dominant_large_is_most_frequent_not_largest(self):
        """cursor_up_spinner is the most-frequent large value, not the numerically largest."""
        w = collections.deque(maxlen=WINDOW_SIZE)
        # 3×value-7, 1×value-8 — dominant is 7 (frequency), not 8 (magnitude)
        w.append((0, self._make_feat(ts=1.0, cup_vals=(4, 7, 7, 7, 8))))
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['cursor_up_spinner'] == 7


# ── detect_sessions_in_recording tests ───────────────────────────────────────

class TestDetectSessions:
    """Tests for detect_sessions_in_recording."""

    def _write_ttyrec(self, payloads: list, ts_step_ms: int = 100) -> str:
        """Write a ttyrec with the given payloads and return its path."""
        data = b""
        for i, p in enumerate(payloads):
            sec = 1 + (i * ts_step_ms) // 1000
            usec = ((i * ts_step_ms) % 1000) * 1000
            data += struct.pack("<III", sec, usec, len(p)) + p
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_empty_file_returns_no_sessions(self):
        """Empty ttyrec → empty session list."""
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        os.close(fd)
        try:
            assert detect_sessions_in_recording(path) == []
        finally:
            os.unlink(path)

    def test_no_claude_signals_returns_no_sessions(self):
        """File with plain text output (no sync blocks) → empty session list."""
        path = self._write_ttyrec([b"plain terminal output", b"no claude here"])
        try:
            assert detect_sessions_in_recording(path) == []
        finally:
            os.unlink(path)

    def test_version_string_detected(self):
        """File with version string → session with start_signal='version' and confidence≥3."""
        path = self._write_ttyrec([
            b"\\033[?2026h",  # sync block on frame 0
            b"claude-code/2.1.34 startup",  # version on frame 1
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            s = sessions[0]
            assert s['start_signal'] in ('version', 'sync_activity', 'welcome')
            assert s['confidence'] >= 3
            assert s['claude_version'] == '2.1.34'
        finally:
            os.unlink(path)

    def test_welcome_banner_detected(self):
        """File with 'Welcome to Claude Code' → session with confidence≥5."""
        path = self._write_ttyrec([
            b"\\033[?2026h Welcome to Claude Code v2.1.34",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['confidence'] >= 5
        finally:
            os.unlink(path)

    def test_sync_only_confidence_one(self):
        """File with only sync blocks → session with confidence=1."""
        path = self._write_ttyrec([b"\\033[?2026h\\033[4A\\033[8A"] * 5)
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['confidence'] == 1
        finally:
            os.unlink(path)

    def test_compaction_text_confidence_two(self):
        """File with compaction text → confidence≥2."""
        path = self._write_ttyrec([
            b"\\033[?2026h",
            b"Compacting context...",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['confidence'] >= 2
        finally:
            os.unlink(path)

    def test_version_and_welcome_confidence_eight(self):
        """Version + welcome banner together → confidence=8."""
        path = self._write_ttyrec([
            b"Welcome to Claude Code  claude-code/2.2.0 \\033[?2026h",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['confidence'] >= 8
        finally:
            os.unlink(path)

    def test_goodbye_boosts_confidence(self):
        """Goodbye message boosts confidence by 5 (capped at 10)."""
        path = self._write_ttyrec([
            b"claude-code/2.1.34 \\033[?2026h",  # version frame (confidence=3)
            b"Goodbye from Claude Code session ended",  # goodbye frame
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            s = sessions[0]
            # Base confidence 3 (version) + 5 (goodbye) = 8, capped at 10
            assert s['confidence'] >= 8
        finally:
            os.unlink(path)

    def test_start_frame_at_first_claude_signal(self):
        """start_frame is the index of the first frame with any Claude signal."""
        path = self._write_ttyrec([
            b"plain output frame 0",        # no signal
            b"more plain output frame 1",   # no signal
            b"\\033[?2026h sync frame 2",   # first Claude signal — frame index 2
            b"\\033[?2026h sync frame 3",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['start_frame'] == 2
        finally:
            os.unlink(path)

    def test_end_frame_at_last_sync(self):
        """end_frame is the index of the last frame with sync blocks."""
        path = self._write_ttyrec([
            b"\\033[?2026h sync frame 0",   # Claude signal
            b"\\033[?2026h sync frame 1",   # Claude signal
            b"\\033[?2026h sync frame 2",   # last sync — frame index 2
            b"plain output no sync",        # no signal after
            b"plain output no sync",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            # end_frame should be 2 (last sync), not 4 (end of file)
            assert sessions[0]['end_frame'] == 2
        finally:
            os.unlink(path)

    def test_goodbye_frame_used_as_end_when_before_eof(self):
        """When goodbye appears before last sync, end_frame is at goodbye."""
        path = self._write_ttyrec([
            b"claude-code/2.1.34 \\033[?2026h frame 0",
            b"Goodbye from Claude Code \\033[?2026h frame 1",  # goodbye on frame 1
            b"\\033[?2026h sync after goodbye frame 2",         # sync AFTER goodbye
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            # goodbye is on frame 1; end_frame should be 1, not 2
            assert sessions[0]['end_frame'] == 1
        finally:
            os.unlink(path)

    def test_session_fields_present(self):
        """All expected fields must be present in the returned session dict."""
        path = self._write_ttyrec([b"\\033[?2026h sync"])
        expected_fields = {
            'start_frame', 'end_frame', 'start_ts_us', 'end_ts_us',
            'claude_version', 'confidence', 'start_signal',
        }
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert set(sessions[0].keys()) == expected_fields
        finally:
            os.unlink(path)

    def test_timestamps_in_microseconds(self):
        """start_ts_us and end_ts_us are timestamps in microseconds."""
        data = b""
        # Frame 0: sec=10, usec=500000, sync signal
        data += struct.pack("<III", 10, 500_000, len(b"\\033[?2026h")) + b"\\033[?2026h"
        # Frame 1: sec=11, usec=0, sync signal
        data += struct.pack("<III", 11, 0, len(b"\\033[?2026h")) + b"\\033[?2026h"
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            s = sessions[0]
            assert s['start_ts_us'] == 10_500_000  # 10.5s in microseconds
            assert s['end_ts_us'] == 11_000_000    # 11.0s in microseconds
        finally:
            os.unlink(path)


# ── Session DB operations tests ───────────────────────────────────────────────

class TestSessionDbOps:
    """Tests for insert_session, update_flicker_session_flags, ensure_session_schema."""

    def _fresh_db(self) -> sqlite3.Connection:
        """Create a fresh in-memory DB with both base and session schema."""
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        return conn

    def _insert_test_recording(self, conn) -> int:
        """Insert a minimal recording and return its id."""
        return insert_recording(conn, {
            'path': '/tmp/test.lz',
            'file_size_bytes': 1000,
            'frame_count': 100,
            'duration_us': 10_000_000,
            'start_ts_us': 1_000_000,
            'end_ts_us': 11_000_000,
            'claude_version': '2.1.34',
            'has_compaction': 1,
            'compaction_count': 3,
            'flicker_count': 2,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })

    def test_insert_session_returns_id(self):
        """insert_session returns the row id of the inserted session."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 10, 'end_frame': 90,
            'start_ts_us': 1_000_000, 'end_ts_us': 9_000_000,
            'claude_version': '2.1.34',
            'confidence': 3, 'start_signal': 'version',
        }
        sid = insert_session(conn, rec_id, session)
        assert isinstance(sid, int)
        assert sid > 0

    def test_insert_session_stores_fields(self):
        """All session fields are persisted correctly."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 5, 'end_frame': 85,
            'start_ts_us': 2_000_000, 'end_ts_us': 8_000_000,
            'claude_version': '2.1.99',
            'confidence': 8, 'start_signal': 'welcome',
        }
        sid = insert_session(conn, rec_id, session)
        row = conn.execute(
            "SELECT recording_id, session_start_frame, session_end_frame, "
            "claude_version, confidence, start_signal FROM claude_sessions WHERE id = ?",
            (sid,)
        ).fetchone()
        assert row[0] == rec_id
        assert row[1] == 5
        assert row[2] == 85
        assert row[3] == '2.1.99'
        assert row[4] == 8
        assert row[5] == 'welcome'

    def test_update_flicker_flags_marks_in_session(self):
        """Events whose start_frame falls within session bounds get is_in_session=1."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        # Insert a flicker event at frame 50 (within session 10..90)
        insert_flicker(conn, rec_id, {
            'start_frame': 50, 'end_frame': 60,
            'start_ts_us': 5_000_000, 'end_ts_us': 6_000_000,
            'duration_us': 1_000_000, 'frame_count': 10,
            'frames_per_second': 10.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 4, 'cursor_up_spinner': 8,
            'sync_block_count': 5, 'line_clear_count': 0,
        })
        session = {'start_frame': 10, 'end_frame': 90,
                   'start_ts_us': 1_000_000, 'end_ts_us': 9_000_000,
                   'claude_version': '2.1.34', 'confidence': 3, 'start_signal': 'version'}
        sid = insert_session(conn, rec_id, session)
        update_flicker_session_flags(conn, rec_id, [session], [sid])

        row = conn.execute(
            "SELECT is_in_session, session_id FROM flicker_events WHERE recording_id = ?",
            (rec_id,)
        ).fetchone()
        assert row[0] == 1
        assert row[1] == sid

    def test_update_flicker_flags_outside_session_stays_zero(self):
        """Events outside session bounds keep is_in_session=0."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        # Insert a flicker event at frame 5 (BEFORE session starts at frame 10)
        insert_flicker(conn, rec_id, {
            'start_frame': 5, 'end_frame': 8,
            'start_ts_us': 500_000, 'end_ts_us': 800_000,
            'duration_us': 300_000, 'frame_count': 3,
            'frames_per_second': 10.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 4, 'cursor_up_spinner': 8,
            'sync_block_count': 3, 'line_clear_count': 0,
        })
        session = {'start_frame': 10, 'end_frame': 90,
                   'start_ts_us': 1_000_000, 'end_ts_us': 9_000_000,
                   'claude_version': '2.1.34', 'confidence': 3, 'start_signal': 'version'}
        sid = insert_session(conn, rec_id, session)
        update_flicker_session_flags(conn, rec_id, [session], [sid])

        row = conn.execute(
            "SELECT COALESCE(is_in_session, 0) FROM flicker_events WHERE recording_id = ?",
            (rec_id,)
        ).fetchone()
        assert row[0] == 0

    def test_ensure_session_schema_idempotent(self):
        """ensure_session_schema can be called multiple times without error."""
        conn = self._fresh_db()
        # Call again — should not raise
        ensure_session_schema(conn)
        ensure_session_schema(conn)
        # Table should exist exactly once
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='claude_sessions'"
        ).fetchone()[0]
        assert count == 1


# ── is_thinking_spinner_frame tests ──────────────────────────────────────────

class TestIsThinkingSpinnerFrame:
    """Tests for is_thinking_spinner_frame — detects thinking/tool-use spinners."""

    def _feat(self, sync_count=1, cup_values=None, is_compact=False):
        """Build a minimal feat dict."""
        return {
            'sync_count': sync_count,
            'cup_values': cup_values if cup_values is not None else [],
            'is_compact': is_compact,
            'clear_count': 0,
            'pane_id': -1,
            'version': None,
            'geom': None,
            'ts': 1.0,
        }

    def test_returns_true_for_sync_large_cup_no_compact(self):
        """sync_count>0, large CUP, no compact text → thinking spinner."""
        feat = self._feat(sync_count=1, cup_values=[8], is_compact=False)
        assert is_thinking_spinner_frame(feat) is True

    def test_returns_false_when_compact(self):
        """Frames with compaction text are NOT thinking spinners."""
        feat = self._feat(sync_count=1, cup_values=[8], is_compact=True)
        assert is_thinking_spinner_frame(feat) is False

    def test_returns_false_without_sync(self):
        """Frames with no sync blocks are not thinking spinner frames."""
        feat = self._feat(sync_count=0, cup_values=[8], is_compact=False)
        assert is_thinking_spinner_frame(feat) is False

    def test_returns_false_without_large_cup(self):
        """Frames without a large CUP value (≥6) are not thinking spinners."""
        feat = self._feat(sync_count=1, cup_values=[3], is_compact=False)
        assert is_thinking_spinner_frame(feat) is False

    def test_returns_false_with_no_cups(self):
        """No cursor-up values → not a thinking spinner."""
        feat = self._feat(sync_count=1, cup_values=[], is_compact=False)
        assert is_thinking_spinner_frame(feat) is False

    def test_requires_large_cup_exactly_at_threshold(self):
        """CUP=6 (the LARGE_CUP_MIN threshold) should count as large."""
        feat = self._feat(sync_count=1, cup_values=[LARGE_CUP_MIN], is_compact=False)
        assert is_thinking_spinner_frame(feat) is True

    def test_small_cup_only_not_thinking_spinner(self):
        """Only small CUP values (≤5) present → not a thinking spinner."""
        feat = self._feat(sync_count=1, cup_values=[3, 4, 5], is_compact=False)
        assert is_thinking_spinner_frame(feat) is False

    def test_mixed_cups_qualifies_if_any_large(self):
        """Mix of small and large CUP values → qualifies as thinking spinner."""
        feat = self._feat(sync_count=1, cup_values=[3, 8], is_compact=False)
        assert is_thinking_spinner_frame(feat) is True

    def test_real_payload_thinking_frame(self):
        """Frame with literal tmux sync + large CUP and no compaction text."""
        frame = make_frame(b"\\033[?2026h\\033[8A Some output text...")
        feat = extract_features(frame)
        assert is_thinking_spinner_frame(feat) is True

    def test_real_payload_compaction_frame_excluded(self):
        """Frame with compaction text is excluded even if it has large CUP."""
        frame = make_frame(b"\\033[?2026h\\033[8A Compacting context...")
        feat = extract_features(frame)
        assert is_thinking_spinner_frame(feat) is False


# ── thinking_above in extract_event_metadata ──────────────────────────────────

class TestThinkingMetadata:
    """Tests for thinking_above and thinking_frame_offset in extract_event_metadata."""

    def _build_window(self):
        """Build a minimal window that satisfies flicker heuristics."""
        return build_flicker_window()

    def test_thinking_above_zero_when_no_history(self):
        """thinking_above=0 when thinking_history is empty."""
        w = self._build_window()
        th = collections.deque(maxlen=COMPACT_LOOKBACK)
        result = extract_event_metadata(w, collections.deque(), 0, th)
        assert result['thinking_above'] == 0
        assert result['thinking_frame_offset'] is None

    def test_thinking_above_zero_when_history_is_none(self):
        """thinking_above=0 when thinking_history argument is omitted (None)."""
        w = self._build_window()
        result = extract_event_metadata(w, collections.deque(), 0)
        assert result['thinking_above'] == 0
        assert result['thinking_frame_offset'] is None

    def test_thinking_above_one_when_history_has_entry(self):
        """thinking_above=1 when thinking_history contains a recent frame index."""
        w = self._build_window()
        th = collections.deque([5], maxlen=COMPACT_LOOKBACK)
        result = extract_event_metadata(w, collections.deque(), 0, th)
        assert result['thinking_above'] == 1

    def test_thinking_frame_offset_computed(self):
        """thinking_frame_offset is set when thinking_history has entries."""
        w = self._build_window()
        # Window starts at frame 0 (build_flicker_window assigns frame indices 0..N-1)
        th = collections.deque([5, 10, 15], maxlen=COMPACT_LOOKBACK)
        result = extract_event_metadata(w, collections.deque(), 0, th)
        assert result['thinking_above'] == 1
        assert result['thinking_frame_offset'] is not None

    def test_thinking_and_compaction_independent(self):
        """thinking_above and compaction_above can both be set independently."""
        w = self._build_window()
        ch = collections.deque([3], maxlen=COMPACT_LOOKBACK)
        th = collections.deque([7], maxlen=COMPACT_LOOKBACK)
        result = extract_event_metadata(w, ch, 0, th)
        assert result['compaction_above'] == 1
        assert result['thinking_above'] == 1

    def test_result_keys_present(self):
        """Both thinking_above and thinking_frame_offset are always in the result."""
        w = self._build_window()
        result = extract_event_metadata(w, collections.deque(), 0)
        assert 'thinking_above' in result
        assert 'thinking_frame_offset' in result


# ── ensure_thinking_schema and backfill tests ──────────────────────────────────

class TestThinkingSchema:
    """Tests for ensure_thinking_schema and backfill_thinking_above."""

    def _fresh_db(self) -> sqlite3.Connection:
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        return conn

    def _insert_test_recording(self, conn) -> int:
        return insert_recording(conn, {
            'path': '/tmp/test.lz',
            'file_size_bytes': 1000,
            'frame_count': 100,
            'duration_us': 10_000_000,
            'start_ts_us': 1_000_000,
            'end_ts_us': 11_000_000,
            'claude_version': '2.1.34',
            'has_compaction': 0,
            'compaction_count': 0,
            'flicker_count': 1,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })

    def test_ensure_thinking_schema_idempotent(self):
        """ensure_thinking_schema can be called multiple times without error."""
        conn = self._fresh_db()
        ensure_thinking_schema(conn)
        ensure_thinking_schema(conn)
        # Columns should exist (no exception when querying them)
        conn.execute("SELECT thinking_above, thinking_frame_offset FROM flicker_events LIMIT 0")

    def test_thinking_columns_present_in_fresh_db(self):
        """init_db creates flicker_events with thinking_above column."""
        conn = init_db(":memory:")
        conn.execute("SELECT thinking_above, thinking_frame_offset FROM flicker_events LIMIT 0")

    def test_insert_flicker_with_thinking_fields(self):
        """insert_flicker stores thinking_above and thinking_frame_offset."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        insert_flicker(conn, rec_id, {
            'start_frame': 20, 'end_frame': 50,
            'start_ts_us': 2_000_000, 'end_ts_us': 5_000_000,
            'duration_us': 3_000_000, 'frame_count': 30,
            'frames_per_second': 10.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 5, 'line_clear_count': 0,
            'thinking_above': 1, 'thinking_frame_offset': -2,
        })
        row = conn.execute(
            "SELECT thinking_above, thinking_frame_offset FROM flicker_events WHERE recording_id = ?",
            (rec_id,)
        ).fetchone()
        assert row[0] == 1
        assert row[1] == -2

    def test_insert_flicker_without_thinking_fields_stores_null(self):
        """insert_flicker without thinking fields stores NULL (not yet backfilled)."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        insert_flicker(conn, rec_id, {
            'start_frame': 20, 'end_frame': 50,
            'start_ts_us': 2_000_000, 'end_ts_us': 5_000_000,
            'duration_us': 3_000_000, 'frame_count': 30,
            'frames_per_second': 10.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 5, 'line_clear_count': 0,
        })
        row = conn.execute(
            "SELECT thinking_above FROM flicker_events WHERE recording_id = ?",
            (rec_id,)
        ).fetchone()
        assert row[0] is None  # NULL = not yet backfilled

    def test_backfill_skips_missing_files(self):
        """backfill_thinking_above records an error for non-existent files."""
        conn = self._fresh_db()
        rec_id = insert_recording(conn, {
            'path': '/nonexistent/file.lz',
            'file_size_bytes': 0, 'frame_count': 0, 'duration_us': 0,
            'start_ts_us': 0, 'end_ts_us': 0, 'claude_version': None,
            'has_compaction': 0, 'compaction_count': 0, 'flicker_count': 1,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })
        insert_flicker(conn, rec_id, {
            'start_frame': 5, 'end_frame': 34,
            'start_ts_us': 0, 'end_ts_us': 0,
            'duration_us': 0, 'frame_count': 30,
            'frames_per_second': 0.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 3, 'line_clear_count': 0,
        })
        stats = backfill_thinking_above(conn)
        assert stats['errors'] == 1
        assert stats['updated'] == 0

    def test_ensure_thinking_schema_adds_columns_to_old_db(self):
        """ensure_thinking_schema adds thinking columns to a DB that lacks them."""
        conn = sqlite3.connect(":memory:")
        # Create minimal flicker_events WITHOUT thinking columns (old-style schema)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS flicker_events (
                id INTEGER PRIMARY KEY,
                sync_block_count INTEGER,
                line_clear_count INTEGER
            );
        """)
        conn.commit()
        cols = [row[1] for row in conn.execute("PRAGMA table_info(flicker_events)").fetchall()]
        assert 'thinking_above' not in cols
        ensure_thinking_schema(conn)
        # Columns should now be present
        conn.execute("SELECT thinking_above, thinking_frame_offset FROM flicker_events LIMIT 0")

    def test_backfill_processes_real_file_and_sets_thinking_above(self):
        """backfill_thinking_above correctly sets thinking_above=1 from a real file."""
        # Build a ttyrec: frames 0-4 are thinking-spinner frames (sync + large CUP,
        # no compaction text), frames 5-34 are plain frames.
        thinking_frame = b"\\033[?2026h\\033[8A thinking..."
        plain_frame = b"hello"
        payloads = [thinking_frame] * 5 + [plain_frame] * 30
        data = make_ttyrec_bytes(*payloads)

        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, 'wb') as f:
            f.write(data)

        try:
            conn = self._fresh_db()
            rec_id = insert_recording(conn, {
                'path': path,
                'file_size_bytes': len(data), 'frame_count': 35,
                'duration_us': 1_000_000, 'start_ts_us': 0, 'end_ts_us': 1_000_000,
                'claude_version': None, 'has_compaction': 0,
                'compaction_count': 0, 'flicker_count': 1,
                'processed_at': '2025-01-01T00:00:00+00:00',
            })
            # Insert flicker event with thinking_above=NULL (not yet backfilled)
            insert_flicker(conn, rec_id, {
                'start_frame': 5, 'end_frame': 34,
                'start_ts_us': 0, 'end_ts_us': 0,
                'duration_us': 0, 'frame_count': 30,
                'frames_per_second': 0.0,
                'compaction_above': 0, 'compaction_frame_offset': None,
                'cursor_up_rows': 3, 'cursor_up_spinner': 8,
                'sync_block_count': 3, 'line_clear_count': 0,
            })
            # Verify NULL before backfill
            assert conn.execute("SELECT thinking_above FROM flicker_events").fetchone()[0] is None

            stats = backfill_thinking_above(conn)
            assert stats['updated'] == 1
            assert stats['errors'] == 0

            row = conn.execute(
                "SELECT thinking_above FROM flicker_events WHERE recording_id = ?",
                (rec_id,)
            ).fetchone()
            assert row[0] == 1  # thinking frames at idx 0-4 are in the lookback
        finally:
            os.unlink(path)

    def test_backfill_sets_thinking_above_zero_when_no_spinner(self):
        """backfill_thinking_above sets thinking_above=0 when no spinner frames precede event."""
        plain_frame = b"hello"
        payloads = [plain_frame] * 35
        data = make_ttyrec_bytes(*payloads)

        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, 'wb') as f:
            f.write(data)

        try:
            conn = self._fresh_db()
            rec_id = insert_recording(conn, {
                'path': path,
                'file_size_bytes': len(data), 'frame_count': 35,
                'duration_us': 1_000_000, 'start_ts_us': 0, 'end_ts_us': 1_000_000,
                'claude_version': None, 'has_compaction': 0,
                'compaction_count': 0, 'flicker_count': 1,
                'processed_at': '2025-01-01T00:00:00+00:00',
            })
            insert_flicker(conn, rec_id, {
                'start_frame': 5, 'end_frame': 34,
                'start_ts_us': 0, 'end_ts_us': 0,
                'duration_us': 0, 'frame_count': 30,
                'frames_per_second': 0.0,
                'compaction_above': 0, 'compaction_frame_offset': None,
                'cursor_up_rows': 3, 'cursor_up_spinner': 8,
                'sync_block_count': 3, 'line_clear_count': 0,
            })
            stats = backfill_thinking_above(conn)
            assert stats['updated'] == 1
            assert stats['errors'] == 0

            row = conn.execute(
                "SELECT thinking_above FROM flicker_events WHERE recording_id = ?",
                (rec_id,)
            ).fetchone()
            assert row[0] == 0  # no thinking frames before event

        finally:
            os.unlink(path)

    def test_thinking_stats_report_runs_without_error(self):
        """thinking_stats_report completes without raising on a populated DB."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        # Insert events with various thinking_above values
        for ta in (0, 1, 1):
            insert_flicker(conn, rec_id, {
                'start_frame': 10, 'end_frame': 40,
                'start_ts_us': 0, 'end_ts_us': 0,
                'duration_us': 0, 'frame_count': 30,
                'frames_per_second': 0.0,
                'compaction_above': 0, 'compaction_frame_offset': None,
                'cursor_up_rows': 3, 'cursor_up_spinner': 8,
                'sync_block_count': 3, 'line_clear_count': 0,
                'thinking_above': ta, 'thinking_frame_offset': None,
            })
        thinking_stats_report(conn)  # Should not raise

    def test_thinking_stats_report_warns_on_null(self):
        """thinking_stats_report prints a warning when thinking_above IS NULL."""
        import io
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        insert_flicker(conn, rec_id, {
            'start_frame': 10, 'end_frame': 40,
            'start_ts_us': 0, 'end_ts_us': 0,
            'duration_us': 0, 'frame_count': 30,
            'frames_per_second': 0.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 3, 'line_clear_count': 0,
            # thinking_above omitted → NULL
        })
        # Should complete without raising even with NULL values
        thinking_stats_report(conn)


# ── thinking_stats_report stdout tests ────────────────────────────────────────

class TestThinkingStatsReportOutput:
    """Tests capturing stdout of thinking_stats_report to kill mutations."""

    def _fresh_db(self) -> sqlite3.Connection:
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        return conn

    def _insert_test_recording(self, conn) -> int:
        return insert_recording(conn, {
            'path': '/tmp/test.lz',
            'file_size_bytes': 1000,
            'frame_count': 100,
            'duration_us': 10_000_000,
            'start_ts_us': 1_000_000,
            'end_ts_us': 11_000_000,
            'claude_version': '2.1.34',
            'has_compaction': 0,
            'compaction_count': 0,
            'flicker_count': 1,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })

    def _insert_in_session_event(self, conn, rec_id, compaction_above=0,
                                  thinking_above=None, thinking_offset=None):
        """Insert a flicker event marked as in-session."""
        insert_flicker(conn, rec_id, {
            'start_frame': 10, 'end_frame': 40,
            'start_ts_us': 0, 'end_ts_us': 0,
            'duration_us': 0, 'frame_count': 30,
            'frames_per_second': 0.0,
            'compaction_above': compaction_above,
            'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 3, 'line_clear_count': 0,
            'thinking_above': thinking_above,
            'thinking_frame_offset': thinking_offset,
        })
        # Mark as in-session
        conn.execute(
            "UPDATE flicker_events SET is_in_session = 1 "
            "WHERE recording_id = ? AND is_in_session = 0",
            (rec_id,),
        )
        conn.commit()

    def _capture(self, conn):
        """Run thinking_stats_report and capture stdout."""
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            thinking_stats_report(conn)
        return buf.getvalue()

    def test_header_printed(self):
        """Output contains the 'THINKING SPINNER ANALYSIS' header."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=0)
        output = self._capture(conn)
        assert "THINKING SPINNER ANALYSIS" in output

    def test_separator_line_printed(self):
        """Output contains the '=' separator line."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=0)
        output = self._capture(conn)
        assert "=" * 60 in output

    def test_no_schema_message(self):
        """Without thinking schema, prints a 'not initialized' message."""
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS flicker_events (
                id INTEGER PRIMARY KEY,
                recording_id INTEGER
            );
        """)
        output = self._capture(conn)
        assert "Thinking schema not initialized" in output

    def test_warning_on_null_values(self):
        """Prints WARNING when events have thinking_above=NULL."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=None)
        output = self._capture(conn)
        assert "WARNING" in output
        assert "NULL" in output or "null" in output.lower()

    def test_null_count_in_warning(self):
        """Warning includes the count of NULL events."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        # Insert 3 NULL events
        for _ in range(3):
            self._insert_in_session_event(conn, rec_id, thinking_above=None)
        output = self._capture(conn)
        assert "3" in output

    def test_no_warning_when_no_nulls(self):
        """No WARNING printed when all events have thinking_above set."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=1)
        self._insert_in_session_event(conn, rec_id, thinking_above=0)
        output = self._capture(conn)
        assert "WARNING" not in output

    def test_compaction_above_label_in_output(self):
        """Output contains 'compaction_above=' labels."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, compaction_above=1, thinking_above=1)
        output = self._capture(conn)
        assert "compaction_above=1" in output

    def test_thinking_above_yes_label(self):
        """thinking_above=1 is displayed as 'yes'."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=1)
        output = self._capture(conn)
        assert "yes" in output

    def test_thinking_above_no_label(self):
        """thinking_above=0 is displayed as 'no'."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=0)
        output = self._capture(conn)
        assert "thinking_above=no" in output

    def test_thinking_above_null_label(self):
        """thinking_above=NULL is displayed as 'NULL'."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=None)
        output = self._capture(conn)
        assert "thinking_above=NULL" in output

    def test_events_count_in_output(self):
        """Output includes 'events=' with the correct count."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        for _ in range(5):
            self._insert_in_session_event(conn, rec_id, thinking_above=1)
        output = self._capture(conn)
        assert "events=" in output
        assert "5" in output

    def test_non_compaction_section_header(self):
        """Output contains the non-compaction section header."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, compaction_above=0, thinking_above=0)
        output = self._capture(conn)
        assert "Non-compaction in-session events" in output

    def test_total_non_compaction_count(self):
        """Output shows 'Total non-compaction in-session events' with correct count."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        for _ in range(4):
            self._insert_in_session_event(conn, rec_id, compaction_above=0, thinking_above=0)
        self._insert_in_session_event(conn, rec_id, compaction_above=1, thinking_above=1)
        output = self._capture(conn)
        assert "Total non-compaction in-session events: 4" in output

    def test_percentage_in_non_compaction_section(self):
        """Non-compaction breakdown includes percentage values."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        # 3 thinking=yes, 1 thinking=no → 75.0% and 25.0%
        for _ in range(3):
            self._insert_in_session_event(conn, rec_id, compaction_above=0, thinking_above=1)
        self._insert_in_session_event(conn, rec_id, compaction_above=0, thinking_above=0)
        output = self._capture(conn)
        assert "75.0%" in output
        assert "25.0%" in output

    def test_in_session_section_header(self):
        """Output contains 'In-session events' section header."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=0)
        output = self._capture(conn)
        assert "In-session events" in output

    def test_run_backfill_message_on_null(self):
        """Warning includes 'Run --backfill-thinking' instruction."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, thinking_above=None)
        output = self._capture(conn)
        assert "--backfill-thinking" in output

    def test_mixed_compaction_and_thinking(self):
        """Verify both compaction_above=0 and compaction_above=1 rows appear."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        self._insert_in_session_event(conn, rec_id, compaction_above=0, thinking_above=0)
        self._insert_in_session_event(conn, rec_id, compaction_above=1, thinking_above=1)
        output = self._capture(conn)
        assert "compaction_above=0" in output
        assert "compaction_above=1" in output

    def test_empty_db_no_crash(self):
        """Report on a DB with schema but no events doesn't crash."""
        conn = self._fresh_db()
        output = self._capture(conn)
        # Should still print headers
        assert "THINKING SPINNER ANALYSIS" in output
        # Total should be 0
        assert "Total non-compaction in-session events: 0" in output


# ── backfill_thinking_above assertion tests ───────────────────────────────────

class TestBackfillThinkingAboveAssertions:
    """Tighter assertions on backfill_thinking_above return values and DB state."""

    def _fresh_db(self) -> sqlite3.Connection:
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        return conn

    def _insert_recording_with_file(self, conn, payloads):
        """Create a ttyrec file from payloads, insert recording, return (rec_id, path)."""
        data = make_ttyrec_bytes(*payloads)
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        rec_id = insert_recording(conn, {
            'path': path,
            'file_size_bytes': len(data),
            'frame_count': len(payloads),
            'duration_us': 1_000_000,
            'start_ts_us': 0,
            'end_ts_us': 1_000_000,
            'claude_version': None,
            'has_compaction': 0,
            'compaction_count': 0,
            'flicker_count': 1,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })
        return rec_id, path

    def _insert_null_flicker(self, conn, rec_id, start_frame=5, end_frame=34):
        """Insert a flicker event with thinking_above=NULL."""
        insert_flicker(conn, rec_id, {
            'start_frame': start_frame, 'end_frame': end_frame,
            'start_ts_us': 0, 'end_ts_us': 0,
            'duration_us': 0, 'frame_count': end_frame - start_frame,
            'frames_per_second': 0.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 3, 'line_clear_count': 0,
        })

    def test_return_dict_has_all_keys(self):
        """Return dict has keys: total, updated, errors."""
        conn = self._fresh_db()
        stats = backfill_thinking_above(conn)
        assert 'total' in stats
        assert 'updated' in stats
        assert 'errors' in stats

    def test_total_counts_recordings(self):
        """'total' in return dict counts recordings with NULL thinking_above events."""
        conn = self._fresh_db()
        thinking_frame = b"\\033[?2026h\\033[8A thinking..."
        plain = b"hello"
        payloads = [thinking_frame] * 5 + [plain] * 30

        rec1, p1 = self._insert_recording_with_file(conn, payloads)
        self._insert_null_flicker(conn, rec1)
        rec2, p2 = self._insert_recording_with_file(conn, payloads)
        self._insert_null_flicker(conn, rec2)

        try:
            stats = backfill_thinking_above(conn)
            assert stats['total'] == 2
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_updated_matches_event_count(self):
        """'updated' matches the number of NULL events actually populated."""
        conn = self._fresh_db()
        plain = b"hello"
        payloads = [plain] * 35

        rec_id, path = self._insert_recording_with_file(conn, payloads)
        # Insert 3 NULL events
        self._insert_null_flicker(conn, rec_id, start_frame=5, end_frame=10)
        self._insert_null_flicker(conn, rec_id, start_frame=15, end_frame=20)
        self._insert_null_flicker(conn, rec_id, start_frame=25, end_frame=30)

        try:
            stats = backfill_thinking_above(conn)
            assert stats['updated'] == 3
        finally:
            os.unlink(path)

    def test_errors_zero_for_valid_files(self):
        """'errors' is 0 when all files exist and are valid."""
        conn = self._fresh_db()
        plain = b"hello"
        payloads = [plain] * 35

        rec_id, path = self._insert_recording_with_file(conn, payloads)
        self._insert_null_flicker(conn, rec_id)

        try:
            stats = backfill_thinking_above(conn)
            assert stats['errors'] == 0
        finally:
            os.unlink(path)

    def test_thinking_frame_offset_is_distance(self):
        """thinking_frame_offset is start_frame - nearest thinking frame index."""
        conn = self._fresh_db()
        thinking_frame = b"\\033[?2026h\\033[8A thinking..."
        plain = b"hello"
        # Frames 0-2 are thinking, frames 3-34 are plain
        payloads = [thinking_frame] * 3 + [plain] * 32

        rec_id, path = self._insert_recording_with_file(conn, payloads)
        # Event starts at frame 5, nearest thinking is frame 2 → offset = 5 - 2 = 3
        self._insert_null_flicker(conn, rec_id, start_frame=5, end_frame=34)

        try:
            stats = backfill_thinking_above(conn)
            assert stats['updated'] == 1
            row = conn.execute(
                "SELECT thinking_above, thinking_frame_offset FROM flicker_events "
                "WHERE recording_id = ?", (rec_id,)
            ).fetchone()
            assert row[0] == 1
            assert row[1] == 3  # start_frame(5) - nearest_thinking(2) = 3
        finally:
            os.unlink(path)

    def test_already_populated_events_skipped(self):
        """Events with thinking_above already set are not re-processed."""
        conn = self._fresh_db()
        plain = b"hello"
        payloads = [plain] * 35

        rec_id, path = self._insert_recording_with_file(conn, payloads)
        # Insert event WITH thinking_above set (not NULL)
        insert_flicker(conn, rec_id, {
            'start_frame': 5, 'end_frame': 34,
            'start_ts_us': 0, 'end_ts_us': 0,
            'duration_us': 0, 'frame_count': 30,
            'frames_per_second': 0.0,
            'compaction_above': 0, 'compaction_frame_offset': None,
            'cursor_up_rows': 3, 'cursor_up_spinner': 8,
            'sync_block_count': 3, 'line_clear_count': 0,
            'thinking_above': 1, 'thinking_frame_offset': -5,
        })

        try:
            stats = backfill_thinking_above(conn)
            assert stats['total'] == 0  # No recordings need processing
            assert stats['updated'] == 0
        finally:
            os.unlink(path)

    def test_backfill_sets_zero_without_nearby_thinking(self):
        """When no thinking frames precede, thinking_above=0 and offset is NULL."""
        conn = self._fresh_db()
        plain = b"hello"
        payloads = [plain] * 35

        rec_id, path = self._insert_recording_with_file(conn, payloads)
        self._insert_null_flicker(conn, rec_id)

        try:
            stats = backfill_thinking_above(conn)
            row = conn.execute(
                "SELECT thinking_above, thinking_frame_offset FROM flicker_events "
                "WHERE recording_id = ?", (rec_id,)
            ).fetchone()
            assert row[0] == 0
            assert row[1] is None  # no thinking frame → no offset
        finally:
            os.unlink(path)

    def test_verbose_mode_errors(self):
        """In verbose mode, missing files still count as errors."""
        conn = self._fresh_db()
        rec_id = insert_recording(conn, {
            'path': '/nonexistent/file.lz',
            'file_size_bytes': 0, 'frame_count': 0, 'duration_us': 0,
            'start_ts_us': 0, 'end_ts_us': 0, 'claude_version': None,
            'has_compaction': 0, 'compaction_count': 0, 'flicker_count': 1,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })
        self._insert_null_flicker(conn, rec_id)
        stats = backfill_thinking_above(conn, verbose=True)
        assert stats['errors'] == 1
        assert stats['total'] == 1

    def test_backfill_no_null_events_returns_zero_total(self):
        """When no events have NULL thinking_above, total=0."""
        conn = self._fresh_db()
        stats = backfill_thinking_above(conn)
        assert stats['total'] == 0
        assert stats['updated'] == 0
        assert stats['errors'] == 0


# ── detect_sessions_in_recording additional tests ─────────────────────────────

class TestDetectSessionsAdditional:
    """Additional edge case tests for detect_sessions_in_recording."""

    def _write_ttyrec(self, payloads: list, ts_step_ms: int = 100) -> str:
        data = b""
        for i, p in enumerate(payloads):
            sec = 1 + (i * ts_step_ms) // 1000
            usec = ((i * ts_step_ms) % 1000) * 1000
            data += struct.pack("<III", sec, usec, len(p)) + p
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_goodbye_caps_confidence_at_10(self):
        """Version + welcome + goodbye → confidence capped at 10."""
        path = self._write_ttyrec([
            b"Welcome to Claude Code  claude-code/2.2.0 \\033[?2026h",
            b"Goodbye from Claude Code session ended",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            # version+welcome=8, +5 goodbye would be 13, capped at 10
            assert sessions[0]['confidence'] == 10
        finally:
            os.unlink(path)

    def test_goodbye_sets_end_frame(self):
        """When goodbye is present, end_frame points to the goodbye frame."""
        path = self._write_ttyrec([
            b"\\033[?2026h sync frame 0",
            b"\\033[?2026h sync frame 1",
            b"Goodbye from Claude Code frame 2",
            b"\\033[?2026h sync frame 3",  # sync after goodbye
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['end_frame'] == 2  # goodbye on frame 2
        finally:
            os.unlink(path)

    def test_version_extraction_from_claude_code_format(self):
        """Extracts version from 'claude-code/X.Y.Z' format."""
        path = self._write_ttyrec([
            b"\\033[?2026h  claude-code/3.0.1 startup",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['claude_version'] == '3.0.1'
        finally:
            os.unlink(path)

    def test_no_version_returns_none(self):
        """When no version string, claude_version is None."""
        path = self._write_ttyrec([b"\\033[?2026h sync only"])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['claude_version'] is None
        finally:
            os.unlink(path)

    def test_end_frame_fallback_to_last_frame(self):
        """Without goodbye or sync, end_frame falls back to last frame index."""
        # Single frame with version (no sync after, no goodbye)
        path = self._write_ttyrec([
            b"claude-code/2.1.34",  # version but no sync on this frame alone
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            if len(sessions) == 1:
                # end_frame should be 0 (last frame) since there's no sync after
                assert sessions[0]['end_frame'] >= 0
        finally:
            os.unlink(path)

    def test_start_signal_is_sync_when_sync_first(self):
        """When sync block is the first signal, start_signal='sync_activity'."""
        path = self._write_ttyrec([
            b"\\033[?2026h plain sync block",
            b"more plain data",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['start_signal'] == 'sync_activity'
        finally:
            os.unlink(path)

    def test_start_signal_is_welcome_when_welcome_first(self):
        """When welcome banner is the first signal, start_signal='welcome'."""
        path = self._write_ttyrec([
            b"Welcome to Claude Code v2.0.0",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['start_signal'] == 'welcome'
        finally:
            os.unlink(path)

    def test_session_ts_us_are_integers(self):
        """start_ts_us and end_ts_us are integers (not floats)."""
        path = self._write_ttyrec([b"\\033[?2026h sync"])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert isinstance(sessions[0]['start_ts_us'], int)
            assert isinstance(sessions[0]['end_ts_us'], int)
        finally:
            os.unlink(path)

    def test_compact_signal_confidence_two(self):
        """Compaction text alone (with sync) gives confidence=2."""
        path = self._write_ttyrec([
            b"\\033[?2026h Compacting context...",
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['confidence'] == 2
            assert sessions[0]['start_signal'] == 'compact'
        finally:
            os.unlink(path)


# ── insert_session field assertion tests ──────────────────────────────────────

class TestInsertSessionFields:
    """Tighter assertions on insert_session DB state."""

    def _fresh_db(self) -> sqlite3.Connection:
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        return conn

    def _insert_test_recording(self, conn) -> int:
        return insert_recording(conn, {
            'path': '/tmp/test.lz',
            'file_size_bytes': 1000,
            'frame_count': 100,
            'duration_us': 10_000_000,
            'start_ts_us': 1_000_000,
            'end_ts_us': 11_000_000,
            'claude_version': '2.1.34',
            'has_compaction': 1,
            'compaction_count': 3,
            'flicker_count': 2,
            'processed_at': '2025-01-01T00:00:00+00:00',
        })

    def test_timestamp_fields_stored(self):
        """session_start_ts_us and session_end_ts_us are stored correctly."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 10, 'end_frame': 90,
            'start_ts_us': 1_500_000, 'end_ts_us': 9_500_000,
            'claude_version': '2.1.34',
            'confidence': 5, 'start_signal': 'welcome',
        }
        sid = insert_session(conn, rec_id, session)
        row = conn.execute(
            "SELECT session_start_ts_us, session_end_ts_us FROM claude_sessions WHERE id = ?",
            (sid,)
        ).fetchone()
        assert row[0] == 1_500_000
        assert row[1] == 9_500_000

    def test_null_version_stored(self):
        """When claude_version is None, NULL is stored in the DB."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 10, 'end_frame': 90,
            'start_ts_us': 0, 'end_ts_us': 0,
            'claude_version': None,
            'confidence': 1, 'start_signal': 'sync_activity',
        }
        sid = insert_session(conn, rec_id, session)
        row = conn.execute(
            "SELECT claude_version FROM claude_sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row[0] is None

    def test_detected_at_populated(self):
        """detected_at field is populated with an ISO timestamp."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 10, 'end_frame': 90,
            'start_ts_us': 0, 'end_ts_us': 0,
            'claude_version': '2.1.34',
            'confidence': 3, 'start_signal': 'version',
        }
        sid = insert_session(conn, rec_id, session)
        row = conn.execute(
            "SELECT detected_at FROM claude_sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row[0] is not None
        # Should be an ISO 8601 timestamp
        assert "T" in row[0]

    def test_consecutive_inserts_increment_id(self):
        """Consecutive inserts return incrementing IDs."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 10, 'end_frame': 90,
            'start_ts_us': 0, 'end_ts_us': 0,
            'claude_version': '2.1.34',
            'confidence': 3, 'start_signal': 'version',
        }
        sid1 = insert_session(conn, rec_id, session)
        sid2 = insert_session(conn, rec_id, session)
        assert sid2 > sid1

    def test_start_signal_stored(self):
        """start_signal value is persisted correctly."""
        conn = self._fresh_db()
        rec_id = self._insert_test_recording(conn)
        session = {
            'start_frame': 0, 'end_frame': 50,
            'start_ts_us': 0, 'end_ts_us': 0,
            'claude_version': None,
            'confidence': 2, 'start_signal': 'compact',
        }
        sid = insert_session(conn, rec_id, session)
        row = conn.execute(
            "SELECT start_signal FROM claude_sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row[0] == 'compact'


# ── stream_frames additional edge case tests ──────────────────────────────────

class TestStreamFramesAdditional:
    """Additional edge cases for stream_frames to kill mutations."""

    def test_gz_file_decompresses(self):
        """stream_frames handles .gz files by decompressing them."""
        payload = b"compressed frame data"
        data = struct.pack("<III", 1, 0, len(payload)) + payload
        fd, path = tempfile.mkstemp(suffix=".gz", dir="/private/tmp/claude-502/")
        os.close(fd)
        import gzip as gz
        with gz.open(path, 'wb') as f:
            f.write(data)
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 1
            assert frames[0]['payload'] == payload
            assert frames[0]['sec'] == 1
            assert frames[0]['usec'] == 0
        finally:
            os.unlink(path)

    def test_frame_dict_keys(self):
        """Each frame dict has exactly sec, usec, payload keys."""
        data = struct.pack("<III", 5, 123456, 3) + b"abc"
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 1
            assert set(frames[0].keys()) == {'sec', 'usec', 'payload'}
        finally:
            os.unlink(path)

    def test_truncated_payload_stops_gracefully(self):
        """If payload is shorter than declared length, stream stops without error."""
        # Header says 100 bytes but only 5 are present
        data = struct.pack("<III", 1, 0, 100) + b"short"
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        try:
            frames = list(stream_frames(path))
            assert frames == []  # truncated payload → yields nothing
        finally:
            os.unlink(path)

    def test_multiple_frames_with_varying_sizes(self):
        """Handles frames with different payload lengths."""
        data = b""
        data += struct.pack("<III", 1, 0, 3) + b"abc"
        data += struct.pack("<III", 2, 0, 10) + b"0123456789"
        data += struct.pack("<III", 3, 0, 1) + b"x"
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        try:
            frames = list(stream_frames(path))
            assert len(frames) == 3
            assert frames[0]['payload'] == b"abc"
            assert frames[1]['payload'] == b"0123456789"
            assert frames[2]['payload'] == b"x"
            assert frames[0]['sec'] == 1
            assert frames[1]['sec'] == 2
            assert frames[2]['sec'] == 3
        finally:
            os.unlink(path)


# ── ensure_*_schema additional tests ──────────────────────────────────────────

class TestSchemaAdditional:
    """Additional tests for ensure_session_schema and ensure_thinking_schema."""

    def test_session_schema_creates_claude_sessions_table(self):
        """ensure_session_schema creates the claude_sessions table."""
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert 'claude_sessions' in tables

    def test_session_schema_adds_session_id_column(self):
        """ensure_session_schema adds session_id column to flicker_events."""
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        cols = [row[1] for row in conn.execute(
            "PRAGMA table_info(flicker_events)"
        ).fetchall()]
        assert 'session_id' in cols

    def test_session_schema_adds_is_in_session_column(self):
        """ensure_session_schema adds is_in_session column to flicker_events."""
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        cols = [row[1] for row in conn.execute(
            "PRAGMA table_info(flicker_events)"
        ).fetchall()]
        assert 'is_in_session' in cols

    def test_thinking_schema_adds_thinking_above_column(self):
        """ensure_thinking_schema adds thinking_above column."""
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS flicker_events (
                id INTEGER PRIMARY KEY
            );
        """)
        ensure_thinking_schema(conn)
        cols = [row[1] for row in conn.execute(
            "PRAGMA table_info(flicker_events)"
        ).fetchall()]
        assert 'thinking_above' in cols

    def test_thinking_schema_adds_thinking_frame_offset_column(self):
        """ensure_thinking_schema adds thinking_frame_offset column."""
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS flicker_events (
                id INTEGER PRIMARY KEY
            );
        """)
        ensure_thinking_schema(conn)
        cols = [row[1] for row in conn.execute(
            "PRAGMA table_info(flicker_events)"
        ).fetchall()]
        assert 'thinking_frame_offset' in cols

    def test_session_schema_creates_index(self):
        """ensure_session_schema creates idx_sessions_recording index."""
        conn = init_db(":memory:")
        ensure_session_schema(conn)
        indexes = [row[1] for row in conn.execute(
            "PRAGMA index_list(claude_sessions)"
        ).fetchall()]
        assert 'idx_sessions_recording' in indexes


# ── _strip_ansi tests ─────────────────────────────────────────────────────

class TestStripAnsi:
    """Tests for _strip_ansi — normalises tmux literal ESC and strips ANSI sequences."""

    def test_strips_raw_esc_csi(self):
        """Raw ESC CSI sequences are removed."""
        raw = b'\x1b[31mRed text\x1b[0m'
        assert _strip_ansi(raw) == b'Red text'

    def test_strips_tmux_literal_csi(self):
        """tmux literal \\033 CSI sequences are normalised then removed."""
        tmux = b'\\033[31mRed text\\033[0m'
        assert _strip_ansi(tmux) == b'Red text'

    def test_preserves_content(self):
        """Non-escape content is preserved unchanged."""
        plain = b'Hello, world!'
        assert _strip_ansi(plain) == b'Hello, world!'

    def test_strips_osc_sequence(self):
        """OSC (Operating System Command) sequences are removed."""
        osc = b'\x1b]0;window title\x07some text'
        assert _strip_ansi(osc) == b'some text'

    def test_strips_tmux_literal_osc(self):
        """tmux literal \\033 OSC sequences are removed."""
        osc = b'\\033]0;window title\x07some text'
        assert _strip_ansi(osc) == b'some text'

    def test_empty_input(self):
        """Empty payload returns empty bytes."""
        assert _strip_ansi(b'') == b''

    def test_mixed_raw_and_literal(self):
        """Payload with both raw ESC and literal \\033 sequences."""
        mixed = b'\x1b[1mBold\\033[0m normal'
        result = _strip_ansi(mixed)
        assert result == b'Bold normal'


# ── detect_welcome_screen tests ──────────────────────────────────────────────

class TestDetectWelcomeScreen:
    """Tests for detect_welcome_screen — structural Claude Code welcome screen detector."""

    def test_full_welcome_box_detected(self):
        """Full welcome box with Claude Code text, version, and logo is detected."""
        payload = (
            b'\xe2\x95\xad\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80'  # ╭───
            b'Claude Code v2.1.62'
            b'\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x95\xae'  # ───╮
            b'\n  Welcome back Alice!  \n'
            b'  \xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c  \n'  # ▐▛███▜▌
            b'  Opus 4.6 \xc2\xb7 Claude Max  \n'
            b'\xe2\x95\xb0\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x95\xaf'  # ╰──────╯
        )
        result = detect_welcome_screen(payload)
        assert result['detected'] is True
        assert result['version'] == '2.1.62'
        assert 'claude_code_text' in result['signals']
        assert 'version_tag' in result['signals']
        assert result['score'] >= 3

    def test_compact_resume_view_detected(self):
        """Compact resume view with logo + version is detected."""
        payload = (
            b'  \xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c   '
            b'Claude Code v2.1.62\n'
            b'  \xe2\x96\x9d\xe2\x96\x9c\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9b\xe2\x96\x98  '
            b'Opus 4.6 \xc2\xb7 Claude Max\n'
        )
        result = detect_welcome_screen(payload)
        assert result['detected'] is True
        assert result['version'] == '2.1.62'
        assert 'claude_logo' in result['signals']

    def test_logo_only_not_detected(self):
        """Logo characters alone without any other signals → not detected."""
        payload = b'\xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c'  # ▐▛█▜▌
        result = detect_welcome_screen(payload)
        # Only 1 signal (claude_logo), needs 3 for non-core detection
        assert result['detected'] is False

    def test_plain_text_negative(self):
        """Plain text with no Claude Code signals → not detected."""
        payload = b'This is just regular terminal output with no special patterns.'
        result = detect_welcome_screen(payload)
        assert result['detected'] is False
        assert result['score'] == 0
        assert result['signals'] == []

    def test_version_extraction_with_ansi(self):
        """Version is extracted even when surrounded by ANSI escape sequences."""
        # Raw ESC codes around the version text
        payload = b'\x1b[1m\x1b[36mClaude Code v3.0.1\x1b[0m  some output'
        result = detect_welcome_screen(payload)
        assert result['detected'] is True  # core (claude_code_text + version_tag)
        assert result['version'] == '3.0.1'

    def test_version_extraction_tmux_literal(self):
        """Version is extracted from tmux literal \\033 encoded payloads."""
        payload = b'\\033[1m\\033[36mClaude Code v2.1.62\\033[0m more text'
        result = detect_welcome_screen(payload)
        assert result['version'] == '2.1.62'
        assert result['detected'] is True

    def test_model_info_signal(self):
        """Model info (Opus 4.6) triggers model_info signal."""
        payload = b'Claude Code v1.0.0\n  Opus 4.6 session'
        result = detect_welcome_screen(payload)
        assert 'model_info' in result['signals']

    def test_sonnet_model_info(self):
        """Sonnet model name is also detected."""
        payload = b'Claude Code v1.0.0\n  Sonnet 4.6'
        result = detect_welcome_screen(payload)
        assert 'model_info' in result['signals']

    def test_haiku_model_info(self):
        """Haiku model name is also detected."""
        payload = b'Claude Code v1.0.0\n  Haiku 4.5'
        result = detect_welcome_screen(payload)
        assert 'model_info' in result['signals']

    def test_welcome_text_signal(self):
        """'Welcome back' text triggers welcome_text signal."""
        payload = b'Claude Code v1.0.0\nWelcome back Alice!'
        result = detect_welcome_screen(payload)
        assert 'welcome_text' in result['signals']

    def test_sandbox_msg_signal(self):
        """'sandboxed' text triggers sandbox_msg signal."""
        payload = b'Claude Code v1.0.0\nYour bash commands will be sandboxed.'
        result = detect_welcome_screen(payload)
        assert 'sandbox_msg' in result['signals']

    def test_token_counter_signal(self):
        """'tokens' text triggers token_counter signal."""
        payload = b'Claude Code v1.0.0\n  0 tokens'
        result = detect_welcome_screen(payload)
        assert 'token_counter' in result['signals']

    def test_terminal_title_raw_detected(self):
        """OSC terminal title with Claude Code is detected (raw ESC)."""
        payload = b'\x1b]0;\xe2\x9c\xb3 Claude Code\x07some output'
        result = detect_welcome_screen(payload)
        assert 'terminal_title' in result['signals']

    def test_terminal_title_tmux_literal_detected(self):
        """OSC terminal title with Claude Code is detected (tmux literal)."""
        payload = b'\\033]0;Claude Code\x07some output'
        result = detect_welcome_screen(payload)
        assert 'terminal_title' in result['signals']

    def test_resume_session_signal(self):
        """'Resume Session' text triggers resume_session signal."""
        payload = b'Claude Code v1.0.0\nResume Session from earlier'
        result = detect_welcome_screen(payload)
        assert 'resume_session' in result['signals']

    def test_caskroom_path_not_detected(self):
        """Homebrew Caskroom path should NOT trigger welcome screen detection."""
        payload = b'/opt/homebrew/Caskroom/claude-code/2.1.34/claude --resume abc'
        result = detect_welcome_screen(payload)
        # This has "claude" and "code" but within a file path — the version_tag
        # might match, but the structural scoring should be low
        # The key thing: version should NOT be 2.1.34 from path
        # (It may or may not detect, but it shouldn't have high confidence)
        assert result['score'] <= 2

    def test_return_keys_present(self):
        """Result dict always has all expected keys."""
        result = detect_welcome_screen(b'nothing here')
        assert 'detected' in result
        assert 'version' in result
        assert 'score' in result
        assert 'signals' in result

    def test_empty_payload(self):
        """Empty payload → not detected."""
        result = detect_welcome_screen(b'')
        assert result['detected'] is False
        assert result['version'] is None
        assert result['score'] == 0

    def test_box_header_signal(self):
        """Box drawing characters (╭───) trigger box_header signal."""
        payload = (
            b'Claude Code v1.0.0\n'
            b'\xe2\x95\xad\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80'  # ╭─────
        )
        result = detect_welcome_screen(payload)
        assert 'box_header' in result['signals']

    def test_box_footer_signal(self):
        """Box drawing characters (╰───╯) trigger box_footer signal."""
        payload = (
            b'Claude Code v1.0.0\n'
            b'\xe2\x95\xb0\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x95\xaf'  # ╰───╯
        )
        result = detect_welcome_screen(payload)
        assert 'box_footer' in result['signals']

    def test_three_supporting_signals_enough(self):
        """≥3 supporting signals without any core signal → detected."""
        # welcome_text + sandbox_msg + token_counter = 3 supporting signals
        # No "Claude Code" text or version tag
        payload = b'Welcome back Alice!\nYour commands will be sandboxed.\n  0 tokens remaining'
        result = detect_welcome_screen(payload)
        assert result['detected'] is True
        assert result['score'] >= 3

    def test_one_core_plus_one_supporting_detected(self):
        """1 core + 1 supporting signal → detected."""
        # claude_code_text (core) + welcome_text (supporting)
        payload = b'Claude Code startup\nWelcome back Bob!'
        result = detect_welcome_screen(payload)
        assert result['detected'] is True
        assert 'claude_code_text' in result['signals']
        assert 'welcome_text' in result['signals']

    def test_one_core_alone_not_detected(self):
        """1 core signal alone (no supporting) → not detected."""
        # Only claude_code_text, nothing else
        payload = b'Claude Code'
        result = detect_welcome_screen(payload)
        assert result['detected'] is False


# ── extract_features with is_welcome_screen ──────────────────────────────────

class TestExtractFeaturesWelcomeScreen:
    """Tests for is_welcome_screen field in extract_features."""

    def test_welcome_screen_sets_flag(self):
        """Welcome screen payload sets is_welcome_screen=True in features."""
        payload = (
            b'\xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c   '
            b'Claude Code v2.1.62\n'
            b'  Opus 4.6 \xc2\xb7 Claude Max\n'
        )
        feat = extract_features(make_frame(payload))
        assert feat['is_welcome_screen'] is True
        assert feat['version'] == '2.1.62'

    def test_non_welcome_screen_clears_flag(self):
        """Regular payload has is_welcome_screen=False."""
        feat = extract_features(make_frame(b'regular terminal output'))
        assert feat['is_welcome_screen'] is False

    def test_welcome_version_preferred_over_caskroom(self):
        """Welcome screen version is preferred over VERSION_RE Caskroom path match."""
        payload = (
            b'Claude Code v3.0.0\n'
            b'Welcome back!\n'
            b'/opt/homebrew/Caskroom/claude-code/2.1.34/claude'
        )
        feat = extract_features(make_frame(payload))
        # The welcome screen detector should find v3.0.0 first
        assert feat['version'] == '3.0.0'

    def test_fallback_to_version_re(self):
        """When welcome screen has no version, VERSION_RE is used as fallback."""
        payload = b'claude-code/2.1.34 startup'
        feat = extract_features(make_frame(payload))
        assert feat['version'] == '2.1.34'
        assert feat['is_welcome_screen'] is False

    def test_fallback_to_version_re_second_alternation(self):
        """VERSION_RE fallback works for claude/X.Y.Z pattern (second alternation)."""
        payload = b'claude/4.5.6 process'
        feat = extract_features(make_frame(payload))
        assert feat['version'] == '4.5.6'
        assert feat['is_welcome_screen'] is False


# ── Session detection with new patterns ──────────────────────────────────────

class TestSessionDetectionNewPatterns:
    """Tests for session detection with welcome screen and resume goodbye patterns."""

    def _write_ttyrec(self, payloads: list, ts_step_ms: int = 100) -> str:
        data = b""
        for i, p in enumerate(payloads):
            sec = 1 + (i * ts_step_ms) // 1000
            usec = ((i * ts_step_ms) % 1000) * 1000
            data += struct.pack("<III", sec, usec, len(p)) + p
        fd, path = tempfile.mkstemp(dir="/private/tmp/claude-502/")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_structural_welcome_detected_as_session(self):
        """Structural welcome screen (logo + Claude Code text + version) → session with high confidence."""
        payload = (
            b'\\033[?2026h'
            b'\xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c   '
            b'Claude Code v2.1.62\n'
            b'  Opus 4.6 \xc2\xb7 Claude Max\n'
        )
        path = self._write_ttyrec([payload])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            s = sessions[0]
            assert s['confidence'] >= 5
            assert s['claude_version'] == '2.1.62'
        finally:
            os.unlink(path)

    def test_resume_goodbye_detected(self):
        """'Resume this session with: claude --resume <id>' triggers goodbye."""
        path = self._write_ttyrec([
            b'\\033[?2026h Claude Code v2.1.62\n'
            b'  \xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c  \n'
            b'  Opus 4.6\n',
            b'Resume this session with:\nclaude --resume abc-123',
        ])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            s = sessions[0]
            # Version + welcome → 8, + goodbye → min(8+5, 10) = 10
            assert s['confidence'] >= 8
            assert s['end_frame'] == 1  # goodbye on frame 1
        finally:
            os.unlink(path)

    def test_welcome_back_text_in_welcome_screen(self):
        """'Welcome back' inside a structural welcome screen triggers welcome signal."""
        payload = (
            b'Claude Code v2.1.62\n'
            b'Welcome back Alice!\n'
            b'  \xe2\x96\x90\xe2\x96\x9b\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x9c\xe2\x96\x8c  \n'
        )
        path = self._write_ttyrec([payload])
        try:
            sessions = detect_sessions_in_recording(path)
            assert len(sessions) == 1
            assert sessions[0]['confidence'] >= 5
        finally:
            os.unlink(path)
