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
