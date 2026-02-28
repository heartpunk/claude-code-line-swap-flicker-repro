# Bug Report: Line-Swap Flicker During Context Compaction in Claude Code

**Repo with full methodology and source:** https://github.com/heartpunk/claude-code-line-swap-flicker-repro
**Affected versions:** Claude Code 2.0.36+, dramatically worse in 2.1.x (confirmed 2.1.4 → 2.1.62)
**Last clean version:** 2.0.70 (no flicker in 11 sessions)
**Platform:** macOS 15 (Darwin 24.6.0), tmux, various terminal emulators

---

## Executive Summary

Claude Code's terminal UI exhibits a reproducible visual glitch during context compaction:
lines approximately **3 rows below the compaction spinner visually reorder ("flicker")**
at high frame rates for the duration of the compaction animation.

The bug was characterized by automated analysis of **1,617 ttyrec session recordings**
spanning mid-2025 through February 2026, with **298 Claude Code sessions** identified
via structural welcome-screen detection. Key numbers:

- **82 recordings (5.1%) contain detectable flicker** — but rate is much higher among
  sessions that actually reached compaction: ~66% of 2.1.34 sessions that compacted had
  measurable flicker
- **58,692 total flicker bursts** detected across the corpus; **54,221 (92.4%) confirmed
  inside a Claude Code session boundary** (tighter boundaries from structural detection)
- **~17 flicker bursts per compaction event** in 2.1.34
- **70% of in-session events** occur during detected compaction; the remaining **30% occur
  during other Claude Code spinner animations** (thinking, tool-use) — same root cause,
  broader trigger surface than previously described
- **99.98% of non-compaction in-session events** have a co-occurring thinking/tool-use
  spinner (`thinking_above=1`); only 3 of 16,478 events have neither compaction nor
  spinner context
- **Low-rate flicker in 2.0.36 and 2.0.76** (2–4% of sessions); **zero in 2.0.13–2.0.70**
- Bug present and dramatically worse in every 2.1.x sub-version from 2.1.4 through 2.1.62
  (55–100% of sessions affected)

---

## What It Looks Like

During compaction, Claude Code displays a spinner (e.g. "Compacting conversation…").
While that animation plays, lines roughly 3 rows below the spinner rapidly jump up and
back, visually swapping with adjacent content, for the entire ~1–2 second duration of
each compaction animation frame. The effect is a stutter or shimmer in the lower portion
of the UI — most visible in a tmux session or any terminal that processes sync blocks.

---

## Root Cause (from Recording Analysis)

Claude Code uses **DEC private mode 2026 (synchronized output)** to batch screen updates,
signalled by `ESC[?2026h` (begin sync) / `ESC[?2026l` (end sync / flush). During
compaction, two independent render paths emit sync blocks in rapid alternation:

**Spinner sync block** — moves cursor up a large number of rows (typically 6–12,
depending on terminal geometry) to repaint the compaction spinner, then returns:

```
ESC[?2026h ... ESC[6A ... <spinner content> ... ESC[?2026l
```

**Input-area sync block** — moves cursor up a small number of rows (typically 3,
sometimes 1–4) to repaint the user input area:

```
ESC[?2026h ... ESC[3A ... <input content> ... ESC[?2026l
```

These two streams **interleave at 37–49 fps** (observed range 12–9,325 fps, latter
being tmux bulk writes). Because each targets a different vertical region, the terminal
alternately renders the input area at "row R" and "row R+3" on successive frames —
producing the visual stutter.

> **Note on ESC encoding in tmux recordings:** tmux's `%extended-output` protocol
> encodes ESC as the literal 4-byte text `\033` (0x5C 0x30 0x33 0x33) rather than
> the actual ESC byte (0x1B). Analysis tooling must match the literal form.

---

## Corpus Analysis

### Dataset

All ttyrec recordings from `~/.ttyrec/` on a single developer machine:

| | |
|---|---|
| Total recordings | 1,617 |
| Compressed (lzip) | 1,608 |
| Uncompressed | 9 |
| Total corpus size | ~2.6 GB |
| Date range | ~mid-2025 through 2026-02-28 |

### Version Attribution

Structural welcome-screen detection (added in this revision) now attributes versions
to 220 of 298 sessions, up from ~80 in the previous analysis which relied on matching
Homebrew Caskroom paths.

| Version | Sessions | Sessions w/ flicker | % affected | Total flicker events |
|---|---|---|---|---|
| 2.0.13 | 1 | 0 | **0%** | 0 |
| 2.0.25 | 9 | 0 | **0%** | 0 |
| 2.0.36 | 1 | 1 | **100%** | 239 |
| 2.0.42 | 9 | 0 | **0%** | 0 |
| 2.0.64 | 20 | 0 | **0%** | 0 |
| 2.0.70 | 11 | 0 | **0%** | 0 |
| 2.0.76 | 48 | 2 | **4%** | 347 |
| 2.1.4 | 10 | 6 | **60%** | 2,499 |
| 2.1.12 | 11 | 6 | **55%** | 2,166 |
| 2.1.14 | 2 | 0 | 0% | 0 |
| 2.1.15 | 8 | 5 | **62%** | 12,145 |
| 2.1.34 | 56 | 37 | **66%** | 27,318 |
| 2.1.50 | 12 | 9 | **75%** | 6,799 |
| 2.1.59 | 2 | 2 | **100%** | 1,543 |
| 2.1.62 | 10 | 9 | **90%** | 1,162 |
| unknown¹ | 78 | 1 | 1.3% | 3 |

¹ "unknown" = sessions where the welcome screen was not captured in any frame.
  Down from 1,418 in the previous analysis thanks to structural welcome-screen detection.

**Note on 2.0.x flicker:** The 2.0.36 session (239 events, all thinking-spinner) and
2.0.76 sessions (347 events, mix of compaction and thinking-spinner) were manually
verified by frame extraction — the welcome screens clearly show `v2.0.36` and `v2.0.76`.
The 2.0.36 session runs on NixOS with a pinned Claude Code version. This contradicts
the earlier "zero flicker in 2.0.x" finding, which was based on broken version detection.

**Revised finding:** Flicker is present but **rare** in late 2.0.x (2–4% of sessions),
**absent** in 2.0.13–2.0.70 (50 sessions, zero flicker), and **dramatically worse**
in all 2.1.x versions (55–100% of sessions). The bug was not introduced in 2.1.x but
was significantly amplified.

### Flicker Intensity in 2.1.34

2.1.34 is the best-sampled version (56 sessions). Among the 37 affected sessions:

| Metric | Value |
|---|---|
| Total flicker events | 27,318 |
| Total compaction events (in same sessions) | ~1,600 |
| Average flicker events per compaction | **~17** |
| Average flicker burst duration | **~1,254 ms** |
| Average fps during flicker | **49.5** |

### Timeline

The earliest flicker in the corpus appears in a 2.0.36 session (thinking-spinner only,
239 events). The rate increases dramatically in 2.1.x, with the earliest 2.1.x flicker
at version 2.1.4. The progression from 2.0.x (rare, 2–4%) to 2.1.x (common, 55–100%)
suggests the interleaved render paths became worse, not that they were newly introduced.

### Cursor-Up Distance Distribution

#### Flicker row (small cursor-up, ≤5)

| Rows | Events | % |
|---|---|---|
| 1 | 13,122 | 22.4% |
| 2 | 12,258 | 20.9% |
| **3** | **26,203** | **44.6%** |
| 4 | 6,363 | 10.8% |
| 5 | 746 | 1.3% |

**3 rows accounts for nearly half of all flicker events**, confirming the subjective
impression that the affected line sits ~3 rows below the spinner.

#### Spinner row (large cursor-up, ≥6) — top values

| Rows | Events |
|---|---|
| 6 | 24,908 |
| 7 | 8,279 |
| 10 | 3,705 |
| 11 | 2,252 |
| 9 | 1,915 |
| **12** | **11,240** |

The bimodal distribution (peaks at 6 and 12) likely reflects different terminal heights
or UI configurations across sessions.

#### Most common (spinner, flicker) pairs

| Spinner rows | Flicker rows | Events |
|---|---|---|
| **6** | **3** | **9,584** |
| 6 | 1 | 4,143 |
| 7 | 3 | 2,786 |
| 7 | 4 | 1,699 |
| 10 | 3 | 1,550 |

The dominant case — spinner at 6 rows up, flicker at 3 rows up — implies the flicker
line sits exactly halfway between the spinner and the cursor baseline.

### Compaction Proximity

| compaction_above | Events | % |
|---|---|---|
| 1 (compaction in recent 600 frames) | 37,743 | 69.6% |
| 0 (no compaction context detected) | 16,478 | 30.4% |

(In-session events only. 70% occur close to a detected compaction event. The remaining
30% were investigated in Stage 3, see below.)

### Session Filtering

Claude Code session boundaries were detected using structural welcome-screen detection
(matching the `╭─── Claude Code vX.Y.Z ───` box header, Claude logo block art, model
info, terminal title, and other signals), plus sync-block activity and compaction text:

| | |
|---|---|
| Total detected Claude Code sessions | 298 (across 291 recordings) |
| Flicker files with detected sessions | 82 / 82 (100%) |
| Flicker events inside a session | 54,221 / 58,692 (92.4%) |
| Flicker events outside any session | 4,471 (7.6%) |

The 4,471 out-of-session events come from recordings where the welcome screen was not
captured (e.g. recording started after Claude Code was already running).

Session confidence distribution:

| Confidence | Sessions | Description |
|---|---|---|
| 10 | 46 | Welcome + version + goodbye |
| 8 | 161 | Welcome + version |
| 5 | 56 | Welcome screen (no version extracted) |
| 3 | 4 | Version string only |
| 2 | 3 | Compact text + sync |
| 1 | 28 | Sync activity only |

**207 of 298 sessions (69%) are at confidence ≥ 8**, a dramatic improvement from the
previous analysis where most sessions were at confidence 1–2.

Non-compaction in-session events by confidence tier:

| Tier | Confidence | Events | Files |
|---|---|---|---|
| `confirmed_claude` (has version/welcome) | ≥ 3 | 16,475 | 59 |
| `uncertain` (sync only) | 1–2 | 3 | 1 |

With structural detection, **99.98% of non-compaction in-session events are now in
confirmed Claude Code sessions** (up from 12% in the previous analysis).

### Stage 3: Non-Compaction Flicker Categorization

A sample of 50 non-compaction events (compaction_above=0) was drawn across confidence
tiers and analyzed by examining a 60-frame window around each event:

| Tier | n | true_flicker | false_positive | unclear |
|---|---|---|---|---|
| confirmed_claude (conf ≥ 3) | 25 | 20 (80%) | 1 (4%) | 4 (16%) |
| mystery_08923efa (conf = 2) | 15 | 10 (67%) | 5 (33%) | 0 |
| uncertain (conf 1–2) | 10 | 6 (60%) | 4 (40%) | 0 |
| **Total** | **50** | **36 (72%)** | **10 (20%)** | **4 (8%)** |

**Zero events were classified as `missed_compaction`** — the non-compaction flicker
is not simply due to compaction occurring outside the 600-frame lookback window.

**Key finding:** The 72% classified as `true_flicker` show the identical interleaved
sync-block pattern (large + small cursor-up values alternating at high fps) but the
spinner text is **not** a compaction spinner. It is Claude Code's thinking/tool-use
spinner (e.g. "Wadling…", "Undulating…", "Actualizing…", "thinking…") and subagent
status displays. The flicker bug fires whenever **any** Claude Code spinner animation
runs concurrently with input-area redraws — compaction is the most common trigger
but not the only one.

**False positives (20%):** Primarily multi-pane tmux recordings where a system
monitor or process viewer occupied one pane (large cursor-up from that pane mixing
with small cursor-up from another). The `08923efa` mystery file is a long recording
that mixes genuine Claude Code sessions with system-monitor content in the same
ttyrec; the Claude Code portions show true flicker.

### Stage 5: Thinking Spinner Confirmation

A structural detector was added to identify "thinking/tool-use spinner frames":
any frame with a sync block **and** large cursor-up (≥6) **but without** compaction
text.  This is analogous to how compaction frames are detected but excludes the
compaction path.

Running a `thinking_above` backfill over all in-session events produced:

| compaction_above | thinking_above | Events |
|---|---|---|
| 1 (compaction) | yes | 37,743 |
| 0 (no compaction) | yes | **16,475** |
| 0 (no compaction) | **no** | **3** |

**16,475 / 16,478 (99.98%) of non-compaction in-session events have a co-occurring
thinking/tool-use spinner.** The remaining 3 events are statistical noise — likely
edge cases in the 600-frame lookback window.

The 37,743 compaction-above events also showing `thinking_above=1` reflects that
thinking spinners often run in the 600-frame window immediately preceding or
surrounding compaction — Claude is thinking, context fills up, then compaction fires.

**Conclusion:** The flicker bug fires whenever **any** Claude Code spinner animation
(compaction or thinking/tool-use) runs concurrently with input-area redraws.  The
trigger surface is not limited to compaction.

---

## Detection Methodology

Full source at https://github.com/heartpunk/claude-code-line-swap-flicker-repro

### Algorithm

A single streaming pass per file with O(1) memory:
- 30-frame sliding window (circular buffer)
- 600-frame bounded deque tracking compaction context

A window is flagged as a flicker event if **all** of:
1. ≥15 frames in window (WINDOW_SIZE // 2)
2. Window spans ≤2.5 seconds wall-clock
3. ≥3 synchronized-output block starts (`ESC[?2026h` in either encoding)
4. ≥5 cursor-up escapes total across all frames
5. ≥2 "small" cursor-up values (≤5 rows)
6. ≥2 "large" cursor-up values (≥6 rows)

Conditions 5 + 6 together are the core signal: the simultaneous presence of both
small and large cursor-up values within one short window is specific to the interleaved
render pattern.

### Validation

- Heuristic tuned against a reference 47-minute session (`df3cbab2-...`) with a typed
  marker string to locate the compaction region precisely
- Spot-checked with `ipbt` frame-by-frame visual rendering
- Test suite: 253 tests (232 unit/integration + 21 Hypothesis property tests), 83.6% mutation kill rate (mutmut 3.x)

---

## Suggested Fix Directions

The interleaving of two independent sync-block streams targeting different screen rows
is the proximate cause. Three approaches to fix it:

1. **Serialize the two render paths** — ensure the spinner redraw and input-area redraw
   are never interleaved; complete one full render cycle before starting the next.

2. **Wrap both redraws in a single sync envelope** — combine the spinner update and
   input-area update under one `ESC[?2026h` … `ESC[?2026l` pair so the terminal
   composites them atomically in one frame.

3. **Throttle the spinner tick rate during compaction** — reduce the frequency at which
   the spinner redraws so it does not interleave with input-area updates faster than
   the terminal can composite them.

---

## Reference Recording

A 47-minute reference ttyrec session containing 82,000+ frames, 184 compaction events,
and 2,466 measured flicker bursts. A nonsense marker string was typed at a known frame
(~12344) to make the compaction region easy to locate with `iris-replay search`.
Recording snippets can be produced if needed.
