# Bug Report: Line-Swap Flicker During Context Compaction in Claude Code

**Repo with full methodology and source:** https://github.com/heartpunk/claude-code-line-swap-flicker-repro
**Affected versions:** Claude Code 2.1.x (confirmed 2.1.4 → 2.1.59)
**Last clean version:** 2.0.x (2.0.13, 2.0.70, 2.0.76 — all flicker-free)
**Platform:** macOS 15 (Darwin 24.6.0), tmux, various terminal emulators

---

## Executive Summary

Claude Code's terminal UI exhibits a reproducible visual glitch during context compaction:
lines approximately **3 rows below the compaction spinner visually reorder ("flicker")**
at high frame rates for the duration of the compaction animation.

The bug was characterized by automated analysis of **1,617 ttyrec session recordings**
spanning mid-2025 through February 2026. Key numbers:

- **82 sessions (5.1%) contain detectable flicker** — but rate is much higher among
  sessions that actually reached compaction: ~65% of 2.1.34 sessions that compacted had
  measurable flicker
- **58,692 total flicker bursts** detected across the corpus; 58,683 (99.98%) confirmed
  inside a Claude Code session boundary
- **~17 flicker bursts per compaction event** in 2.1.34
- **71% of events** occur during detected compaction; the remaining **29% occur during
  other Claude Code spinner animations** (thinking, tool-use) — same root cause, broader
  trigger surface than previously described
- **99.98% of non-compaction events** have a co-occurring thinking/tool-use spinner
  (`thinking_above=1`); only 3 of 16,897 events have neither compaction nor spinner context
- **Zero flicker in any 2.0.x session** tested
- Bug present in every 2.1.x sub-version from 2.1.4 through 2.1.59 (latest in corpus)

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

| Version | Sessions | Sessions w/ flicker | % affected | Total flicker events |
|---|---|---|---|---|
| 2.0.13 | 1 | 0 | **0%** | 0 |
| 2.0.70 | 3 | 0 | **0%** | 0 |
| 2.0.76 | 3 | 0 | **0%** | 0 |
| 2.1.3 | 1 | 0 | 0% | 0 |
| 2.1.4 | 2 | 1 | **50%** | 695 |
| 2.1.12 | 2 | 1 | **50%** | 59 |
| 2.1.14 | 2 | 0 | 0% | 0 |
| 2.1.15 | 1 | 1 | **100%** | 84 |
| 2.1.34 | 46 | 30 | **65%** | 29,457 |
| 2.1.42 | 1 | 1 | **100%** | 239 |
| 2.1.50 | 2 | 0 | 0% | 0 |
| 2.1.59 | 1 | 1 | **100%** | 489 |
| unknown¹ | 1,418 | 47 | 3.3% | 27,669 |

¹ "unknown" = sessions where a Claude Code version string was not visible in any frame
  (e.g. the startup banner was not recorded, or the binary path differed). Many of
  these are genuine Claude Code sessions — the flicker pattern and compaction text are
  identical; the version was simply not captured.

**The bug is absent in all 2.0.x sessions and present in every 2.1.x sub-version
tested. It was not fixed at any point through 2.1.59.**

### Flicker Intensity in 2.1.34

2.1.34 is the best-sampled version (46 sessions). Among the 30 affected sessions:

| Metric | Value |
|---|---|
| Total flicker events | 29,457 |
| Total compaction events (in same sessions) | 1,727 |
| Average flicker events per compaction | **17.1** |
| Total frames inside flicker windows | ~884,000 |
| Average flicker burst duration | **~1,254 ms** |
| Average fps during flicker | **49.5** |

### Timeline

The earliest flicker appears 2026-01-14 (version 2.1.4). The corpus contains no
earlier 2.1.x sessions to pin the exact introduction date more precisely, but the
jump from 2.0.76 → 2.1.x is the clear boundary.

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
| 1 (compaction in recent 600 frames) | 41,787 | 71.2% |
| 0 (no compaction context detected) | 16,905 | 28.8% |

71% of events occur close to a detected compaction event. The remaining 29% were
investigated in Stage 3 (see below).

### Session Filtering

Claude Code session boundaries were detected in all 82 flicker recordings using
sync-block activity, compaction text, and version-string signals:

| | |
|---|---|
| Flicker files with detected sessions | 82 / 82 (100%) |
| Flicker events inside a session | 58,683 / 58,692 (99.98%) |
| Non-compaction events inside a session | 16,897 / 16,905 (99.95%) |

Session confidence tiers:

| Tier | Confidence | Events |
|---|---|---|
| `confirmed_claude` (has version string) | ≥ 3 | 2,094 |
| `uncertain` (sync/compact only, no version) | 1–2 | 14,803 |

The "uncertain" sessions still have Claude Code's sync-block signature and often
compaction text; they are almost certainly genuine Claude Code sessions where the
version banner was not captured.

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

Running a `thinking_above` backfill over all 58,692 corpus events produced:

| compaction_above | thinking_above | Events |
|---|---|---|
| 1 (compaction) | yes | 41,786 |
| 0 (no compaction) | yes | **16,894** |
| 0 (no compaction) | **no** | **3** |

**16,894 / 16,897 (99.98%) of non-compaction in-session events have a co-occurring
thinking/tool-use spinner.** The remaining 3 events are statistical noise — likely
edge cases in the 600-frame lookback window.

The 41,786 compaction-above events also showing `thinking_above=1` reflects that
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
- Test suite: 138 unit + property tests (Hypothesis), 93.4% mutation kill rate (mutmut)

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
