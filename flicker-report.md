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
- **58,692 total flicker bursts** detected across the corpus
- **~17 flicker bursts per compaction event** in 2.1.34
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

71% of events occur close to a detected compaction event. The remaining 29% may be:
(a) rapid streaming tool output producing the same interleaved sync pattern without
compaction; (b) sessions where compaction text was present in different frames than
the cursor-up bursts; or (c) a minority of heuristic false positives.

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
- Test suite: 99 unit + property tests (Hypothesis), 93.4% mutation kill rate (mutmut)

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

A 47-minute reference ttyrec session is available on request. It contains 82,000+
frames, 184 compaction events, and 2,466 measured flicker bursts. A marker string
(`jfkld;sajklads;jkl;dsfjdalsk;jklasd;jklasd;`) was typed at a known point to make
the compaction region easy to locate:

```bash
iris-replay search <file> "jfkld"      # → frame ~12344
iris-replay search <file> "Compacting" # → frames ~3492–3498 (first compaction)
```
