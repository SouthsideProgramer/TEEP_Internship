# CLAUDE.md — Cardiorespiratory Separation×Classification Paper

Context file for Claude Code when working in this repo. Read this before touching `report/`, `results/`, or `spike/`.

---

## 0. Current status (single source of truth)

> ⚠️ **The 13 Sep draft (`report/paper.tex`) has been superseded. DO NOT edit that file until further notice.**
> Current source of truth = repo `/home/satya/teep-spike` (commit `29f79e9` → `4554a1e`) + the "C2 re-scoped" decision below.

| | |
|---|---|
| Target venue | IEEE Signal Processing Letters |
| Submission deadline | ~5 Oct 2026 |
| Page limit | 4 pages (13 Sep draft is 12 pages / 18 tables — way over) |
| Before submission | Replicate on CirCor DigiScope (PhysioNet 2022) |
| Authors | Satya Adhiyaksa, Nguyễn Quốc Thắng, Jenq-Shiou Leu |

---

## 1. Evolution of the Contributions (C1–C4)

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │   STAGE 0 — Original proposal (before 19 Aug)                │
                    ├─────────────────────────────────────────────────────────────┤
  C1 (headline)  →  │ "145 real native triplets, not synthetic ones"                │
  C2 (supporting)→  │ SDR → accuracy curve, positioned as a secondary illustration  │
  C3             →  │ Compute cost (stretch)                                       │
                    └─────────────────────────────────────────────────────────────┘
                                          │
                                          │  Sprint 0: audit Mix.csv
                                          │  → only 36/145 rows are genuinely paired (24.8%)
                                          ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │   STAGE 1 — Post Sprint 0 (19 Aug, "C1/C2 swap weight")      │
                    ├─────────────────────────────────────────────────────────────┤
  C2 (HEADLINE)  →  │ SDR → accuracy curve = MAIN CONTRIBUTION                     │
                    │ Does not need real native pairing — uses self-built          │
                    │ mixtures, difficulty becomes a controllable parameter        │
  C1 (supporting)→  │ 6-method separation benchmark (SDR/SIR/SAR + dual CI:        │
                    │ large-n synthetic + small-n native 36 rows)                  │
  C4 (spin-off)  →  │ HLS-CMDS pairing audit → submitted separately to IEEE Data   │
                    │ Descriptions, NOT part of the main paper                     │
  C3 (stretch)   →  │ MACs / params / latency                                      │
                    └─────────────────────────────────────────────────────────────┘
                                          │
                                          │  13 Sep draft submitted for internal review
                                          │  → Satya's 14 Sep review: REJECTED for submission
                                          │  → found Architecture 2 has a fake "collapse"
                                          │    (36 identical predictions per fold)
                                          │  → the "knee" curve lacks resolution
                                          │    (only ~6 rows actually distinguish it)
                                          ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │   STAGE 2 — Re-scoped 14 Sep (CURRENT)                       │
                    ├─────────────────────────────────────────────────────────────┤
  C2 (HEADLINE)  →  │ ❌ Drop the "knee-point curve"                                │
      NEW        →  │ ✅ "Separation loses classification accuracy mainly through   │
                    │    BAND-LIMITING, which BSS Eval SDR does not see. Training  │
                    │    on matched (band-limited) input recovers it."             │
                    │    Evidence: bandpass 20–200Hz on CLEAN audio reproduces     │
                    │    the same collapse (62%→14%); lower-SDR lung interference  │
                    │    barely matters (60%); retraining on band-limited input    │
                    │    recovers 62%                                              │
  C1             →  │ (same supporting role, needs citation fixes — see §4)        │
  C4             →  │ (unchanged, add the S4-06 section by 21 Sep)                 │
  C3             →  │ (unchanged, stretch — needs re-verification after retracting │
                    │  the contended-machine latency numbers)                      │
                    └─────────────────────────────────────────────────────────────┘
```

---

## 2. Quick status table — the 4 C's

| # | Name | Current role | Status | Risk notes |
|---|-----|----|----|----|
| **C1** | Multi-method separation benchmark (6 baselines, SDR/SIR/SAR) | Supporting | 🟡 Citation fixes needed (see §4) | Cannot stand alone if separated out |
| **C2** | Separation → classification loss = band-limiting effect (retraining on matched input recovers it) | **HEADLINE** | 🟠 Spike experiments running, not yet merged into main draft | This is a NEW claim, not fully reviewed yet; the "blind-tuned" SVM's R2 result does not hold ([+0.0, +33.3]) |
| **C3** | Compute cost (MACs/params/latency) | Stretch | 🟢 OK but retracted once (contended machine → off by 10–340×) | Table XVII has been re-measured on an idle machine |
| **C4** | HLS-CMDS pairing audit (36/145 pass) | Separate publication, used as evidence in the main paper | 🟡 Needs the comment finished for IEEE Data Descriptions | Due 21 Sep |

Legend: 🟢 stable · 🟡 needs work but not blocking · 🟠 blocking submission · 🔴 serious error

---

## 3. Numbers to keep straight (to avoid self-contradiction while writing)

```
HLS-CMDS Dataset
├── Mix.csv: 145 rows
│   ├── Genuinely additive (m ≈ a·(h+l)):   36  (24.8%)
│   └── Not matching named sources:        109  (75.2%)
├── Actual sample rate: 4000 Hz  (descriptor states 22,050 Hz — a release/documentation error, not this copy)
└── 29 leak-groups across 145 rows (grouped by shared heart/lung recording)

New spike results (14 Sep, not yet in the main draft):
├── Bandpass 20–200Hz on CLEAN audio → still collapses 62% → 14%  (proves: the failure is band-limiting, NOT separation itself)
├── Retraining on band-limited input   → recovers to 62%
└── Held-out set (57 Mix/ files, no content-hash overlap with the 50 HS files):
    ├── Small CNN:          R1 +29.8 [+19.5,+40.1]   R2 +28.4 [+16.8,+40.0]
    ├── Yaqub-style ResNet: R1 +16.2 [+5.0,+27.4]    R2 +12.2 [+2.8,+21.7]
    └── Blind-tuned SVM:    R1 holds               R2 does NOT hold [+0.0,+33.3]  ⚠️
```

**Don't confuse the two substrates** (a common mistake when writing): synthetic (n=1500, for large-scale SDR comparison) vs. native additive (n=36, used for downstream classification + the spike). SDR values across the two **must never be ranked against each other**.

---

## 4. Citation checklist to fix (from Satya's review)

- [x] `[5]` — wrong author initials (applied 17 Sep, verified against the paper PDF). Correct: **B. Han, W. Quan, B. Matuszewski, D. Corbett** (same group as `[6]`)
- [ ] `Table II` — Han & Quan's dataset is currently listed as "n.s.", should be **HLS-CMDS**
- [ ] `[2]` — is a single-author PhD thesis (Torabi); verify the citation format for a thesis

---

## 5. Action items (from the 14 Sep review) — by deadline

| Due | Task | Related file/command |
|---|---|---|
| 14 Sep (today) | Commit + push the entire working tree (9 untracked scripts, `level_normalize` fix, Makefile targets S1-12) | `git add -A && git commit -m "S1-12: track scripts + level_normalize fix"` |
| 14 Sep | Send ORCID iD to Satya (create one at orcid.org if you don't have it) | — |
| 16 Sep 18:00 | Independent code review of `spike/` — **do not re-run anything, do not edit Satya's copy** | see the review command in §6 |
| 21 Sep | Finish the C4 audit comment (S4-06) + apply the citation fixes from §4 | `report/c4_audit_comment.tex` (create if missing) |
| Ongoing | Resume the Daily Log (last entry: 24 Aug) | `docs/daily_log.md` |

---

## 6. Reusable prompts / commands for Claude Code in this repo

Copy the prompts below for recurring tasks. Drop them into `.claude/commands/` if you want them as real slash commands.

### `/review-spike` — independent review requested by Satya
```
Review the code in spike/HYPOTHESIS.md, spike/CONFIRM.md, spike/EXTEND.md,
and the scripts confirm.py, extend_common.py, resnet_arch3.py, svm_tuned.py.

Do NOT re-run anything. Do NOT edit any file.
Only look for:
1. Wrong condition construction (e.g. off-by-one in fold logic, wrong SDR target)
2. Label or fold mistakes (mismatched class indices, wrong fold assignment)
3. Any leakage between train and test (shared recordings, shared leak-groups,
   fitting on data later evaluated)

For each finding, output: file, line number, one-sentence description of the issue.
If nothing is found in a file, say so explicitly — do not skip silently.
```

### `/check-c2-claim` — verify the new C2 claim is consistent with the data
```
Read the current C2 claim in this repo (band-limiting causes the
classification collapse; retraining on matched band-limited input recovers
accuracy). Cross-check every number cited for this claim against
results/canonical_rows.csv and the spike/ output files. Flag any number in
prose that does not match a number in the underlying CSV, and flag any claim
phrased more strongly than the confidence interval supports (e.g. "recovers
accuracy" when the interval includes zero, as with the blind-tuned SVM R2 result).
```

### `/sanity-numbers` — before committing any table into paper.tex
```
Take table [X] from report/paper.tex. Trace every number in it back to
results/canonical_rows.csv or the named script that produced it. Report any
number that cannot be traced, and any number that differs from the source
file by more than rounding error.
```

### `/audit-citation` — after fixing citations
```
Check every reference in report/paper.tex against the actual cited paper
(PDFs in papers/). Confirm author names, dataset used, and whether the paper
reports a separation-quality metric and/or a downstream classification
result, matching what Table II claims for that row.
```

### `/scope-check` — before adding any new claim
```
Given this new claim I want to add to the paper: "[paste claim]"
Check it against:
1. Does it depend on the 109 excluded (non-additive) rows? If yes, flag it.
2. Does it treat SDR as causally determining accuracy, rather than as an
   aggregate index along one interpolation path? If yes, rewrite the
   hedging language.
3. Is it supported by a bootstrap interval, or only a point estimate?
   If only a point estimate, say so explicitly in the claim.
```

---

## 7. General writing principles — for LATER, not now

> ⚠️ **We are not writing the paper at this stage.** Per §0, `report/paper.tex` is frozen until Satya lifts the freeze. Right now the only work is code (§5): commit the working tree, review `spike/`, finish the C4 comment. The principles below are recorded so that whoever resumes prose writing doesn't repeat the mistakes from the last two review rounds — they are not a green light to start editing `paper.tex` today.

Once editing resumes, apply these:

1. **Never report a number without a confidence interval** (95% bootstrap CI resampled by leak-group, not by row).
2. **Never pool the synthetic and native substrates** into one table to compare SDR directly.
3. **Every crossing/knee point is a "protocol-specific point estimate"**, not a universal threshold — always say so explicitly when mentioning one.
4. **SDR is not the sole cause** of classification accuracy — it is only an aggregate index along one specific interpolation path.
5. When retracting a number (e.g. the contended-machine latency figures), **keep it in the paper with a retraction note**, don't silently delete it.

**Signal that the freeze is lifted:** Satya says so explicitly (e.g. "you can edit paper.tex again") or C2's re-scoped claim (§1, Stage 2) has been confirmed by the 16 Sep independent code review and merged into a result file both of you agree on. Until then, don't draft new prose for the paper — recording findings in the Daily Log (§5) is fine and encouraged.


## — 16 Sep (Sprint 4): review returned and answered

| # | Finding | Disposition |
| --- | --- | --- |
| 1 | Training and test mixtures draw lungs from the same 50-file pool (S2 tests) | Accepted, no re-run. Touches only S2_B1/B4/B5, which `EXTEND.md` already dropped. R1/R2 contain no lung on either side. |
| 2 | B3 is internally bandpassed 50–1800 Hz | Accepted. **Paper was wrong** and is corrected: §II said "B3 does not" band-limit, §IV said "full-band NMF". |
| 3 | Test labels checked only against the 53 HS copies | Accepted, no change. §II already says exactly this. |
| 4 | Exact-PCM leakage guard weaker than the claim (latent) | Accepted. His check stands: max abs correlation between a test and a training file is 0.128. |
| 5 | Bootstrap resamples recordings, not leak groups | Accepted as a limit, not an error. Pre-registered unit stands (`HYPOTHESIS.md:39-40`), and I quantified it. See below. |
| 6 | BAND missing from `confirm_levels.csv` | **Fixed.** BAND recomputed and merged: 20.3 dB SDR / 13.1 dB SI-SDR, exactly Table II. The numbers were right, the artifact was not. |
| 7 | Cache key carries no seed or lung path (latent) | **Fixed** by a parameter stamp beside the cache, not a key change, which would have invalidated 1,625 cached waveforms. |
| 8 | The 57 held-out recordings have carried three analyses | Accepted, now disclosed in §II of the paper, not only in `EXTEND.md`. |
| 9 | Nothing found, nine items checked | Noted. |

### Finding 5 is the one with teeth

The 57 test recordings are distinct files (max abs correlation between any two 0.232, median 0.012, no near-duplicates), but they are takes of only **ten manikin sound types**. Resampling those ten types as clusters instead of the 57 recordings (`spike/review_checks.py`, `out/review_bootstrap.csv`):

| Model | Test | Estimate | Recording bootstrap (pre-registered) | Ten source types, clustered |
| --- | --- | --- | --- | --- |
| CNN | R1 | +29.8 | [+19.4, +40.0] | [+15.2, +44.4] |
| CNN | R2 | +28.4 | [+17.2, +40.0] | [+8.9, +52.7] |
| ResNet-18 | R1 | +16.2 | [+5.0, +27.4] | [−8.3, +37.5] |
| ResNet-18 | R2 | +12.2 | [+2.7, +21.6] | [−9.6, +33.9] |
| SVM tuned | R1 | +19.3 | [+2.8, +35.4] | [−15.6, +49.8] |
| SVM tuned | R2 | +15.8 | [0.0, +33.3] | [−11.9, +42.9] |

Under the stricter unit only the CNN survives on HLS-CMDS. That does not sink the paper: CirCor already resamples 244 patient groups and gives R1 +11.1 [+8.7, +13.2] and R2 +17.6 [+15.2, +19.5]. It does mean the manikin arm no longer carries ResNet-18 on its own, and the paper now says so instead of waiting for a reviewer to say it.

Paper changes are in commit `575cbdd` (still 3 pages of the 4): the B3 correction in §II and §IV, the sensitivity paragraph in §III, the three-analyses disclosure in §II, and ten distinct sounds added to the limitations.