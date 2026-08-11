# Log: backbone-only CORE metric, re-derived threshold, full re-score (Part 3c)

Date: 2026-08-09
Plan: `docs/plans/2026-08-09-backbone-core-metric.md`

## What happened

The full 10,542-design fold screen (job 238450) and its first scoring pass (job
238484, all-heavy-atom `motif_rmsd`) ran clean and exposed two errors in the spec and
log's current claims about the active-site gate.

**Error 1 — a wrong prediction from a biased sample.** The spec and log said motif
RMSD is "a gross-failure catch, not a ranking metric" and that "a near-100% pass rate
is expected, not a quality result" (2026-08-08 ruling). That was inferred from 21
probe designs, **all of which came from a single cell**, `TadA8e_FULL_pt1.0` — the
easiest of sixteen (full motif frozen, lowest re-noising, one parent). The full-set
measurement falsifies it: the gate discriminates strongly, including within its own
probe cell (65.6% pass there, not 21/21).

**Error 2 — the measured set was incompatible with half the design arms.** All 5,166
`MIN`-arm designs — 100% of them — landed in `unscorable`. `CORE_MOTIF` measures 17
residues (`CATALYTIC ∪ POCKET`) as all heavy atoms, but `MIN` freezes only the 4
`CATALYTIC` residues; LigandMPNN redesigned the other 13, so the reference's sidechain
atoms do not exist in the `MIN`-arm predictions. `motif_rmsd` raised `KeyError` — by
design, since a silently shrunk measured set would report a falsely good number — so
every `MIN` design failed as `unscorable`, not on any measured geometry.

## What changed

Code changes were made under the plan's earlier tasks (this log's own scope is Task 4,
docs) and are only summarized here for provenance:

1. **Task 1** (`1d03a97`): `score_structure.motif_rmsd` gained an `atom_names=None`
   keyword filter, default unchanged (byte-identical prior behaviour). Still raises on
   any missing *requested* atom — a missing backbone atom is a broken structure, not a
   design choice.
2. **Task 2** (`6f4382b`, `d7b6f63`): `constants.BACKBONE_ATOMS = ("N", "CA", "C",
   "O")` added. `reference_baseline.py` reports the backbone-only CORE RMSD alongside
   the all-heavy-atom value. `MOTIF_RMSD_MAX` re-derived from measurement, not carried
   over: parent offset max(TadA8e 0.7348 Å, TadA9 0.6464 Å) + fold-to-fold jitter
   median 0.2135 Å (10 pairs, `scatter_full/seed{1..5}/TadA8e_rep.cif`) = 0.9483 Å
   floor -> **`MOTIF_RMSD_MAX = 1.0 Å`**, one tick above the floor, same margin
   convention as the retired 2.0 Å derivation. The same 5 replicates re-measure the
   all-heavy-atom jitter at 0.5486 Å against the previously recorded 0.563 Å figure — a
   ~2.5% drift, recorded here rather than papered over; it changes no conclusion.
   `MOTIF_RMSD_MAX = 1.0 Å` numerically coincides with the pre-existing, unrelated
   `BACKBONE_MOTIF_RMSD_MAX = 1.0 Å` (`filter_backbones.py`'s RFdiffusion-backbone
   gate, a different reference distribution) — a coincidence, not a shared meaning.
3. **Task 3** (`53e265d`): `score_folds.score_one` now calls `motif_rmsd(...,
   atom_names=constants.BACKBONE_ATOMS)` instead of the all-heavy-atom default, plus
   `score_folds.slurm` (CPU-only array; scoring never touches the GPU and was measured
   ~80x slower on the login node than on a compute node).
4. **This task (4, docs)**: corrected `docs/specs/2026-08-05-tada-redesign-design.md`
   in place — superseded claims kept visible (struck through, labelled), new
   correction blocks added at "The central finding" (Errors 1 and 2, the fix, the
   re-derived threshold, the full-population results table, the three findings below)
   and at Stage 4 / the compute-budget table (Rosetta's input is now a measurement,
   2,517 designs, not a projection). `score_folds.py`'s module docstring and its
   `[score_folds] N/M passed` print statement no longer predict a pass rate — the
   docstring names the sampling error and points at this log; the print now reports the
   run's own measured rate (`{n_pass}/{total} ({rate:.1%})`) instead of asserting
   "near-100% is expected." No constants and no scoring logic were touched in this
   task; the print-string and docstring edits are the only code change.

## Fold-screen provenance (job 238450)

11 shards, all COMPLETED ~55 min each, 10,542/10,542 folded, zero per-design failures.
Before submission, 21 stale reduced-sampling folds were found already sitting in
`fold_screen/` and moved to `retired_20260809_reduced_sampling_shard001/` — leaving
them in place would have let `--skip-existing` silently adopt them at retired
settings, contaminating 21 designs' fold results with the wrong sampling depth.

## Results — backbone-only re-score (job 238496, 5:23 elapsed)

**Overall: 2,517/10,542 passed (23.9%)**, up from the all-heavy-atom run's
1,375/10,542 (13.0%, now superseded). Status breakdown: ok 2517, low_plddt 6338,
motif_drift 1687, **unscorable 0** (all-heavy-atom run: ok 1375, low_plddt 3621,
motif_drift 380, unscorable 5166 — 100% of the `MIN` arm).

| partial_t | FULL | MIN |
|---|---|---|
| 1.0 | 787/1344 (58.6%) | 1016/1344 (75.6%) |
| 2.0 | 371/1344 (27.6%) | 280/1344 (20.8%) |
| 4.0 | 39/1344 (2.9%) | 22/1344 (1.6%) |
| 6.0 | 2/1344 (0.1%) | 0/1134 (0.0%) |

By arm: FULL 5,376 designs, 1,199 passed (22.3%), median RMSD 1.359 Å, low_plddt 3621,
motif_drift 556. MIN 5,166 designs, 1,318 passed (25.5%), median RMSD 1.444 Å,
low_plddt 2717, motif_drift 1131.

## Findings (results, not asides)

1. **Clean monotonic dose-response in re-noising, in both arms.** `pt6.0` is
   effectively dead (2 of 2,478 designs) — that bounds the protocol's usable
   re-noising range for future generation rounds.
2. **`MIN` and `FULL` are now comparable, and `MIN` is not worse.** 25.5% vs 22.3%
   pass, medians 1.444 vs 1.359 Å. Freezing only the 4 catalytic residues preserves
   CORE backbone geometry nearly as well as freezing all 17 — a comparison that was
   impossible before this task (the `MIN` arm was 100% unscorable). At `pt1.0`, `MIN`
   passes MORE (75.6% vs 58.6%) and, by arm overall, has fewer low-pLDDT rejections
   than `FULL` (2717 vs 3621) — consistent with, but not proof of, LigandMPNN finding
   better-folding sequences when given more design freedom. Stated as an observation
   with a plausible mechanism, not a proven one.
3. **pLDDT is the dominant filter** (6,338 rejections vs 1,687 motif drift): fold
   confidence, not active-site geometry, does most of the selecting in this screen.

## Honesty ceiling

Backbone-only RMSD measures whether the CORE backbone geometry is preserved; it says
nothing about sidechain rotamer quality (deferred to the Rosetta stage), stability,
solubility, or enzymatic activity. pLDDT is model confidence, not a physical
measurement. No wet-lab validation has been performed. **2,517 designs passing this
geometry-and-confidence screen is not evidence any of them are stable or active.**

## Outcome

- Test suite: **170 passed, 1 deselected** (run by the user, not by this task; not
  re-run here per instruction).
- Spec corrected in place with superseded text kept visible; this log created.
- No constants, scoring logic, or generation artifacts were touched by this task.
  `score_folds.py`'s docstring and print-string wording are the only code edited here.
- No SLURM job was submitted and no PyRosetta/RFdiffusion/AlphaFold ran as part of
  this task; all numbers above were measured in Tasks 1-3 and are recorded, not
  re-derived, here.

## Follow-up items

- Stage 4 (Rosetta) compute planning should use the measured 2,517-design input, not
  either the retired ~2,000-survivor budget or the (also now-superseded) ~10,542
  projection.
- `cleft_clearance` is still measured and recorded but not wired into `gate()` —
  unchanged by this task, remains an open scope decision.
- `BACKBONE_MOTIF_RMSD_MAX` (the unrelated `filter_backbones.py` RFdiffusion-backbone
  gate) still needs re-derivation under the corrected anchor before any future
  generation round — unchanged by this task, flagged previously in the 2026-08-08 log.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
