# Plan — backbone-only CORE metric (Part 3c)

**Date:** 2026-08-09
**Branch:** main (campaign repo)
**Base:** `f41f193`

## Why this plan exists

The full 10,542-design screen (job 238450) and its scoring (job 238484) ran clean and
produced a result that falsifies two claims in the current spec and log.

**1. All 5,166 MIN-arm designs are unscorable — 100% of them.** `CORE_MOTIF` measures
17 residues (CATALYTIC ∪ POCKET) as **all heavy atoms**, but the MIN arm freezes only the
4 CATALYTIC residues. The other 13 were designable, LigandMPNN changed their identities,
and so the reference's sidechain atoms do not exist in the prediction. `motif_rmsd` raises
`KeyError` by design — a silently shrunk measured set would report a falsely good number —
so every MIN design lands in `unscorable` and fails. This is a defect in how the measured
set was paired with the design arms, not a property of the designs.

**2. The gate DOES discriminate.** The prior conclusion ("motif RMSD is a gross-failure
catch; a near-100% pass rate is expected") was inferred from 21 probe designs that all came
from a single cell — `TadA8e_FULL_pt1.0`, the easiest of sixteen. Measured over the full
set:

| FULL arm | n | passed |
|---|---|---|
| pt1.0 | 1344 | 881 (65.6%) |
| pt2.0 | 1344 | 418 (31.1%) |
| pt4.0 | 1344 | 66 (4.9%) |
| pt6.0 | 1344 | 10 (0.7%) |

Overall 1,375/10,542 = 13.0%. Among scorable designs the dominant rejection is
`low_plddt` (3,621), not `motif_drift` (380). Even the probe's own cell passes at 65.6%,
so the 21/21 draw was unrepresentative within its cell as well as across cells.

## Decisions (repo owner, 2026-08-09)

- **Backbone-only RMSD at CORE for every design.** Measure N/CA/C/O at the 17 CORE
  residues. Identity-independent, so it works for both arms and makes FULL and MIN
  directly comparable. Sidechain rotamer quality becomes a Rosetta-stage question.
- **Keep the high-noise cells** and report the dose-response as a finding about the
  protocol's usable re-noising range.

## Global Constraints

- **The threshold must be re-derived.** Backbone-only RMSD is systematically smaller than
  all-heavy-atom RMSD, so carrying `MOTIF_RMSD_MAX = 2.0` across the metric change would
  repeat exactly the defect that produced Part 3b: a threshold calibrated against a metric
  that no longer means the same thing.
- Preserve superseded text visibly; never silently rewrite a refuted claim.
- Honesty ceiling: motif RMSD measures geometry, pLDDT measures model confidence. Neither
  is stability, solubility, or enzymatic activity. No wet-lab validation.
- Tests: `conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"`
  (166 passed, 1 deselected at base).
- Both trailers on every commit; push to `origin` only.

## Task 1 — backbone-only measurement

**Files:** `tada_redesign/score_structure.py`, `tada_redesign/constants.py`, tests.

Add an `atom_names` filter to `motif_rmsd` (default unchanged, so existing callers and the
`filter_backbones` path keep their current behaviour), plus a
`constants.BACKBONE_ATOMS = ("N", "CA", "C", "O")`.

Requirements:
- It must still RAISE when a requested atom is missing. Backbone atoms are present in every
  complete model, so a missing one is a broken structure, not a design choice — the loud
  failure is the point and must not be softened to a skip.
- The superposition anchor is unchanged (iterative refinement, `ANCHOR_OUTLIER_CUTOFF`).
- Tests: a design whose CORE sidechains differ from the reference must now score rather than
  raise; a structure missing a backbone atom must still raise; assertions literal-based.

## Task 2 — re-derive the threshold under the new metric

**Files:** `tada_redesign/constants.py`, `tada_redesign/reference_baseline.py`, tests, docs.

Measure, do not guess. All inputs already exist on disk — no folding, no SLURM:
- **Parent offset:** backbone CORE RMSD of `baseline/TadA8e__fold.cif` and
  `TadA9__fold.cif` vs `constants.RMSD_REFERENCE` (job 238437).
- **Fold-to-fold jitter:** the 5 parent replicates at `scatter_full/seed{1..5}/TadA8e_rep.cif`
  — 10 pairs, the same source as the retired 0.563 Å heavy-atom figure. Report the
  backbone-only median.
- Set `MOTIF_RMSD_MAX` = parent offset + jitter, one tick above the floor, and state the
  arithmetic in the constant's comment exactly as the current one does.
- Extend `reference_baseline.py` to report the backbone-only value too, so the number stays
  reproducible from committed code — the provenance rule established in the last round.

Report both the old and new values side by side. Do NOT tune toward a target pass rate.

## Task 3 — re-score and report

Re-run `score_folds` over all 10,542 (batch, `score_folds.slurm`; it took 4:57). Report the
pass distribution **by arm and by partial_t**, and confirm MIN designs are now scorable.
State plainly whether the MIN arm's structural quality differs from FULL — that comparison
was impossible before and is the main thing this change buys.

## Task 4 — docs

Correct the spec and log. Both currently assert a near-100% expected pass rate and that the
gate does not discriminate; both are falsified. Record the 13.0% result, the per-cell
dose-response, the MIN unscorable defect and its cause, and the sampling error that produced
the wrong prediction (21 probes drawn from one cell). Keep the superseded claims visible.
