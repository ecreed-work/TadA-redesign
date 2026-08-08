# Plan: apply fixes from the final whole-branch review

Date: 2026-08-08

## Goal

Apply six fixes surfaced by a final review of the `domain-insertion` branch's
gate-fix work (HEAD `3b301c4`). Two are blocking correctness/provenance gaps in
the test suite and in the campaign's headline parent-RMSD numbers; the rest
are doc-accuracy corrections. No measured value, gate constant, or generation
artifact changes.

## Approach

1. **Pin gate constants (`==`).** `tests/test_constants.py` currently only
   floor-bounds `MOTIF_RMSD_MAX` and pins nothing for the three anchor
   constants. Add `==` assertions for `MOTIF_RMSD_MAX == 2.0`,
   `ANCHOR_OUTLIER_CUTOFF == 5.0`, `ANCHOR_MAX_ITER == 10`,
   `ANCHOR_MIN_RETAINED_FRAC == 0.60`, keeping the existing floor check.
2. **Make the parent baseline RMSD reproducible.** Extend
   `reference_baseline.py` to additionally score each parent's CORE motif RMSD
   against `constants.RMSD_REFERENCE` via `score_structure.motif_rmsd` (after
   `align_numbering`), write it into the output table/provenance, and print
   it. Add a flag to score existing CIFs without re-folding (folding is
   forbidden here — GPU work). Run it against the already-existing
   `outputs/20260806_tada_redesign_gen1/baseline/{TadA8e,TadA9}__fold.cif`.
   Must reproduce 1.354/1.357 Å or STOP and report BLOCKED.
3. **Resolve cleft-clearance spec/code contradiction.** Correct the spec
   (`docs/specs/2026-08-05-tada-redesign-design.md:286-296`) to say clearance
   is measured/recorded but does not currently gate (matches
   `score_folds.gate`, which only checks pLDDT + motif RMSD). Add the
   anchor-staleness caveat cross-referenced to the existing
   `BACKBONE_MOTIF_RMSD_MAX` staleness caveat, and note the fold stage now has
   one working geometric gate, not two.
4. **Fix stale survivor arithmetic.** Correct spec `:371` ("Stage 4 —
   survivors") and `:496` ("~2,000 designs x 3 replicates") to reflect the
   21/21 pass rate implication: ~10,542 survivors, a 5x compute miss versus
   the old two-tier-screen-era budget.
5. **Operator-facing pass-rate caveat.** Update `score_folds.py`'s module
   docstring and the `print(f"[score_folds] {n_pass}/{len(designs)} passed...`
   line with a short caveat that a near-100% pass rate reflects the gate's
   gross-failure-catch re-role, not evidence the designs are good.
6. **Minor cleanups.** Spec `:419` `fold_full.slurm` → remove/correct (only
   `fold_screen.slurm`, `rfd_partial.slurm`, `run_ligandmpnn.slurm` exist).
   `score_structure.py:3-4` docstring: drop "full-sampling re-fold" language
   (retired two-tier). `reference_baseline.py`'s vestigial `MODES = {"fold":
   ...}` single-entry dict: simplify to a plain settings reference.

Preserve superseded doc text as visibly struck-through/labelled; never
silently delete prior claims.

## Files affected

- `tada_redesign/tests/test_constants.py` (pin constants)
- `tada_redesign/reference_baseline.py` (CORE RMSD scoring + flag; drop
  vestigial `MODES` dict)
- `tada_redesign/score_folds.py` (docstring + printed caveat)
- `tada_redesign/score_structure.py` (docstring correction)
- `docs/specs/2026-08-05-tada-redesign-design.md` (fixes 2, 3, 4, 6)
- `docs/logs/20260806_tada_redesign_part3a.md` (fix 2 wording correction +
  new entry for this task)
- New log entry in this file's companion log:
  `docs/logs/20260808_tada_redesign_final_review_fixes.md`

## Key decisions

- No SLURM submission, no PyRosetta/RFdiffusion/AlphaFold, no re-fold: score
  existing baseline CIFs only.
- Do not change any of `MOTIF_RMSD_MAX`, `ANCHOR_OUTLIER_CUTOFF`,
  `ANCHOR_MAX_ITER`, `ANCHOR_MIN_RETAINED_FRAC`, `BACKBONE_MOTIF_RMSD_MAX`, or
  any measured value.
- Do not wire cleft clearance into the gate — fix 3 is a doc correction, not a
  scope change.
- STOP condition: if extended `reference_baseline` does not reproduce
  1.354/1.357 Å against the existing baseline CIFs, halt and report BLOCKED
  before touching any doc wording that depends on those numbers.
