# Log: apply fixes from the final whole-branch review

Date: 2026-08-08
Plan: `docs/plans/20260808_tada_redesign_final_review_fixes.md`

## What changed

1. **Pinned the gate constants with `==`.** `tada_redesign/tests/test_constants.py`:
   `test_motif_threshold_exceeds_the_parents_own_offset_and_jitter` now additionally
   asserts `MOTIF_RMSD_MAX == 2.0` (floor check kept, not replaced). New
   `test_anchor_constants_are_pinned_to_their_measured_values` pins
   `ANCHOR_OUTLIER_CUTOFF == 5.0`, `ANCHOR_MAX_ITER == 10`,
   `ANCHOR_MIN_RETAINED_FRAC == 0.60`. Before this, none of the three anchor
   constants were pinned by any test, and `MOTIF_RMSD_MAX` was only floor-bounded
   (`> floor`), so raising it to e.g. 50.0 — the flattering direction — would have
   passed the suite.

2. **Made the parent baseline RMSD reproducible from committed code.**
   `tada_redesign/reference_baseline.py` previously folded both parents and recorded
   only pLDDT; it never called `score_structure.motif_rmsd`. The 1.3542/1.3568 Å
   figures quoted throughout the spec/log came from a separate, unversioned probe
   script (`.superpowers/sdd/2026-08-06-tada-redesign-part3a-gatefix/task-3b-report.md`),
   not from `reference_baseline.py` as the docs claimed. Added:
   - `core_motif_residues()` — `motif.arm_residues(motif.CORE_MOTIF, ...)`, identical
     to what `score_folds.py` measures.
   - `score_baseline_rmsd(run_dir, parents=None, residues=None)` — for each parent,
     reads `baseline/<parent>__fold.cif` if present, scores CORE motif RMSD against
     `constants.RMSD_REFERENCE` via `heavy_atoms_from_cif` -> `align_numbering` ->
     `motif_rmsd`, the SAME sequence `score_folds.score_one` uses for a design. Never
     folds.
   - `--score-only` CLI flag on `main()` — skips the `fold_many.py` subprocess entirely
     (no GPU, no SLURM), scores whatever CIFs already exist, writes
     `baseline/baseline_summary.tsv` (parent, plddt, core_motif_rmsd) and adds
     `core_motif_rmsd` to the provenance `extra` dict, and prints both values.
   - `MODES = {"fold": ...}` (vestigial single-entry dict, fix 6) removed; `main()`
     now reads `constants.ESMFOLD_SETTINGS` directly.
   - 5 new tests in `test_reference_baseline.py`: empty-result on a missing CIF,
     `align_numbering` wiring (rigid transform + renumber-from-1 round-trips to ~0 Å),
     a known 0.6 Å local perturbation measured exactly (anchor larger than the
     measured set, so the check is not vacuous), and a `main(["--score-only", ...])`
     integration test asserting `subprocess.run` is never called with `fold_many.py`
     and that the summary TSV/provenance are written correctly.
   - **Ran it against the existing baseline folds**
     (`outputs/20260806_tada_redesign_gen1/baseline/{TadA8e,TadA9}__fold.cif`, job
     238437) via `python -m tada_redesign.reference_baseline --score-only --run-dir
     outputs/20260806_tada_redesign_gen1`. **Reproduced exactly: TadA8e 1.3542 Å,
     TadA9 1.3568 Å** — matching the quoted figures. Not blocked.

3. **Resolved the cleft-clearance spec/code contradiction.**
   `docs/specs/2026-08-05-tada-redesign-design.md` previously said clearance is
   "gated relative to each parent's own measured clearance"; `score_folds.gate()`
   only checks pLDDT and motif RMSD and never reads `cleft_clearance`. Corrected the
   spec (struck-through/labelled, not silently rewritten) to say clearance is
   measured and recorded in `fold_screen.tsv` but not currently gated, and added the
   second-order caveat: the recorded parent clearances (2.211/2.271 Å,
   `constants.py` commit `91d3041`) predate the anchor-refinement fix
   (`ad88b35`/`7cb69e4`) and were never re-recorded against it, the same documentation
   gap already flagged for `BACKBONE_MOTIF_RMSD_MAX`. Noted for the record:
   `test_score_structure.py::test_both_relaxed_parents_clear_their_own_cleft_gate`
   independently reproduces 2.211/2.271 Å under the CURRENT fixed anchor, so the
   numbers are not shown to be wrong — only their provenance comment predates the fix.
   No gate was wired in; that stays a scope decision.

4. **Fixed the stale survivor arithmetic.** Spec `Stage 4` header and the compute
   budget table's Rosetta row both said "~2,000 designs" / "survivors", inherited
   from the retired two-tier screen. Corrected to state the honest projection is
   ~10,542 designs (a ~5x compute miss), following directly from the 21/21 pass rate.

5. **Added the operator-facing pass-rate caveat.** `score_folds.py`'s module
   docstring and the `[score_folds] N/10542 passed` print line now state that
   `MOTIF_RMSD_MAX` is a gross-failure catch, not a ranking metric, and a near-100%
   pass rate is the expected consequence of that re-role, not a quality result.

6. **Minor cleanups.** Spec: `fold_full.slurm` (never existed — leftover from the
   retired two-tier design) struck through and corrected in the module inventory.
   `score_structure.py`'s docstring: dropped "the full-sampling re-fold" language,
   now names the single fold stage. `reference_baseline.py`'s vestigial
   `MODES = {"fold": ...}` removed (done as part of item 2's edit).

7. **Corrected "production code path" wording** in
   `docs/specs/2026-08-05-tada-redesign-design.md` and
   `docs/logs/20260806_tada_redesign_part3a.md`: both previously claimed the parent
   RMSD numbers were measured "through the actual production code path
   (`reference_baseline.py`)". Added correction blocks stating accurately that the
   fold came from the production path but the RMSD scoring at the time did not (an
   unversioned probe script), and noting it is now reproducible via this module
   (item 2).

## Why

A final whole-branch review found that the test suite let the two most important
gate constants drift arbitrarily in the flattering (lenient) direction with the
suite fully green, and that the campaign's headline parent-RMSD numbers — quoted
throughout the spec and log as coming from "the actual production code path" — in
fact came from an unversioned side-script, the exact provenance failure this branch
already faulted the retired 1.468 Å figure for. Both are now fixed: the constants are
pinned, and the numbers are reproducible from committed code.

## Outcome

- Test suite: **166 passed, 1 deselected** (up from 160/1; 6 new tests, 0 removed,
  0 modified in a way that weakens an assertion).
- `reference_baseline.py --score-only` against the existing job-238437 baseline CIFs
  reproduces **TadA8e 1.3542 Å, TadA9 1.3568 Å** exactly. Not BLOCKED.
- No measured value, gate constant, generation artifact, or SLURM submission was
  touched. No PyRosetta/RFdiffusion/AlphaFold ran.
- Flag for the record (fix 3): with motif RMSD now re-roled as a gross-failure catch
  and cleft clearance not wired into `gate()`, the fold stage currently has **one**
  working geometric gate, not two.

## Follow-up items

- `CLEFT_CLEARANCE_MARGIN` is defined and measured but unused as a gate; wiring it
  in (or formally deciding not to) is an open scope decision for a future task.
- The recorded parent cleft clearances (2.211/2.271 Å) should be explicitly
  re-verified and re-recorded against the current anchor in `constants.py`'s own
  comment before they are relied on again, mirroring `BACKBONE_MOTIF_RMSD_MAX`'s
  existing re-derivation note.
- Stage 4 (Rosetta) compute planning should use the corrected ~10,542-design
  projection, not the retired ~2,000-survivor budget.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
