# 2026-08-06 — tada-redesign final fix wave (post whole-branch review)

Plan: `docs/plans/20260806_tada_redesign_final_fix_wave.md`.

## What changed and why

**Finding 1 (CRITICAL, revised mid-investigation)** — RFD3 partial diffusion
consumes chain D for `select_hotspots` orientation but never emits it
(measured 2026-08-05: input `TadA8e.pdb` has 139 chain-D atoms including the
8AZ; output `.cif.gz` has 1227 rows = 1226 protein + 1 Zn, zero DNA). A first
fix added the DNA to both `unindex` and `select_fixed_atoms` in
`prep_rfd_inputs.py::build_spec`; this passed a CPU-only
`DesignInputSpecification.build()` smoke test (measured: the retained 6
DT/DC tokens landed on chain `G`), but a real RFD3 debug run on that spec
(job 233942, `sbatch --wait --array=1-1 --export=ALL,MODE=debug
rfd_partial.slurm`) FAILED at a later inference-pipeline stage:
```
AssertionError: Cannot unindex non-protein token
rfd3/transforms/conditioning_base.py::expand_unindexed_motifs, line 371
```
Retaining a nucleic-acid chain through RFD3 partial diffusion is not
available at any pipeline stage without changing the model's own
conditioning code -- out of scope. `prep_rfd_inputs.py::build_spec` was
reverted to exactly its prior form (its docstring now states the measured
consume-but-don't-emit behavior).

**Revised fix**: `prep_mpnn_inputs.py::graft_substrate` places the crystal
chain-D context onto each designed backbone by rigid-body superposition
(the relaxed parent's CA onto the designed backbone's CA, same transform
applied to the crystal DNA) before the LigandMPNN inputs are written. This
is a deliberate design decision: partial diffusion barely perturbs the motif
(`motif_rmsd` 0.036-0.057 A at `partial_t=1.0`, measured), so a cleft cannot
collapse during diffusion itself; the missing substrate matters at sequence
design, where an unoccupied groove reads to LigandMPNN as unsatisfied
surface to pack with hydrophobics.

Verified on the real debug artifact (job 233862, no new GPU spend):
- Identity check (designed == reference): grafted chain-D coordinates match
  the crystal's own to within 1.03e-5 A (139/139 atoms matched).
- On the real diffused backbone (`cell_TadA8e_FULL_pt1.0..._model_0`), the
  grafted `mpnn_in/pdb/*.pdb` has chain F (1226 atoms), chain B (1 Zn atom),
  and chain D (139 atoms, resids 23-29 including the 8AZ at 26) -- measured
  directly from both debug backbones.
- The grafted 8AZ-to-catalytic-tetrad minimum distance (3.158 A) matches the
  reference's own (3.144 A) within 0.014 A -- the substrate sits in the
  pocket, not merely somewhere in the file.

**Finding 2 (CRITICAL)** — `filter_backbones`' degraded-run refusal was
gated on the FILTER PASS RATE (`provenance.output_path(out_path, len(paths),
n_ok)`), so `partial_t=6`'s expected-poor yield would rename the canonical
`backbones.tsv` to `.degraded.tsv` and halt the campaign on its own
documented result. Now gated on rows written vs backbones attempted
(`n_written`, tracked around a try/except on `io.append_row`); the pass
count/rate moved to `provenance.write`'s `extra` (`n_passed`, `pass_rate`).

**Finding 3 (IMPORTANT)** — `filter_backbones.main` now enumerates expected
cells from `rfd_in/rfd_inputs.yaml` (falling back to observed cells if that
file is absent) and warns `ZERO backbones present` for any expected cell
with no files at all -- previously a dead RFD3 array task was simply absent
from the per-cell report. `collect_designs.main` now compares rows written
against an expected count derived from the manifest (backbones x
`len(MPNN_TEMPS)*SEQS_PER_TEMP + CONTROL_SEQS_PER_BACKBONE`, grouped by arm),
not `len(fastas)` -- the latter made `is_degraded` mathematically unreachable
since `n_rows` is always ~20x the fasta count by design.

**Finding 4 (IMPORTANT)** — `run_ligandmpnn.slurm`'s control-task branch now
reads `constants.CONTROL_SEQS_PER_BACKBONE` (via an inline `python3 -c`)
instead of `NBATCH` for `--number_of_batches`, so the control set stays
sized 1:1 with the backbone set regardless of what `NBATCH` the biased
submission used.

**Finding 7 (IMPORTANT)** — `filter_backbones.evaluate` now distinguishes
`metal_missing` (no Zn found; all three donor distances nan) and
`metal_ambiguous` (`metal_xyz`'s `ValueError` from >1 match, matched on the
"expected one" substring so an unrelated parse failure like "Empty file."
still falls back like before) from `zn_displaced`, so a measurement failure
no longer reads as the same shape as a real geometric rejection.

**Honesty fixes** — `prep_mpnn_inputs.py`'s and `run_ligandmpnn.slurm`'s
docstrings/comments no longer claim the DNA rode through diffusion; they
describe the graft and cite the measured chain composition above. Softened
the buried-position bias claim in `prep_mpnn_inputs.py`: exposure is derived
ONCE from the parent's RASA, not re-measured per backbone (deferred to
Part 3). `docs/logs/20260805_tada_redesign_part2.md` got a correction
section (original text preserved) stating that the "no `:` in any design
sequence" inference was invalid -- there was no DNA present to mis-parse.

## Tests added (10)

- `test_filter_backbones.py`: `test_evaluate_reports_metal_missing_when_no_zn_is_found`,
  `test_evaluate_reports_metal_ambiguous_on_more_than_one_zn`,
  `test_main_low_pass_rate_with_every_row_written_is_not_degraded`,
  `test_main_a_genuine_row_loss_still_triggers_degraded`,
  `test_main_warns_when_an_expected_cell_produced_no_files_at_all`.
- `test_collect_designs.py`: `test_n_expected_designs_is_manifest_backbones_times_seqs_per_backbone`,
  `test_main_flags_degraded_when_most_of_the_manifest_has_no_fasta`.
- `test_prep_mpnn_inputs.py`: `test_graft_substrate_reproduces_the_crystal_dna_when_designed_is_the_reference`,
  `test_graft_substrate_writes_chain_d_with_exactly_the_context_residues`,
  `test_graft_substrate_places_the_substrate_in_the_pocket`.
- `test_prep_rfd_inputs.py`: one assertion added (`"unindex" not in spec`)
  to the existing spec test, documenting the reverted decision.

## Verification

Fast suite, re-verified live:
```
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"
116 passed, 1 deselected in 33.78s
```
(106 baseline + 10 new.)

`filter_backbones` and `prep_mpnn_inputs` were re-run against the real debug
artifact (job 233862) after the fixes: `filter_backbones` now correctly
warns on all 15 expected-but-absent debug-scope cells (this campaign's debug
gate only ran cell `TadA8e_FULL_pt1.0`) while still passing both real
backbones (`ok=2`); `prep_mpnn_inputs`'s `mpnn_in/pdb/*.pdb` measured chain
composition matches the grafting report above exactly.

No GPU compute beyond the one already-approved debug array task (job
233942, which failed as documented above and was not retried) was consumed
by this fix wave.

## Follow-up items

1. Per-backbone RASA recomputation for the solubility bias (deferred to
   Part 3, per the softened honesty note in `prep_mpnn_inputs.py`).
2. The 16-cell RFD3 array and the 10-task LigandMPNN array remain
   unsubmitted; this wave only re-verified the existing single debug cell.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
