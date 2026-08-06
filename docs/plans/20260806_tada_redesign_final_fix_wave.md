# 2026-08-06 — tada-redesign final fix wave (post whole-branch review)

## Goal

Apply the fix wave from a final whole-branch review of `tada-redesign`
(submodule) before any batch RFD3/LigandMPNN array is submitted. Seven
findings, in priority order:

1. **CRITICAL** — the ssDNA substrate never survives RFD3 partial diffusion,
   so it reaches neither generative stage despite the spec requiring it.
2. **CRITICAL** — the degraded-run gate is wired to the FILTER PASS RATE, so
   `partial_t=6`'s expected-poor yield halts the campaign on its own
   documented result.
3. **IMPORTANT** — a missing RFD3 cell or a lost LigandMPNN shard is
   undetectable (`filter_backbones` globs, never enumerates expected cells;
   `collect_designs`'s degraded math is unreachable).
4. **IMPORTANT** — the LigandMPNN control set would be submitted 5x
   oversized (`NBATCH` applies to controls too; `CONTROL_SEQS_PER_BACKBONE`
   is never read).
5. **IMPORTANT** — metal-identification failures (`metal_xyz` finding 0 or
   >1 Zn) are reported as the same geometric verdict (`zn_displaced`) a real
   rejection uses.
6. Three docstring/log claims asserting the DNA context reaches LigandMPNN,
   or that it was "correctly read as ligand context," are false for the
   Finding-1 root cause and must be corrected without deletion.
7. Soften `prep_mpnn_inputs`' claim that bias never touches a buried
   position: exposure is derived once from the parent, not per-backbone.

## Approach

- **Finding 1**: read the installed `rfd3` source
  (`design/foundry/models/rfd3/src/rfd3/inference/input_parsing.py`) to
  determine how partial diffusion could retain a non-protein chain, rather
  than guessing from the from-scratch `denovo_tada` precedent (which is a
  different diffusion mode). Confirmed: `_build_init`'s partial-diffusion
  branch unconditionally subsets the input to protein-only tokens BEFORE any
  contig/unindex is consulted; `unindex` tokens are fetched from the
  UNFILTERED array and glued back on, but only atoms ALSO flagged
  `is_motif_atom_with_fixed_coord` (set by `select_fixed_atoms`) survive
  that gluing. A first attempt added the DNA to both `unindex` and
  `select_fixed_atoms` and PASSED a CPU-only `DesignInputSpecification.build()`
  smoke test, but a real RFD3 debug run on that spec failed at a later
  inference stage (`AssertionError: Cannot unindex non-protein token`,
  `rfd3/transforms/conditioning_base.py::expand_unindexed_motifs`) --
  retaining a nucleic-acid chain through RFD3 partial diffusion is not
  available at all, at any pipeline stage, without changing the model's own
  conditioning code. **Revised fix, by design decision**: leave
  `prep_rfd_inputs.py::build_spec` as it always was (RFD3 still consumes
  chain D for `select_hotspots` orientation only, per the existing spec);
  instead graft the crystal chain-D context onto each DESIGNED backbone by
  superposition, in a new `prep_mpnn_inputs.py::graft_substrate`, since
  partial diffusion barely perturbs the motif (`motif_rmsd` 0.036-0.057 A at
  `partial_t=1.0`, measured) and the missing substrate matters at sequence
  design (LigandMPNN reading an empty groove as unsatisfied surface), not at
  diffusion. Verified on the real debug artifact (job 233862, no new GPU
  spend): grafted output has chain F (1226 atoms), chain B (1 Zn), chain D
  (139 DNA atoms including the 8AZ), and the grafted 8AZ-to-tetrad distance
  (3.158 A) matches the reference's own (3.144 A) within 0.014 A.
- **Finding 2**: change `filter_backbones.main`'s degraded gate to compare
  ROWS WRITTEN against backbones ATTEMPTED (`len(paths)` vs an explicit
  `n_written` tracked around `io.append_row`), moving the pass count/rate
  into `provenance`'s `extra`. Add tests for both a low-pass-rate/no-loss
  case (not degraded) and a genuine row-loss case (still degraded).
- **Finding 3**: `filter_backbones.main` enumerates expected cells from
  `rfd_in/rfd_inputs.yaml` (falling back to observed cells if that file is
  absent, for backward compatibility with existing unit tests) and warns on
  any expected cell with zero files. `collect_designs.main` computes an
  expected design count from the manifest (backbones x
  `constants.MPNN_TEMPS`x`SEQS_PER_TEMP` + `CONTROL_SEQS_PER_BACKBONE`,
  grouped by arm) instead of `len(fastas)`.
- **Finding 4**: `run_ligandmpnn.slurm`'s control-task branch reads
  `constants.CONTROL_SEQS_PER_BACKBONE` (via an inline `python3 -c`) instead
  of `NBATCH` for `--number_of_batches`.
- **Finding 5 (numbered 7 in the review)**: `filter_backbones.evaluate`
  distinguishes `metal_ambiguous` (`metal_xyz` raised `ValueError`) and
  `metal_missing` (all three donor distances nan) from a real `zn_displaced`
  geometric rejection.
- **Honesty fixes**: correct `prep_mpnn_inputs.py`'s docstring and
  `run_ligandmpnn.slurm`'s header comment; append (not rewrite) a correction
  section to `docs/logs/20260805_tada_redesign_part2.md`; soften the
  buried-position bias claim with the per-backbone-RASA scope limitation
  (no per-backbone recomputation implemented -- deferred to Part 3).

## Files affected

- `tada_redesign/prep_rfd_inputs.py` (module docstring only -- `build_spec`
  is unchanged from before this wave; see the revised Finding-1 approach)
- `tada_redesign/prep_mpnn_inputs.py` (new `graft_substrate`, wired into
  `main`, module docstring)
- `tada_redesign/filter_backbones.py` (`evaluate`, `main`, module docstring)
- `tada_redesign/collect_designs.py` (`seqs_per_backbone`,
  `n_expected_designs`, `main`, module docstring)
- `run_ligandmpnn.slurm` (header comment, control-task `--number_of_batches`)
- `tada_redesign/tests/test_prep_mpnn_inputs.py`, `test_filter_backbones.py`,
  `test_collect_designs.py` (new/updated tests)
- `docs/logs/20260805_tada_redesign_part2.md` (correction appended)
- `docs/logs/20260806_tada_redesign_final_fix_wave.md` (this wave's log)

## Key decisions

- Do NOT try to make RFD3 retain the DNA through partial diffusion --
  confirmed blocked at the model's own conditioning-transform level, not
  just an input-spec limitation. `write_input_pdb` keeps writing chain D
  (RFD3 still needs it for `select_hotspots`); `prep_rfd_inputs.py::build_spec`
  is otherwise unchanged.
- Graft the substrate at sequence design instead, by rigid-body superposition
  of the relaxed parent's CA onto the designed backbone's CA, applied to the
  crystal's chain-D atoms. This is a decision that partial diffusion's small
  motif perturbation doesn't need the substrate physically present to stay
  correct, but LigandMPNN's sequence choices do.
- Degraded-run refusal is a ROW-LOSS detector, not a quality gate; pass rate
  moves to provenance `extra`, never removed.
- `rfd_in/rfd_inputs.yaml` absence degrades the expected-cells check
  gracefully to "whatever was observed," rather than making it a hard
  dependency that would break existing synthetic-directory unit tests.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
