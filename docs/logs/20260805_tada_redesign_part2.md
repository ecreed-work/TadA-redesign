# 2026-08-05 — tada-redesign Part 2 (generation stages) + measured debug gate

## What Part 2 built

The generation chain that consumes Part 1's foundation (`io`, `provenance`,
`substrate`, `motif`, `score_structure`, `preflight`):

- `tada_redesign/prep_rfd_inputs.py` — RFdiffusion3 partial-diffusion input
  specs for the 16-cell sweep (2 parents x 2 arms x 4 `partial_t` levels),
  each with `select_fixed_atoms` holding the arm's motif (+ `ZN`) rigid.
- `tada_redesign/filter_backbones.py` — four cheap CPU gates on every RFD3
  backbone before it costs sequence-design or folding time: motif heavy-atom
  RMSD to the relaxed parent (`<= 1.0` A), no CA-CA break `> 4.2` A, length
  in `150-175`, Zn within `2.0-2.6` A of all three donors. Per-cell
  pass/fail counts are printed so a cell whose every backbone failed is
  visible as a zero, not silently absent.
- `tada_redesign/prep_mpnn_inputs.py` — LigandMPNN multi-JSON inputs per arm
  (one process designs every backbone of an arm at one temperature), plus
  the narrow solubility bias restricted to `EXPOSED & MODELED - arm_residues`
  and a zero-bias control set carried alongside it.
- `tada_redesign/collect_designs.py` — parses LigandMPNN FASTA output into
  `designs.tsv`, skipping the non-design input record in each `.fa` and
  raising (not silently accepting) if a design sequence contains `:`
  (multi-chain output would mean the DNA context was parsed as a designed
  protein chain).
- `rfd_partial.slurm`, `run_ligandmpnn.slurm` — the two SLURM array scripts,
  each with a debug mode (`MODE=debug`: 1 batch x 2 backbones, 20 timesteps;
  `NBATCH=2`: 2 sequences/backbone) distinct from the full-scale mode.

## Fast-suite total (verified live)

```
conda run -n ligandmpnn_env python -m pytest -q -m "not slow"
........................................................................ [ 71%]
.............................                                            [100%]
101 passed, 1 deselected in 11.37s
```

## Step 1 — preflight (verbatim, live, exit 0)

```
[PASS] TADA_MONOREPO                           /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
[PASS] masks.json                              FULL=24 residues (expected 24), MIN=(57, 59, 87, 90)
[PASS] reference parents                       all present
[PASS] Zn coordination geometry                both parents pass
[PASS] 6VPC structure                          /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada8e-cas9-Interface-design/structural_analysis/structures/deaminase/pdb6vpc.ent
[PASS] RFD3 checkpoint                         /research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/claude/foundry_ckpt/rfd3_latest.ckpt
[PASS] LigandMPNN checkpoint                   /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
[PASS] AF3 SIF                                 /hpcf/authorized_apps/rhel8_apps/alphafold/3.0.2/alphafold.3.0.2.sif
[PASS] AF3 weights                             /lustre_scratch/reference/public/alphafold_data/af3/models/af3.bin
[PASS] ESMFold2 HF cache                       /home/ecreed/.cache/huggingface/hub/models--biohub--ESMFold2
[PASS] ESMFold2 ligand support                 /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tools/esmfold2/fold.py
[PASS] known-bad PDB not referenced            clean
[PASS] RMSD reference completeness             all FULL-arm residues present in both references
[PASS] conda env 'ligandmpnn_env' has Bio.PDB  ok
[PASS] conda env 'pyrosetta' has pyrosetta     ok
[PASS] conda env 'cas9-pam-design' has rfd3    ok
[PASS] conda env 'ligandmpnn-sc' has prody     ok
[PASS] conda env 'esmfold2' has transformers   ok

all 18 checks pass
exit=0
```

## Step 2 — RFD3 input generation

```
[prep_rfd_inputs] wrote 16 specs -> outputs/20260805_tada_redesign/rfd_in/rfd_inputs.yaml
16 cells
TadA8e_FULL_pt1.0 -> 25 fixed keys, partial_t 1.0
```
Matches the expected 16 cells and 25 fixed keys (24 residues + `ZN`) exactly.

## Step 3 — RFD3 debug run, MEASURED

`sbatch --array=1-1 --export=ALL,MODE=debug rfd_partial.slurm` (job
233862, array task `233862_1`, cell `TadA8e_FULL_pt1.0`, node `nodegpu311`,
H100). **Measured from `sacct` (not a stopwatch):**

```
sacct -n -P -j 233862 --format=JobID,State,Elapsed,ExitCode,Start,End
233862_1|COMPLETED|00:02:36|0:0|2026-08-06T10:28:05|2026-08-06T10:30:41
```

**Elapsed: 2 min 36 s, exit 0.** Produced exactly 2 backbones as expected
(`n_batches=1 x diffusion_batch_size=2`). Breaking down the 156 s from the
job log: ~118 s SLURM dispatch + conda activation + `rfd3`/hydra import
(before the first log line), ~33 s structure prevalidation/CCD lookup, then
`rfd3.engine` logged **"Finished inference batch in 5.10 seconds"** for the
actual diffusion forward pass (batch_size=2, 20 timesteps) — the rest is
output writing/cleanup. So the fixed per-task overhead (~150 s) dominates
completely at debug scale; the diffusion compute itself is a small fraction
of the wall time here.

### RFD3 output filename finding (explicitly verified against real files)

`filter_backbones.main` assumed RFD3 names outputs `<spec_key>_<n>.cif.gz`
and derives a backbone's cell by
`os.path.basename(path).split(".")[0].rsplit("_", 1)[0]`. **This assumption
does NOT hold.** The actual files on disk:

```
cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_0.cif.gz
cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_0.json
cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_1.cif.gz
cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_1.json
```

This matches RFD3's documented convention exactly
(`foundry/models/rfd3/docs/intro_inference_calculations.md`):
`<name of the yaml file>_<settings group name>_<batch_number>_model_n.<suffix>`.
`rfd_partial.slurm` writes the per-cell input as `cell_<CELL>.yaml` with a
single settings group also named `<CELL>`, so both the file-stem and the
group name equal the cell key (`TadA8e_FULL_pt1.0`), and the key itself
contains an embedded `.` (`pt1.0`) — RFD3 was never going to produce a
`<spec_key>_<n>.cif.gz` file. Confirmed downstream: see Step 4 below, where
this is not a cosmetic mismatch but a hard crash.

## Step 4 — filter_backbones: BLOCKED (real crash, not a "zero passed" case)

```
conda run -n ligandmpnn_env python -m tada_redesign.filter_backbones
Traceback (most recent call last):
  ...
  File ".../filter_backbones.py", line 158, in main
    constants.RMSD_REFERENCE[parent])
KeyError: 'cell'
```

Root cause: `os.path.basename(path).split(".")[0]` splits on the FIRST `.`
in the filename, which lands inside `pt1.0` (not at the `.cif.gz`
extension), giving `"cell_TadA8e_FULL_pt1"`; `.rsplit("_", 1)[0]` then
strips `"pt1"`, giving `cell = "cell_TadA8e_FULL"`. `cell.split("_")` then
yields `parent = "cell"`, `arm = "TadA8e"` — both wrong (`RMSD_REFERENCE` is
keyed by `TadA8e`/`TadA9`, `arm_residues` expects `FULL`/`MIN`). The crash
happens on the very first path processed, before any row is written:
**`backbones.tsv` was never created; zero rows exist.** No
`filter_backbones.provenance.json` was written either (that call is after
the loop).

**No `motif_rmsd` values were obtained** — the crash occurs before
`evaluate()` is ever called on either of the two debug backbones, so there
is no noise-ladder evidence to report from this run.

**Steps 5 and 6 were NOT executed.** `prep_mpnn_inputs` requires
`backbones.tsv`; with zero rows (in fact, no file at all), there is nothing
for it to consume. Per this task's own ground rules, the fix belongs to
whoever plans the next Part-2 remediation, not to this debug-gate task
(scope: run and measure, not modify), and guessing at a patch here would
defeat the purpose of running the debug gate on 2 files instead of 512. No
`prep_mpnn_inputs`, `run_ligandmpnn.slurm`, or `collect_designs` run was
attempted; none of their elapsed times were measured; `designs.tsv` does
not exist.

## Projected cost of the full arrays (derived from the ONE measurement taken)

**16-cell RFD3 array.** Full mode differs from debug on three axes:
`n_batches` 1->4, `diffusion_batch_size` 2->8, and `inference_sampler.num_timesteps`
20 (explicit in debug) -> 200 (`rfdiffusion3.yaml` default, unset in
`MODE=run`). The measured per-task fixed overhead (dispatch + env + import +
prevalidation, ~150 s) is paid once per array task regardless of batch
count. The measured diffusion compute itself was 5.10 s for one batch of 2
at 20 timesteps. Two bounding extrapolations for one full-scale cell
(4 batches, 8/batch, 200 timesteps), stated as a range because batch-size
scaling on one H100 is not linear and was not measured at that size:
- **Pessimistic (linear in both timesteps and batch size):**
  `5.10 s x 10 (timesteps) x 4 (batch size) = 204 s/batch x 4 batches = 816 s`
  compute + ~150 s overhead = **~966 s (~16.1 min) per cell**.
- **Optimistic (batch size ~free on GPU, only timesteps scale):**
  `5.10 s x 10 = 51 s/batch x 4 batches = 204 s` compute + ~150 s overhead =
  **~354 s (~5.9 min) per cell**.
- For 16 cells run as a SLURM array (each task its own GPU node), total
  wall time for the array is roughly the per-cell time above plus queue
  wait (not the sum, since tasks run concurrently subject to node
  availability); total GPU-node-time consumed is 16x per-cell, i.e.
  **~1.6-4.3 node-hours**. The script's current `-t 08:00:00` request has
  30-80x headroom over either bound.
- This projection is invalidated if the filename/derivation bug above is
  not fixed first, because `filter_backbones` would crash identically on
  every one of the 512 full-scale backbones, not just the 2 debug ones.

**10-task LigandMPNN array.** Not projected. Steps 5-6 were never reached,
so zero LigandMPNN wall-clock data exists to derive a projection from;
fabricating one would violate this task's own honesty requirement.

## Honesty ceiling

These metrics measure structural plausibility and energetic ranking, not
biological function. No wet-lab validation has been performed.

## No batch array was submitted

Only `--array=1-1` (RFD3) was submitted, and it ran once. The 16-cell RFD3
array and the 10-task LigandMPNN array were not submitted, consistent with
CLAUDE.md's pre-job debug gate — and in this case the gate did exactly its
job: it caught a real, hard-crashing bug in `filter_backbones`'s filename
assumption before it could be discovered on 512 backbones instead of 2.

## Follow-up items (for whoever plans the Part 2 remediation / Part 3)

1. **Blocking bug, confirmed live:** `filter_backbones.main`'s cell
   derivation from the RFD3 output filename is wrong for the filenames RFD3
   actually produces (`cell_<CELL>_<CELL>_<batch>_model_<n>.cif.gz`, where
   `<CELL>` itself contains a `.`). Needs a real fix (e.g. parse using the
   known `<CELL>` values from `rfd_inputs.yaml` rather than positional
   string splitting), then this task's Steps 4-6 need to be re-run at debug
   scale to get real `motif_rmsd` values and LigandMPNN timing before any
   batch array is sized or submitted.
2. LigandMPNN debug timing (Step 5) is still unmeasured; needed before the
   10-task array's `-t` can be sized with evidence rather than the current
   placeholder `08:00:00`.
3. The RFD3 full-scale batch-size-vs-wall-time scaling (linear vs sublinear)
   was not measured at `diffusion_batch_size=8`; the "Pessimistic"/"Optimistic"
   bounds above should be narrowed with a real intermediate-scale timing
   once the filter bug is fixed, before committing to the full 16-cell `-t`.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

## Fix wave — cell-from-filename bug, corrected, and a second real finding (2026-08-06)

The coordinator's diagnosis went further than the crash reported above and
found a SECOND, worse defect in the same line: even with a correct cell,
`os.path.basename(cif_path).split(".")[0]` was also used for the per-backbone
id in `evaluate()`, and it collides `..._model_0` and `..._model_1` of the
same cell onto one id (both truncate at the embedded `.` in `pt1.0`) — that
one would have silently overwritten rows instead of crashing, which is worse
than the `KeyError` this task hit.

**Fix applied** (no filename parsing at all):
- `rfd_partial.slurm`: `OUT` is now `"$RUN/rfd/$CELL"`, computed after `$CELL`
  is known, with its own `mkdir -p`; each cell's two RFD3 outputs land in
  their own directory. `MODE=run`'s and `MODE=debug`'s `rfd3 design` calls
  and sampling budgets are unchanged.
- `tada_redesign/filter_backbones.py`: added `backbone_id(path)`, which strips
  a known suffix (`.cif.gz`/`.cif`/`.pdb`/`.ent`) rather than splitting on the
  first `.`. `evaluate()` uses it for the backbone id. `main()` globs
  `<rfd_subdir>/*/*.cif.gz` and reads the **cell from the parent directory
  name** (`os.path.basename(os.path.dirname(path))`), never from the
  filename.
- Two new tests in `test_filter_backbones.py`:
  `test_backbone_id_survives_a_float_in_the_cell_name` (proves the two models
  of one cell now get distinct ids) and
  `test_main_derives_the_cell_from_the_directory_not_the_filename` (an
  end-to-end `main()` run against a synthetic per-cell directory, asserting
  the correct `cell`/`parent`/`arm`/`partial_t`/`status`).

**Fast suite, re-verified live:**
```
conda run -n ligandmpnn_env python -m pytest -q -m "not slow"
103 passed, 1 deselected in 12.22s
```
(101 + 2 new, as expected.)

**No GPU re-run.** The four existing debug artifacts (2 `.cif.gz` + 2 `.json`
from job 233862) were moved byte-for-byte into the new layout:
```
mkdir -p outputs/20260805_tada_redesign/rfd/TadA8e_FULL_pt1.0
mv outputs/20260805_tada_redesign/rfd/cell_TadA8e_FULL_pt1.0_* \
   outputs/20260805_tada_redesign/rfd/TadA8e_FULL_pt1.0/
```

### Step 4, re-run: filter_backbones now runs correctly, but 0/2 pass

```
[filter_backbones] TadA8e_FULL_pt1.0: zn_displaced=2
[filter_backbones] WARNING TadA8e_FULL_pt1.0: ZERO backbones passed
[filter_backbones] 0/2 passed -> outputs/20260805_tada_redesign/backbones.tsv
[filter_backbones] DEGRADED: >20% of backbones failed; wrote
  outputs/20260805_tada_redesign/backbones.degraded.tsv instead of the
  canonical path
```

The cell/id fix is confirmed working: both rows show `cell=TadA8e_FULL_pt1.0`,
`parent=TadA8e`, `arm=FULL`, `partial_t=1.0`, and DISTINCT backbone ids
(`..._model_0`, `..._model_1`) — no collision.

**`motif_rmsd` values, reported prominently as instructed: 0.036 A and
0.057 A for the two debug backbones.** Both are well under the 1.0 A
`BACKBONE_MOTIF_RMSD_MAX` gate at `partial_t=1.0` — this is exactly the
noise-ladder behaviour expected at the lowest partial_t level and a good
sign for `select_fixed_atoms`. `n_res=156` (inside `150-175`) and
`max_ca_break=3.924` A (under `4.2` A) also pass for both.

**Both backbones failed on `zn_displaced`** — `zn_57ND1`, `zn_87SG`,
`zn_90SG` are all `nan` for both. Traced directly (not guessed): loading
`cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_0.cif.gz` with
`filter_backbones.load_backbone` finds the three donor sidechain atoms
present with real coordinates on chain F, but zero `ZN` atoms on chain F.
Grepping the decompressed CIF directly shows why:
```
HETATM ZN ZN . ZN B 0 161 . 161 ZN B ZN 1226 ...
```
The Zn is present in the file — on `auth_asym_id B`, not `F`. This matches
the RFD3 job log exactly: `"Chain F contains both polymer and non-polymer
residues; separating them for processing, naming the non-polymer residues
as B."` `prep_rfd_inputs.py`'s docstring already documents that "AtomWorks
renames a hetero atom's chain when it shares a chain letter with protein
residues" as the reason `select_fixed_atoms` keys the Zn by CCD name rather
than chain+resid for the RFD3 **input** — but that note covers only the
input-spec side. `score_structure.heavy_atoms_from_cif` /
`heavy_atoms_from_pdb`, which `filter_backbones.load_backbone` calls, filter
strictly to `chain=constants.SCAFFOLD_CHAIN` ("F") and therefore never see
the Zn in RFD3's **output**, on every backbone, unconditionally — this is
not specific to this cell or this run.

Whether the relabelled letter is deterministically "B" was not established;
the job log shows chain D's non-polymer residues got "A" and chain F's got
"B" in the same run, consistent with sequential assignment of the next free
letter rather than a fixed contract. Hardcoding "B" would be guessing at a
fix for an internal AtomWorks convention, not something either the RFD3 docs
or `prep_rfd_inputs.py`'s existing note commit to. Per this task's own rule
("the filter finds zero backbones ... report the counts as-is"), no
workaround was attempted.

**Steps 5 and 6 remain NOT executed.** `backbones.tsv` (canonical path) does
not exist — only `backbones.degraded.tsv`, which `prep_mpnn_inputs` does not
read (it reads the canonical path only). 0 backbones are available to design
sequences on. No `run_ligandmpnn.slurm` job was submitted; no LigandMPNN
elapsed time was measured; `collect_designs` was not run; 0 designs
collected.

### Updated projections

The 16-cell RFD3 timing projection in the section above is unaffected by
this fix wave (no RFD3 job was re-run). The 10-task LigandMPNN projection
remains unmeasured. Both remain blocked on a real fix — now specifically
the Zn output-chain gap, not the filename bug (which is resolved and
covered by tests).

### Follow-up items, updated

1. ~~`filter_backbones.main`'s filename-to-cell derivation~~ **RESOLVED**:
   cell now comes from the per-cell RFD3 output directory, never from
   parsing a filename; covered by two new tests.
2. **New, blocking:** `score_structure.heavy_atoms_from_cif` /
   `heavy_atoms_from_pdb` need a way to find the catalytic Zn in RFD3 output
   when it has been relabelled onto a different `auth_asym_id` than
   `constants.SCAFFOLD_CHAIN`. A real fix should look up the Zn by CCD name
   (`constants.ZN_RESNAME`) across ALL chains in the output structure, not
   assume a fixed chain letter, and should be re-verified against BOTH
   parents (TadA8e and TadA9) since the relabelled letter is not established
   to be constant. Confirmed via a live job log and the raw CIF, not
   inferred.
3. `motif_rmsd` evidence at `partial_t=1.0` is good (0.036, 0.057 A) and
   should be preserved as a regression baseline once backbones can pass the
   Zn gate.
4. LigandMPNN debug timing (Step 5) is still unmeasured; still needed before
   the 10-task array's `-t` can be sized with evidence.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

## CORRECTION (2026-08-06, from the final whole-branch review) — Step 5's DNA-context claim was invalid

The "Step 5" section below states: **"No `:` in any of the 4 design
sequences — verified directly ... The DNA/Zn context was correctly read as
ligand context, not as a second designed protein chain."** That conclusion
is INVALID and is preserved above, uncorrected, as the record of what was
actually written at the time — this section supersedes it rather than
silently rewriting it.

What was actually measured at the time: the RFD3 output backbones used in
that debug run (job 233862) contained the Zn (1227 atoms: 1226 protein + 1
Zn) and **zero DNA atoms** — confirmed by independently inspecting the
`.cif.gz` output. Partial diffusion's `_build_init` unconditionally subsets
its re-noised input to protein-only tokens before any contig/unindex is even
consulted (`rfd3/inference/input_parsing.py`), and nothing in the spec at
that time retained the DNA against that filter. So the "no `:`" check
observed zero DNA atoms because there was no DNA present to mis-parse in the
first place — it is equally consistent with "correctly read as ligand
context" and with "there was nothing there to read." The absence of `:`
proved nothing about LigandMPNN's ligand-context handling; it only proved
the DNA context requirement had already silently failed one stage earlier.

Consequence for every downstream measurement in this log: every backbone,
design, and confidence value recorded above (Steps 3-6, both fix waves) was
generated with the catalytic Zn present but the ssDNA substrate context
ABSENT — the spec's requirement that both be fixed context in the generative
stages was unmet. The MIN arm in particular had nothing but Zn holding its
cleft open. This is a data-validity note about what those runs actually
modeled, not a retraction of the measured numbers themselves (motif_rmsd,
Zn-donor distances, timings) — those are real measurements of what was
diffused, on a substrate-free active site.

Fixed separately (see `docs/plans/20260806_tada_redesign_final_fix_wave.md`
and `docs/logs/20260806_tada_redesign_final_fix_wave.md`) -- and the fix
changed direction mid-investigation, which is itself worth recording: an
`unindex` + `select_fixed_atoms` retention was implemented and PASSED a
CPU-only `DesignInputSpecification.build()` smoke test, but a real RFD3
debug run on that spec (job 233942) FAILED at a later inference-pipeline
stage (`AssertionError: Cannot unindex non-protein token`,
`rfd3/transforms/conditioning_base.py::expand_unindexed_motifs`) -- RFD3
genuinely cannot retain a nucleic-acid chain through partial diffusion, at
any stage, not just the one the smoke test happened to reach. The design
decision this settled on: leave `prep_rfd_inputs.py::build_spec` as it always
was (RFD3 consumes chain D for `select_hotspots` orientation only), and
instead graft the crystal chain-D context onto each designed backbone by
superposition in `prep_mpnn_inputs.py::graft_substrate`, since partial
diffusion barely perturbs the motif in the first place
(`motif_rmsd` 0.036-0.057 A at `partial_t=1.0`, measured above) and the
missing substrate matters at sequence design, not at diffusion. Verified on
the real debug backbone from job 233862: the grafted PDB has chain F (1226
atoms), chain B (1 Zn), and chain D (139 DNA atoms including the 8AZ), and
the grafted 8AZ-to-catalytic-tetrad distance (3.158 A) matches the
reference's own (3.144 A) within 0.014 A.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

## Fix wave 2 — Zn located by identity, not by chain; the gate finally runs end to end (2026-08-06)

Root cause, confirmed by the coordinator independently: the raw output CIF
has exactly one Zn row (`HETATM ZN ZN . ZN B 0 161`) on `auth_asym_id B`,
while the protein and its donor sidechains sit on chain F. This was already
half-documented in `prep_rfd_inputs.py` (AtomWorks renames a hetero atom's
chain when it shares a letter with protein on INPUT), but the consequence on
the OUTPUT side — every chain-scoped reader silently missing the Zn — had
not been carried through.

**Fix applied:**
- `score_structure.py`: new `metal_xyz(path, resname="ZN")` locates the metal
  by CCD identity across ALL chains (PDB and mmCIF, gz-aware), returns `None`
  if absent, raises `ValueError` if more than one match (so an unexpected
  multi-metal input fails loudly rather than silently picking one).
- `filter_backbones.py`: `zn_donor_distances(atoms, zn_xyz=None)` now accepts
  an explicit metal position, falling back to the old atoms-dict lookup (so
  synthetic test fixtures are unaffected). `evaluate()` calls
  `score_structure.metal_xyz(cif_path)` and passes the result through,
  catching `(OSError, ValueError)` so the existing tests' nonexistent
  `"x.cif.gz"` paths still exercise the fallback correctly.
- Four new tests: `test_metal_xyz_finds_the_zn_on_a_foreign_chain`,
  `test_metal_xyz_returns_none_when_absent` (both in
  `test_score_structure.py`), `test_zn_donor_distances_honours_an_explicit_metal_position`
  (`test_filter_backbones.py`).

**Fast suite, re-verified live:**
```
conda run -n ligandmpnn_env python -m pytest -q -m "not slow"
106 passed, 1 deselected in 13.41s
```
(103 + 3 new — the coordinator's estimate of "~106" landed exactly on the
real count.)

**No GPU re-run.** `filter_backbones` was re-run on the SAME two moved
`.cif.gz` artifacts from job 233862.

### Step 4, re-run again: 2/2 PASS

```
[filter_backbones] TadA8e_FULL_pt1.0: ok=2
[filter_backbones] 2/2 passed -> outputs/20260805_tada_redesign/backbones.tsv
```

| backbone | motif_rmsd (A) | zn_57ND1 (A) | zn_87SG (A) | zn_90SG (A) | max_ca_break (A) | n_res | status |
|---|---|---|---|---|---|---|---|
| `..._model_0` | 0.036 | 2.194 | 2.387 | 2.418 | 3.924 | 156 | ok |
| `..._model_1` | 0.057 | 2.194 | 2.387 | 2.418 | 3.924 | 156 | ok |

All three Zn-donor distances land inside `ZN_DONOR_RANGE` (2.0-2.6 A) for
both backbones. `motif_rmsd` values are unchanged from the earlier
(pre-Zn-fix) run — as expected, since that gate never touched the Zn.

**Both backbones report IDENTICAL Zn-donor distances to the atom.** This is
not a bug: `select_fixed_atoms` locks the motif + Zn's INTERNAL geometry
while letting that rigid group re-orient in space during partial diffusion
(`prep_rfd_inputs.py`'s documented semantics), so the two independent
diffusion samples of the same cell are expected to preserve the donor-to-Zn
distances exactly while differing elsewhere. Read as confirmation that
`select_fixed_atoms` is doing what the spec says, not as a suspicious
coincidence.

`provenance.json`: `is_degraded: false`, `n_in: 2`, `n_out: 2`.

### Step 5: prep_mpnn_inputs + LigandMPNN debug run, MEASURED

```
[prep_mpnn_inputs] FULL: 2 backbones, 67 biased positions, fixed='F28 F30 F46 F54 F57 F58 F59 F84 F85 F86 ...'
[prep_mpnn_inputs] MIN: 0 backbones, 81 biased positions, fixed='F57 F59 F87 F90...'
```
(MIN is 0 because only the FULL cell was run at debug scale — expected, not
a bug.)

`sbatch --array=1-1 --export=ALL,NBATCH=2 run_ligandmpnn.slurm` -> job
**233870**, array task `233870_1` (ARM=FULL, T=0.1, biased=1 — array index 1
maps here per the script's `IDX = SLURM_ARRAY_TASK_ID - 1` scheme). Queue:
straight to RUNNING, no pending wait.

```
sacct -n -P -j 233870 --format=JobID,State,Elapsed,ExitCode
233870_1|COMPLETED|00:00:17|0:0
233870_1.batch|COMPLETED|00:00:17|0:0
```
**Elapsed: 00:00:17, exit 0.** Produced 2 `.fa` files (one per backbone),
each with the input record + 2 design records, exactly as expected.

**No `:` in any of the 4 design sequences** — verified directly
(`grep -c ":" *.fa` -> 0 for both files). The DNA/Zn context was correctly
read as ligand context, not as a second designed protein chain.

Sample header (backbone `model_0`, design 1):
```
>cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_0, id=1, T=0.1, seed=111,
 overall_confidence=0.4770, ligand_confidence=0.2986, seq_rec=0.5000
ANTNEDWMAQALELARKAKDENEVPVGAVLVKDNKVLGRGYNTRKRDNDPTAHAENLALRQGAEVEKNPKLTDATLYVTFEPCVYCAQAAIDAQIGKVVIGAPNSKRGAAGSIKNVLQDPSNSHKVAIEKGVLIDEACELLEEFYNQPRQRFNCRK
```

### Step 6: collect_designs

```
[collect_designs] 4 designs from 2 fastas -> outputs/20260805_tada_redesign/designs.tsv
```
**4 designs** (2 backbones x 2 sequences), all `seq_len = 156`, inside
`LENGTH_RANGE` (150-175). `provenance.json`: `is_degraded: false`,
`n_in: 2`, `n_out: 4`, `skipped_fastas: []`.

### Updated projections

RFD3 16-cell projection: unchanged (no RFD3 job re-run this fix wave).
LigandMPNN 10-task projection, now with one real data point: the debug run
(1 arm, T=0.1, biased, 2 backbones, `NBATCH=2`) took **17 s total**
(dominated by fixed overhead — env activation, checkpoint load — the same
pattern as RFD3). A full run_ligandmpnn.slurm task designs the SAME set of
backbones per cell but presumably with a larger `NBATCH` (the full-scale
`NBATCH` default is 5 per `run_ligandmpnn.slurm`, vs. 2 here) and, at full
scale, up to 32 backbones/cell x however many cells pass the filter (up to
16), all funneled through ONE LigandMPNN process per (arm, temperature) via
the `_multi` JSON flags — so wall time scales with total backbone count fed
to that one process, not with the number of SLURM tasks (only 10 tasks
exist regardless of backbone count). A precise per-backbone rate was not
isolated from this single 17 s data point (it includes one-time model load),
so the 10-task array is not yet confidently sized from this alone; a
second timing at a larger backbone count would narrow it.

## Honesty ceiling (restated, unchanged)

These metrics measure structural plausibility and energetic ranking, not
biological function. `overall_confidence`/`ligand_confidence`/`seq_rec` are
LigandMPNN's own sequence scores, not stability or activity measurements.
No wet-lab validation has been performed.

## No batch array was submitted (restated)

Only `--array=1-1` was submitted for RFD3 (job 233862) and for LigandMPNN
(job 233870), each once. The 16-cell RFD3 array and the 10-task LigandMPNN
array were not submitted.

### Follow-up items, updated again

1. ~~`filter_backbones.main`'s filename-to-cell derivation~~ **RESOLVED**
   (fix wave 1).
2. ~~Zn relabelled to a foreign chain in RFD3 output~~ **RESOLVED**:
   `score_structure.metal_xyz` locates it by CCD identity; both debug
   backbones now pass all four gates.
3. LigandMPNN per-backbone wall-time rate is still not cleanly isolated
   from fixed overhead; recommend timing a second debug point with more
   backbones or a higher `NBATCH` before committing to the 10-task array's
   `-t`.
4. The RFD3 full-scale batch-size-vs-wall-time scaling (Fix wave 1's
   "Pessimistic"/"Optimistic" bounds) is still unmeasured at
   `diffusion_batch_size=8`.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

## Final review gate — the two never-exercised cells, MEASURED (2026-08-06)

Submodule HEAD unchanged throughout (`b973d9e`, suite 116 passed) — this was
verification only, no fixes. The final review required exercising the two
paths the first gate never touched: **index 5 = `TadA8e_MIN_pt1.0`**
(tetrad-only arm, never run) and **index 4 = `TadA8e_FULL_pt6.0`** (top of the
noise ladder, never run). Confirmed against the sorted `rfd_inputs.yaml` key
list that these indices map to exactly those cells.

**Preflight**: all 18 checks pass, exit 0.

**`sbatch --wait --array=4,5 --export=ALL,MODE=debug rfd_partial.slurm`** —
both completed cleanly, no PENDING delay (GPUs idle):

| JobID | Elapsed | State | ExitCode |
|---|---|---|---|
| 233963_4 (`FULL_pt6.0`) | 00:00:49 | COMPLETED | 0:0 |
| 233963_5 (`MIN_pt1.0`) | 00:00:48 | COMPLETED | 0:0 |

Each produced 2 backbones (debug scale) in its own per-cell output directory.

**`filter_backbones` (all cells present)**: 6/6 backbones passed (2 from the
pre-existing `FULL_pt1.0` gate-1 run + these 4 new ones). Full results:

| cell | motif_rmsd (A) | status |
|---|---|---|
| `TadA8e_FULL_pt1.0` model_0/1 | 0.036 / 0.057 | ok / ok |
| `TadA8e_FULL_pt6.0` model_0/1 | 0.292 / 0.274 | ok / ok |
| `TadA8e_MIN_pt1.0` model_0/1 | 0.040 / 0.036 | ok / ok |

**(b) MIN arm is NOT a zero-pass cell**: `TadA8e_MIN_pt1.0` passed 2/2 — the
tetrad-only arm (4 residues frozen instead of 24) survived `partial_t=1.0`
cleanly at debug scale.

**(c) `partial_t=6.0` real yield**: `TadA8e_FULL_pt6.0` passed 2/2, motif_rmsd
0.292/0.274 A — well under the 1.0 A gate, and NOT the cliff a related
campaign measured at `partial_t=8` (1.1 -> 2.5-6.75 A). At this small (n=2)
debug sample the aggressive end of the noise ladder did not degrade the
motif; this does not by itself establish the full-scale (n=32) yield.

**(d) canonical path held**: only `backbones.tsv` exists, no
`backbones.degraded.tsv`, even though this run mixed a good cell with the
untested aggressive one. Provenance: `n_passed=6`, `pass_rate=1.0`,
`is_degraded=false`.

**(a) substrate reaches the MIN design**: `prep_mpnn_inputs` ran the
substrate graft on all 6 backbones. A `TadA8e_MIN_pt1.0` grafted PDB shows
chain F = 1226 atoms (protein), chain B = 1 atom (Zn, `HETATM ZN B 161`),
chain D = 139 atoms (22 `8AZ` + 57 `DC` + 60 `DT`) — identical on both MIN
backbones and on a `FULL_pt6.0` backbone, matching the counts documented in
`run_ligandmpnn.slurm`'s header comment.

**LigandMPNN, array 5 (MIN biased, `NBATCH=2`) and array 10 (MIN control,
`NBATCH=2`)**:

| JobID | Elapsed | State | ExitCode |
|---|---|---|---|
| 233967_5 (MIN, T=0.1, biased) | 00:00:13 | COMPLETED | 0:0 |
| 233969_10 (MIN, control) | 00:00:08 | COMPLETED | 0:0 |

**(e) control sizing holds on real data**: despite both submissions using
`NBATCH=2`, the control fastas carry exactly **1** design record per backbone
(2 total across 2 backbones) while the biased task carries **2** per backbone
(4 total) — `constants.CONTROL_SEQS_PER_BACKBONE` is read for the control
regardless of `NBATCH`, confirmed by counting `id=` records directly in each
`.fa`.

**`collect_designs`**: 10 designs collected from 6 fastas (4 pre-existing
`FULL_pt1.0` + 4 new `MIN_pt1.0` biased + 2 new `MIN_pt1.0` control); 126
expected from the full manifest (only 2 of 10 LigandMPNN array slots were
run, so `is_degraded=true` here is an artifact of the deliberately partial
submission, not a defect). **Zero occurrences of `:`** in any of the 10
collected sequences — checked directly against every `.fa` sequence line, not
just via the absence of a parse error. This check is now meaningful because
the ssDNA genuinely rides in every backbone PDB this time (see (a)), unlike
the earlier CORRECTION above where the same check on the same-looking output
proved nothing because the DNA was absent.

**Total designs collected: 10.** Full report with every command's output,
`sacct` times, the complete `backbones.tsv`, per-chain compositions, FASTA
record counts, and `designs.tsv`:
`.superpowers/sdd/2026-08-05-tada-redesign-part2/two-cell-gate-report.md`.

No defects found. No fixes applied. Only `--array=4,5` (RFD3) and
`--array=5`, `--array=10` (LigandMPNN) were submitted — no full/batch array.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
