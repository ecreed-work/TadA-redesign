# 2026-08-06 — tada-redesign Part 3a (folding & gating): preflight, debug shard, BLOCKED

Plan: `docs/plans/2026-08-06-tada-redesign-part3a.md`, Task 6 (final task).

## What Part 3a built (Tasks 1-5, already committed)

- `enrich_designs.py` added `identity_to_parent`/`mutation_count` to `designs.tsv`
  and pointed `RUN_DIR_NAME` at the production run (`20260806_tada_redesign_gen1`,
  10,542 designs).
- `tools/esmfold2/fold_many.py` (monorepo) folds many designs in one process so
  the model load is amortised instead of paid per design.
- `fold_screen.py` + `fold_screen.slurm`: a contiguous, deterministic,
  SLURM-array-sharded screen over `designs.tsv`.
- `reference_baseline.py`: both parents folded in both `screen`/`full` modes, so
  the pLDDT gate has a same-mode baseline.
- `score_folds.py`: applies Part 1's `score_structure` geometric gates
  (motif RMSD, cleft clearance) plus the relative pLDDT gate to folded models.
- Task 6 (this entry): two preflight checks (`fold_many available`,
  `designs.tsv enriched`), a `constants.FOLD_SHARDS` comment fix, and the debug
  shard that is this task's actual deliverable.

## Measured costs (Task 2 / Task 4, already committed)

- Model load: **20.8 s warm / 104.5 s cold**; per-fold: **0.90 s** (job 234196,
  `fold_probe/timing.json`, 10 real 156-mers).
- Cost projection at `FOLD_BATCH_SIZE=250`: amortised `(20.8 + 250*0.90)/250 =
  0.983 s/design`; `10,542 * 0.983 / 3600 = 2.88 GPU-hours` total, spread over
  `FOLD_SHARDS=44` shards. **`constants.FOLD_SHARDS`'s comment previously said
  `ceil(10542/250)`, which is 43, not 44** — a comment-arithmetic error, not a
  functional one (44 partitions 10,542 into 26 shards of 240 + 18 of 239, no
  empty shard, and safely exceeds the `ceil(n_designs/FOLD_BATCH_SIZE)` floor).
  The comment now says so explicitly; the value 44 is unchanged.
- Parent baselines (job 234208): TadA8e full **0.8968** / screen **0.9005**;
  TadA9 full **0.8882** / screen **0.8900**. This **refutes** the plan's working
  assumption that reduced sampling substantially depresses pLDDT (that belief
  came from one uncontrolled 78-mer fold on 2026-08-04 that returned 0.45, which
  reflected that peptide's difficulty, not the sampling settings): here, sampling
  depth moves pLDDT by under 0.004 for both parents, and `screen` was in fact
  marginally *higher* than `full` for both. The relative baseline is kept for
  correctness/symmetry, not because a large penalty needs correcting.

## Task 6: preflight and suite

```
cd tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"
147 passed, 1 deselected in 74.04s
```
(146 baseline + 1 new: `test_new_part3_checks_are_registered`.)

```
cd tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.preflight; echo exit=$?
```
All **20/20** checks pass, including the two new ones (`fold_many available`,
`designs.tsv enriched`) and all five `conda env` probes.

## Task 6: the debug shard — BLOCKED

```
sbatch --wait --array=1-1 --export=ALL,SHARDS=527 fold_screen.slurm
```
Job **234216** (shard 1/527, 21 designs — the first cell's `FULL`-arm
`partial_t=1.0` designs). **COMPLETED, elapsed 00:42:02**, 21/21 folded, 0
failed. The model load for this run took **2397.1 s** (~40 min) rather than the
20.8-104.5 s measured in Task 2 — traced live to severe contention on the
allocated node (`nodegpu312`): load average ~35 on 112 cores, two other users'
processes each pinning ~49 GB / ~99% on two of the node's eight H100s, and four
of this user's own unrelated `gmx mdrun` jobs on the same node. `strace`-level
inspection (`/proc/<pid>/status`, `nvidia-smi`, `rchar`) showed continuous, if
extremely slow, forward progress throughout (RSS climbing, no crash, CUDA
context eventually initialized) rather than a hang — this is infrastructure
contention, not a code defect, and is **not** counted against the Task 2 cost
model above. Per-fold time once running was **1.24 s/design** (`fold_screen/
timing.json`), close to the earlier 0.90 s baseline.

**Scoring, however, surfaced a real, reproducible blocking defect.**

```
conda run -n ligandmpnn_env python -m tada_redesign.score_folds
```
Every one of the 21 folded designs came back `status=unscorable`, all raised
from `score_structure.motif_rmsd`'s `KeyError("prediction is missing measured
atoms: [...]")`, and all 21 have `motif_rmsd=nan`, `passed=False`. pLDDT itself
was fine and in range (median 0.877, range **0.835-0.902**, all on the 0-1
scale) — only the geometric gate failed.

**Root cause, verified directly, not inferred:** the folded CIF's residue
numbering is offset by 4 from the numbering `score_structure`/`motif.py`
expect. `motif.arm_residues(ARM_FULL)` names 24 residues including 85 and 111.
For design `cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_0__T0.1__1`:
- `designs.tsv`'s `sequence` field (campaign convention, chain F starts at
  residue 5, so `resnum = index + 5`) has **Arg** at position 111
  (`seq[111-5]=seq[106]='R'`) and **Glu** at position 85 (`seq[85-5]=seq[80]
  ='E'`) — i.e. the "frozen" motif positions genuinely were preserved by
  LigandMPNN in the sequence that was handed to the folder.
- The folded CIF's chain F residue **111 is GLY** (backbone atoms only, no
  side chain) and residue **85 is ARG** — and `seq[111-1]=seq[110]='G'` matches
  the CIF's residue 111 exactly. **The prediction is numbered `resnum =
  index + 1`, not `index + 5`.**

`tools/esmfold2/fold_many.py` (monorepo, out of this task's file scope) builds
`ProteinInput(id="F", sequence=job["sequence"])` with no residue-number offset,
so ESMFold2 numbers chain F starting at 1. Every other artifact in this
campaign — `constants.RMSD_REFERENCE`, `PARENT_SEQUENCE`, `motif.arm_residues`,
`designs.tsv` itself — uses the crystal's numbering, where chain F starts at 5.
This is exactly the class of defect this task's brief warned about ("an earlier
stage's tool silently relabelled the catalytic zinc's chain, turning every
metal measurement into nan and rejecting everything for a plausible-looking
reason") — here it is a 4-residue numbering shift instead of a chain
relabelling, caught on this task's 21 designs rather than propagating silently
through all 10,542.

Chain F itself is present in the output (`{'F', 'B'}` in the CIF), so criterion
4 is not separately triggered, and no pLDDT fell outside `[0, 1]` (criterion 3
not triggered). Criterion 2 — every design coming back unmeasurable — **is**
triggered, with the mechanism fully identified.

**No fix was attempted in `fold_many.py`.** It is in the monorepo, outside this
task's file scope, and it is handed only a sequence — numbering from 1 is
correct behaviour for a generic folder, and baking a campaign-specific offset
into a shared monorepo tool would be wrong. The reconciliation belongs where
the folded model is read.

## Fix: `score_structure.align_numbering` (applied after the report above)

Added `align_numbering(ref_atoms, pred_atoms)` to `score_structure.py`, called
from `score_folds.score_one` before `motif_rmsd`/`cleft_clearance`. The offset
is **derived** from the two residue-number sets, never hardcoded: it computes
`ref_res[0] - pred_res[0]` and verifies that shift reconciles the *entire* two
sets before applying it, raising `ValueError` (caught by `score_one`'s existing
`try`, so it becomes an `unscorable` row, not a stage-killing exception) if a
single uniform shift cannot explain the mismatch — a hardcoded `+4` would have
"fixed" this shard while silently mis-aligning any future model of a different
length. Four tests added to `test_score_structure.py` covering the shift, the
already-matching no-op case, a residue-count mismatch, and a gapped/non-uniform
mismatch that must raise rather than be forced into agreement.

**Re-scored the same 21 already-folded models, no GPU spend:**
```
conda run -n ligandmpnn_env python -m tada_redesign.score_folds \
  --run-dir outputs/20260806_tada_redesign_gen1
```
No more `unscorable` rows — `align_numbering` resolves cleanly (offset +4,
verified as a single uniform shift across all 156 residues) for every design,
confirming the root cause was fully and correctly identified. But the real
geometric measurement it unblocks is **not a pass**:

- **Status distribution (21 designs): `motif_drift` 18, `low_plddt` 3, `passed`
  0.**
- **pLDDT: median 0.877, range 0.835-0.902** (unchanged from the blocked run,
  as expected — pLDDT never depended on numbering).
- **motif_rmsd: median 8.863 A, range 3.898-9.847 A.** Every value but one
  (3.898 A) is above 7 A. This is far beyond `SCREEN_MOTIF_RMSD_MAX` (1.5 A) and
  far beyond the crystal-vs-relaxed-parent reference point (2.166 A, this
  campaign's own worst previously-seen case). `cleft_clearance` values
  (0.27-1.54 A) stayed in a plausible range, which is some evidence the
  superposition itself is not separately broken.

**This is reported as a real, different finding, not treated as success or as
evidence of a further software defect.** The numbering fix is verified
correct (independently confirmed against `designs.tsv`'s own sequence field
in the earlier root-cause section); what it exposes is that this shard's
screen-mode ESMFold2 refolds of these `FULL`-arm, `partial_t=1.0` designs do
not preserve the intended active-site geometry within tolerance, despite
decent per-residue confidence (pLDDT ~0.84-0.90) and despite the underlying
RFD3 backbones themselves having been measured at only 0.036-0.057 A backbone
drift from the parent (`docs/logs/20260806_tada_redesign_final_fix_wave.md`).
Candidate explanations not investigated here — reduced sampling (`num_loops=4,
num_sampling_steps=20`) being far noisier for this motif than the constants'
own docstring anticipated, or the redesigned (non-frozen) 132 positions
genuinely destabilizing the frozen arm's packing — are a question for whoever
decides how to proceed with Part 3b, not something resolved by this task.
**The full 44-shard screen still has NOT been submitted** and should not be
until this is understood: if it is representative, `SCREEN_SURVIVORS=2000`
would not be reached and the screen would need to fall back to the spec's
documented MPNN-score-first gating, which is a human decision.

## Suite count

151 passed, 1 deselected (`-m "not slow"`), out of 152 collected (147 + 4 new
`align_numbering` tests).

## Honesty ceiling

pLDDT measures model confidence; motif RMSD measures active-site geometry.
Neither is stability, solubility, or deaminase activity, and nothing in this
task constitutes wet-lab validation. Once `align_numbering` made the geometric
measurement possible, it showed active-site drift far outside this screen's own
tolerance for every design in this shard — a confident fold (pLDDT ~0.84-0.90)
is not evidence the active site sits where the design intended, and this task
does not attempt to explain why it does not.

## Follow-up items

1. **Re-verify Task 2's baseline** (`fold_probe`, `reference_baseline`) fold
   outputs against `align_numbering` — those were scored by inspection of
   pLDDT only, never by `motif_rmsd`, so a numbering offset there (if any)
   would not have been caught. `reference_baseline`'s parent folds use the
   exact same `PARENT_SEQUENCE` strings and `fold_many.py` path as the designs,
   so they likely carry the identical offset; whether their own motif geometry
   also drifts by several angstroms after alignment is untested.
2. **Understand the motif-drift finding above** before deciding whether/how to
   proceed with the screen: is 3.9-9.8 A drift specific to this cell
   (`TadA8e_FULL_pt1.0`), to `screen`-mode's reduced sampling, or general? A
   next debug shard drawn from a different cell (e.g. a `MIN`-arm or higher
   `partial_t`) would help distinguish those. This is a human decision, not
   something this task resolves.
3. Re-measure whether node contention adds meaningful average cost to the 2.88
   GPU-hour projection; today's single data point (40 min cold load under heavy
   contention vs. 104.5 s previously) suggests requesting a less-loaded node or
   a higher `-t` margin may matter operationally, though it does not change the
   GPU-second cost model itself.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

---

## 2026-08-08 — Part 3a gate-fix (Tasks 1-4, 3b): the active-site gate shipped broken twice, now fixed and re-roled

Plan: `docs/plans/2026-08-06-tada-redesign-part3a-gatefix.md`. Continues directly from
the BLOCKED finding above (parent scored 7.673 Å against a 1.5 Å gate). This entry
records both defects, the controls that exposed each, and the final, honest result.

### Defect 1 — the measured motif included the disordered chain terminus

The `FULL`-motif measurement above scored the **unmodified parent** at **7.673 Å**
against a 1.5 Å gate. Root cause: residue 156's ring atoms deviate 19-23 Å between a
predicted structure and the relaxed crystal reference — free-flapping near the chain
end — dominating a 202-atom average. Fix (Task 1): `motif.CORE_MOTIF`, `CATALYTIC ∪
POCKET` intersected with `MODELED` — 17 residues (28, 30, 46, 54, 57, 58, 59, 84, 85,
86, 87, 88, 90, 108, 110, 111, 149), dropping 109, 148, 152, 153, 154, 156, 157. This
took the parent from 7.673 Å to a then-measured 1.414/1.468 Å (screen/full).

Task 2 also retired the two-tier reduced/full sampling screen: parent-vs-parent
structural noise at the core is 2.020 Å median at reduced sampling vs 0.563 Å at full
(3.6x worse), and pLDDT was flat between the two modes (0.899-0.908 vs 0.835-0.902) —
confidence and structural reproducibility are different quantities, and a
confidence-only check could not have revealed this. Shards: `FOLD_BATCH_SIZE` 250→1000,
`FOLD_SHARDS` 44→11 (jobs 234250 screen scatter, 234277 full scatter).

### Defect 2 — the superposition anchor carried the same defect, one level down

Tasks 1-3 removed the disordered tail from the **measured** set but left it in the
**superposition anchor**: `score_structure._anchor_arrays` defaulted to every shared
Cα, and Kabsch is least-squares, so the tail (deviating 9-36 Å from the crystal) still
dragged the fit that was supposed to be the core's own reference frame. Re-measured
2026-08-08 through the actual production code path (`reference_baseline.py`, baseline
job **238437**, COMPLETED 00:03:22, pLDDT 0.897/0.888):

- **Parent vs crystal, CORE, under the unfixed all-CA anchor: TadA8e 3.555 Å, TadA9
  3.523 Å — both FAIL the 2.1 Å gate Task 3 had derived**, while 4/21 debug designs
  passed. This is the identical defect as Defect 1, in the other half of the same
  calculation — the measured-set fix and the anchor fix were the same fix, and only one
  half had been applied.
- Task 4's own STOP rule ("the gate is only fixed if the parent passes") caught this
  live and reported **BLOCKED** rather than shipping a gate that looked fixed but
  wasn't (see the BLOCKED task-4-report entry this continues from). No spec/log edit
  or push happened at that point.

**The fix (Task 3b):** `_anchor_arrays` now does iterative outlier-rejecting
refinement — Kabsch-fit on the current included Cα set, drop residues deviating beyond
`ANCHOR_OUTLIER_CUTOFF = 5.0 Å`, refit, repeat until the included set stops changing,
capped at `ANCHOR_MAX_ITER = 10` — with a self-fitting guard
(`ANCHOR_MIN_RETAINED_FRAC = 0.60`) that raises rather than silently returning an
anchor that has collapsed onto the measured set, plus a non-convergence raise. `5.0 Å`
was chosen by measuring candidate cutoffs (1.0, 2.5, 3.0, 5.0, 8.0, 15.0, 20.0, 50.0 Å)
against retained-fraction and resulting CORE RMSD, not guessed: it sits in the flat,
low-RMSD plateau and the resulting parent value (1.354/1.357 Å) agrees with an
independent, fixed CA 5-150 anchor (1.349/1.359 Å) to within 0.03 Å — i.e. it is not
self-fitting.

### The CORE definition (final, unchanged since Task 1)

`motif.CORE_MOTIF` = `CATALYTIC ∪ POCKET`, intersected with `MODELED`: 17 residues —
28, 30, 46, 54, 57, 58, 59, 84, 85, 86, 87, 88, 90, 108, 110, 111, 149 — spanning
28-149, clear of the chain end at 160.

### The derived threshold, with its arithmetic

Final, verified numbers, measured through the production code path after the anchor
fix (baseline job 238437; probe folds job 238335, `gatefix_probe/`):

- Parents vs crystal, CORE: **TadA8e 1.354 Å, TadA9 1.357 Å — both PASS.**
- `MOTIF_RMSD_MAX = 2.0 Å` = max(1.354, 1.357) + 0.563 Å fold-to-fold jitter = **1.920 Å
  floor, one tick above.**
- 21 probe designs, CORE: min 1.296, median 1.517, max 1.713 Å — **21/21 pass.**
- Test suite: **160 passed, 1 deselected** (run by the repo owner; not re-run here per
  the task brief).

### Backbone filter re-score — repo-owner ruling: keep the 502

`filter_backbones.py`'s `BACKBONE_MOTIF_RMSD_MAX = 1.0 Å` gate had already run against
the unfixed anchor (502/512 backbones passed, all 10 rejections `motif_drift` at
`partial_t=6.0` in `MIN` cells). Re-scored the same 512 backbones under the corrected
anchor: **502 → 506** — seven `motif_drift` rejections became passes and three passes
became rejections, all ten changes confined to `TadA8e_MIN_pt6.0`/`TadA9_MIN_pt6.0`
and all marginal (old scores 0.881-1.265 Å against the 1.0 Å gate). **Repo-owner
ruling, 2026-08-08: keep the 502-backbone set; `backbones.tsv` and
`filter_backbones.py` are NOT re-run** — the 502 already seeded LigandMPNN and the
10,542 downstream designs, and the fold-stage gate above is the filter that actually
selects, not the backbone filter. **Caveat carried forward:**
`BACKBONE_MOTIF_RMSD_MAX = 1.0` was itself derived under the now-retired anchor and is
therefore calibrated against a retired metric; it must be re-derived before any future
generation round. Leaving it as-is is a recorded choice for this round's
already-generated designs, not an endorsement of the constant.

### The central finding — not a success story

With the corrected anchor, the 21 debug designs span **1.30-1.71 Å against a parent at
1.35 Å with a 0.563 Å fold-to-fold jitter.** The frozen motif worked — the CORE site is
preserved in all 21 — but **ESMFold2 cannot resolve a design-from-parent difference at
the active site.** Motif RMSD is therefore re-roled as a **gross-failure catch, not a
ranking metric** (repo owner's ruling, 2026-08-08): a design whose core has genuinely
collapsed will still fail this gate, but a passing score says nothing about whether a
design is better than, worse than, or different from its parent. 21/21 passing must
not be read as evidence the designs were shown to be good — real discrimination between
designs must come from the deferred Rosetta stability stage, not from this geometry
gate.

### Honesty ceiling

Motif RMSD measures geometry; pLDDT measures model confidence. Neither is stability,
solubility, or enzymatic activity, and no wet-lab validation has been performed anywhere
in this campaign.

### What has and has NOT been submitted

Baseline fold job **238437** (COMPLETED, 00:03:22, pLDDT 0.897/0.888). Scatter jobs
**234250** (screen) and **234277** (full), from Task 2. Debug/probe folds only —
**the full 11-shard screen over all 10,542 designs has NOT been submitted.** No
PyRosetta, RFdiffusion, or AlphaFold run occurred in this task.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
