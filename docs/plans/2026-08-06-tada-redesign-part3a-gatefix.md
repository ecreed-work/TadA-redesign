# TadA Redesign — Part 3a Gate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active-site geometry gate able to distinguish a design from the unmodified parent — which it currently cannot — by restricting the measured motif to the catalytic core, folding at full sampling, and deriving the threshold from measured scatter instead of assumption.

**Architecture:** Three evidence-backed changes to existing modules, then a measurement task that sets the threshold, then a verification task that proves the gate discriminates. No new pipeline stages. The two-tier screen collapses to a single full-sampling pass, and the shard count drops from 44 to 11 because model load — not folding — dominates wall time under cluster contention.

**Tech Stack:** Python 3.12, numpy, Biopython, pytest. ESMFold2 (env `esmfold2`), SLURM `gpu` partition (H100).

**Spec:** `docs/specs/2026-08-05-tada-redesign-design.md` (this repo) — Task 4 updates it.
**Prior:** Part 3a is implemented (151 tests, preflight 20/20). This plan fixes a defect its debug shard exposed.

## Why this plan exists

Part 3a's debug shard folded 21 designs successfully and scored **0/21 passing**. The control that isolated the cause: the **unmodified parent**, folded and measured the same way, scored **7.673 Å** against a gate threshold of 1.5 Å. The metric could not tell a redesigned protein from the original, so no threshold on it means anything.

Two causes, both now measured and separable:

1. **Terminal residues dominated the FULL motif.** Residue 156's ring atoms deviated 19–23 Å between a predicted structure and the relaxed crystal reference — free-flapping near the chain end, swamping a 202-atom average. Restricting to the catalytic core drops the parent from **7.673 Å → 1.414 Å**.
2. **Reduced sampling is structurally noisy.** The same sequence folded twice with different seeds varies by **2.020 Å median** at the core. Designs (2.844 Å) sat inside that band. Full sampling cuts it to **0.563 Å**.

Note the trap this plan closes: pLDDT is essentially identical between sampling modes (0.899–0.908 full vs 0.835–0.902 screen), so a confidence-only check said the two modes were equivalent. **Confidence and structural reproducibility are different quantities.** An earlier ruling in this campaign generalised from the first to the second and was wrong to.

## Global Constraints

- Residue numbering is **Met = 1**; scaffold chain is **F**. Folded models arrive numbered from 1 and are reconciled by `score_structure.align_numbering`, which derives the offset and raises unless a single uniform shift reconciles both residue sets.
- Every table goes through `io.append_row`/`io.write_tsv` (atomic) and every stage writes `provenance.write(...)`.
- The degraded gate compares rows WRITTEN to inputs, never pass rate. A low pass rate is a measurement, not a stage failure.
- Submit with `sbatch`, never `bsub`. Partition `gpu`.
- All work in this repo; docs live here under `docs/`, not the monorepo.
- Push over `ssh://git@ssh.github.com:443/...` to `origin` (this repo's only remote, `ecreed-work/TadA-redesign`). The `tsailab` rule applies to the monorepo, not here.
- Both trailers on every commit:
  `Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>` and
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Tests: `cd <repo> && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"`. Each command needs its own `cd`.
- **The full screen is NOT submitted by this plan.** Task 4 verifies the gate on the existing 21 debug designs only.
- Honesty ceiling in every touched docstring: motif RMSD measures geometry and pLDDT measures model confidence. Neither is stability, solubility, or enzymatic activity.

## Measured facts this plan is built on

All measured 2026-08-06 on the production run. Do not re-derive; do not "improve" a constant away from these.

| Quantity | Value |
|---|---|
| `CORE` motif | `CATALYTIC ∪ POCKET`, intersected with `MODELED` = **17 residues**: 28, 30, 46, 54, 57, 58, 59, 84, 85, 86, 87, 88, 90, 108, 110, 111, 149 — spanning 28–149, clear of the chain end at 160 |
| Dropped vs the FULL motif | 109, 148, 152, 153, 154, 156, 157 (DNA-face-only, terminal-adjacent) |
| Noise floor, parent vs parent, CORE, **screen** sampling (10 pairs) | median **2.020 Å** (0.396–3.445) |
| Noise floor, parent vs parent, CORE, **full** sampling (10 pairs) | median **0.563 Å** (0.314–2.821) |
| Parent vs crystal reference, CORE | ~~screen **1.414 Å**, full **1.468 Å**~~ — **SUPERSEDED 2026-08-08, see below** |
| Parent vs crystal reference, FULL motif | **7.673 Å** (the defect) |
| 21 debug designs vs crystal, CORE, screen sampling | median **2.844 Å** (1.626–3.157); 1/21 under 2.0 Å |
| pLDDT, full sampling | 0.8992–0.9079 (screen: 0.835–0.902) |
| Fold rate | full **3.93 s/design**; screen 1.79 s/design |
| Full-sampling cost, 10,542 designs | **~11.5 GPU-hours** of folding |
| Model load | **22.4 s** quiet; measured at 104.5 s and ~2,400 s under contention |
| Jobs behind these numbers | 234250 (screen scatter), 234277 (full scatter), 234216 (debug shard) |

### Correction, 2026-08-08 — the anchor carried the same defect as the measured set

Tasks 1–3 removed the disordered C-terminal residues from the **measured** set but left
them in the **superposition anchor**. `_anchor_arrays` defaults to every shared CA and
Kabsch is least-squares, so the tail — which deviates 9–36 Å from the crystal — drags the
fit that is supposed to be the core's reference frame. The measured-set fix and the anchor
fix are the same defect in the two halves of one calculation; only one half was repaired.

Superseded rows above are left visible on purpose. The 1.414/1.468 Å parent figures came
from a side-script, not from `reference_baseline.py`'s production path, and were never
reproducible through the code the campaign actually runs. Re-measured 2026-08-08 through
the production path (baseline job 238437, COMPLETED 00:03:22, pLDDT 0.897/0.888):

| Quantity | all-CA anchor (shipped) | CA 5–150, tail excluded |
|---|---|---|
| TadA8e parent vs crystal, CORE | **3.555 Å** | 1.349 Å |
| TadA9 parent vs crystal, CORE | **3.523 Å** | 1.359 Å |
| 21 debug designs, CORE | median 2.791 (1.486–3.521) | median 1.515 (1.301–1.714) |
| Pass at `MOTIF_RMSD_MAX = 2.1` | 4/21 | 21/21 |

**The shipped path fails the unmodified parent at 3.55 Å against its own 2.1 Å gate while
admitting 4 designs** — the original defect intact, now selecting on superposition noise.
That the 1.35 Å is not self-fitting: a CORE-only anchor gives 1.319 Å and the independent
133-residue 5–150 anchor gives 1.349 Å, agreeing to 0.03 Å. The fold's own per-residue
confidence agrees — the tail scores pLDDT 29–48 against a 94.8 median.

Two consequences. `MOTIF_RMSD_MAX = 2.1` was derived in Task 3 from the corrupted
distribution and carries no meaning. And once the anchor is corrected the designs span
1.30–1.71 Å against a parent at 1.35 Å with 0.563 Å fold-to-fold jitter: **the CORE site is
preserved in all 21 and the gate does not discriminate among them.** That is a real result
about the protocol, and it re-roles motif RMSD as a gross-failure catch rather than a
ranking metric. Discrimination must come from the deferred stability stage.

**Why the shard count changes.** Each shard pays the model load once, so shard COUNT multiplies load overhead. At 44 shards a contended run spends ~29 GPU-hours loading weights against ~11.5 folding. At 11 shards that overhead drops to ~7.3. Fold cost is unchanged.

## File Structure

- `tada_redesign/motif.py` — add the `CORE` set alongside the existing arms.
- `tada_redesign/constants.py` — retire the two-tier sampling split; batch/shards; the threshold constant.
- `tada_redesign/score_folds.py` — gate on CORE, single mode.
- `tada_redesign/reference_baseline.py` — single mode.
- `tada_redesign/fold_screen.py` + `fold_screen.slurm` — fold at full sampling.
- `docs/specs/2026-08-05-tada-redesign-design.md`, `docs/logs/20260806_tada_redesign_part3a.md` — Task 4.
- Tests alongside each.

---

### Task 1: The `CORE` motif set

**Files:**
- Modify: `tada_redesign/motif.py`, `tada_redesign/tests/test_motif.py`

**Interfaces:**
- Consumes: `masks` from `motif.load_masks()`.
- Produces: `motif.CORE_MOTIF = "CORE"`, and `motif.arm_residues("CORE", masks)` returning the 17-residue tuple. `measured_residues(arm, masks)` continues to work for every arm.

- [ ] **Step 1: Write the failing tests**

Append to `tada_redesign/tests/test_motif.py`:

```python
CORE_EXPECTED = (28, 30, 46, 54, 57, 58, 59, 84, 85, 86, 87, 88, 90,
                 108, 110, 111, 149)


def test_core_is_the_catalytic_machinery_and_substrate_pocket(masks):
    """CATALYTIC | POCKET, intersected with MODELED. Measured 2026-08-06."""
    assert motif.arm_residues(motif.CORE_MOTIF, masks) == CORE_EXPECTED
    assert len(CORE_EXPECTED) == 17


def test_core_excludes_the_terminal_dna_face_residues(masks):
    """Residue 156's ring atoms deviated 19-23 A between a predicted structure
    and the relaxed reference -- free-flapping near the chain end, swamping the
    average. Dropping the DNA-face-only positions took the parent's own score
    from 7.673 A to 1.414 A."""
    core = set(motif.arm_residues(motif.CORE_MOTIF, masks))
    full = set(motif.arm_residues(motif.ARM_FULL, masks))
    assert sorted(full - core) == [109, 148, 152, 153, 154, 156, 157]
    assert 156 not in core


def test_core_stays_clear_of_the_chain_terminus(masks):
    """The measured artifact was terminal flapping, so the guard is on where the
    core sits, not merely on which mask it came from."""
    core = motif.arm_residues(motif.CORE_MOTIF, masks)
    assert max(core) <= max(masks["MODELED"]) - 10
    assert min(core) >= min(masks["MODELED"]) + 10


def test_core_contains_the_catalytic_tetrad(masks):
    core = set(motif.arm_residues(motif.CORE_MOTIF, masks))
    assert {57, 59, 87, 90} <= core


def test_core_is_a_subset_of_the_full_arm(masks):
    assert set(motif.arm_residues(motif.CORE_MOTIF, masks)) <= set(
        motif.arm_residues(motif.ARM_FULL, masks))
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_motif.py -q
```
Expected: FAIL — `AttributeError: module 'tada_redesign.motif' has no attribute 'CORE_MOTIF'`.

- [ ] **Step 3: Add `CORE` to `motif.py`**

Add the constant beside the existing arms and the mask tuple to `_ARM_MASKS`:

```python
ARM_FULL = "FULL"
ARM_MIN = "MIN"

# The MEASURED subset, used for scoring rather than for freezing. The FULL arm's
# DNA-face residues sit near the chain terminus, where a predicted structure and
# a relaxed crystal-derived one diverge freely: residue 156's ring atoms alone
# deviated 19-23 A, dominating a 202-atom average and pushing the UNMODIFIED
# parent to 7.673 A against a 1.5 A gate. Restricted to the catalytic machinery
# and the substrate pocket, the same parent measures 1.414 A. Measured
# 2026-08-06; see docs/plans/2026-08-06-tada-redesign-part3a-gatefix.md.
CORE_MOTIF = "CORE"

_ARM_MASKS = {
    ARM_FULL: ("CATALYTIC", "POCKET", "DNA_FACE"),
    ARM_MIN: ("CATALYTIC",),
    CORE_MOTIF: ("CATALYTIC", "POCKET"),
}
```

`arm_residues` already intersects with `MODELED`, so no other change is needed. Add a sentence to the module docstring noting that `CORE_MOTIF` is what gets MEASURED, while `ARM_FULL`/`ARM_MIN` are what get FROZEN during design — they are different jobs and must not be conflated.

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_motif.py -q
```
Expected: 14 passed (9 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/motif.py tada_redesign/tests/test_motif.py
git commit -F - <<'EOF'
feat: add the CORE motif set, measured rather than frozen

The FULL arm is what gets FROZEN during design; CORE is what gets MEASURED when
scoring a predicted structure. Conflating them was the defect: FULL includes
DNA-face residues near the chain terminus, where a prediction and a relaxed
crystal-derived reference diverge freely. Residue 156's ring atoms alone
deviated 19-23 A, dominating a 202-atom average and scoring the UNMODIFIED
parent at 7.673 A against a 1.5 A gate -- so the metric could not distinguish a
redesigned protein from the original.

CORE = CATALYTIC | POCKET intersected with MODELED: 17 residues spanning 28-149,
clear of the chain end at 160. The same parent measures 1.414 A over CORE.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: Single-tier full sampling, larger shards, CORE gate

**Files:**
- Modify: `tada_redesign/constants.py`, `tada_redesign/score_folds.py`, `tada_redesign/reference_baseline.py`, `tada_redesign/fold_screen.py`, `fold_screen.slurm`
- Test: `tada_redesign/tests/test_constants.py`, `tada_redesign/tests/test_score_folds.py`, `tada_redesign/tests/test_reference_baseline.py` (all append/adjust)

**Interfaces:**
- Produces: `constants.ESMFOLD_SETTINGS` (single dict, replacing the SCREEN/FULL pair), `constants.FOLD_BATCH_SIZE = 1000`, `constants.FOLD_SHARDS = 11`, `constants.MOTIF_RMSD_MAX` (set provisionally here, DERIVED in Task 3), `constants.PLDDT_MARGIN`.
- `reference_baseline.MODES` collapses to a single `"fold"` mode; `baseline_id(parent)` loses its mode argument.
- `score_folds` measures `motif.CORE_MOTIF`, not the design's own arm.

- [ ] **Step 1: Write the failing tests**

Append to `tada_redesign/tests/test_constants.py`:

```python
def test_single_sampling_setting_at_full_depth():
    """The two-tier screen is retired. Measured 2026-08-06: reduced sampling has
    a 2.020 A parent-vs-parent noise floor at the core, vs 0.563 A at full --
    3.6x worse, and above any usable threshold. pLDDT was flat between the modes
    (0.899-0.908 vs 0.835-0.902), so confidence alone could not reveal this."""
    assert constants.ESMFOLD_SETTINGS == {"num_loops": 20, "num_sampling_steps": 100}
    assert not hasattr(constants, "ESMFOLD_SCREEN")
    assert not hasattr(constants, "ESMFOLD_FULL")


def test_shard_count_reflects_load_dominated_cost():
    """Each shard pays the model load once (22.4 s quiet, ~2400 s contended), so
    shard COUNT multiplies that overhead while fold cost stays fixed."""
    assert constants.FOLD_BATCH_SIZE == 1000
    assert constants.FOLD_SHARDS == 11
    assert constants.FOLD_SHARDS * constants.FOLD_BATCH_SIZE >= 10542


def test_motif_threshold_exceeds_the_parents_own_offset_and_jitter():
    """The gate must admit the unmodified parent, which measures 1.468 A against
    the crystal reference at full sampling with a 0.563 A median fold-to-fold
    jitter. A threshold below that rejects the parent and is meaningless."""
    assert constants.MOTIF_RMSD_MAX > 1.468
```

Append to `tada_redesign/tests/test_score_folds.py`:

```python
def test_gate_uses_the_single_motif_threshold():
    ok, status = sf.gate(_row(rmsd=constants.MOTIF_RMSD_MAX + 0.01), parent_plddt=0.80)
    assert ok is False and status == "motif_drift"
    ok, _ = sf.gate(_row(rmsd=constants.MOTIF_RMSD_MAX - 0.01), parent_plddt=0.80)
    assert ok is True
```

Append to `tada_redesign/tests/test_reference_baseline.py`:

```python
def test_baseline_is_a_single_fold_per_parent():
    """One sampling mode means one baseline per parent."""
    jobs = rb.baseline_jobs()
    assert {j["parent"] for j in jobs} == set(constants.PARENTS)
    assert len(jobs) == len(constants.PARENTS)
```

Existing tests that reference `ESMFOLD_SCREEN`, `ESMFOLD_FULL`, `SCREEN_MOTIF_RMSD_MAX`, `FINAL_MOTIF_RMSD_MAX`, `SCREEN_PLDDT_MARGIN`, or a two-mode baseline must be UPDATED to the new names, not deleted. Where a test asserted the screen threshold was looser than the final one, replace it with the single-threshold assertion above and note in its docstring that the two-tier design was retired on measured evidence.

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"
```
Expected: failures in the new assertions plus every test still naming the retired constants.

- [ ] **Step 3: Update `constants.py`**

Replace the two-tier block:

```python
# ONE sampling setting. The two-tier screen (reduced sampling, then re-fold
# survivors at full) was retired on 2026-08-06 measurement: the parent-vs-parent
# noise floor at the core motif is 2.020 A at reduced sampling versus 0.563 A at
# full -- 3.6x worse, and above any threshold that could admit the parent.
# pLDDT was FLAT between the modes (0.899-0.908 full, 0.835-0.902 screen), so a
# confidence check alone said they were equivalent; confidence and structural
# reproducibility are different quantities. Full sampling costs 3.93 s/design
# (~11.5 GPU-h for 10,542) against 1.79 s/design -- affordable, and load time
# dominates wall clock anyway.
ESMFOLD_SETTINGS = {"num_loops": 20, "num_sampling_steps": 100}

# Each shard pays the model load once: 22.4 s on a quiet node, measured at
# 104.5 s and ~2400 s under contention. Shard COUNT therefore multiplies that
# overhead while total fold cost stays fixed, so fewer, larger shards win.
# 11 shards over 10,542 designs gives 958-959 each, under the batch cap.
FOLD_BATCH_SIZE = 1000
FOLD_SHARDS = 11

# Gate thresholds. MOTIF_RMSD_MAX is DERIVED from measured scatter, not assumed:
# the unmodified parent measures 1.468 A against the crystal reference at full
# sampling, with a 0.563 A median fold-to-fold jitter. A gate below that rejects
# the parent. Set provisionally at 2.5 and re-derived in Task 3 of the gate-fix
# plan from the designs' own full-sampling distribution.
MOTIF_RMSD_MAX = 2.5
PLDDT_MARGIN = 0.05
```

Delete `ESMFOLD_SCREEN`, `ESMFOLD_FULL`, `SCREEN_MOTIF_RMSD_MAX`, `FINAL_MOTIF_RMSD_MAX`, `SCREEN_PLDDT_MARGIN` and `SCREEN_SURVIVORS`. Keep `ESMFOLD_PLDDT_SCALE`/`AF3_PLDDT_SCALE` — the 0–1 vs 0–100 hazard is unchanged.

- [ ] **Step 4: Update the consumers**

`reference_baseline.py`: collapse `MODES` to one fold per parent. `baseline_id(parent)` drops its mode argument; `read_baseline(run_dir, require=None)` returns `{parent: plddt}` keyed by parent alone. Update its docstring to state the measured reason the two modes were retired.

`score_folds.py`: measure `motif.arm_residues(motif.CORE_MOTIF, masks)` — the CORE set — rather than the design's own arm. Add a comment making the distinction explicit: the arm is what was FROZEN during design; CORE is what gets MEASURED. Use `constants.MOTIF_RMSD_MAX` and `constants.PLDDT_MARGIN`, and `read_baseline(...)[parent]`.

`fold_screen.py` and `fold_screen.slurm`: pass `constants.ESMFOLD_SETTINGS`; default `--n-shards` to the new `FOLD_SHARDS`.

- [ ] **Step 5: Run the full suite**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"
```
Report the ACTUAL count. Every previously-passing test must still pass, updated to the new names where required. If a test cannot be updated without changing what it asserts, STOP and report — that would mean the rename changed behaviour, not just names.

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/constants.py tada_redesign/score_folds.py tada_redesign/reference_baseline.py tada_redesign/fold_screen.py fold_screen.slurm tada_redesign/tests/
git commit -F - <<'EOF'
feat: single full-sampling tier, CORE gate, 11 larger shards

Retires the two-tier screen on measured evidence: the parent-vs-parent noise
floor at the core motif is 2.020 A at reduced sampling versus 0.563 A at full.
Designs sat inside the reduced-sampling band, so no threshold could separate
design quality from seed jitter. pLDDT was flat between the modes, so a
confidence check alone said they were equivalent -- confidence and structural
reproducibility are different quantities, and an earlier ruling in this campaign
generalised from the first to the second and was wrong to.

score_folds now measures the CORE set rather than the design's own arm: the arm
is what was FROZEN during design, CORE is what gets MEASURED.

FOLD_BATCH_SIZE 250 -> 1000, FOLD_SHARDS 44 -> 11. Each shard pays the model
load once (22.4 s quiet, ~2400 s contended), so shard count multiplies that
overhead while fold cost stays fixed. Full sampling costs ~11.5 GPU-h for
10,542 designs.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: Derive the threshold from the designs' own full-sampling distribution

The threshold must come from data. Task 2 set it provisionally at 2.5 Å; this task re-folds the existing 21 debug designs at full sampling, measures their CORE distribution against the crystal reference, and sets the constant from that plus the parent's own numbers.

**Files:**
- Modify: `tada_redesign/constants.py` (the `MOTIF_RMSD_MAX` value and its citation), `tada_redesign/tests/test_constants.py`

- [ ] **Step 1: Re-fold the 21 debug designs at full sampling**

The designs already exist in `designs.tsv`; only the folds need redoing at the new settings. Fold them into a separate directory so the screen-sampling models remain as evidence.

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
RUN=outputs/20260806_tada_redesign_gen1
mkdir -p $RUN/gatefix_probe
head -1 $RUN/fold_screen/jobs_shard001.tsv > $RUN/gatefix_probe/jobs.tsv
tail -n +2 $RUN/fold_screen/jobs_shard001.tsv >> $RUN/gatefix_probe/jobs.tsv
wc -l $RUN/gatefix_probe/jobs.tsv
sbatch --wait --partition=gpu --gres=gpu:1 -c 4 --mem=48G -t 03:00:00 \
  -o logs/gatefix_probe.out -e logs/gatefix_probe.err \
  --wrap "source /research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3/bin/activate esmfold2 && python /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tools/esmfold2/fold_many.py --jobs /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign/outputs/20260806_tada_redesign_gen1/gatefix_probe/jobs.tsv --out-dir /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign/outputs/20260806_tada_redesign_gen1/gatefix_probe --ligand-ccd ZN --num-loops 20 --num-sampling-steps 100"
cat $RUN/gatefix_probe/timing.json
```
If the job sits PENDING beyond ~25 minutes, stop waiting and report DONE_WITH_CONCERNS with the job id. Do not resubmit or cancel.

- [ ] **Step 2: Measure the distribution**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python3 -c "
from tada_redesign import constants, motif, score_structure as ss
import glob, os, statistics as st
masks = motif.load_masks()
core = motif.arm_residues(motif.CORE_MOTIF, masks)
run = os.path.join('outputs', constants.RUN_DIR_NAME)
ref = ss.heavy_atoms_from_pdb(constants.RMSD_REFERENCE['TadA8e'])
v = []
for f in sorted(glob.glob(os.path.join(run, 'gatefix_probe', '*.cif'))):
    pred = ss.align_numbering(ref, ss.heavy_atoms_from_cif(f))
    v.append(ss.motif_rmsd(ref, pred, core))
v.sort()
print(f'designs at FULL sampling, CORE vs crystal (n={len(v)})')
print(f'  min {v[0]:.3f}  q1 {v[len(v)//4]:.3f}  median {st.median(v):.3f}  q3 {v[3*len(v)//4]:.3f}  max {v[-1]:.3f}')
for t in (1.5, 2.0, 2.5, 3.0):
    print(f'  under {t}: {sum(x < t for x in v)}/{len(v)}')
print('reference points: parent vs crystal 1.468 | parent-vs-parent jitter median 0.563')
"
```

- [ ] **Step 3: Set the threshold and cite it**

Choose `MOTIF_RMSD_MAX` by this rule, and STATE which case applied:

- It must exceed the parent's own offset plus its jitter — i.e. **at least 1.468 + 0.563 ≈ 2.03 Å**. Never set it below that: a gate the unmodified parent fails is meaningless.
- Within that floor, prefer the value that makes the gate *discriminating* rather than permissive. If the design distribution has a clear shoulder, put the threshold there. If designs are broadly indistinguishable from the parent, set it at the floor (2.1 Å) and say plainly in the commit message that the gate is admitting nearly everything — that is a finding about the designs, not a reason to tighten past the parent.

Update the constant with the measured citation replacing the provisional note, and update `test_motif_threshold_exceeds_the_parents_own_offset_and_jitter` to assert the chosen value against the same floor.

- [ ] **Step 4: Run the suite and commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"
```

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/constants.py tada_redesign/tests/test_constants.py
git commit -F - <<'EOF'
feat: derive MOTIF_RMSD_MAX from the measured full-sampling distribution

The threshold is now an empirical quantity rather than an assumed one. Floor:
the unmodified parent measures 1.468 A against the crystal reference at full
sampling with a 0.563 A median fold-to-fold jitter, so any gate below ~2.03 A
rejects the parent itself and measures nothing.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3b: Robust iterative superposition (added 2026-08-08)

Added after Task 4 reported BLOCKED. Runs **before** Task 4 completes; existing task numbers
are left alone so the ledger and commit history stay readable.

**Files:**
- Modify: `tada_redesign/score_structure.py`, `tada_redesign/constants.py`, `tada_redesign/tests/test_score_structure.py`

- [ ] **Step 1: Implement robust iterative superposition**

Replace the all-CA anchor with an iteratively refined one, in `_anchor_arrays` or a helper
it calls. The scheme, decided by the repo owner 2026-08-08 over a fixed residue span:

1. Start with every shared CA.
2. Kabsch-fit on the current included set; compute each included residue's deviation.
3. Drop residues deviating beyond the cutoff; refit.
4. Repeat until the included set stops changing, capped at `ANCHOR_MAX_ITER = 10`.

Cutoff: `ANCHOR_OUTLIER_CUTOFF`, a new constant in `constants.py`. **Derive it by
measurement, do not guess** — report the retained-set size and the parent's resulting CORE
RMSD for at least three candidate cutoffs and pick from that evidence. A correct cutoff puts
both parents near 1.35 Å (the independently-anchored value) and retains the structured core.

Two guards, each with a test:

- **Self-fitting guard.** The retained anchor must not collapse onto the measured set — that
  would shrink the reported quantity by construction. Require the retained set to keep at
  least `ANCHOR_MIN_RETAINED_FRAC` (start at 0.60) of the shared CAs, and raise if it does not.
- **Determinism.** Same inputs must give the same anchor and the same RMSD, every run.

Keep the existing `anchor_residues=` override and the `>= 3 shared CA` check.

- [ ] **Step 2: Measure the effect on `filter_backbones`, do NOT silently change it**

`filter_backbones.py:162` calls `motif_rmsd` on RFD3 backbones against
`BACKBONE_MOTIF_RMSD_MAX = 1.0`, and that gate has already run: **502 of 512 backbones
passed, all 10 rejections `motif_drift` at `partial_t=6.0` in MIN cells.** Re-score those
512 backbones under the new superposition and report whether the pass set changes.

If it changes, STOP and report — retroactively moving a completed filter result is a
campaign-level decision, not an implementation detail. If it does not change, say so with
the re-measured count.

**It changed, and the repo owner ruled on 2026-08-08: keep the 502.** Re-measured
independently: **502 → 506**, seven `motif_drift` rejections becoming passes and three
passes becoming rejections, all ten inside `TadA8e_MIN_pt6.0` / `TadA9_MIN_pt6.0` and all
marginal (old 0.881–1.265 against the 1.0 Å gate; the seven that improved did so sharply,
e.g. 1.265 → 0.628, the signature of a corrupted frame being repaired).

Anchor retention on backbones: **100% at `partial_t` 1.0 and 2.0** — the new fit is
numerically identical to the old one there — falling to a minimum of 0.821 at `pt6.0`,
still well clear of the 0.60 floor. The change is a no-op outside the highest-noise cells.

`backbones.tsv` and `filter_backbones.py` are NOT re-run. The 502 already seeded LigandMPNN
and the 10,542 designs; regenerating for a 0.8% change in the seed set is not worth the GPU
time, and the fold-stage gate is the filter that actually selects.

**Carry this caveat forward:** `BACKBONE_MOTIF_RMSD_MAX = 1.0` was itself derived under the
superseded superposition, so it is now calibrated against a retired metric. It must be
re-derived before any future generation round — leaving it as-is is a deliberate, recorded
choice for THIS round's already-generated designs, not an endorsement of the constant.

- [ ] **Step 3: Re-derive `MOTIF_RMSD_MAX` as a gross-failure catch**

The repo owner's ruling, 2026-08-08: the gate catches designs whose core actually collapsed;
it does **not** rank. Derive it from the parent's own value plus fold-to-fold jitter
(0.563 Å), state the arithmetic in the constant's comment, and record the resulting pass
count on the 21 probes.

A high pass rate is the honest finding that freezing the motif worked — report it as a
measurement. **Do not tune the threshold toward a target pass fraction**; with the whole
distribution inside fold-to-fold jitter that would rank noise and present it as active-site
quality. Retire the "designs must be distinguishable" framing from Task 4 Step 2: this task
establishes that ESMFold2 cannot resolve design-vs-parent differences at the CORE site, so
real discrimination belongs to the deferred stability stage.

### Task 4: Prove the gate discriminates, then update the spec and log

**Files:**
- Modify: `docs/specs/2026-08-05-tada-redesign-design.md`, `docs/logs/20260806_tada_redesign_part3a.md`

- [ ] **Step 1: Re-fold the parent baselines at the new settings and re-score**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
sbatch --wait --partition=gpu --gres=gpu:1 -c 4 --mem=48G -t 03:00:00 \
  -o logs/gatefix_baseline.out -e logs/gatefix_baseline.err \
  --wrap "source /research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3/bin/activate esmfold2 && cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && PYTHONPATH=. python -m tada_redesign.reference_baseline"
```
Then point `score_folds` at the full-sampling probe folds and report the status distribution. The probe folds live in `gatefix_probe/`, so either pass a `--run-dir` whose `fold_screen/` holds them or copy them into place — say which you did.

- [ ] **Step 2: The discrimination check — this is the deliverable**

Report all four numbers together:

- the parent's own CORE RMSD against the crystal reference at full sampling,
- the designs' CORE distribution,
- the resulting pass count,
- and whether the parent would pass its own gate.

**The gate is only fixed if the parent passes.** If it does not, STOP and report BLOCKED — a threshold that rejects the unmodified parent is the exact defect this plan exists to remove, and shipping one would be worse than the original bug because it would look like it had been fixed.

*This rule did its job on 2026-08-08: the parent measured 3.55 Å against the 2.1 Å gate, the task reported BLOCKED rather than shipping, and Task 3b exists because of it.*

Also report honestly whether designs and parent are *distinguishable*. If the design median sits within the parent's fold-to-fold jitter, say so plainly: it means ESMFold2 cannot resolve a difference between them at the active site, which is a real limitation of the readout, not a pass.

**Superseded by Task 3b Step 3 (2026-08-08).** The answer is now measured and settled: they are NOT distinguishable — designs span 1.30–1.71 Å against a parent at 1.35 Å with 0.563 Å jitter. Task 4 records that finding rather than re-testing for it.

- [ ] **Step 3: Update the spec**

In `docs/specs/2026-08-05-tada-redesign-design.md`, correct every place the retired design is described: the two-tier screen, `SCREEN_MOTIF_RMSD_MAX`/`FINAL_MOTIF_RMSD_MAX`, the reduced-sampling rationale, and the shard sizing. Preserve what was previously believed as superseded text rather than silently rewriting it — the campaign's record has twice been improved by keeping a refuted claim visible next to its correction.

- [ ] **Step 4: Update the log, commit, push**

Record in `docs/logs/20260806_tada_redesign_part3a.md`: the defect, the control that exposed it (parent at 7.673 Å), both scatter measurements with their job ids, the CORE definition, the derived threshold with its floor, the discrimination result, and the honesty ceiling. State plainly that the full screen has still NOT been submitted.

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add docs/specs/2026-08-05-tada-redesign-design.md docs/logs/20260806_tada_redesign_part3a.md
git commit -F - <<'EOF'
docs: record the gate fix, its measurements, and the superseded design

The two-tier screen and its thresholds are retired with the measurements that
retired them, and the previously-believed rationale is kept visible as
superseded rather than rewritten away.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git push origin main
```

---

## Self-Review

**Spec coverage.** The two causes the control isolated are each addressed: terminal-residue domination → Task 1's CORE set; reduced-sampling noise → Task 2's single full-sampling tier. The threshold becomes empirical in Task 3, and Task 4 proves the parent passes its own gate — the property whose absence was the original defect.

**Deliberately NOT done.** The predicted-reference idea from option 3 is dropped: the measurement showed the crystal reference is fine over CORE (parent 1.414 Å screen, 1.468 Å full), so it would add a moving reference for no benefit. Also not done: the full 44→11 shard screen submission, per-backbone bias re-evaluation, and everything in Part 3b.

**Type consistency.** `motif.arm_residues(arm, masks)` takes CORE exactly as it takes FULL/MIN — no new signature. `read_baseline` changes shape from `{(parent, mode): plddt}` to `{parent: plddt}`, and Task 2 Step 4 updates its single consumer in `score_folds`. `constants.MOTIF_RMSD_MAX`/`PLDDT_MARGIN` replace four retired names, all deleted rather than left as aliases so a stale reference fails loudly.

**Known risk, stated.** Task 3 may find that designs are indistinguishable from the parent at full sampling. That is not a plan failure — it is a real possible result, and Task 4's Step 2 requires reporting it plainly rather than tuning the threshold until the distribution looks discriminating. Tuning a threshold to manufacture separation would be the worst outcome available here.
