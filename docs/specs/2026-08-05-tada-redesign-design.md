# TadA Redesign Campaign — Design Spec

Date: 2026-08-05
Branch: `domain-insertion` (spec + submodule pointer land here)
New code: `tada-redesign/`, a **git submodule** whose upstream is
`git@github.com:ecreed-work/TadA-redesign` (empty as of 2026-08-05, verified)
Status: approved design, pending implementation plan

## Goal

Produce more thermodynamically stable and more soluble variants of **TadA-8e** and
**TadA-9** by *diffusing new backbones around a frozen active site* and redesigning
the sequence on them — a structurally generative alternative to
`tada-stability/`'s point-mutation campaign.

The optimization target is unchanged from that campaign: **absolute gain over the
evolved parent itself**, never against wild-type TadA. Both parents are already-evolved,
catalytically active deaminases.

Two first-class criteria:

1. **Folding free energy** — PyRosetta `ref2015_cart` absolute score, as REU/residue
   and as Δ vs. the identically-processed parent. Readout: Tm by nanoDSF.
2. **Solubility / aggregation** — pursued at design time (LigandMPNN surface biasing)
   and reported via `tada_stability.score_solubility`. Readout: soluble expression
   yield, SEC monodispersity.

Design context is the **monomer** plus its catalytic Zn plus the ssDNA substrate.
Dimerization is not a property to be preserved (the 6VPC E–F interface is ordinary
exposed surface for this campaign).

## Honesty ceiling

Every number this pipeline produces measures **structural plausibility and energetic
ranking within a fold family**. Specifically:

- Rosetta `ref2015_cart` absolute score is **not a physical ΔG of folding** and does
  not convert to Tm. It ranks designs of similar length and fold; it is not a
  measurement.
- MPNN-designed sequences relax to good Rosetta scores close to by construction.
  This is exactly why the two orthogonal folding models are in the funnel, and why
  Rosetta score alone can never be the sole gate.
- pLDDT and pTM measure model confidence / designability, not stability.
- No wet-lab validation has been performed. **Deaminase function is not demonstrated
  by any number here.** The active-site RMSD gate says the site is geometrically
  intact in a model, nothing more.
- Relaxing the isolated protomer lets formerly-buried pocket residues settle into a
  non-native conformation. Parent and design get identical treatment so the Δ stays
  comparable, but absolute values carry this caveat.

Every scorer module repeats this ceiling in its docstring.

## Relationship to prior work

| Source | What this campaign takes from it |
|---|---|
| `tada-stability/` | The locked TadA facts: `masks.json`, `constants` (parent sequences, `TADA9_MUTATIONS`, `SCAFFOLD_CHAIN`), the three tracked reference PDBs (`chainF_raw.pdb`, `TadA8e.pdb`, `TadA9.pdb`), Zn-constrained relax + `check_zn_geometry`, `sasa`, `score_solubility`. Imported live, not forked. |
| `domain-insertion/denovo_tada/` | RFD3 input-spec semantics (`unindex` / `select_fixed_atoms` / Zn-by-CCD-name / AtomWorks residue dropping / partial-diffusion spec shape), the numpy-compat LigandMPNN wrapper, and the SLURM sharding + incremental-TSV conventions. Copied with citation, **not** imported — that package is Cas9/RuvC-coupled throughout. |
| `tools/esmfold2/` | Local ESMFold2 folding. Requires a small extension (below). |

This campaign does **not** touch, block, or supersede `tada-stability/`. That
campaign's own blocker (real LigandMPNN generation → `candidates.tsv`) is unrelated
and unaffected.

## Verified inputs and assets

All paths below were verified to exist on 2026-08-05 unless marked otherwise.

| Asset | Path | Notes |
|---|---|---|
| RFD3 checkpoint | `/research_jude/.../common/claude/foundry_ckpt/rfd3_latest.ckpt` | driven by `rfd3 design` in env `cas9-pam-design` |
| LigandMPNN ckpt | `design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt` | ligand-aware; env `ligandmpnn-sc` |
| AF3 | `/hpcf/authorized_apps/rhel8_apps/alphafold/3.0.2/alphafold.3.0.2.sif` | weights `/lustre_scratch/reference/public/alphafold_data/af3/models/af3.bin` |
| ESMFold2 | `biohub/ESMFold2` via the Biohub `transformers` fork, env `esmfold2` | weights cached at `~/.cache/huggingface/hub/models--biohub--ESMFold2`; pLDDT on a **0–1** scale |
| PyRosetta | env `pyrosetta` | `ref2015_cart`, `-auto_setup_metals`, Biopython 1.81 |
| Structure source | `tada8e-cas9-Interface-design/structural_analysis/structures/deaminase/pdb6vpc.ent` | 6VPC; chain **F** is the catalytic protomer (its Zn is 2.12 Å from the 8AZ mimic vs. 17.3 Å for chain E), modeled 5–160 gap-free |
| Masks | `tada-stability/outputs/20260728_tada_stability/masks.json` | locked by regression test |
| Reference parents | `tada-stability/reference/{TadA8e,TadA9}.pdb` | committed, tracked, Zn geometry recorded |

**Known-bad input, never referenced:**
`tada8e-cas9-Interface-design/interface_design/abe_tadA_ruvc/inputs/6vpc_dCas9_TadA8e.pdb`
(chains B/E only, Zn stripped). `preflight.py` asserts it appears nowhere in this
package.

**Cluster reality:** SLURM has exactly two partitions — `gpu` (14 nodes,
`gpu:h100:8`, 1568 CPUs, ~2 TB/node, no time limit) and `hpcf_test`. There is no
CPU-only partition, so the Rosetta stage runs on `gpu` **without** `--gres`.
Submission is `sbatch`, never `bsub`.

**Network reality:** github.com is unreachable from `splpslurm03` on both port 22 and
port 443/HTTPS. SSH over `ssh.github.com:443` works and authenticates as
`ecreed-work` (verified). The submodule's remote URL must therefore pin
`ssh://git@ssh.github.com:443/ecreed-work/TadA-redesign.git`. `gh` is not installed.

## Fixed motif — two arms in parallel

Both arms freeze their residues in **both** senses: fixed atomic geometry during
diffusion (`select_fixed_atoms`, which injects zero noise so internal geometry is
preserved while the group may still move rigid-body), and locked WT identity during
sequence design (`--fixed_residues`).

| Arm | Definition | Size |
|---|---|---|
| `FULL` | `CATALYTIC ∪ POCKET ∪ DNA_FACE`, intersected with `MODELED` | **24** residues |
| `MIN` | `CATALYTIC` only (His57, Glu59, Cys87, Cys90) | **4** residues |

The catalytic Zn is fixed in both arms via RFD3's `ligand` keyword, keyed by **CCD
name** (`"ZN": "ALL"`), never by chain+resid — AtomWorks renames a hetero atom's chain
when it shares a chain letter with protein residues, so a `F201` key can never resolve
post-load. This is a measured RFD3 failure, not a precaution.

`MIN` exists to measure what the extra 20 frozen positions actually cost in stability
headroom. Its known risk: with only the tetrad pinned, nothing but the substrate
context prevents the binding cleft from collapsing — which is why the ssDNA is present
in both arms.

`motif.py` is the **single source of truth** for these sets and emits all three
consumer formats (RFD3 `select_fixed_atoms`, LigandMPNN `--fixed_residues`, the
scorer's atom list) from one definition. The worst defects in the predecessor campaign
came from a generator and a gate disagreeing about the same residue set.

## Substrate and ligand context

Present as fixed context in both generative stages:

- **Zn** — via `ligand` (RFD3) and as a context atom (LigandMPNN).
- **ssDNA** — 6VPC chain D, truncated to the pocket neighborhood.

**A measured limitation to design around:** RFD3's AtomWorks loader **drops
non-standard residues**, and the 8-azanebularine target-base analogue (`8AZ`, chain D
residue 26) is one of them — confirmed by a real RFD3 run raising
`ValidationError: [component=D26] Residue D26 not found in atom array`. So RFD3 never
sees the target base itself; the cleft is held open by the flanking nucleotides plus
frozen pocket geometry, and chain D's contig must be emitted as maximal contiguous
subranges that split around D26 as well as around 6VPC's real numbering gaps (a
labeled range spanning a missing resid raises `ComponentValidationError`).

LigandMPNN, by contrast, reads non-protein atoms directly and *can* see the 8AZ atoms
as context. AF3 receives DNA as a `dnaSequence` with an ordinary adenine at the target
position. ESMFold2 folds protein + Zn only in the screen (DNA optional via `DNAInput`,
not enabled by default — see Prerequisite 2).

RFD3 `select_hotspots` aimed at the target-adenine position is used to orient the
pocket, mirroring `denovo_tada`'s validated use.

## Design matrix

| Axis | Values | N |
|---|---|---|
| Parent | TadA-8e, TadA-9 | 2 |
| Motif arm | `FULL` (24), `MIN` (4) | 2 |
| `partial_t` (Å of re-noise) | 1, 2, 4, 6 | 4 |
| LigandMPNN temperature | 0.1, 0.15, 0.2, 0.3 | 4 |

16 diffusion cells × 32 backbones/cell (`n_batches=4` × `diffusion_batch_size=8`) =
**512 backbones**. × 20 sequences/backbone (5 per temperature) = **10,240 designs**,
plus the 512-design zero-bias control set defined in Stage 2 = **10,752 total**.
Every design row carries its full cell coordinates, so "what did more noise or a
hotter temperature buy" is a query against `designs.tsv`, not a re-run.

**Why the ladder tops out at 6 Å.** `partial_t` is Ångströms of injected re-noise
(RFD3 recommends ≤15). `denovo_tada`'s 2026-08-04 debug gate measured that
`partial_t=8` degraded the TadA active site from ~1.1 Å to **2.5–6.75 Å** RMSD, failing
a 1.5 Å gate on every design. 6 Å is therefore the deliberate high end of a range that
still plausibly survives, and its yield is expected to be poor — that outcome is a
measurement this campaign will report, not a surprise.

## Pipeline and gates

Thresholds live in `constants.py`. Every gate below is stated with whether it is
measured or assumed.

### Stage 0 — `preflight.py` (blocking, read-only, cheap)

RFD3 ckpt; LigandMPNN ckpt; AF3 SIF + `af3.bin`; ESMFold2 HF cache; `masks.json`;
both reference parents present **and passing `check_zn_geometry`**; and each conda env
importing what its own stage needs (`cas9-pam-design`→`rfd3`, `ligandmpnn-sc`,
`esmfold2`→the transformers fork, `pyrosetta`→`pyrosetta` + `Bio`). Known-bad-PDB
absence assertion. Refuses to let any batch stage run until green.

### Stage 1 — diffusion + backbone filter

`prep_rfd_inputs.py` → `rfd_inputs.yaml` (one partial-diffusion spec per cell:
`input` = the parent's relaxed reference PDB, `partial_t`, `select_fixed_atoms`,
`ligand`, `select_hotspots`; no contig, per partial-diffusion semantics) →
`rfd_partial.slurm` → 512 backbones.

`filter_backbones.py` rejects, per backbone:

- motif heavy-atom RMSD to the parent > **1.0 Å** (RFD3 does not perfectly honor fixed
  atoms — verify, never assume)
- any consecutive Cα–Cα distance > **4.2 Å** (chain break)
- length outside **150–175**
- Zn not within **2.0–2.6 Å** of all three donors

Rejection counts are printed **per cell**, so a cell that silently produced nothing is
visible rather than quietly absent from the results.

### Stage 2 — sequence design

`prep_mpnn_inputs.py` → per-backbone PDB (protein + Zn + DNA context), fixed-residue
string, and the solubility bias spec. `run_ligandmpnn.slurm` sweeps the four
temperatures, 5 sequences each. `collect_designs.py` → `designs.tsv`: `design_id`,
cell coordinates, backbone, sequence, MPNN score, identity to parent, mutation count.

**Solubility at design time** is LigandMPNN amino-acid biasing on *non-frozen,
solvent-exposed* positions only (`EXPOSED ∩ designable`): a negative bias on the
strongly hydrophobic set and a mild positive bias on polar/charged residues. Frozen
positions and buried core positions are never biased — biasing a core position would
trade the very stability the campaign is trying to buy. Bias magnitudes are recorded
in provenance.

**Zero-bias control set.** The bias is an assumption, not a known good, so it gets a
control: **one** additional zero-bias sequence per backbone at T=0.15, i.e. **512
extra designs**, tracked with `bias=none` and carried through every scoring stage.
Total designs entering Stage 3 is therefore **10,752** (10,240 biased + 512 control).
The control set is reported separately and is eligible for the shortlist — if
unbiased designs win, that is a finding, not an error.

### Stage 3 — ESMFold2 screen (all 10,752) → survivors re-folded

Screen at reduced sampling (`num_loops=4`, `num_sampling_steps=20` — the settings
verified working in the 2026-08-04 debug fold); full sampling is the wrapper's default
`num_loops=20`, `num_sampling_steps=100`. Reduced sampling **depresses pLDDT
substantially** (the debug fold returned 0.45 on a 78-mer for this reason), so the
parent is folded in the **identical** reduced mode and the gate is relative to it.
Gates:

- pLDDT ≥ `parent_screen_plddt − 0.05` (0–1 scale)
- motif heavy-atom RMSD ≤ **1.5 Å** — deliberately looser than the final gate, because
  reduced sampling is noisier

Top ~2,000 survive. **The cap is a compute decision, and the dropped count is logged**
(no silent truncation). Survivors are re-folded at full sampling, and only the
full-sampling numbers carry forward.

`score_structure.py` is the one geometric scorer, shared by this stage and AF3:

- **Heavy-atom motif RMSD** after Kabsch superposition on all designed Cα. Heavy-atom
  is tractable precisely because motif identity is locked, so atom names match
  one-to-one — an upgrade over the Cα-only compromise `gate_fold.py` had to accept.
  Superposing on the full backbone rather than on the measured motif is deliberate:
  fitting on the measured points trivially shrinks whatever is measured.
- Tetrad-only RMSD, reported alongside.
- Zn coordination via `relax_scaffolds.check_zn_geometry`.
- **Cleft openness**: map the crystal 8AZ coordinates into the design frame through
  the same superposition, then measure the minimum distance from any 8AZ atom to any
  design heavy atom, **excluding the catalytic Zn**. The Zn coordinates the target
  base at ~2.12 A as a matter of correct catalytic geometry, not a clash; counting it
  makes a correctly-placed metal read as a collapsed cleft. Gated **relative to each
  parent's own measured clearance** (`constants.CLEFT_CLEARANCE_MARGIN = 0.3` A), not
  against an absolute floor — native substrate H-bond contacts sit at 2.2-2.4 A, so an
  absolute floor near that range leaves no real headroom on the parents themselves.
  Measured on the committed references (Zn excluded): crystal `chainF_raw.pdb` vs
  itself 2.330 A (closest protein atom Arg111:NH1), TadA8e 2.211 A, TadA9 2.271 A. A
  design fails when its own clearance is worse than its parent's by more than the
  margin — the specific failure the `MIN` arm is exposed to.

### Stage 4 — Rosetta (survivors)

Relax each survivor's full-sampling model with explicit Zn constraints; both parents
travel the **identical** ESMFold2→relax path. Comparing a designed *prediction*
against a *crystal* baseline would confound the Δ with prediction error, so
`reference_baseline.py` exists solely to prevent that.

- **≥3 relax replicates**, mean ± sd. The predecessor campaign measured ~17.6 kcal/mol
  of relax noise on a naive protocol; any design whose sd exceeds its own Δ is
  **flagged**, never silently ranked.
- Accept: Δ(REU/residue) < 0 vs. its own parent **and** `check_zn_geometry` passing on
  the relaxed design.
- Report: total REU, REU/residue, Δ, `exposed_hydrophobicity`, net charge, pI, each vs.
  parent.

### Stage 5 — AF3 (top 200 + ~100 stratified controls)

Single-sequence mode (`unpairedMsa: ""`, `templates: []`) — an MSA for a redesigned
sequence is meaningless and the search is the expensive part — plus the `ZN` ligand
and the DNA chain.

**Final acceptance:** AF3 pLDDT ≥ parent **and** motif heavy-atom RMSD ≤ **1.0 Å**
**and** Zn geometry pass **and** Rosetta Δ < 0.

The ~100 controls are drawn by stratified sampling across the full Rosetta Δ range,
**including rejects**. They exist only so the correlation is not range-restricted, and
can never enter the shortlist.

### Stage 6 — correlate and report

`correlate.py`: Spearman and Pearson of ESMFold2 vs. AF3 on pLDDT and on motif RMSD
across the both-folded set, reporting n, the Δ range spanned, and **the top-200
correlation separately from the full stratified one** — so range restriction is visible
instead of hidden.

`rank.py` / `report.py`: gated shortlist sorted by Rosetta Δ, a markdown report, and
one relaxed PDB per finalist. Any fallback or truncation prints itself.

## Module inventory — `tada-redesign/tada_redesign/`

`constants.py` · `motif.py` · `prep_rfd_inputs.py` · `filter_backbones.py` ·
`prep_mpnn_inputs.py` · `_run_ligandmpnn.py` (vendored numpy-compat wrapper, cited) ·
`collect_designs.py` · `fold_screen.py` · `score_structure.py` ·
`reference_baseline.py` · `score_rosetta.py` · `prep_af3.py` · `correlate.py` ·
`rank.py` · `report.py` · `preflight.py` · `tests/`

SLURM: `rfd_partial.slurm` · `run_ligandmpnn.slurm` · `fold_screen.slurm` (array) ·
`fold_full.slurm` · `score_rosetta.slurm` (array, no `--gres`) · `af3_infer.slurm`
(array) · `merge_*.sh`.

## Data flow

```
masks.json + reference PDBs
  → motif.py → rfd_inputs.yaml → [RFD3] → backbones/ → backbones.tsv
  → lmpnn_inputs.tsv → [LigandMPNN] → designs.tsv
  → [ESMFold2 screen] → fold_screen.tsv → [ESMFold2 full] → fold_full.tsv
  → [Rosetta] → rosetta.tsv
  → [AF3] → af3.tsv
  → correlation.md + shortlist.tsv + shortlist_structures/
```

Run dir: `tada-redesign/outputs/20260805_tada_redesign/`. Every stage keys rows by
`design_id` and appends; a killed array resumes rather than restarting.

## Error handling and provenance

- Array stages write **incrementally** (header + flush per row), one shard TSV per
  task, merged by `merge_*.sh`.
- A per-design exception emits a **failed row** with a status column and message; it
  never kills the shard.
- `skip_existing` on every stage.
- **Degraded-run refusal**: a stage that fails to produce output for **more than 20%**
  of its inputs refuses the canonical output path, writing `*.degraded.tsv` plus
  provenance naming what was lost — the pattern `tada_stability.vote` established.
  20% is a deliberate choice: high enough to tolerate the per-design failures a 10k
  batch will always have, low enough that a systematically broken stage cannot pass
  itself off as a completed one.
- Every stage writes `<stage>.provenance.json`: submodule git SHA, checkpoint paths +
  mtimes, conda env, thresholds used, input and output row counts, and (for MPNN) the
  bias magnitudes.
- Every TSV reader skips `#` comment lines.

## Testing

TDD per module; `pytest` in `ligandmpnn_env`; pure-Python paths testable without GPU
or PyRosetta. Fixtures that carry real weight:

- `motif.py` emits three formats that agree on one residue set (the anti-divergence
  test).
- Heavy-atom RMSD: a **rigid-body invariance** test (known rotation + translation →
  RMSD ≈ 0) and a **local-perturbation** test (same transform plus one perturbed atom
  → RMSD reflects exactly that perturbation). These two are non-negotiable: their
  absence is precisely how the predecessor's `pocket_rmsd` shipped with no
  superposition at all.
- Backbone-filter rejection for each reject reason.
- Cleft-openness detects a deliberately closed cleft.
- Stratified control sampling actually spans the range.
- `#`-comment skipping in every reader.
- `--mock` mode on every stage that invokes a heavy tool.

**Debug gate before any batch (CLAUDE.md, mandatory).** One cell × 2 backbones × 2
sequences, end to end: RFD3 at `n_batches=1 × diffusion_batch_size=2`,
`num_timesteps=20`; LigandMPNN 2 sequences; ESMFold2 reduced sampling; Rosetta on 1
design; AF3 with `--run_data_pipeline` only before any inference. This run **measures**
per-design wall time at every stage, and all shard sizes are set from those
measurements. ESMFold2 throughput at 10k scale is the single largest unknown in this
plan and is explicitly not guessed.

## Compute budget

Per-design costs are **unmeasured** for every GPU stage on this input size; the debug
gate exists to measure them, and the shard widths below are placeholders to be replaced
by measured values before any batch is submitted.

| Stage | Work | Hardware |
|---|---|---|
| RFD3 partial | 16 cells × 32 backbones | `gpu`, 1×H100 per cell, array of 16 |
| LigandMPNN | 512 backbones × 4 temps | `gpu`, 1×H100; `denovo_tada` measured ~1 h per 64 backbones per temperature sweep, so this stage shards by backbone |
| ESMFold2 screen | 10,752 folds, reduced sampling | `gpu`, sharded array — **the dominant unknown**; if the measured per-fold cost makes 10,752 unaffordable, the documented fallback is to gate on MPNN score first and report exactly how many designs were dropped unfolded |
| ESMFold2 full | ~2,000 folds, full sampling | `gpu`, sharded array |
| Rosetta | ~2,000 designs × 3 replicates | `gpu` partition **without** `--gres`, CPU-only array |
| AF3 | ~300 single-sequence + ligand | `gpu`, 1×H100, array |

## Repository mechanics

- `tada-redesign/` is a git submodule; remote
  `ssh://git@ssh.github.com:443/ecreed-work/TadA-redesign.git` (port 443 is mandatory
  here — 22 is blocked).
- Per CLAUDE.md: commit **inside the submodule first**, then move the root pointer.
- Cross-repo assets (`masks.json`, the reference PDBs, `tada_stability` modules)
  resolve through a `TADA_MONOREPO` environment variable defaulting to
  `/research/rgs01/home/clusterHome/ecreed/claude-proteindesign`, mirroring the `R=`
  pattern the existing SLURM scripts already use. `preflight.py` validates it.
- The submodule keeps its own `requirements.md` documenting which conda env each stage
  needs, following `tada-stability`'s precedent that env drift on this cluster is real.
- Nothing is pushed to GitHub until this spec is approved.

## Prerequisites (must be resolved before implementation)

1. **`tada-stability/tada_stability/relax_scaffolds.py` has uncommitted changes** in
   the working tree (the `_apply_zn_constraints` factoring and the paired-local ΔΔG
   path, from the 2026-08-04 ΔΔG rewrite). This campaign depends on
   `_apply_zn_constraints` and `check_zn_geometry`. That work must be committed, or
   this pipeline depends on an untracked file.
2. **`tools/esmfold2/fold.py` must be extended to carry the Zn** (a parent-repo
   change, outside the submodule). Confirmed feasible: `esm.models.esmfold2` exports
   `LigandInput(id, smiles, ccd)` and `DNAInput(id, sequence)`, and
   `StructurePredictionInput.sequences` accepts them alongside `ProteinInput` — so
   `LigandInput(id="B", ccd=["ZN"])`. Must be verified by one debug fold before the
   batch.
3. The GitHub repo `ecreed-work/TadA-redesign` exists and is empty (verified); it
   needs its first commit and a `main` branch.

## Explicitly out of scope

- Wild-type-TadA calibration (the objective is gain over the evolved parent).
- Full de novo motif scaffolding — considered and rejected; partial diffusion only, so
  the result remains recognizably TadA.
- Plain ProteinMPNN — no ligand channel, so it cannot see the catalytic Zn.
- The crystallographic dimer interface as a design objective.
- MD / MM-PBSA / OpenMM finalist refinement. Deferred; may be added once a shortlist
  exists.
- Any claim about editing activity or base-editor function.
