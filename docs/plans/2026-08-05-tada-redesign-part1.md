# TadA Redesign — Part 1 (Foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `tada-redesign` submodule with its shared-truth modules — constants, the frozen-motif definition, the structural scorer, and the dependency preflight — all unit-tested, plus the two parent-repo prerequisites the campaign depends on.

**Architecture:** A new git submodule at `tada-redesign/` (upstream `ecreed-work/TadA-redesign`) holding a `tada_redesign` Python package. Part 1 builds only the pure-Python foundation that later stages consume: one definition of the frozen motif rendered into three consumer formats, one geometric scorer shared by both folding stages, and a preflight gate. No GPU stage, no design job, and no heavy tool run except a single cheap ESMFold2 debug fold to verify the Zn extension.

**Tech Stack:** Python 3.12, numpy, Biopython (`Bio.PDB`), pytest. Conda envs: `ligandmpnn_env` (tests), `esmfold2` (the one debug fold). Git submodules over SSH port 443.

**Spec:** `docs/superpowers/specs/2026-08-05-tada-redesign-design.md`

## Global Constraints

- Residue numbering is **Met = 1** (UniProt P68398) everywhere. Never use literature TadA-8e mutation labels — they are relative to TadA-7.10.
- Scaffold chain is **F** (6VPC's catalytic protomer). Chain E is never designed.
- The catalytic Zn is referenced in RFD3 specs by **CCD name** (`"ZN"`), never by chain+resid (`"F201"`) — AtomWorks renames a hetero atom's chain when it shares a letter with protein residues.
- `masks.json`'s `FROZEN` key (36 residues, includes `ZN_PROXIMITY`) is **not** this campaign's motif. The `FULL` arm is `CATALYTIC ∪ POCKET ∪ DNA_FACE` intersected with `MODELED` = **24** residues, verified 2026-08-05.
- Cross-repo paths resolve through `TADA_MONOREPO`, defaulting to `/research/rgs01/home/clusterHome/ecreed/claude-proteindesign`.
- github.com is unreachable from this cluster on ports 22 and 443/HTTPS. All git remotes must use `ssh://git@ssh.github.com:443/...`.
- Submit with `sbatch`, never `bsub`. SLURM has only the `gpu` partition (`gpu:h100:8`) and `hpcf_test`.
- Commits carry both trailers:
  `Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>` and
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Per CLAUDE.md: commit **inside the submodule first**, then move the root pointer. Write a `docs/logs/20260805_tada_redesign.md` entry in the monorepo when Part 1 completes.
- Honesty ceiling, repeated in every scorer docstring: these metrics measure structural plausibility, not function. Rosetta absolute score is not a physical ΔG.

## File Structure

**Parent repo (monorepo):**
- Modify: `tools/esmfold2/fold.py` — add ligand (Zn) and DNA inputs; factor input-building into a testable function.
- Create: `tools/esmfold2/tests/test_fold_inputs.py` — GPU-free tests for the new input builder.
- Create: `.gitmodules` entry + `tada-redesign` pointer.
- Create: `docs/logs/20260805_tada_redesign.md`.

**Submodule (`tada-redesign/`):**
- `README.md`, `CLAUDE.md`, `requirements.md`, `.gitignore`, `pytest.ini`
- `tada_redesign/__init__.py`
- `tada_redesign/constants.py` — every path, threshold, and sweep axis. No logic.
- `tada_redesign/motif.py` — one motif definition, three renderers.
- `tada_redesign/score_structure.py` — Kabsch, structure loaders, motif RMSD, cleft clearance.
- `tada_redesign/preflight.py` — dependency gate.
- `tada_redesign/tests/{__init__.py,test_constants.py,test_motif.py,test_score_structure.py,test_preflight.py}`

**Deferred to later plans:** Part 2 (generation — `prep_rfd_inputs`, `filter_backbones`, `prep_mpnn_inputs`, `_run_ligandmpnn`, `collect_designs`, and their SLURM scripts). Part 3 (scoring/report — `fold_screen`, `reference_baseline`, `score_rosetta`, `prep_af3`, `correlate`, `rank`, `report`). Each is independently testable and gets its own plan once Part 1 lands.

---

### Task 1: Commit the `tada-stability` ΔΔG prerequisite

The 2026-08-04 ΔΔG rewrite (in-process PyRosetta, common-reference local relax) is uncommitted in the working tree. Part 3's Rosetta stage depends on its `_apply_zn_constraints` factoring. Part 1's code does **not** depend on it — `check_zn_geometry` is already committed — so if this task's tests fail, stop and report rather than proceeding to fix that campaign's code.

**Files:**
- Modify (commit only, no edits): `tada-stability/tada_stability/relax_scaffolds.py`, `tada-stability/tada_stability/score_ddg.py`, `tada-stability/tada_stability/tests/test_relax_scaffolds.py`, `tada-stability/tada_stability/tests/test_score_ddg.py`, `tada-stability/score_ddg.bsub`, `tada-stability/_debug_ddg_m70l.bsub`

**Interfaces:**
- Consumes: nothing.
- Produces: committed `relax_scaffolds._apply_zn_constraints(pose, ...)` and `relax_scaffolds.check_zn_geometry(relaxed_pdb, raw_pdb)` — the latter returns a dict of measured distances and raises `ValueError` listing every measured value on failure.

- [ ] **Step 1: Confirm what is actually uncommitted**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git status --short -- tada-stability
git diff --stat -- tada-stability
```

Expected: the six files listed above, five modified and `_debug_ddg_m70l.bsub` untracked.

- [ ] **Step 2: Run the covering tests**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-stability
conda run -n ligandmpnn_env python -m pytest \
  tada_stability/tests/test_score_ddg.py \
  tada_stability/tests/test_relax_scaffolds.py -v
```

Expected: all pass (the 2026-08-04 log records 16 passing in `test_score_ddg.py`; the one `test_relax_scaffolds.py` case needing a live PyRosetta pose is skipped in this env via `pytest.importorskip`).

**If anything fails: STOP.** Report the failure and do not commit. Part 1 continues at Task 2 regardless.

- [ ] **Step 3: Import-check both modules**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-stability
PYTHONPATH=. conda run -n ligandmpnn_env python3 -c \
  "import tada_stability.score_ddg; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 4: Commit, by explicit path only**

The working tree has many unrelated modified files (submodule pointers, `domain-insertion/`, `docs/presentations/`). Stage nothing but these six.

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git add tada-stability/tada_stability/relax_scaffolds.py \
        tada-stability/tada_stability/score_ddg.py \
        tada-stability/tada_stability/tests/test_relax_scaffolds.py \
        tada-stability/tada_stability/tests/test_score_ddg.py \
        tada-stability/score_ddg.bsub \
        tada-stability/_debug_ddg_m70l.bsub
git diff --cached --stat
```

Expected: exactly six files staged. Then:

```bash
git commit -F - <<'EOF'
feat(tada-stability): in-process PyRosetta ddG with a common-reference local relax

Replaces cartesian_ddg with ddg_local_round(): both arms clone ONE relaxed
reference pose and descend by gradient (PackRotamers over a mobile sphere +
cartesian LBFGS MinMover), so the bulk energy cancels and the paired
difference is the mutation's local effect.

Two measured predecessors this fixes:
  - whole-protein independent relax: M70L = +22 kcal/mol (global relaxation
    noise dominated the subtraction)
  - local sphere, independent FastRelax: M70L mean +1.46 kcal/mol but
    sd 51.75 REU (~17.6 kcal/mol) -- the mean was luck

Debug gate (job 311454316, 31 s wall): M70L = -0.525 REU = -0.18 kcal/mol,
sd 0.0 over 3 iterations; both WT and MUT poses passed check_zn_geometry.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: Teach `tools/esmfold2/fold.py` to hold the Zn

ESMFold2's local wrapper folds protein chains only. The campaign needs holo folds so the Zn-geometry check applies to all designs, not just the AF3 subset. `esm.models.esmfold2` exports `LigandInput(id, smiles, ccd)` and `DNAInput(id, sequence)`, and `StructurePredictionInput.sequences` accepts them alongside `ProteinInput` (verified 2026-08-05 by reading `esm/utils/structure/input_builder.py`).

**Files:**
- Modify: `tools/esmfold2/fold.py`
- Create: `tools/esmfold2/tests/test_fold_inputs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fold.next_chain_id(used: set[str]) -> str`, `fold.parse_dna_spec(spec: str) -> tuple[str, str]`, and `fold.build_inputs(records: list[tuple[str, str]], ligand_ccds: list[str], dna_specs: list[str]) -> list` returning a list of `ProteinInput`/`LigandInput`/`DNAInput` in that order. New CLI flags `--ligand-ccd` (repeatable) and `--dna CHAIN:SEQ` (repeatable).

- [ ] **Step 1: Write the failing tests**

Create `tools/esmfold2/tests/test_fold_inputs.py`. These are GPU-free: they import the input classes but never load the model.

```python
"""Input-building tests for tools/esmfold2/fold.py -- no GPU, no model load."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fold  # noqa: E402


def test_next_chain_id_skips_used_letters():
    assert fold.next_chain_id({"A"}) == "B"
    assert fold.next_chain_id({"A", "B", "D"}) == "C"


def test_next_chain_id_raises_when_exhausted():
    with pytest.raises(ValueError):
        fold.next_chain_id(set(chr(c) for c in range(ord("A"), ord("Z") + 1)))


def test_parse_dna_spec_splits_chain_and_sequence():
    assert fold.parse_dna_spec("D:ACGT") == ("D", "ACGT")


def test_parse_dna_spec_rejects_a_missing_colon():
    with pytest.raises(ValueError):
        fold.parse_dna_spec("ACGT")


def test_build_inputs_protein_only_is_unchanged():
    seqs = fold.build_inputs([("A", "MKV")], [], [])
    assert len(seqs) == 1
    assert seqs[0].id == "A"
    assert seqs[0].sequence == "MKV"


def test_build_inputs_appends_a_zn_ligand_on_a_fresh_chain():
    seqs = fold.build_inputs([("A", "MKV")], ["ZN"], [])
    assert len(seqs) == 2
    assert seqs[1].ccd == ["ZN"]
    assert seqs[1].id == "B"          # must not collide with the protein chain


def test_build_inputs_keeps_the_requested_dna_chain_id():
    seqs = fold.build_inputs([("F", "MKV")], ["ZN"], ["D:ACGT"])
    kinds = [type(s).__name__ for s in seqs]
    assert kinds == ["ProteinInput", "LigandInput", "DNAInput"]
    assert seqs[2].id == "D"
    assert seqs[2].sequence == "ACGT"


def test_build_inputs_rejects_a_dna_chain_that_collides_with_a_protein_chain():
    with pytest.raises(ValueError):
        fold.build_inputs([("D", "MKV")], [], ["D:ACGT"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
conda run -n esmfold2 python -m pytest tools/esmfold2/tests/test_fold_inputs.py -v
```

Expected: FAIL — `AttributeError: module 'fold' has no attribute 'next_chain_id'`.

- [ ] **Step 3: Implement the input builder in `fold.py`**

Add these three functions above `main()`:

```python
_CHAIN_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def next_chain_id(used):
    """First unused single-letter chain id, so an added ligand/DNA chain can
    never silently collide with a protein chain from the FASTA."""
    for letter in _CHAIN_LETTERS:
        if letter not in used:
            return letter
    raise ValueError("no single-letter chain ids left")


def parse_dna_spec(spec):
    """`CHAIN:SEQUENCE` -> `(chain, sequence)`."""
    chain, sep, seq = spec.partition(":")
    if not sep or not chain or not seq:
        raise ValueError(f"--dna expects CHAIN:SEQUENCE, got {spec!r}")
    return chain, seq


def build_inputs(records, ligand_ccds, dna_specs):
    """Protein chains, then one ligand chain per CCD code, then DNA chains.

    Ligand chain ids are auto-assigned around the ids already in use; DNA
    chain ids are taken from the caller (the campaign needs chain D to stay
    chain D so its numbering matches 6VPC).
    """
    from esm.models.esmfold2 import DNAInput, LigandInput, ProteinInput

    seqs = [ProteinInput(id=c, sequence=s) for c, s in records]
    used = {c for c, _ in records}
    for ccd in ligand_ccds:
        cid = next_chain_id(used)
        used.add(cid)
        seqs.append(LigandInput(id=cid, ccd=[ccd]))
    for spec in dna_specs:
        cid, seq = parse_dna_spec(spec)
        if cid in used:
            raise ValueError(f"--dna chain {cid!r} collides with an existing chain")
        used.add(cid)
        seqs.append(DNAInput(id=cid, sequence=seq))
    return seqs
```

- [ ] **Step 4: Wire the flags into `main()`**

Add the two arguments next to the existing sampling flags:

```python
    ap.add_argument("--ligand-ccd", action="append", default=[],
                    help="CCD code to fold alongside the protein (repeatable), "
                         "e.g. --ligand-ccd ZN. Each gets its own chain.")
    ap.add_argument("--dna", action="append", default=[],
                    help="DNA chain as CHAIN:SEQUENCE (repeatable), e.g. --dna D:ACGT")
```

Replace the existing `spi = StructurePredictionInput(...)` construction with:

```python
    spi = StructurePredictionInput(
        sequences=build_inputs(records, args.ligand_ccd, args.dna)
    )
```

Delete the now-unused `ProteinInput` name from the top-level import inside `main()` if it becomes unreferenced there — `build_inputs` imports what it needs. Keep `StructurePredictionInput` and `ESMFold2InputBuilder` imported in `main()`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
conda run -n esmfold2 python -m pytest tools/esmfold2/tests/test_fold_inputs.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Verify with one real debug fold (GPU, cheap)**

The API acceptance of `LigandInput` is confirmed statically only. Fold one short sequence with the Zn at reduced sampling. This is a single cheap invocation, which CLAUDE.md's debug gate explicitly allows.

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
mkdir -p tools/esmfold2/debug
cat > tools/esmfold2/debug/zn_probe.fasta <<'EOF'
>F
MSEVEFSHEYWMRHALTLAKRAWDEREVPVGAVLVHNNRVIGEGWNRPIGRHDPTAHAEIMALRQGG
EOF
sbatch --wait --partition=gpu --gres=gpu:1 -c 4 --mem=32G -t 00:40:00 \
  -o tools/esmfold2/debug/zn_probe.out -e tools/esmfold2/debug/zn_probe.err \
  --wrap "source /research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3/bin/activate esmfold2 && python /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tools/esmfold2/fold.py /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tools/esmfold2/debug/zn_probe.fasta -o /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tools/esmfold2/debug/zn_probe.cif --ligand-ccd ZN --num-loops 4 --num-sampling-steps 20"
cat tools/esmfold2/debug/zn_probe.out
grep -c "^ATOM\|^HETATM" tools/esmfold2/debug/zn_probe.cif 2>/dev/null || \
  python3 -c "print(open('tools/esmfold2/debug/zn_probe.cif').read().count('ZN'))"
```

Expected: exit 0, a valid mmCIF written, and at least one `ZN` atom present in it.

**If the fold fails on the ligand input**, capture the exact traceback and stop — do not work around it. The whole holo-fold decision rests on this working, and a silent fallback to apo would invalidate the ESMFold2↔AF3 comparison the campaign is built on.

- [ ] **Step 7: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git add tools/esmfold2/fold.py tools/esmfold2/tests/test_fold_inputs.py
git commit -F - <<'EOF'
feat(esmfold2): fold ligand (Zn) and DNA chains alongside protein

fold.py handled protein chains only, so a metalloenzyme could only be folded
apo. esm.models.esmfold2 exports LigandInput(id, smiles, ccd) and
DNAInput(id, sequence), both accepted by StructurePredictionInput.sequences.

Adds --ligand-ccd (repeatable) and --dna CHAIN:SEQUENCE (repeatable), with
input construction factored into build_inputs() so chain-id assignment and
spec parsing are testable without a GPU. Ligand chain ids are auto-assigned
around the FASTA's ids; DNA chain ids come from the caller so chain D can
stay chain D and keep 6VPC numbering.

Verified by a real reduced-sampling fold with --ligand-ccd ZN.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: Create the `tada-redesign` submodule and push it

**Files:**
- Create (in a scratch clone, then pushed): `README.md`, `CLAUDE.md`, `requirements.md`, `.gitignore`, `pytest.ini`, `tada_redesign/__init__.py`, `tada_redesign/tests/__init__.py`
- Modify: `~/.ssh/config` (add an `ssh.github.com` host block)
- Create: `.gitmodules` + the `tada-redesign` pointer in the monorepo

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `tada_redesign` package at `<monorepo>/tada-redesign/tada_redesign/`, and a working `git push` path for the submodule.

- [ ] **Step 1: Add the SSH host block for port 443**

Port 22 to github.com is blocked from this cluster; `ssh.github.com:443` works. Append to `~/.ssh/config`:

```
Host ssh.github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_id_ed25519
    IdentitiesOnly yes
    PKCS11Provider none
```

Verify:

```bash
ssh -T -o ConnectTimeout=20 git@ssh.github.com 2>&1 | head -2
```

Expected: `Hi ecreed-work! You've successfully authenticated, but GitHub does not provide shell access.`

- [ ] **Step 2: Build the repo skeleton in a scratch directory**

Build it outside the monorepo first, push it, then add it back as a submodule — `git submodule add` needs a remote that already has a branch.

Shell variables do NOT persist between separate command invocations, so every step below uses the absolute path literally rather than `$SCRATCH`.

```bash
mkdir -p /tmp/claude-97545/-research-rgs01-home-clusterHome-ecreed-claude-proteindesign/c5117087-affd-40d5-b792-99bfb8feb2b2/scratchpad/TadA-redesign/tada_redesign/tests
git -C /tmp/claude-97545/-research-rgs01-home-clusterHome-ecreed-claude-proteindesign/c5117087-affd-40d5-b792-99bfb8feb2b2/scratchpad/TadA-redesign init -b main
```

Write `.gitignore`:

```
outputs/
logs/
__pycache__/
*.pyc
.pytest_cache/
```

Write `pytest.ini`:

```ini
[pytest]
testpaths = tada_redesign/tests
python_files = test_*.py
```

Write `tada_redesign/__init__.py` and `tada_redesign/tests/__init__.py` as empty files.

Write `README.md`:

```markdown
# TadA-redesign

Diffusion-based stabilization of the ABE8e/ABE9 TadA deaminase domain:
motif-frozen RFdiffusion3 partial diffusion, Zn-aware LigandMPNN sequence
design, orthogonal ESMFold2 + AlphaFold3 validation, and PyRosetta
absolute-score ranking against the identically-processed parents.

Design spec: `docs/superpowers/specs/2026-08-05-tada-redesign-design.md` in the
parent monorepo.

## Honesty ceiling

Every metric here measures **structural plausibility and energetic ranking
within a fold family** — not function. Rosetta `ref2015_cart` absolute score is
not a physical ΔG of folding and does not convert to Tm. pLDDT measures model
confidence, not stability. No wet-lab validation has been performed, and
adenine deaminase activity is not demonstrated by any number in this
repository.

## Layout

- `tada_redesign/` — the pipeline package
- `outputs/` — run directories, `YYYYMMDD_` prefixed (gitignored)

## Environment

See `requirements.md`. This repo is a submodule of a monorepo and resolves
shared TadA assets (`masks.json`, reference PDBs, `tada_stability` modules)
through the `TADA_MONOREPO` environment variable.
```

Write `CLAUDE.md`:

```markdown
# CLAUDE.md — tada-redesign

Submodule of `claude-proteindesign`. Read the parent repo's `CLAUDE.md` first;
its rules (plans/logs, the pre-job checklist, `sbatch` not `bsub`, dated output
dirs, absolute paths) all apply here.

## Submodule-specific rules

- Commit **here first**, then update the parent repo's pointer.
- The remote must use `ssh://git@ssh.github.com:443/...` — github.com port 22
  and HTTPS are both blocked from this cluster.
- Cross-repo assets resolve through `TADA_MONOREPO`; never hardcode a path
  into the parent repo outside `constants.py`.
- `masks.json`'s `FROZEN` key is NOT this campaign's motif (it is 36 residues
  and includes `ZN_PROXIMITY`). Use `motif.arm_residues()`.
- Tests run in the `ligandmpnn_env` conda env.
```

Write `requirements.md`:

```markdown
# Environment requirements — tada-redesign

Split across conda envs on purpose; this cluster has a history of envs being
rebuilt out from under a project. Update this file whenever a stage adds a
dependency.

| Env | Location | Used for | Needs |
|---|---|---|---|
| `ligandmpnn_env` | `~/.conda/envs` | the test suite, all pure-Python modules | numpy, Biopython (`Bio.PDB`), pytest |
| `cas9-pam-design` | shared miniforge3 | RFD3 partial diffusion (`rfd3 design`) | rfd3 / foundry |
| `ligandmpnn-sc` | shared miniforge3 | LigandMPNN sequence design | ml_collections, prody, torch |
| `esmfold2` | shared miniforge3 | ESMFold2 folding | Biohub `transformers` fork, torch |
| `pyrosetta` | shared miniforge3 | Zn-constrained relax + `ref2015_cart` | pyrosetta, Biopython |

AF3 runs from a singularity image, not a conda env:
`/hpcf/authorized_apps/rhel8_apps/alphafold/3.0.2/alphafold.3.0.2.sif`.

`Bio.PDB.ShrakeRupley` is broken in `ligandmpnn_env` (uses the removed `np.int`
alias); `tada_stability.sasa` vendors its own Shrake-Rupley and is used instead.
```

- [ ] **Step 3: Commit and push the skeleton**

```bash
cd /tmp/claude-97545/-research-rgs01-home-clusterHome-ecreed-claude-proteindesign/c5117087-affd-40d5-b792-99bfb8feb2b2/scratchpad/TadA-redesign
git add -A
git -c user.name="Ethan Creed" -c user.email="ethan.creed@stjude.org" commit -F - <<'EOF'
chore: repository skeleton for the TadA redesign campaign

Package layout, pytest config, environment inventory, and the honesty ceiling
that governs every metric this pipeline reports.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git remote add origin ssh://git@ssh.github.com:443/ecreed-work/TadA-redesign.git
git push -u origin main
```

Expected: the push succeeds and reports `main -> main`. The repo was verified empty on 2026-08-05, so there is nothing to overwrite. **If the push is rejected as non-fast-forward, stop** — that means the repo is no longer empty, and clobbering someone else's commit is not this task's call.

- [ ] **Step 4: Add it to the monorepo as a submodule**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git submodule add ssh://git@ssh.github.com:443/ecreed-work/TadA-redesign.git tada-redesign
git submodule status tada-redesign
cat .gitmodules | tail -5
```

Expected: `tada-redesign` checked out at the pushed commit, and a `.gitmodules` entry whose URL includes `:443`.

- [ ] **Step 5: Verify the package imports from the submodule path**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python3 -c "import tada_redesign; print('import OK')"
```

Expected: `import OK`.

- [ ] **Step 6: Commit the pointer in the monorepo**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git add .gitmodules tada-redesign
git diff --cached --stat
git commit -F - <<'EOF'
feat(tada-redesign): add the campaign submodule

tada-redesign/ tracks ecreed-work/TadA-redesign. Its remote pins
ssh://git@ssh.github.com:443 because github.com is unreachable from this
cluster on both port 22 and HTTPS.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 4: `constants.py`

Every path, threshold, and sweep axis in one place, with no logic. Later tasks import from here rather than repeating a literal.

**Files:**
- Create: `tada-redesign/tada_redesign/constants.py`
- Test: `tada-redesign/tada_redesign/tests/test_constants.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MONOREPO`, `MASKS_JSON`, `REFERENCE_DIR`, `PARENT_PDB` (dict parent→path), `CHAINF_RAW`, `PDB6VPC`, `KNOWN_BAD_PDB`, `SCAFFOLD_CHAIN`, `ZN_RESNAME`, `SUBSTRATE_CHAIN`, `SUBSTRATE_RESID`, `SUBSTRATE_RESNAME`, `PARENTS`, `ARMS`, `PARTIAL_T`, `MPNN_TEMPS`, `SEQS_PER_TEMP`, `BACKBONES_PER_CELL`, `RFD_N_BATCHES`, `RFD_BATCH_SIZE`, `CONTROL_TEMP`, `CONTROL_SEQS_PER_BACKBONE`, `BACKBONE_MOTIF_RMSD_MAX`, `CA_BREAK_MAX`, `LENGTH_RANGE`, `ZN_DONOR_RANGE`, `SCREEN_PLDDT_MARGIN`, `SCREEN_MOTIF_RMSD_MAX`, `FINAL_MOTIF_RMSD_MAX`, `SCREEN_SURVIVORS`, `AF3_TOP_N`, `AF3_CONTROL_N`, `ROSETTA_REPLICATES`, `DEGRADED_FRACTION`, `CLEFT_CLEARANCE_MIN`, `ESMFOLD_SCREEN`, `ESMFOLD_FULL`, `RUN_DIR_NAME`, `ENV_*`, `RFD3_CKPT`, `LIGANDMPNN_CKPT`, `AF3_SIF`, `AF3_DB`, `ESMFOLD_HF_CACHE`, `n_designs()`, `n_control_designs()`

- [ ] **Step 1: Write the failing test**

Create `tada-redesign/tada_redesign/tests/test_constants.py`:

```python
"""Constants are load-bearing: several encode a measured cluster or tool fact
that a plausible-looking edit would silently break."""
import importlib
import os

from tada_redesign import constants


def test_monorepo_defaults_to_the_known_root():
    assert constants.MONOREPO.endswith("claude-proteindesign")


def test_monorepo_honours_the_env_override(monkeypatch):
    monkeypatch.setenv("TADA_MONOREPO", "/somewhere/else")
    reloaded = importlib.reload(constants)
    try:
        assert reloaded.MONOREPO == "/somewhere/else"
        assert reloaded.MASKS_JSON.startswith("/somewhere/else")
    finally:
        monkeypatch.delenv("TADA_MONOREPO")
        importlib.reload(constants)


def test_both_parents_have_a_reference_pdb():
    assert set(constants.PARENT_PDB) == set(constants.PARENTS)
    for path in constants.PARENT_PDB.values():
        assert path.endswith(".pdb")


def test_partial_t_ladder_stays_below_the_measured_cliff():
    # denovo_tada's 2026-08-04 debug gate measured partial_t=8 A degrading the
    # TadA active site to 2.5-6.75 A RMSD (from ~1.1 A seeds), failing a 1.5 A
    # gate on every design. Nothing in this ladder may reach 8.
    assert max(constants.PARTIAL_T) < 8.0
    assert min(constants.PARTIAL_T) > 0.0


def test_design_counts_match_the_spec():
    assert constants.n_designs() == 10240
    assert constants.n_control_designs() == 512
    assert constants.n_designs() + constants.n_control_designs() == 10752


def test_rfd_batch_product_equals_backbones_per_cell():
    assert constants.RFD_N_BATCHES * constants.RFD_BATCH_SIZE == \
        constants.BACKBONES_PER_CELL


def test_screen_rmsd_gate_is_looser_than_the_final_gate():
    # Reduced-sampling folds are noisier; a screen gate tighter than the final
    # one would discard designs the full-sampling fold would have kept.
    assert constants.SCREEN_MOTIF_RMSD_MAX > constants.FINAL_MOTIF_RMSD_MAX


def test_esmfold_screen_is_cheaper_than_full():
    assert constants.ESMFOLD_SCREEN["num_loops"] < constants.ESMFOLD_FULL["num_loops"]
    assert constants.ESMFOLD_SCREEN["num_sampling_steps"] < \
        constants.ESMFOLD_FULL["num_sampling_steps"]


def test_known_bad_pdb_is_recorded_so_preflight_can_forbid_it():
    assert "6vpc_dCas9_TadA8e.pdb" in os.path.basename(constants.KNOWN_BAD_PDB)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_constants.py -v
```

Expected: FAIL — `ImportError: cannot import name 'constants' from 'tada_redesign'`.

- [ ] **Step 3: Write `constants.py`**

```python
"""Paths, sweep axes, and gate thresholds for the TadA redesign campaign.

No logic beyond the two design-count helpers. Several values encode a MEASURED
tool or cluster fact and carry that citation inline -- do not "tidy" them.

Residue numbering is Met = 1 (UniProt P68398) throughout.
"""
import os

# ---------------------------------------------------------------- cross-repo
# This package is a submodule; the shared TadA assets live in the parent
# monorepo. Resolved through an env var, mirroring the `R=` pattern the
# existing SLURM scripts already use.
MONOREPO = os.environ.get(
    "TADA_MONOREPO",
    "/research/rgs01/home/clusterHome/ecreed/claude-proteindesign")

TADA_STABILITY = os.path.join(MONOREPO, "tada-stability")
MASKS_JSON = os.path.join(
    TADA_STABILITY, "outputs/20260728_tada_stability/masks.json")
REFERENCE_DIR = os.path.join(TADA_STABILITY, "reference")

PARENTS = ("TadA8e", "TadA9")
PARENT_PDB = {p: os.path.join(REFERENCE_DIR, f"{p}.pdb") for p in PARENTS}
CHAINF_RAW = os.path.join(REFERENCE_DIR, "chainF_raw.pdb")

PDB6VPC = os.path.join(
    MONOREPO,
    "tada8e-cas9-Interface-design/structural_analysis/structures/"
    "deaminase/pdb6vpc.ent")

# Chains B and E only, with the Zn STRIPPED. Any ddG or MPNN run from this file
# would model an apo, partner-less deaminase. preflight asserts it is
# referenced nowhere in this package.
KNOWN_BAD_PDB = os.path.join(
    MONOREPO,
    "tada8e-cas9-Interface-design/interface_design/abe_tadA_ruvc/inputs/"
    "6vpc_dCas9_TadA8e.pdb")

# ------------------------------------------------------------------ structure
SCAFFOLD_CHAIN = "F"        # 6VPC's catalytic protomer: its Zn is 2.12 A from
                            # the 8AZ mimic vs 17.3 A for chain E
ZN_RESNAME = "ZN"           # RFD3 keys the ion by CCD name, never chain+resid
SUBSTRATE_CHAIN = "D"
SUBSTRATE_RESID = 26
SUBSTRATE_RESNAME = "8AZ"   # 8-azanebularine target-base analogue

# ---------------------------------------------------------------- sweep axes
ARMS = ("FULL", "MIN")

# Angstroms of re-noise (RFD3 partial diffusion; the tool recommends <=15).
# denovo_tada's 2026-08-04 debug gate measured partial_t=8 degrading the TadA
# active site from ~1.1 A to 2.5-6.75 A RMSD, failing a 1.5 A gate on EVERY
# design. 6 is therefore the deliberate high end of a survivable range; its
# yield is expected to be poor, and that is a result to report.
PARTIAL_T = (1.0, 2.0, 4.0, 6.0)

MPNN_TEMPS = (0.1, 0.15, 0.2, 0.3)
SEQS_PER_TEMP = 5
RFD_N_BATCHES = 4
RFD_BATCH_SIZE = 8
BACKBONES_PER_CELL = 32

# Solubility biasing is an assumption, not a known good, so it gets a control:
# one unbiased sequence per backbone at a single mid temperature.
CONTROL_TEMP = 0.15
CONTROL_SEQS_PER_BACKBONE = 1

# ---------------------------------------------------------------- gates
BACKBONE_MOTIF_RMSD_MAX = 1.0    # A; RFD3 does not perfectly honour fixed atoms
CA_BREAK_MAX = 4.2               # A between consecutive CA
LENGTH_RANGE = (150, 175)        # residues
ZN_DONOR_RANGE = (2.0, 2.6)      # A, Zn to each of its three donors

SCREEN_PLDDT_MARGIN = 0.05       # 0-1 scale (ESMFold2 reports 0-1, not 0-100)
SCREEN_MOTIF_RMSD_MAX = 1.5      # A, looser: reduced sampling is noisier
FINAL_MOTIF_RMSD_MAX = 1.0       # A, full sampling / AF3
SCREEN_SURVIVORS = 2000          # compute decision; the dropped count is logged
CLEFT_CLEARANCE_MIN = 2.2        # A from any mapped 8AZ atom to any design
                                 # heavy atom; below this the cleft has closed

AF3_TOP_N = 200
AF3_CONTROL_N = 100              # stratified across the full Rosetta range,
                                 # rejects included, so the ESMFold2<->AF3
                                 # correlation is not range-restricted
ROSETTA_REPLICATES = 3

# A stage failing on more than this fraction of its inputs refuses the
# canonical output path and writes *.degraded.tsv instead.
DEGRADED_FRACTION = 0.20

# --------------------------------------------------------------- fold budgets
# Screen settings are the ones verified working in the 2026-08-04 debug fold.
# Reduced sampling depresses pLDDT substantially (that fold returned 0.45 on a
# 78-mer), which is why the parent is folded in the IDENTICAL mode and the gate
# is relative to it.
ESMFOLD_SCREEN = {"num_loops": 4, "num_sampling_steps": 20}
ESMFOLD_FULL = {"num_loops": 20, "num_sampling_steps": 100}

# ------------------------------------------------------------------- runtime
RUN_DIR_NAME = "20260805_tada_redesign"

ENV_TEST = "ligandmpnn_env"
ENV_RFD3 = "cas9-pam-design"
ENV_MPNN = "ligandmpnn-sc"
ENV_ESM = "esmfold2"
ENV_ROSETTA = "pyrosetta"

RFD3_CKPT = ("/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/"
             "common/claude/foundry_ckpt/rfd3_latest.ckpt")
LIGANDMPNN_CKPT = os.path.join(
    MONOREPO, "design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
AF3_SIF = "/hpcf/authorized_apps/rhel8_apps/alphafold/3.0.2/alphafold.3.0.2.sif"
AF3_DB = "/lustre_scratch/reference/public/alphafold_data/af3"
ESMFOLD_HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--biohub--ESMFold2")
ESMFOLD_FOLD_PY = os.path.join(MONOREPO, "tools/esmfold2/fold.py")


def n_designs():
    """Biased designs: cells x backbones x temperatures x sequences."""
    return (len(PARENTS) * len(ARMS) * len(PARTIAL_T) * BACKBONES_PER_CELL
            * len(MPNN_TEMPS) * SEQS_PER_TEMP)


def n_control_designs():
    """Zero-bias control designs: one per backbone at CONTROL_TEMP."""
    return (len(PARENTS) * len(ARMS) * len(PARTIAL_T) * BACKBONES_PER_CELL
            * CONTROL_SEQS_PER_BACKBONE)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_constants.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit (in the submodule)**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/constants.py tada_redesign/tests/test_constants.py
git commit -F - <<'EOF'
feat: campaign constants, paths, sweep axes and gate thresholds

Single home for every path, threshold and sweep value. Tests lock the facts a
plausible edit would break: the partial_t ladder staying below the measured
8-A active-site cliff, the screen RMSD gate staying looser than the final one,
the design counts matching the spec (10240 + 512 control), and TADA_MONOREPO
overriding correctly.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 5: `motif.py` — one motif definition, three renderers

**Files:**
- Create: `tada-redesign/tada_redesign/motif.py`
- Test: `tada-redesign/tada_redesign/tests/test_motif.py`

**Interfaces:**
- Consumes: `constants.MASKS_JSON`, `constants.SCAFFOLD_CHAIN`, `constants.ZN_RESNAME`.
- Produces: `ARM_FULL = "FULL"`, `ARM_MIN = "MIN"`, `FIXED_ATOM_KEYWORD = "ALL"`, `load_masks(path=None) -> dict[str, list[int]]`, `arm_residues(arm: str, masks: dict) -> tuple[int, ...]`, `rfd_select_fixed_atoms(arm, masks, chain=None, ligand=None) -> dict[str, str]`, `mpnn_fixed_residues(arm, masks, chain=None) -> str`, `measured_residues(arm, masks) -> tuple[int, ...]`.

- [ ] **Step 1: Write the failing test**

Create `tada-redesign/tada_redesign/tests/test_motif.py`:

```python
"""The motif set is the campaign's single source of truth. Its predecessor's
worst defects came from a generator and a gate disagreeing about one residue
set, so these tests assert the three renderers cannot diverge."""
import pytest

from tada_redesign import constants, motif

# Locked 2026-08-05 by computing (CATALYTIC | POCKET | DNA_FACE) & MODELED
# against the on-disk masks.json.
FULL_EXPECTED = (28, 30, 46, 54, 57, 58, 59, 84, 85, 86, 87, 88, 90,
                 108, 109, 110, 111, 148, 149, 152, 153, 154, 156, 157)
TETRAD = (57, 59, 87, 90)


@pytest.fixture
def masks():
    return motif.load_masks()


def test_full_arm_is_the_locked_24_residues(masks):
    assert motif.arm_residues(motif.ARM_FULL, masks) == FULL_EXPECTED
    assert len(FULL_EXPECTED) == 24


def test_min_arm_is_the_catalytic_tetrad(masks):
    assert motif.arm_residues(motif.ARM_MIN, masks) == TETRAD


def test_full_arm_is_not_masks_json_frozen(masks):
    """masks.json's own FROZEN key is 36 residues and includes ZN_PROXIMITY --
    it exists to keep cartesian_ddg away from the metal site, a failure mode
    diffusion + LigandMPNN do not exhibit. Reading it here would silently
    freeze 12 extra designable positions."""
    assert len(masks["FROZEN"]) == 36
    assert set(motif.arm_residues(motif.ARM_FULL, masks)) != set(masks["FROZEN"])


def test_all_three_renderers_agree_on_one_residue_set(masks):
    for arm in constants.ARMS:
        residues = set(motif.arm_residues(arm, masks))
        rfd = motif.rfd_select_fixed_atoms(arm, masks)
        rfd_residues = {int(k[1:]) for k in rfd if k != constants.ZN_RESNAME}
        mpnn_residues = {int(tok[1:])
                         for tok in motif.mpnn_fixed_residues(arm, masks).split()}
        assert rfd_residues == residues
        assert mpnn_residues == residues
        assert set(motif.measured_residues(arm, masks)) == residues


def test_zn_is_keyed_by_ccd_name_not_chain_resid(masks):
    """AtomWorks renames a hetero atom's chain when it shares a chain letter
    with protein residues, so a "F201" key can never resolve post-load -- a
    real RFD3 ValidationError, not a precaution."""
    rfd = motif.rfd_select_fixed_atoms(motif.ARM_FULL, masks)
    assert rfd[constants.ZN_RESNAME] == motif.FIXED_ATOM_KEYWORD
    assert "F201" not in rfd


def test_every_motif_residue_is_fixed_backbone_and_sidechain(masks):
    """ALL, not TIP: tada-stability commit 132509f only passed
    check_zn_geometry once the donors' BACKBONE was frozen too."""
    rfd = motif.rfd_select_fixed_atoms(motif.ARM_FULL, masks)
    assert set(rfd.values()) == {"ALL"}


def test_mpnn_fixed_residues_uses_the_scaffold_chain(masks):
    tokens = motif.mpnn_fixed_residues(motif.ARM_MIN, masks).split()
    assert tokens == [f"{constants.SCAFFOLD_CHAIN}{r}" for r in TETRAD]


def test_unknown_arm_raises(masks):
    with pytest.raises(ValueError):
        motif.arm_residues("EVERYTHING", masks)


def test_arm_residues_are_all_modeled(masks):
    modeled = set(masks["MODELED"])
    for arm in constants.ARMS:
        assert set(motif.arm_residues(arm, masks)) <= modeled
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_motif.py -v
```

Expected: FAIL — `ImportError: cannot import name 'motif' from 'tada_redesign'`.

- [ ] **Step 3: Write `motif.py`**

```python
"""Single source of truth for the frozen active-site motif.

Emits the SAME residue set in the three formats its consumers need: RFD3
`select_fixed_atoms`, LigandMPNN `--fixed_residues`, and the structural
scorer's measured-residue list. The predecessor campaign's worst defects came
from a generator and a gate disagreeing about one set (see
docs/logs/20260728_tada_stabilization.md, fix rounds 1-2), so there is exactly
one definition here and three renderers over it.

TRAP -- masks.json's own "FROZEN" key is NOT this campaign's motif. There it
means CATALYTIC | POCKET | DNA_FACE | ZN_PROXIMITY (36 residues) and exists to
keep `cartesian_ddg` away from the metal site. Diffusion + LigandMPNN do not
exhibit that failure, so ZN_PROXIMITY is deliberately designable here and the
FULL arm is CATALYTIC | POCKET | DNA_FACE intersected with MODELED (24).

Honesty ceiling: freezing a residue set constrains geometry and identity. It
does not make the resulting protein catalytically active, and nothing in this
module measures function.
"""
import json

from . import constants

ARM_FULL = "FULL"
ARM_MIN = "MIN"

# Every motif residue is fixed with the ALL keyword (backbone + sidechain),
# never TIP. Freeing the Zn donors' backbone is MEASURED to let the metal
# migrate: tada-stability commit 132509f ("freeze donor backbone too -- both
# parents pass check_zn_geometry") only went green once the donor backbone was
# frozen as well.
FIXED_ATOM_KEYWORD = "ALL"

_ARM_MASKS = {
    ARM_FULL: ("CATALYTIC", "POCKET", "DNA_FACE"),
    ARM_MIN: ("CATALYTIC",),
}


def load_masks(path=None):
    """The `masks` block of masks.json: {mask_name: [resnum, ...]}."""
    with open(path or constants.MASKS_JSON) as fh:
        return json.load(fh)["masks"]


def arm_residues(arm, masks):
    """Sorted residue numbers frozen by `arm`, intersected with MODELED.

    The intersection matters: EVOLVED positions 166/167 and the disordered
    termini are outside chain F's modelled 5-160 span and can never be frozen
    or designed.
    """
    try:
        names = _ARM_MASKS[arm]
    except KeyError:
        raise ValueError(
            f"unknown arm {arm!r}; expected one of {tuple(_ARM_MASKS)}") from None
    selected = set()
    for name in names:
        selected |= set(masks[name])
    return tuple(sorted(selected & set(masks["MODELED"])))


def rfd_select_fixed_atoms(arm, masks, chain=None, ligand=None):
    """RFD3 `select_fixed_atoms` mapping: {"F57": "ALL", ..., "ZN": "ALL"}.

    The ion is keyed by CCD NAME, not chain+resid. RFD3's AtomWorks loader
    renames a hetero atom's chain when it shares a chain letter with protein
    residues (our Zn is `HETATM ZN F 201`), so a "F201" key raises
    `ValidationError: [component=F201] Residue F201 not found in atom array` --
    confirmed by a real RFD3 run. `fetch_mask_from_name` matches res_name
    chain-independently and is tried first, so a CCD key resolves regardless.
    """
    chain = chain or constants.SCAFFOLD_CHAIN
    ligand = ligand or constants.ZN_RESNAME
    fixed = {f"{chain}{resnum}": FIXED_ATOM_KEYWORD
             for resnum in arm_residues(arm, masks)}
    fixed[ligand] = FIXED_ATOM_KEYWORD
    return fixed


def mpnn_fixed_residues(arm, masks, chain=None):
    """LigandMPNN `--fixed_residues` string: space-separated "F57 F59 ..."."""
    chain = chain or constants.SCAFFOLD_CHAIN
    return " ".join(f"{chain}{resnum}"
                    for resnum in arm_residues(arm, masks))


def measured_residues(arm, masks):
    """Residues whose heavy atoms the structural scorer measures.

    Identical to the frozen set by construction: the gate must measure exactly
    what the generators were told to hold, or a divergence between the two
    becomes invisible.
    """
    return arm_residues(arm, masks)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_motif.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/motif.py tada_redesign/tests/test_motif.py
git commit -F - <<'EOF'
feat: single-source frozen-motif definition with three renderers

One definition of the frozen active site, rendered into RFD3
select_fixed_atoms, LigandMPNN --fixed_residues, and the scorer's measured
residue list. A test asserts the three cannot diverge -- the predecessor
campaign's worst defects were a generator and a gate disagreeing about one set.

Also guards two measured traps: masks.json's FROZEN key is 36 residues
(includes ZN_PROXIMITY) and is NOT this campaign's motif; and the Zn must be
keyed by CCD name because AtomWorks renames its chain on load.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: `score_structure.py` — the shared geometric scorer

Used by both the ESMFold2 and AF3 stages, so there is one implementation of superposition and RMSD, not two.

**Files:**
- Create: `tada-redesign/tada_redesign/score_structure.py`
- Test: `tada-redesign/tada_redesign/tests/test_score_structure.py`

**Interfaces:**
- Consumes: `constants.SCAFFOLD_CHAIN`.
- Produces:
  - `kabsch(P, Q) -> (R, P_mean, Q_mean)` — `(Q - Q_mean) @ R.T + P_mean` superposes `Q` onto `P`.
  - `apply_transform(X, R, P_mean, Q_mean) -> np.ndarray`
  - `heavy_atoms_from_pdb(path, chain=None) -> dict[tuple[int, str], np.ndarray]` keyed by `(resnum, atom_name)`
  - `heavy_atoms_from_cif(path, chain=None) -> dict[tuple[int, str], np.ndarray]`
  - `ca_map(atoms) -> dict[int, np.ndarray]`
  - `motif_rmsd(ref_atoms, pred_atoms, residues, anchor_residues=None) -> float`
  - `cleft_clearance(ref_atoms, pred_atoms, substrate_xyz, anchor_residues=None) -> float`

- [ ] **Step 1: Write the failing test**

Create `tada-redesign/tada_redesign/tests/test_score_structure.py`:

```python
"""Geometry tests. The two rigid-body tests are non-negotiable: their absence
is exactly how the predecessor's pocket_rmsd shipped with no superposition at
all, which would have emptied a shortlist after real GPU spend."""
import numpy as np
import pytest

from tada_redesign import score_structure as ss


def _synthetic_atoms(n_res=6, seed=0):
    """{(resnum, atom): xyz} with CA + one sidechain atom per residue."""
    rng = np.random.default_rng(seed)
    atoms = {}
    for i in range(1, n_res + 1):
        base = rng.normal(size=3) * 5.0
        atoms[(i, "CA")] = base
        atoms[(i, "CB")] = base + np.array([1.5, 0.0, 0.0])
    return atoms


def _rotation(deg=37.0):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _transformed(atoms, R, shift):
    return {k: R @ v + shift for k, v in atoms.items()}


def test_kabsch_recovers_a_known_rigid_transform():
    ref = _synthetic_atoms()
    moved = _transformed(ref, _rotation(), np.array([10.0, -4.0, 7.0]))
    keys = sorted(ref)
    P = np.array([ref[k] for k in keys])
    Q = np.array([moved[k] for k in keys])
    R, P_mean, Q_mean = ss.kabsch(P, Q)
    back = ss.apply_transform(Q, R, P_mean, Q_mean)
    assert np.allclose(back, P, atol=1e-8)


def test_motif_rmsd_is_invariant_to_rigid_body_motion():
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(51.0), np.array([-30.0, 12.0, 3.0]))
    rmsd = ss.motif_rmsd(ref, pred, residues=(2, 3, 4))
    assert rmsd == pytest.approx(0.0, abs=1e-8)


def test_motif_rmsd_reports_a_local_deviation_not_the_global_offset():
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(51.0), np.array([-30.0, 12.0, 3.0]))
    # perturb ONE measured atom by a known 0.6 A in the prediction's own frame
    R = _rotation(51.0)
    pred[(3, "CB")] = pred[(3, "CB")] + R @ np.array([0.6, 0.0, 0.0])
    # 6 measured atoms over residues 2,3,4 -> rmsd = sqrt(0.6^2 / 6)
    rmsd = ss.motif_rmsd(ref, pred, residues=(2, 3, 4))
    assert rmsd == pytest.approx(np.sqrt(0.36 / 6), abs=1e-6)


def test_motif_rmsd_raises_when_a_measured_residue_is_absent():
    ref = _synthetic_atoms()
    pred = _synthetic_atoms()
    del pred[(3, "CB")]
    with pytest.raises(KeyError):
        ss.motif_rmsd(ref, pred, residues=(2, 3, 4))


def test_motif_rmsd_raises_below_three_anchor_points():
    ref = _synthetic_atoms()
    pred = _synthetic_atoms()
    with pytest.raises(ValueError):
        ss.motif_rmsd(ref, pred, residues=(1,), anchor_residues=(1, 2))


def test_cleft_clearance_is_large_when_the_pocket_is_open():
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(20.0), np.array([4.0, 4.0, 4.0]))
    far = np.array([[100.0, 100.0, 100.0]])
    assert ss.cleft_clearance(ref, pred, far) > 50.0


def test_cleft_clearance_detects_a_closed_cleft():
    """A substrate atom placed on top of a design atom (in the ref frame) must
    come back as a sub-Angstrom clearance after superposition."""
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(20.0), np.array([4.0, 4.0, 4.0]))
    on_top = np.array([ref[(3, "CB")] + np.array([0.3, 0.0, 0.0])])
    assert ss.cleft_clearance(ref, pred, on_top) == pytest.approx(0.3, abs=1e-6)


def test_heavy_atoms_from_pdb_skips_hydrogens_and_other_chains(tmp_path):
    # These four lines are COLUMN-EXACT: chain in col 22, resSeq 23-26,
    # coords 31-54, element right-justified in 77-78. Verified by generating
    # them from a column-aware writer -- an eyeballed HETATM line put the Zn's
    # element column at 'N' instead of 'ZN', which made an earlier version of
    # this fixture pass for the wrong reason.
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA F  57       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  HA  ALA F  57       1.500   2.000   3.000  1.00  0.00           H\n"
        "ATOM      3  CA  ALA E  57       9.000   9.000   9.000  1.00  0.00           C\n"
        "HETATM    4  ZN   ZN F 201       4.000   5.000   6.000  1.00  0.00          ZN\n"
        "END\n")
    atoms = ss.heavy_atoms_from_pdb(str(pdb), chain="F")
    assert (57, "CA") in atoms
    assert (57, "HA") not in atoms          # hydrogen dropped
    assert (201, "ZN") in atoms             # hetero heavy atom kept
    assert all(k[0] in (57, 201) for k in atoms)   # chain E excluded
    assert np.allclose(atoms[(57, "CA")], [1.0, 2.0, 3.0])


def test_heavy_atoms_from_cif_reads_the_atom_site_loop(tmp_path):
    cif = tmp_path / "mini.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "ATOM   C  CA ALA F 57 1.000 2.000 3.000\n"
        "ATOM   H  HA ALA F 57 1.500 2.000 3.000\n"
        "ATOM   C  CA ALA E 57 9.000 9.000 9.000\n"
        "HETATM ZN ZN ZN  F 201 4.000 5.000 6.000\n")
    atoms = ss.heavy_atoms_from_cif(str(cif), chain="F")
    assert (57, "CA") in atoms
    assert (57, "HA") not in atoms
    assert (201, "ZN") in atoms
    assert np.allclose(atoms[(57, "CA")], [1.0, 2.0, 3.0])
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_score_structure.py -v
```

Expected: FAIL — `ImportError: cannot import name 'score_structure' from 'tada_redesign'`.

- [ ] **Step 3: Write `score_structure.py`**

```python
"""Shared geometric scorer for both folding stages.

One implementation of superposition and RMSD, called by the ESMFold2 screen,
the full-sampling re-fold, and the AF3 stage. Two independent forward passes of
any structure predictor share no global frame, so EVERY comparison here
superposes before measuring. The predecessor campaign shipped a `pocket_rmsd`
that summed raw coordinate differences with no superposition at all; at a 1.0-A
threshold that silently fails every design after the GPU spend is already
gone (docs/logs/20260728_tada_stabilization.md, fix round 2).

Heavy-atom rather than CA-only RMSD is tractable here precisely because motif
identity is LOCKED by `motif.py`, so atom names match one-to-one between
reference and design.

Honesty ceiling: an intact motif geometry in a predicted model is not evidence
of catalytic activity. Nothing in this module measures function.
"""
import numpy as np

from . import constants

_HYDROGEN = {"H", "D"}


def kabsch(P, Q):
    """Rotation and centroids such that `(Q - Q_mean) @ R.T + P_mean`
    least-squares superposes `Q` onto `P`.

    Mean-centred SVD with a reflection correction, mirroring
    `tada_stability.gate_fold._kabsch` and
    `denovo_tada/rf3_gate.py::ca_rmsd_over_resids` rather than introducing a
    third implementation of the same arithmetic.
    """
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    if len(P) != len(Q):
        raise ValueError(f"point count mismatch: {len(P)} vs {len(Q)}")
    if len(P) < 3:
        raise ValueError(f"Kabsch is ill-posed on {len(P)} points; need >= 3")
    P_mean, Q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - P_mean, Q - Q_mean
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return R, P_mean, Q_mean


def apply_transform(X, R, P_mean, Q_mean):
    return (np.asarray(X, dtype=float) - Q_mean) @ R.T + P_mean


def _element_of(atom_name, type_symbol=None):
    if type_symbol:
        return type_symbol.strip().upper()
    return atom_name.strip()[0].upper()


def heavy_atoms_from_pdb(path, chain=None):
    """{(resnum, atom_name): xyz} for one chain's non-hydrogen atoms.

    Hetero atoms (the Zn) are kept: the metal's position is a measured quantity
    for this campaign, not a decoration.
    """
    chain = chain or constants.SCAFFOLD_CHAIN
    atoms = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21] != chain:
                continue
            atom_name = line[12:16].strip()
            element = _element_of(atom_name, line[76:78] if len(line) > 77 else "")
            if element in _HYDROGEN:
                continue
            resnum = int(line[22:26])
            atoms[(resnum, atom_name)] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return atoms


def heavy_atoms_from_cif(path, chain=None):
    """Same mapping, from an mmCIF `_atom_site` loop.

    ESMFold2 and AF3 both emit mmCIF. Uses Biopython's MMCIF2Dict (as
    `denovo_tada/filter_backbones.py::_load_atom_site` does) but requires only
    the standard columns -- RFD3's custom `is_motif_atom_with_fixed_seq` column
    is absent from folding-model output.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    chain = chain or constants.SCAFFOLD_CHAIN
    d = MMCIF2Dict(path)

    def col(*candidates):
        for name in candidates:
            key = f"_atom_site.{name}"
            if key in d:
                return d[key]
        raise KeyError(f"mmCIF {path} has none of {candidates}")

    chains = col("auth_asym_id", "label_asym_id")
    resids = col("auth_seq_id", "label_seq_id")
    names = col("auth_atom_id", "label_atom_id")
    elements = col("type_symbol")
    xs, ys, zs = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")

    atoms = {}
    for i in range(len(chains)):
        if chains[i] != chain:
            continue
        if elements[i].strip().upper() in _HYDROGEN:
            continue
        atoms[(int(resids[i]), names[i].strip())] = np.array(
            [float(xs[i]), float(ys[i]), float(zs[i])])
    return atoms


def ca_map(atoms):
    """{resnum: xyz} over CA atoms only."""
    return {resnum: xyz for (resnum, name), xyz in atoms.items() if name == "CA"}


def _anchor_arrays(ref_atoms, pred_atoms, anchor_residues):
    """Paired CA coordinate arrays for the superposition anchor.

    Default anchor is EVERY CA shared by the two structures -- the full
    modelled backbone, deliberately NOT the (much smaller) set being measured.
    Fitting on the measured points would trivially shrink the very quantity
    being reported.
    """
    ref_ca, pred_ca = ca_map(ref_atoms), ca_map(pred_atoms)
    shared = sorted(set(ref_ca) & set(pred_ca))
    if anchor_residues is not None:
        shared = [r for r in sorted(anchor_residues) if r in ref_ca and r in pred_ca]
    if len(shared) < 3:
        raise ValueError(
            f"superposition anchor has {len(shared)} shared CA; need >= 3")
    P = np.array([ref_ca[r] for r in shared])
    Q = np.array([pred_ca[r] for r in shared])
    return P, Q


def motif_rmsd(ref_atoms, pred_atoms, residues, anchor_residues=None):
    """Heavy-atom RMSD (A) over `residues`, after CA superposition.

    Raises `KeyError` if any atom of a measured residue is missing from either
    structure -- a silently shrunk measured set would report a falsely good
    number on a broken design.
    """
    P, Q = _anchor_arrays(ref_atoms, pred_atoms, anchor_residues)
    R, P_mean, Q_mean = kabsch(P, Q)

    keys = sorted(k for k in ref_atoms if k[0] in set(residues))
    if not keys:
        raise KeyError(f"no reference atoms for residues {tuple(residues)}")
    missing = [k for k in keys if k not in pred_atoms]
    if missing:
        raise KeyError(f"prediction is missing measured atoms: {missing}")

    ref_xyz = np.array([ref_atoms[k] for k in keys])
    pred_xyz = apply_transform(np.array([pred_atoms[k] for k in keys]),
                               R, P_mean, Q_mean)
    return float(np.sqrt(np.mean(np.sum((ref_xyz - pred_xyz) ** 2, axis=1))))


def cleft_clearance(ref_atoms, pred_atoms, substrate_xyz, anchor_residues=None):
    """Minimum distance (A) from any substrate atom to any design heavy atom.

    `substrate_xyz` is in the REFERENCE frame (the crystal 8AZ coordinates).
    The prediction is superposed onto the reference, so the substrate's
    crystallographic position is effectively mapped into the design's frame; a
    small value means the design's own atoms now occupy the space the target
    base must sit in, i.e. the substrate cleft closed. This is the specific
    failure the tetrad-only MIN arm is exposed to, since nothing but the
    substrate context holds that cleft open during design.
    """
    P, Q = _anchor_arrays(ref_atoms, pred_atoms, anchor_residues)
    R, P_mean, Q_mean = kabsch(P, Q)

    keys = sorted(pred_atoms)
    pred_xyz = apply_transform(np.array([pred_atoms[k] for k in keys]),
                               R, P_mean, Q_mean)
    sub = np.atleast_2d(np.asarray(substrate_xyz, dtype=float))
    d = np.linalg.norm(pred_xyz[None, :, :] - sub[:, None, :], axis=2)
    return float(d.min())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_score_structure.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Prove the rigid-body test would catch a missing superposition**

RED evidence for the test that matters most. Temporarily break `motif_rmsd` by removing the superposition:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
python3 - <<'EOF'
import re, pathlib
p = pathlib.Path("tada_redesign/score_structure.py")
s = p.read_text()
s = s.replace("""    pred_xyz = apply_transform(np.array([pred_atoms[k] for k in keys]),
                               R, P_mean, Q_mean)
    return float(np.sqrt""", """    pred_xyz = np.array([pred_atoms[k] for k in keys])
    return float(np.sqrt""")
p.write_text(s)
EOF
conda run -n ligandmpnn_env python -m pytest \
  tada_redesign/tests/test_score_structure.py::test_motif_rmsd_is_invariant_to_rigid_body_motion -v
```

Expected: FAIL — the RMSD comes back dominated by the ~33 Å translation instead of ≈ 0.

Restore and confirm green:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git checkout -- tada_redesign/score_structure.py 2>/dev/null || true
git diff --stat -- tada_redesign/score_structure.py
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_score_structure.py -v
```

If `score_structure.py` is not yet committed, `git checkout` cannot restore it — in that case re-apply the correct two lines by hand and re-run. Expected final state: 9 passed, no diff.

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/score_structure.py tada_redesign/tests/test_score_structure.py
git commit -F - <<'EOF'
feat: shared geometric scorer -- Kabsch, motif RMSD, cleft clearance

One superposition-and-RMSD implementation for both folding stages. Heavy-atom
motif RMSD after CA superposition on the full shared backbone (never on the
measured subset, which would trivially shrink what it reports), plus a cleft
clearance metric that maps the crystal 8AZ position into the design frame to
catch a collapsed substrate pocket.

Includes the two tests whose absence let the predecessor's pocket_rmsd ship
with no superposition at all: rigid-body invariance and local-perturbation.
RED evidence captured by temporarily removing the superposition.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 7: `preflight.py` — the dependency gate

CLAUDE.md forbids submitting any job before every dependency is verified. This module is that gate, and it runs read-only and cheap so it can be re-run before every stage.

**Files:**
- Create: `tada-redesign/tada_redesign/preflight.py`
- Test: `tada-redesign/tada_redesign/tests/test_preflight.py`

**Interfaces:**
- Consumes: `constants.*`, `motif.load_masks`, and `tada_stability.relax_scaffolds.check_zn_geometry` (reached via `TADA_MONOREPO` on `sys.path`).
- Produces: `Check = namedtuple("Check", "name ok detail")`, `run_checks() -> list[Check]`, `main() -> int` (0 if every check passes, 1 otherwise).

- [ ] **Step 1: Write the failing test**

Create `tada-redesign/tada_redesign/tests/test_preflight.py`:

```python
"""Preflight must fail loudly and specifically. A check that reports success
when its subject is absent is worse than no check.

Every test here passes `with_env_probes=False`. The env probes shell out to
`conda run` twice, which takes tens of seconds each; four tests calling
run_checks() with them enabled would turn a unit suite into a multi-minute one.
The probes are exercised directly in test_conda_env_check_* against a
monkeypatched subprocess, and for real by the Step 5 live run.
"""
import subprocess

from tada_redesign import preflight


def test_run_checks_returns_named_checks():
    checks = preflight.run_checks(with_env_probes=False)
    assert checks
    for c in checks:
        assert isinstance(c.name, str) and c.name
        assert isinstance(c.ok, bool)
        assert isinstance(c.detail, str)


def test_env_probes_are_appended_only_when_requested():
    without = {c.name for c in preflight.run_checks(with_env_probes=False)}
    assert not any(n.startswith("conda env") for n in without)


def test_conda_env_check_passes_on_returncode_zero(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    c = preflight._conda_env_check("someenv", "numpy")
    assert c.ok is True
    assert c.name == "conda env 'someenv' has numpy"


def test_conda_env_check_fails_on_nonzero_returncode(monkeypatch):
    """The predecessor's cartesian_ddg check never inspected returncode and so
    reported success while singularity was never running the binary."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "ModuleNotFoundError"))
    c = preflight._conda_env_check("someenv", "nope")
    assert c.ok is False
    assert "ModuleNotFoundError" in c.detail


def test_every_expected_check_is_present():
    names = {c.name for c in preflight.run_checks(with_env_probes=False)}
    for expected in ("TADA_MONOREPO", "masks.json", "reference parents",
                     "Zn coordination geometry", "RFD3 checkpoint",
                     "LigandMPNN checkpoint", "AF3 SIF", "AF3 weights",
                     "ESMFold2 HF cache", "ESMFold2 ligand support",
                     "known-bad PDB not referenced"):
        assert expected in names, f"missing check: {expected}"


def test_missing_path_check_reports_not_ok(tmp_path):
    c = preflight._path_check("nonexistent thing", str(tmp_path / "nope"))
    assert c.ok is False
    assert "nope" in c.detail


def test_present_path_check_reports_ok(tmp_path):
    f = tmp_path / "here"
    f.write_text("x")
    assert preflight._path_check("present thing", str(f)).ok is True


def test_known_bad_pdb_is_not_referenced_in_this_package():
    checks = {c.name: c for c in preflight.run_checks(with_env_probes=False)}
    assert checks["known-bad PDB not referenced"].ok is True


def test_main_returns_nonzero_when_a_check_fails(monkeypatch):
    monkeypatch.setattr(
        preflight, "run_checks",
        lambda: [preflight.Check("fake", False, "deliberately failing")])
    assert preflight.main() == 1


def test_main_returns_zero_when_all_checks_pass(monkeypatch):
    monkeypatch.setattr(
        preflight, "run_checks",
        lambda: [preflight.Check("fake", True, "fine")])
    assert preflight.main() == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_preflight.py -v
```

Expected: FAIL — `ImportError: cannot import name 'preflight' from 'tada_redesign'`.

- [ ] **Step 3: Write `preflight.py`**

```python
"""Verify every external dependency before any job is submitted.

CLAUDE.md's pre-job gate exists because this cluster has a history of conda
envs being rebuilt and reference data moving. Every check here is read-only and
cheap, so it can run before every stage rather than once at the start.

A check that cannot prove its subject is present reports FAILURE. It never
reports success on an unverifiable condition -- that is the specific defect
that let the predecessor's `cartesian_ddg -help` check pass while singularity
was never actually running the binary.
"""
import glob
import os
import subprocess
import sys
from collections import namedtuple

from . import constants, motif

Check = namedtuple("Check", "name ok detail")

_THIS_PACKAGE = os.path.dirname(os.path.abspath(__file__))


def _path_check(name, path):
    return Check(name, os.path.exists(path), path)


def _masks_check():
    try:
        masks = motif.load_masks()
    except (OSError, ValueError, KeyError) as exc:
        return Check("masks.json", False, f"{constants.MASKS_JSON}: {exc}")
    full = motif.arm_residues(motif.ARM_FULL, masks)
    ok = len(full) == 24 and motif.arm_residues(motif.ARM_MIN, masks) == (57, 59, 87, 90)
    return Check("masks.json", ok,
                 f"FULL={len(full)} residues (expected 24), MIN={motif.arm_residues(motif.ARM_MIN, masks)}")


def _reference_parents_check():
    missing = [p for p in list(constants.PARENT_PDB.values()) + [constants.CHAINF_RAW]
               if not os.path.exists(p)]
    return Check("reference parents", not missing,
                 "all present" if not missing else f"missing: {missing}")


def _zn_geometry_check():
    """Both relaxed parents must still have a chemically sane Zn site.

    Reuses tada_stability.relax_scaffolds.check_zn_geometry rather than
    duplicating the thresholds -- it is pure Biopython/numpy, so this runs
    without PyRosetta.
    """
    sys.path.insert(0, constants.TADA_STABILITY)
    try:
        from tada_stability.relax_scaffolds import check_zn_geometry
    except ImportError as exc:
        return Check("Zn coordination geometry", False,
                     f"cannot import check_zn_geometry: {exc}")
    problems = []
    for parent, pdb in constants.PARENT_PDB.items():
        try:
            check_zn_geometry(pdb, constants.CHAINF_RAW)
        except (ValueError, OSError) as exc:
            problems.append(f"{parent}: {exc}")
    return Check("Zn coordination geometry", not problems,
                 "both parents pass" if not problems else "; ".join(problems))


def _esmfold_ligand_support_check():
    """The extended fold.py must expose ligand input, or every fold is apo and
    the Zn-geometry metric silently becomes unavailable."""
    path = constants.ESMFOLD_FOLD_PY
    if not os.path.exists(path):
        return Check("ESMFold2 ligand support", False, f"missing {path}")
    src = open(path).read()
    ok = "LigandInput" in src and "--ligand-ccd" in src
    return Check("ESMFold2 ligand support", ok, path)


def _known_bad_pdb_check():
    """The Zn-stripped 6VPC file must appear nowhere in this package."""
    needle = os.path.basename(constants.KNOWN_BAD_PDB)
    hits = []
    for py in glob.glob(os.path.join(_THIS_PACKAGE, "**", "*.py"), recursive=True):
        if os.path.basename(py) == "constants.py":
            continue          # constants records it precisely so this check can run
        if needle in open(py).read():
            hits.append(py)
    return Check("known-bad PDB not referenced", not hits,
                 "clean" if not hits else f"referenced in {hits}")


def _conda_env_check(name, module):
    try:
        r = subprocess.run(
            ["conda", "run", "-n", name, "python3", "-c", f"import {module}"],
            capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(f"conda env '{name}' has {module}", False, str(exc))
    return Check(f"conda env '{name}' has {module}", r.returncode == 0,
                 (r.stderr or r.stdout).strip()[-200:] or "ok")


def run_checks(with_env_probes=True):
    """Every check. `with_env_probes=False` skips the two `conda run` probes,
    which cost tens of seconds each -- the unit suite uses that; the real
    pre-job gate (main()) always runs them.
    """
    checks = [
        Check("TADA_MONOREPO", os.path.isdir(constants.MONOREPO), constants.MONOREPO),
        _masks_check(),
        _reference_parents_check(),
        _zn_geometry_check(),
        _path_check("6VPC structure", constants.PDB6VPC),
        _path_check("RFD3 checkpoint", constants.RFD3_CKPT),
        _path_check("LigandMPNN checkpoint", constants.LIGANDMPNN_CKPT),
        _path_check("AF3 SIF", constants.AF3_SIF),
        _path_check("AF3 weights", os.path.join(constants.AF3_DB, "models", "af3.bin")),
        _path_check("ESMFold2 HF cache", constants.ESMFOLD_HF_CACHE),
        _esmfold_ligand_support_check(),
        _known_bad_pdb_check(),
    ]
    if with_env_probes:
        checks.append(_conda_env_check(constants.ENV_TEST, "Bio.PDB"))
        checks.append(_conda_env_check(constants.ENV_ROSETTA, "pyrosetta"))
    return checks


def main():
    checks = run_checks()
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{'PASS' if c.ok else 'FAIL'}] {c.name:<{width}}  {c.detail}")
    failed = [c.name for c in checks if not c.ok]
    if failed:
        print(f"\n{len(failed)} of {len(checks)} checks FAILED: {failed}")
        return 1
    print(f"\nall {len(checks)} checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_preflight.py -v
```

Expected: 10 passed, in seconds rather than minutes (the `conda run` probes are skipped in the unit suite).

- [ ] **Step 5: Run preflight for real (read-only)**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m tada_redesign.preflight; echo "exit=$?"
```

Expected: every check PASS, `exit=0`. Record the actual output in the log.

**If the Zn-geometry check fails**, do not loosen its thresholds — the reference parents are committed as passing (`tada-stability/reference/README.md` records their measured geometry), so a failure means either the wrong file is being read or those references changed. Investigate and report.

- [ ] **Step 6: Run the whole Part 1 suite together**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -v
```

Expected: 37 passed (9 constants + 9 motif + 9 score_structure + 10 preflight).

- [ ] **Step 7: Commit and push the submodule, then update the pointer**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/preflight.py tada_redesign/tests/test_preflight.py
git commit -F - <<'EOF'
feat: dependency preflight gate

Fourteen read-only checks covering TADA_MONOREPO, masks.json (asserting the
FULL arm is 24 residues, not masks.json's 36-residue FROZEN key), the reference
parents and their Zn coordination geometry, RFD3/LigandMPNN/AF3/ESMFold2
assets, ESMFold2's ligand support, the known-bad Zn-stripped PDB being
referenced nowhere, and two conda env import probes.

Every check reports FAILURE when it cannot prove its subject present -- the
predecessor's cartesian_ddg check passed while singularity never ran the
binary, because it ignored the return code.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git push origin main
```

Then move the parent pointer:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git add tada-redesign
git commit -F - <<'EOF'
chore(tada-redesign): bump submodule to the Part 1 foundations

constants, single-source motif definition, shared geometric scorer, and the
dependency preflight gate. 34 tests pass; preflight green.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 8: Write the monorepo log entry**

Create `docs/logs/20260805_tada_redesign.md` recording: what Part 1 built, the ESMFold2 debug-fold result (including whether the Zn appeared in the output mmCIF), the real preflight output, the 34-test result, the RED evidence from Task 6 Step 5, and any follow-up items. Commit it with both trailers.

---

## Self-Review

**Spec coverage (Part 1 scope).** Submodule + GitHub remote → Task 3. Prerequisite 1 (ΔΔG commit) → Task 1. Prerequisite 2 (ESMFold2 Zn) → Task 2. `constants.py` → Task 4. `motif.py` and the two arms → Task 5. `score_structure.py`, heavy-atom motif RMSD, Kabsch anchoring, cleft openness → Task 6. `preflight.py` → Task 7. Honesty ceiling → in `README.md` (Task 3) and every module docstring. `requirements.md` → Task 3. Log entry → Task 7 Step 8.

**Deliberately deferred**, each to a named later plan: `prep_rfd_inputs`, `filter_backbones`, `prep_mpnn_inputs`, `_run_ligandmpnn`, `collect_designs` and their SLURM scripts (Part 2); `fold_screen`, `reference_baseline`, `score_rosetta`, `prep_af3`, `correlate`, `rank`, `report` (Part 3). The spec's degraded-run refusal, incremental shard writes, and provenance JSON belong to those stages, since Part 1 has no array stage to apply them to.

**Type consistency.** `check_zn_geometry(relaxed_pdb, raw_pdb)` is called with that signature in Task 7 and matches the committed source. `kabsch` returns `(R, P_mean, Q_mean)` in Task 6 and is consumed with that convention by `apply_transform`, `motif_rmsd`, and `cleft_clearance`. `motif.arm_residues(arm, masks)` takes masks as its second argument everywhere it appears, including inside `preflight._masks_check`. `Check(name, ok, detail)` field order is consistent between `_path_check`, `run_checks`, `main`, and the tests. `build_inputs(records, ligand_ccds, dna_specs)` matches its test calls in Task 2.

**Verified rather than assumed.** Every numeric assertion in Task 6 was checked against a live implementation of these exact functions before this plan was written: the Kabsch round-trip (`allclose` at `atol=1e-8`), rigid-body invariance (RMSD `2.8e-15`), the local-perturbation value (`0.24494897427831838` vs. expected `sqrt(0.36/6)`), open-cleft clearance (`168.08`), closed-cleft clearance (`0.3000000000000025`), and both raise paths. The mmCIF fixture was parsed with the real `Bio.PDB.MMCIF2Dict` in `ligandmpnn_env` and yields exactly `[(57, 'CA'), (201, 'ZN')]`.

**One asymmetry, stated rather than hidden:** `heavy_atoms_from_pdb` derives the element from columns 77-78 when present and falls back to the atom name's first character. For a `ZN` hetero atom the fallback would read `Z`, which is not in `_HYDROGEN`, so the atom is kept either way — but the fixture is column-exact so the primary path (element `ZN`) is what actually gets exercised. This was a real defect in the first draft of the fixture, caught by parsing it rather than reading it.
