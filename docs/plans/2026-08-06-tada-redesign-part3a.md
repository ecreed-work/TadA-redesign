# TadA Redesign — Part 3a (Folding & Gating) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold all 10,542 designs with ESMFold2 at a measured, affordable cost, and gate them on active-site geometry — producing a survivor set and the per-stage numbers that Part 3b's Rosetta/AF3 stages will be sized from.

**Architecture:** A batching fold wrapper that loads the model ONCE per process (the single decision that makes this affordable), a sharded SLURM screen over `designs.tsv`, a scorer that applies Part 1's existing geometric gates to folded models, and a reference baseline that puts both parents through the identical path. Rosetta ΔΔG, AF3, correlation and ranking are deliberately **Part 3b** — their shard widths cannot be planned until this stage's survivor count and cost are measured.

**Tech Stack:** Python 3.12, numpy, Biopython, pytest. ESMFold2 via the Biohub `transformers` fork (env `esmfold2`), SLURM `gpu` partition (H100).

**Spec:** `docs/specs/2026-08-05-tada-redesign-design.md` (in this repo)
**Prior parts:** Part 1 (foundations) and Part 2 (generation) are complete. Suite is 116 passing.

## Global Constraints

- Residue numbering is **Met = 1**; scaffold chain is **F**.
- Everything is measured against `constants.RMSD_REFERENCE` (the RELAXED parents), never the crystal.
- **ESMFold2 reports pLDDT on 0–1; AF3 reports 0–100.** `SCREEN_PLDDT_MARGIN = 0.05` is on ESMFold2's scale. Never compare across scales without converting via `ESMFOLD_PLDDT_SCALE` / `AF3_PLDDT_SCALE`.
- Every table is written through `io.append_row` (header + flushed row) and every stage writes `provenance.write(...)`; a stage losing more than `DEGRADED_FRACTION` of its inputs diverts to `*.degraded.tsv`.
- Submit with `sbatch`, never `bsub`. Partition `gpu`; CPU-only stages omit `--gres`.
- All work is in this repo (`tada-redesign`). Docs live here under `docs/`, NOT in the monorepo — see this repo's `CLAUDE.md`.
- Push over `ssh://git@ssh.github.com:443/...`.
- Both trailers on every commit:
  `Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>` and
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Tests: `cd <repo> && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"`. Each command needs its own `cd`.
- **No large fold run is submitted by this plan** until Task 2 has measured the cost and Task 6's debug shard has passed.
- Honesty ceiling in every new module docstring: pLDDT measures model confidence, motif RMSD measures geometry. Neither measures stability, solubility or deaminase activity.

## Verified inputs this plan is built on

Measured 2026-08-06 on the production run; do not re-derive.

| Fact | Value |
|---|---|
| Production run dir | `outputs/20260806_tada_redesign_gen1/` — **not** `constants.RUN_DIR_NAME`, which still says `20260805_tada_redesign` |
| `designs.tsv` | 10,542 rows; columns `design_id, backbone, cell, parent, arm, partial_t, temperature, bias, seed, mpnn_id, sequence, seq_len, overall_confidence, ligand_confidence, seq_rec` |
| Design composition | 10,040 biased + 502 control; FULL 5,376 / MIN 5,166; 2,510 per temperature; 502 backbones, 16 cells |
| Every `seq_len` | 156 |
| Backbones | 502 of 512 passed the filter; all 10 rejections were `motif_drift` at `partial_t=6.0` in MIN cells |
| `fold.py` semantics | **One complex per invocation.** Each FASTA record becomes a CHAIN of the same complex, so N designs cannot be folded by one N-record FASTA. |
| `fold.py` CLI | `fold.py <fasta> -o <out.cif> [--num-loops N] [--num-sampling-steps N] [--ligand-ccd CCD] [--dna CHAIN:SEQ] [--seed N]` |
| Debug fold cost | job 233015: 3:46 wall for one 67-mer at `--num-loops 4 --num-sampling-steps 20`, **including model load** |
| `relax_core` (Part 3b) | `relax_core(pdb_raw, mutations=(), cycles=5, out_pdb=None, mock=False) -> dict` with `free_score` (constraint-free `ref2015_cart`), `score`, `zn_geometry`, `out_pdb` |

**The cost problem this plan exists to solve.** At 3:46 per invocation including model load, 10,542 designs is ~660 GPU-hours — unaffordable. Model load is a fixed cost paid once per *process*, not per fold. Task 2 measures the split and Task 3 exploits it by folding many designs per process.

## File Structure

- `tada_redesign/constants.py` — modify: production run dir, parent sequences, fold-batch settings.
- `tada_redesign/enrich_designs.py` — new: adds `identity_to_parent` and `mutation_count` to `designs.tsv` (spec-required, absent since Part 2).
- `tools/esmfold2/fold_many.py` — new, in the MONOREPO: fold N sequences in ONE process. The single change that makes the screen affordable.
- `tada_redesign/fold_screen.py` — new: shard-aware driver over `designs.tsv`.
- `fold_screen.slurm` — new: sharded GPU array.
- `tada_redesign/score_folds.py` — new: applies Part 1's `score_structure` gates to folded models.
- `tada_redesign/reference_baseline.py` — new: both parents through the identical fold path.
- `tada_redesign/preflight.py` — modify: two checks for this stage.
- Tests: one module each under `tada_redesign/tests/`.

**Deferred to Part 3b** (cannot be sized until this plan's numbers exist): `score_rosetta`, `prep_af3`, `af3_infer.slurm`, `correlate`, `rank`, `report`, and the per-backbone bias re-evaluation the final review deferred.

---

### Task 1: Production run dir, parent sequences, and the two missing `designs.tsv` columns

`constants.RUN_DIR_NAME` still points at the Part 2 *debug* directory, so every Part 3 module would default to the wrong run. And the spec requires `identity_to_parent` and `mutation_count` on each design; Part 2 shipped without them, and `constants` carries no parent sequence to compute them from.

**Files:**
- Modify: `tada_redesign/constants.py`
- Create: `tada_redesign/enrich_designs.py`, `tada_redesign/tests/test_enrich_designs.py`
- Test: `tada_redesign/tests/test_constants.py` (append)

**Interfaces:**
- Consumes: `io.read_tsv`, `io.write_tsv`, `provenance.write`, `prep_scaffolds.parent_sequence` (via `TADA_MONOREPO`).
- Produces:
  - `constants.RUN_DIR_NAME = "20260806_tada_redesign_gen1"`
  - `constants.PARENT_SEQUENCE: dict[str, str]` — parent → 156-aa chain-F sequence
  - `constants.FOLD_BATCH_SIZE`, `constants.FOLD_SHARDS`
  - `enrich_designs.identity_to_parent(seq, parent_seq) -> float`
  - `enrich_designs.mutation_count(seq, parent_seq) -> int`
  - `enrich_designs.main(argv=None) -> int` — rewrites `designs.tsv` with the two columns appended

- [ ] **Step 1: Derive the parent sequences from the real reference PDBs**

Do NOT hand-type them. Run this and paste the output into `constants.py`:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python3 -c "
import sys; sys.path.insert(0, '/research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-stability')
from tada_stability import prep_scaffolds
from tada_redesign import constants
for p, pdb in constants.PARENT_PDB.items():
    s = prep_scaffolds.parent_sequence(pdb)
    print(f'{p}: len={len(s)}'); print(s)
"
```
Expected: two 156-residue sequences (chain F spans 5–160). If either is not 156, STOP and report — every design is `seq_len=156` and a mismatch means the comparison basis is wrong.

- [ ] **Step 2: Write the failing tests**

Append to `tada_redesign/tests/test_constants.py`:

```python
def test_run_dir_points_at_the_production_run():
    """Part 2's production generation wrote 20260806_tada_redesign_gen1; the old
    default was the debug dir and would silently score the wrong run."""
    assert constants.RUN_DIR_NAME == "20260806_tada_redesign_gen1"


def test_parent_sequences_are_present_and_the_right_length():
    assert set(constants.PARENT_SEQUENCE) == set(constants.PARENTS)
    for parent, seq in constants.PARENT_SEQUENCE.items():
        assert len(seq) == 156, (parent, len(seq))
        assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_tada9_differs_from_tada8e_at_exactly_its_two_defining_positions():
    """TadA-9 = TadA-8e + N108Q + L145T. Chain F starts at residue 5, so
    sequence index = resnum - 5."""
    a, b = constants.PARENT_SEQUENCE["TadA8e"], constants.PARENT_SEQUENCE["TadA9"]
    diffs = {i + 5 for i, (x, y) in enumerate(zip(a, b)) if x != y}
    assert diffs == {108, 145}, diffs
    assert (a[108 - 5], b[108 - 5]) == ("N", "Q")
    assert (a[145 - 5], b[145 - 5]) == ("L", "T")
```

Create `tada_redesign/tests/test_enrich_designs.py`:

```python
"""identity_to_parent and mutation_count are spec-required and were missing from
Part 2's designs.tsv, so Part 3 would have had to recompute them ad hoc."""
import pytest

from tada_redesign import constants, enrich_designs as en, io as tio


def test_mutation_count_counts_substitutions():
    assert en.mutation_count("AAAA", "AAAA") == 0
    assert en.mutation_count("AAAA", "AAAC") == 1
    assert en.mutation_count("ACDE", "WWWW") == 4


def test_identity_is_the_complement_of_mutation_rate():
    assert en.identity_to_parent("AAAA", "AAAA") == pytest.approx(1.0)
    assert en.identity_to_parent("AAAA", "AAAC") == pytest.approx(0.75)


def test_length_mismatch_raises_rather_than_truncating():
    """zip() would silently compare only the overlap and report a falsely high
    identity for a truncated design."""
    with pytest.raises(ValueError):
        en.mutation_count("AAA", "AAAA")


def test_a_real_parent_sequence_is_self_identical():
    seq = constants.PARENT_SEQUENCE["TadA8e"]
    assert en.identity_to_parent(seq, seq) == pytest.approx(1.0)
    assert en.mutation_count(seq, seq) == 0


def test_main_appends_both_columns_and_preserves_every_row(tmp_path):
    rows = [{"design_id": "d1", "parent": "TadA8e",
             "sequence": constants.PARENT_SEQUENCE["TadA8e"]},
            {"design_id": "d2", "parent": "TadA9",
             "sequence": constants.PARENT_SEQUENCE["TadA9"]}]
    src = tmp_path / "designs.tsv"
    tio.write_tsv(str(src), rows, ("design_id", "parent", "sequence"))
    assert en.main(["--designs", str(src)]) == 0
    out = tio.read_tsv(str(src))
    assert len(out) == 2
    assert out[0]["mutation_count"] == "0"
    assert float(out[0]["identity_to_parent"]) == pytest.approx(1.0)
    assert "sequence" in out[0]          # original columns retained


def test_main_compares_each_design_against_ITS_OWN_parent(tmp_path):
    """A TadA9 design scored against TadA8e would read as 2 spurious mutations."""
    rows = [{"design_id": "d1", "parent": "TadA9",
             "sequence": constants.PARENT_SEQUENCE["TadA9"]}]
    src = tmp_path / "d.tsv"
    tio.write_tsv(str(src), rows, ("design_id", "parent", "sequence"))
    en.main(["--designs", str(src)])
    assert tio.read_tsv(str(src))[0]["mutation_count"] == "0"
```

- [ ] **Step 3: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_enrich_designs.py tada_redesign/tests/test_constants.py -q -m "not slow"
```
Expected: FAIL — `ImportError: cannot import name 'enrich_designs'` plus the three new constants assertions.

- [ ] **Step 4: Update `constants.py`**

Change the run dir, keeping the old value visible as history:

```python
# Part 2's PRODUCTION generation ran here. The previous value
# ("20260805_tada_redesign") was the debug run: 6 backbones at reduced RFD3
# sampling. Scoring that directory would silently rank debug artifacts.
RUN_DIR_NAME = "20260806_tada_redesign_gen1"
```

Add the parent sequences (paste the exact strings from Step 1):

```python
# Chain F residues 5-160 of each relaxed reference, read from the tracked
# reference PDBs by prep_scaffolds.parent_sequence (never hand-typed).
# TadA-9 = TadA-8e + N108Q + L145T, asserted by test_constants.
PARENT_SEQUENCE = {
    "TadA8e": "<paste>",
    "TadA9": "<paste>",
}
```

Add the fold batching settings:

```python
# ESMFold2 loads its weights once per PROCESS, not once per fold, so the screen
# folds many designs per invocation. FOLD_SHARDS is the SLURM array width;
# FOLD_BATCH_SIZE is how many designs one process folds before exiting (a cap,
# so a crash loses at most this much work).
FOLD_BATCH_SIZE = 250
FOLD_SHARDS = 44          # ceil(10542 / 250); re-derive if the design count moves
```

- [ ] **Step 5: Write `enrich_designs.py`**

```python
"""Add the two spec-required per-design columns Part 2 shipped without.

`identity_to_parent` and `mutation_count` are named in the design spec's Stage 2
but were never emitted, so every later stage would have recomputed them ad hoc --
and `constants` carried no parent sequence to compute them from.

Each design is compared against ITS OWN parent. TadA-9 differs from TadA-8e at
two positions (N108Q, L145T), so comparing a TadA9 design against TadA8e would
report two mutations that are not the designer's doing.

Honesty ceiling: sequence identity is a similarity measure. It says nothing about
whether a design folds, is stable, or is active.
"""
import argparse
import os

from . import constants, io, provenance


def mutation_count(seq, parent_seq):
    """Substitutions between equal-length sequences.

    Raises on a length mismatch: `zip` would silently compare only the overlap
    and report a falsely high identity for a truncated design.
    """
    if len(seq) != len(parent_seq):
        raise ValueError(
            f"length mismatch: design {len(seq)} vs parent {len(parent_seq)}")
    return sum(1 for a, b in zip(seq, parent_seq) if a != b)


def identity_to_parent(seq, parent_seq):
    return 1.0 - mutation_count(seq, parent_seq) / float(len(parent_seq))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--designs", default=os.path.join(
        sub, "outputs", constants.RUN_DIR_NAME, "designs.tsv"))
    args = ap.parse_args(argv)

    rows = io.read_tsv(args.designs)
    if not rows:
        raise SystemExit(f"[enrich_designs] no rows in {args.designs}")

    failed = []
    for row in rows:
        parent_seq = constants.PARENT_SEQUENCE[row["parent"]]
        try:
            row["mutation_count"] = str(mutation_count(row["sequence"], parent_seq))
            row["identity_to_parent"] = f"{identity_to_parent(row['sequence'], parent_seq):.4f}"
        except (ValueError, KeyError) as exc:
            row["mutation_count"] = io.MISSING
            row["identity_to_parent"] = io.MISSING
            failed.append((row["design_id"], str(exc)))

    columns = tuple(rows[0].keys())
    io.write_tsv(args.designs, rows, columns)
    for design_id, why in failed[:10]:
        print(f"[enrich_designs] WARNING {design_id}: {why}")
    print(f"[enrich_designs] enriched {len(rows) - len(failed)}/{len(rows)} rows "
          f"-> {args.designs}")
    provenance.write(os.path.dirname(args.designs), "enrich_designs",
                     len(rows), len(rows) - len(failed),
                     extra={"failed": failed[:50]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run to verify pass, then enrich the real file**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_enrich_designs.py tada_redesign/tests/test_constants.py -q -m "not slow"
```
Expected: 6 new enrich tests + the constants module green.

Then enrich the production file and report the distribution:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.enrich_designs
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python3 -c "
import csv, statistics as st, collections
rows=list(csv.DictReader(open('outputs/20260806_tada_redesign_gen1/designs.tsv'), delimiter='\t'))
m=[int(r['mutation_count']) for r in rows]
print('rows', len(rows), '| mutation_count median', st.median(m), 'min', min(m), 'max', max(m))
for arm in ('FULL','MIN'):
    v=[int(r['mutation_count']) for r in rows if r['arm']==arm]
    print(f'  {arm}: median {st.median(v)}  range {min(v)}-{max(v)}')
"
```
Report the actual numbers. **Expect MIN to carry more mutations than FULL** — it freezes 4 residues instead of 24, so more positions are designable. If FULL shows MORE mutations than MIN, STOP and report: that would mean the frozen set is not being honoured, which contradicts Part 2's verification.

- [ ] **Step 7: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/constants.py tada_redesign/enrich_designs.py tada_redesign/tests/test_enrich_designs.py tada_redesign/tests/test_constants.py
git commit -F - <<'EOF'
feat: point at the production run and add the two missing design columns

RUN_DIR_NAME still named the Part 2 DEBUG directory (6 backbones at reduced RFD3
sampling), so every Part 3 module would have defaulted to scoring debug
artifacts. Now the production run, with the old value kept visible as history.

Adds PARENT_SEQUENCE, read from the tracked reference PDBs rather than
hand-typed, and asserted to differ between the two parents at exactly N108Q and
L145T. enrich_designs then fills in identity_to_parent and mutation_count --
named in the spec's Stage 2 but never emitted by Part 2 -- comparing each design
against ITS OWN parent, since a TadA9 design scored against TadA8e would report
two mutations the designer never made.

mutation_count raises on a length mismatch rather than zipping to the shorter
sequence, which would report a falsely high identity for a truncated design.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: `fold_many.py` — fold N sequences per process, and MEASURE the split

The decision this whole plan turns on. `tools/esmfold2/fold.py` folds ONE complex per invocation, and the 2026-08-04 debug fold took 3:46 wall for a single 67-mer *including model load*. At that rate 10,542 designs is ~660 GPU-hours. Model load is paid once per process, so folding many designs per process is what makes the screen affordable — but the actual load-vs-fold split has never been measured, and the whole funnel's viability depends on it.

**Files:**
- Create (in the MONOREPO): `tools/esmfold2/fold_many.py`, `tools/esmfold2/tests/test_fold_many.py`

**Interfaces:**
- Consumes: `esm.models.esmfold2.{ESMFold2InputBuilder, ProteinInput, LigandInput, StructurePredictionInput}`, `transformers.models.esmfold2.modeling_esmfold2.ESMFold2Model`.
- Produces:
  - `fold_many.read_jobs(tsv) -> list[dict]` — rows with `design_id` and `sequence`
  - `fold_many.out_paths(out_dir, design_id) -> tuple[str, str]` — `(cif_path, metrics_path)`
  - `fold_many.main(argv) -> int` — CLI: `--jobs <tsv> --out-dir <dir> [--ligand-ccd ZN] [--num-loops N] [--num-sampling-steps N] [--limit N] [--skip-existing]`
  - Writes one `<design_id>.cif` + `<design_id>.metrics.json` per design, and a `timing.json` recording model-load seconds and per-fold seconds.

- [ ] **Step 1: Write the failing test (GPU-free)**

`tools/esmfold2/tests/test_fold_many.py`:

```python
"""Job-list and path handling for the batch folder. No GPU, no model load."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fold_many  # noqa: E402


def _jobs_tsv(tmp_path, n=3):
    p = tmp_path / "jobs.tsv"
    lines = ["design_id\tsequence"]
    lines += [f"d{i}\tMKV{'A' * i}" for i in range(n)]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_read_jobs_returns_id_and_sequence(tmp_path):
    jobs = fold_many.read_jobs(_jobs_tsv(tmp_path))
    assert [j["design_id"] for j in jobs] == ["d0", "d1", "d2"]
    assert jobs[1]["sequence"] == "MKVA"


def test_read_jobs_skips_comment_lines(tmp_path):
    p = tmp_path / "c.tsv"
    p.write_text("# note\ndesign_id\tsequence\nd0\tMKV\n")
    assert len(fold_many.read_jobs(str(p))) == 1


def test_read_jobs_rejects_a_missing_sequence_column(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("design_id\tfoo\nd0\tx\n")
    with pytest.raises(ValueError):
        fold_many.read_jobs(str(p))


def test_out_paths_are_derived_from_the_design_id(tmp_path):
    cif, metrics = fold_many.out_paths(str(tmp_path), "d0")
    assert cif.endswith("d0.cif")
    assert metrics.endswith("d0.metrics.json")


def test_skip_existing_filters_completed_designs(tmp_path):
    jobs = fold_many.read_jobs(_jobs_tsv(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    # d1 already has BOTH artifacts -> it is done
    (out / "d1.cif").write_text("x")
    (out / "d1.metrics.json").write_text("{}")
    # d2 has only a cif -> incomplete, must be redone
    (out / "d2.cif").write_text("x")
    remaining = fold_many.filter_done(jobs, str(out))
    assert [j["design_id"] for j in remaining] == ["d0", "d2"]


def test_timing_summary_reports_load_and_per_fold(tmp_path):
    path = fold_many.write_timing(str(tmp_path), load_s=120.0, fold_s=[2.0, 4.0])
    doc = json.load(open(path))
    assert doc["model_load_s"] == 120.0
    assert doc["n_folded"] == 2
    assert doc["mean_fold_s"] == pytest.approx(3.0)
    # the number that decides the batch size
    assert doc["amortised_s_per_design_at_this_batch"] == pytest.approx(63.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign && conda run -n esmfold2 python -m pytest tools/esmfold2/tests/test_fold_many.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fold_many'`.

- [ ] **Step 3: Write `fold_many.py`**

```python
#!/usr/bin/env python
"""Fold many designed sequences in ONE process, holding the catalytic Zn.

`fold.py` folds one complex per invocation, and ESMFold2's weights load once per
PROCESS. Folding 10,542 designs one invocation at a time therefore pays the model
load 10,542 times -- at the 2026-08-04 debug fold's 3:46 wall for a single
67-mer, roughly 660 GPU-hours. This script loads once and folds a list, so the
load is amortised across the batch.

Each design is an INDEPENDENT single-chain fold. Do not be tempted to put several
designs in one StructurePredictionInput: `fold.py`'s FASTA records become CHAINS
OF ONE COMPLEX, which would fold them as a hetero-oligomer and produce garbage.

Writes per design: `<design_id>.cif` and `<design_id>.metrics.json` (pLDDT, pTM,
iptm on ESMFold2's 0-1 scale). Writes `timing.json` once per process, which is
what the SLURM array width is sized from.

Honesty ceiling: pLDDT is model confidence. It is not stability, and a confident
fold of a designed sequence is not evidence of function.
"""
import argparse
import csv
import json
import os
import time


def read_jobs(tsv):
    """Rows with `design_id` and `sequence`; `#` comment lines skipped."""
    with open(tsv, newline="") as fh:
        clean = (ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#"))
        rows = list(csv.DictReader(clean, delimiter="\t"))
    if not rows:
        return []
    for required in ("design_id", "sequence"):
        if required not in rows[0]:
            raise ValueError(f"{tsv}: missing required column {required!r}")
    return [{"design_id": r["design_id"], "sequence": r["sequence"]} for r in rows]


def out_paths(out_dir, design_id):
    return (os.path.join(out_dir, f"{design_id}.cif"),
            os.path.join(out_dir, f"{design_id}.metrics.json"))


def filter_done(jobs, out_dir):
    """Drop designs that have BOTH artifacts.

    Requiring both matters: a cif without its metrics means the process died
    mid-write, and that design must be refolded rather than silently accepted.
    """
    remaining = []
    for job in jobs:
        cif, metrics = out_paths(out_dir, job["design_id"])
        if not (os.path.exists(cif) and os.path.exists(metrics)):
            remaining.append(job)
    return remaining


def write_timing(out_dir, load_s, fold_s):
    n = len(fold_s)
    doc = {
        "model_load_s": load_s,
        "n_folded": n,
        "mean_fold_s": (sum(fold_s) / n) if n else None,
        "total_s": load_s + sum(fold_s),
        # the number that decides the SLURM array width
        "amortised_s_per_design_at_this_batch": ((load_s + sum(fold_s)) / n) if n else None,
    }
    path = os.path.join(out_dir, "timing.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True, help="TSV with design_id + sequence")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ligand-ccd", action="append", default=[])
    ap.add_argument("--num-loops", type=int, default=4)
    ap.add_argument("--num-sampling-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = read_jobs(args.jobs)
    if args.skip_existing:
        jobs = filter_done(jobs, args.out_dir)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[fold_many] {len(jobs)} designs to fold -> {args.out_dir}", flush=True)
    if not jobs:
        return 0

    from esm.models.esmfold2 import (ESMFold2InputBuilder, LigandInput,
                                     ProteinInput, StructurePredictionInput)
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    t0 = time.time()
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    builder = ESMFold2InputBuilder()
    load_s = time.time() - t0
    print(f"[fold_many] model loaded in {load_s:.1f}s", flush=True)

    fold_s, failed = [], []
    for i, job in enumerate(jobs, 1):
        design_id = job["design_id"]
        cif_path, metrics_path = out_paths(args.out_dir, design_id)
        seqs = [ProteinInput(id="F", sequence=job["sequence"])]
        for j, ccd in enumerate(args.ligand_ccd):
            seqs.append(LigandInput(id=chr(ord("B") + j), ccd=[ccd]))
        try:
            t1 = time.time()
            result = builder.fold(
                model, StructurePredictionInput(sequences=seqs),
                num_loops=args.num_loops,
                num_sampling_steps=args.num_sampling_steps,
                num_diffusion_samples=1, seed=args.seed)
            open(cif_path, "w").write(result.complex.to_mmcif())
            metrics = {"design_id": design_id,
                       "plddt": float(result.plddt),
                       "ptm": float(result.ptm)}
            try:
                metrics["iptm"] = float(result.iptm)
            except (AttributeError, TypeError):
                metrics["iptm"] = None      # single-entity folds have no iptm
            json.dump(metrics, open(metrics_path, "w"), indent=1)
            fold_s.append(time.time() - t1)
        except Exception as exc:            # noqa: BLE001 - one bad design must
            failed.append((design_id, repr(exc)))   # not kill the whole shard
            print(f"[fold_many] FAILED {design_id}: {exc!r}", flush=True)
        if i % 25 == 0:
            print(f"[fold_many] {i}/{len(jobs)} done", flush=True)

    path = write_timing(args.out_dir, load_s, fold_s)
    print(f"[fold_many] folded {len(fold_s)}, failed {len(failed)}; timing -> {path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign && conda run -n esmfold2 python -m pytest tools/esmfold2/tests/test_fold_many.py -q
```
Expected: 6 passed.

- [ ] **Step 5: MEASURE the real cost — this is the deliverable**

Fold 10 real designs in one process and read the timing split.

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
mkdir -p outputs/20260806_tada_redesign_gen1/fold_probe
conda run -n ligandmpnn_env python3 -c "
import csv
rows=list(csv.DictReader(open('outputs/20260806_tada_redesign_gen1/designs.tsv'), delimiter='\t'))[:10]
import sys
w=csv.DictWriter(open('outputs/20260806_tada_redesign_gen1/fold_probe/jobs.tsv','w',newline=''),
                 fieldnames=['design_id','sequence'], delimiter='\t', lineterminator='\n')
w.writeheader()
for r in rows: w.writerow({'design_id': r['design_id'], 'sequence': r['sequence']})
print('wrote 10 jobs')
"
sbatch --wait --partition=gpu --gres=gpu:1 -c 4 --mem=48G -t 02:00:00 \
  -o logs/fold_probe.out -e logs/fold_probe.err \
  --wrap "source /research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3/bin/activate esmfold2 && python /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tools/esmfold2/fold_many.py --jobs /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign/outputs/20260806_tada_redesign_gen1/fold_probe/jobs.tsv --out-dir /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign/outputs/20260806_tada_redesign_gen1/fold_probe --ligand-ccd ZN --num-loops 4 --num-sampling-steps 20"
cat outputs/20260806_tada_redesign_gen1/fold_probe/timing.json
```

Then compute and REPORT the projection:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && python3 -c "
import json, math
t=json.load(open('outputs/20260806_tada_redesign_gen1/fold_probe/timing.json'))
load, mean = t['model_load_s'], t['mean_fold_s']
print(f'model load {load:.0f}s | mean fold {mean:.1f}s')
for B in (100, 250, 500, 1000):
    per = (load + B*mean)/B; total = 10542*per/3600
    print(f'  batch {B:5d}: {per:6.1f}s/design amortised, {total:7.1f} GPU-h total, {math.ceil(10542/B):3d} shards')
"
```

**Decision rule, to be applied and reported, not guessed:**
- If total GPU-hours at `FOLD_BATCH_SIZE=250` is **under ~100**, proceed as planned.
- If it is **100–300**, raise `FOLD_BATCH_SIZE` (fewer, longer shards) and report the chosen value with its arithmetic.
- If it exceeds ~300 even at batch 1000, **STOP and report**. The spec's documented fallback is to gate on MPNN score first and fold a subset, logging exactly how many designs were dropped unfolded. That is a funnel-reshaping decision for the human, not something to absorb quietly.

Also confirm from the probe: 10 `.cif` + 10 `.metrics.json` written, and each metrics file has a `plddt` in [0, 1] (ESMFold2's scale — a value above 1 would mean the scale assumption is wrong and `SCREEN_PLDDT_MARGIN` is meaningless).

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git add tools/esmfold2/fold_many.py tools/esmfold2/tests/test_fold_many.py
git commit -F - <<'EOF'
feat(esmfold2): fold many sequences per process, with timing instrumentation

fold.py folds ONE complex per invocation and ESMFold2 loads its weights once per
process, so folding 10,542 designs one at a time pays the model load 10,542
times -- roughly 660 GPU-hours at the 2026-08-04 debug fold's measured rate.
fold_many loads once and folds a list.

Each design is an independent single-chain fold: putting several in one
StructurePredictionInput would fold them as a hetero-oligomer, because
fold.py's FASTA records are chains of one complex.

timing.json records the model-load / per-fold split and the amortised cost per
design, which is what the SLURM array width gets sized from. --skip-existing
requires BOTH the cif and its metrics, so a design whose process died mid-write
is refolded rather than silently accepted.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: `fold_screen.py` + `fold_screen.slurm` — the sharded screen

**Files:**
- Create: `tada_redesign/fold_screen.py`, `tada_redesign/tests/test_fold_screen.py`, `fold_screen.slurm`

**Interfaces:**
- Consumes: `io.read_tsv`, `io.write_tsv`, `provenance.write`, `constants.{RUN_DIR_NAME,FOLD_BATCH_SIZE,FOLD_SHARDS,ESMFOLD_SCREEN,ESMFOLD_FOLD_PY}`.
- Produces:
  - `fold_screen.shard_of(designs, shard, n_shards) -> list[dict]` — deterministic contiguous split
  - `fold_screen.write_shard_jobs(rows, path) -> str` — the `design_id`/`sequence` TSV `fold_many` consumes
  - `fold_screen.main(argv=None) -> int` — CLI `--shard N --n-shards M [--run-dir DIR]`

- [ ] **Step 1: Write the failing test**

```python
"""Sharding must be deterministic and lossless: every design folded exactly once
across the array, or the screen silently drops designs."""
import pytest

from tada_redesign import fold_screen as fs, io as tio


def _rows(n):
    return [{"design_id": f"d{i}", "sequence": "MKV", "parent": "TadA8e"} for i in range(n)]


def test_shards_partition_every_design_exactly_once():
    rows = _rows(1000)
    seen = []
    for s in range(1, 8):
        seen += [r["design_id"] for r in fs.shard_of(rows, s, 7)]
    assert sorted(seen) == sorted(r["design_id"] for r in rows)
    assert len(seen) == len(set(seen))       # no design folded twice


def test_shards_are_balanced_within_one():
    rows = _rows(1000)
    sizes = [len(fs.shard_of(rows, s, 7)) for s in range(1, 8)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_is_deterministic():
    rows = _rows(100)
    assert fs.shard_of(rows, 3, 7) == fs.shard_of(rows, 3, 7)


def test_shard_index_is_one_based_and_validated():
    rows = _rows(10)
    with pytest.raises(ValueError):
        fs.shard_of(rows, 0, 4)
    with pytest.raises(ValueError):
        fs.shard_of(rows, 5, 4)


def test_write_shard_jobs_emits_only_the_two_needed_columns(tmp_path):
    path = fs.write_shard_jobs(_rows(3), str(tmp_path / "jobs.tsv"))
    rows = tio.read_tsv(path)
    assert set(rows[0]) == {"design_id", "sequence"}
    assert len(rows) == 3
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_fold_screen.py -q
```
Expected: FAIL on import.

- [ ] **Step 3: Write `fold_screen.py`**

```python
"""Shard designs.tsv across a SLURM array and hand each shard to fold_many.

Sharding is a contiguous, deterministic split of the row order. Contiguity keeps
a shard's designs adjacent in designs.tsv, which makes a partial run easy to
reason about; determinism means a resubmitted shard folds exactly the same
designs, so `--skip-existing` composes correctly.

The screen folds at ESMFOLD_SCREEN settings (reduced sampling). Those numbers
depress pLDDT substantially, which is why the gate is RELATIVE to a parent
folded in the identical mode -- see reference_baseline.

Honesty ceiling: this module moves sequences to a GPU and files back. It measures
nothing.
"""
import argparse
import os
import subprocess

from . import constants, io, provenance


def shard_of(rows, shard, n_shards):
    """Contiguous 1-based shard `shard` of `n_shards`, balanced within one row."""
    if not 1 <= shard <= n_shards:
        raise ValueError(f"shard {shard} out of range 1..{n_shards}")
    n = len(rows)
    base, extra = divmod(n, n_shards)
    start = (shard - 1) * base + min(shard - 1, extra)
    size = base + (1 if shard - 1 < extra else 0)
    return rows[start:start + size]


def write_shard_jobs(rows, path):
    io.write_tsv(path, [{"design_id": r["design_id"], "sequence": r["sequence"]}
                        for r in rows], ("design_id", "sequence"))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(sub, "outputs", constants.RUN_DIR_NAME))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=constants.FOLD_SHARDS)
    args = ap.parse_args(argv)

    rows = io.read_tsv(os.path.join(args.run_dir, "designs.tsv"))
    if not rows:
        raise SystemExit("[fold_screen] designs.tsv is empty or missing")
    mine = shard_of(rows, args.shard, args.n_shards)

    shard_dir = os.path.join(args.run_dir, "fold_screen")
    os.makedirs(shard_dir, exist_ok=True)
    jobs = write_shard_jobs(mine, os.path.join(
        shard_dir, f"jobs_shard{args.shard:03d}.tsv"))

    cmd = ["python", os.path.join(constants.MONOREPO, "tools/esmfold2/fold_many.py"),
           "--jobs", jobs, "--out-dir", shard_dir, "--ligand-ccd", constants.ZN_RESNAME,
           "--num-loops", str(constants.ESMFOLD_SCREEN["num_loops"]),
           "--num-sampling-steps", str(constants.ESMFOLD_SCREEN["num_sampling_steps"]),
           "--skip-existing"]
    print(f"[fold_screen] shard {args.shard}/{args.n_shards}: {len(mine)} designs")
    print("[fold_screen] " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode

    done = sum(os.path.exists(os.path.join(shard_dir, f"{r['design_id']}.cif"))
               and os.path.exists(os.path.join(shard_dir, f"{r['design_id']}.metrics.json"))
               for r in mine)
    provenance.write(shard_dir, f"fold_screen_shard{args.shard:03d}",
                     len(mine), done, extra={"returncode": rc})
    print(f"[fold_screen] shard {args.shard}: {done}/{len(mine)} folded")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_fold_screen.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Write `fold_screen.slurm` (do NOT submit)**

```bash
#!/bin/bash
# ESMFold2 screen over designs.tsv, one array task per shard. Each task folds its
# shard in ONE process so the model load is amortised (see fold_many.py).
#   SHARDS=44 sbatch --array=1-44 fold_screen.slurm
# Set -t from the measured amortised cost in fold_probe/timing.json, with margin.
#SBATCH --job-name=tada_fold_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/tada_fold_screen_%A_%a.out
#SBATCH -e logs/tada_fold_screen_%A_%a.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
SUB="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign"
SHARDS="${SHARDS:-44}"

cd "$SUB"; mkdir -p logs
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" esmfold2

echo "[fold_screen.slurm] shard ${SLURM_ARRAY_TASK_ID:?run as an array job}/$SHARDS host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
PYTHONPATH="$SUB" python -m tada_redesign.fold_screen \
  --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$SHARDS"
```

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/fold_screen.py tada_redesign/tests/test_fold_screen.py fold_screen.slurm
git commit -F - <<'EOF'
feat: sharded ESMFold2 screen over designs.tsv

One array task per shard, each folding its shard in a single process so the
model load is amortised rather than paid per design. Sharding is contiguous and
deterministic, and a test asserts the shards partition every design exactly once
-- a sharding bug would silently drop designs from the screen, which no later
stage could detect.

fold_screen.slurm written, not submitted; its -t is to be set from the measured
amortised cost.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 4: `reference_baseline.py` — both parents through the identical path

The screen's pLDDT gate is relative to a parent folded the same way. Reduced sampling depresses pLDDT substantially (the 2026-08-04 debug fold returned 0.45 on a 78-mer for that reason), so an absolute floor would be meaningless, and a parent folded at FULL sampling would make every design look bad.

**Files:**
- Create: `tada_redesign/reference_baseline.py`, `tada_redesign/tests/test_reference_baseline.py`

**Interfaces:**
- Consumes: `constants.{PARENT_SEQUENCE,ESMFOLD_SCREEN,ESMFOLD_FULL,RUN_DIR_NAME}`, `io`, `provenance`.
- Produces:
  - `reference_baseline.baseline_jobs() -> list[dict]` — one row per parent per mode
  - `reference_baseline.baseline_id(parent, mode) -> str` — e.g. `"TadA8e__screen"`
  - `reference_baseline.read_baseline(run_dir) -> dict[tuple[str, str], float]` — `(parent, mode) -> plddt`
  - `reference_baseline.main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

```python
"""The baseline must be folded in the SAME mode as the designs it gates, or the
comparison is meaningless."""
import json

import pytest

from tada_redesign import constants, reference_baseline as rb


def test_baseline_covers_both_parents_in_both_modes():
    jobs = rb.baseline_jobs()
    assert {(j["parent"], j["mode"]) for j in jobs} == {
        (p, m) for p in constants.PARENTS for m in ("screen", "full")}


def test_baseline_sequences_are_the_real_parent_sequences():
    for job in rb.baseline_jobs():
        assert job["sequence"] == constants.PARENT_SEQUENCE[job["parent"]]
        assert len(job["sequence"]) == 156


def test_baseline_id_round_trips():
    assert rb.baseline_id("TadA8e", "screen") == "TadA8e__screen"
    assert rb.baseline_id("TadA9", "full") == "TadA9__full"


def test_read_baseline_parses_metrics_files(tmp_path):
    d = tmp_path / "baseline"
    d.mkdir()
    json.dump({"plddt": 0.71}, open(d / "TadA8e__screen.metrics.json", "w"))
    json.dump({"plddt": 0.83}, open(d / "TadA9__full.metrics.json", "w"))
    got = rb.read_baseline(str(tmp_path))
    assert got[("TadA8e", "screen")] == pytest.approx(0.71)
    assert got[("TadA9", "full")] == pytest.approx(0.83)


def test_read_baseline_raises_when_a_required_baseline_is_missing(tmp_path):
    """Gating designs against an absent baseline would silently pass or fail
    everything."""
    (tmp_path / "baseline").mkdir()
    with pytest.raises(FileNotFoundError):
        rb.read_baseline(str(tmp_path), require=[("TadA8e", "screen")])
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_reference_baseline.py -q
```
Expected: FAIL on import.

- [ ] **Step 3: Write `reference_baseline.py`**

```python
"""Fold both parents through the IDENTICAL path the designs take.

The screen's pLDDT gate is relative: `plddt >= parent_plddt - SCREEN_PLDDT_MARGIN`.
That only means anything if the parent was folded the same way. Reduced sampling
depresses pLDDT substantially (a 2026-08-04 debug fold returned 0.45 on a 78-mer
purely from reduced sampling), so an absolute floor is meaningless and a parent
folded at full sampling would make every screened design look bad by comparison.

Both modes are folded here: `screen` gates the screen, `full` gates the
full-sampling re-fold of survivors.

Honesty ceiling: this is a confidence reference, not a stability reference. The
energetic baseline is Part 3b's Rosetta stage.
"""
import argparse
import json
import os
import subprocess

from . import constants, io, provenance

MODES = {"screen": constants.ESMFOLD_SCREEN, "full": constants.ESMFOLD_FULL}


def baseline_id(parent, mode):
    return f"{parent}__{mode}"


def baseline_jobs():
    return [{"parent": parent, "mode": mode,
             "design_id": baseline_id(parent, mode),
             "sequence": constants.PARENT_SEQUENCE[parent]}
            for parent in constants.PARENTS for mode in sorted(MODES)]


def read_baseline(run_dir, require=None):
    """{(parent, mode): plddt}. Raises if any `require`d pair is absent."""
    out = {}
    d = os.path.join(run_dir, "baseline")
    for parent in constants.PARENTS:
        for mode in MODES:
            path = os.path.join(d, f"{baseline_id(parent, mode)}.metrics.json")
            if os.path.exists(path):
                out[(parent, mode)] = float(json.load(open(path))["plddt"])
    for key in (require or []):
        if key not in out:
            raise FileNotFoundError(
                f"missing baseline fold for {key}; gating designs against an "
                f"absent baseline would pass or fail all of them")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(sub, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.run_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    jobs = baseline_jobs()

    for mode, settings in sorted(MODES.items()):
        mode_jobs = [j for j in jobs if j["mode"] == mode]
        jobs_tsv = os.path.join(out_dir, f"jobs_{mode}.tsv")
        io.write_tsv(jobs_tsv, [{"design_id": j["design_id"], "sequence": j["sequence"]}
                                for j in mode_jobs], ("design_id", "sequence"))
        cmd = ["python", os.path.join(constants.MONOREPO, "tools/esmfold2/fold_many.py"),
               "--jobs", jobs_tsv, "--out-dir", out_dir,
               "--ligand-ccd", constants.ZN_RESNAME,
               "--num-loops", str(settings["num_loops"]),
               "--num-sampling-steps", str(settings["num_sampling_steps"]),
               "--skip-existing"]
        print(f"[reference_baseline] {mode}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False)

    got = read_baseline(args.run_dir)
    for (parent, mode), plddt in sorted(got.items()):
        print(f"[reference_baseline] {parent} {mode}: pLDDT {plddt:.4f}")
    provenance.write(out_dir, "reference_baseline", len(jobs), len(got),
                     extra={f"{p}__{m}": v for (p, m), v in got.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass, then fold the real baselines**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_reference_baseline.py -q
```
Expected: 5 passed.

Then fold the four baselines on a GPU (4 folds, cheap):

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
sbatch --wait --partition=gpu --gres=gpu:1 -c 4 --mem=48G -t 03:00:00 \
  -o logs/baseline.out -e logs/baseline.err \
  --wrap "source /research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3/bin/activate esmfold2 && cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && PYTHONPATH=. python -m tada_redesign.reference_baseline"
tail -6 logs/baseline.out
```
Report the four pLDDT values. **Expect `full` > `screen` for both parents** — that is the reduced-sampling penalty this module exists to control for. If `screen` ≥ `full`, STOP and report: the sampling settings are not doing what the constants claim.

- [ ] **Step 5: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/reference_baseline.py tada_redesign/tests/test_reference_baseline.py
git commit -F - <<'EOF'
feat: fold both parents through the identical path as a relative baseline

The screen gates on plddt >= parent_plddt - SCREEN_PLDDT_MARGIN, which is only
meaningful if the parent was folded the same way. Reduced sampling depresses
pLDDT substantially, so an absolute floor is meaningless and a full-sampling
parent would make every screened design look bad.

Both modes are folded, and read_baseline raises on a missing required pair
rather than returning a partial dict -- gating against an absent baseline would
silently pass or fail every design.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 5: `score_folds.py` — apply the geometric gates to folded models

Part 1's `score_structure` already provides Kabsch superposition, heavy-atom motif RMSD, cleft clearance and the chain-agnostic metal lookup. This task applies them to folded models and emits `fold_screen.tsv`.

**Files:**
- Create: `tada_redesign/score_folds.py`, `tada_redesign/tests/test_score_folds.py`

**Interfaces:**
- Consumes: `score_structure.{heavy_atoms_from_cif,heavy_atoms_from_pdb,motif_rmsd,cleft_clearance,metal_xyz}`, `motif.{load_masks,arm_residues}`, `substrate.substrate_xyz`, `reference_baseline.read_baseline`, `constants.*`, `io`, `provenance`.
- Produces:
  - `score_folds.COLUMNS`
  - `score_folds.gate(row, parent_plddt) -> tuple[bool, str]` — `(passed, status)`
  - `score_folds.score_one(cif, metrics, ref_atoms, residues, substrate) -> dict`
  - `score_folds.main(argv=None) -> int` — writes `<run>/fold_screen.tsv`

- [ ] **Step 1: Write the failing test**

```python
"""The screen's gate decides what survives to the expensive stages, so its
threshold logic is asserted directly rather than inferred from a run."""
import numpy as np
import pytest

from tada_redesign import constants, score_folds as sf


def _row(plddt=0.80, rmsd=0.5, clearance=2.5, status="ok"):
    return {"plddt": plddt, "motif_rmsd": rmsd, "cleft_clearance": clearance,
            "status": status}


def test_gate_passes_a_good_design():
    ok, status = sf.gate(_row(), parent_plddt=0.80)
    assert ok is True and status == "ok"


def test_gate_rejects_low_plddt_relative_to_the_parent():
    """Relative, not absolute: reduced sampling depresses pLDDT for everything."""
    ok, status = sf.gate(_row(plddt=0.80 - constants.SCREEN_PLDDT_MARGIN - 0.01),
                         parent_plddt=0.80)
    assert ok is False and status == "low_plddt"


def test_gate_accepts_plddt_exactly_at_the_margin():
    ok, _ = sf.gate(_row(plddt=0.80 - constants.SCREEN_PLDDT_MARGIN), parent_plddt=0.80)
    assert ok is True


def test_gate_rejects_motif_drift_using_the_SCREEN_threshold():
    """The screen uses the looser SCREEN_MOTIF_RMSD_MAX, not the final one,
    because reduced-sampling folds are noisier."""
    assert constants.SCREEN_MOTIF_RMSD_MAX > constants.FINAL_MOTIF_RMSD_MAX
    ok, status = sf.gate(_row(rmsd=constants.SCREEN_MOTIF_RMSD_MAX + 0.01),
                         parent_plddt=0.80)
    assert ok is False and status == "motif_drift"
    ok, _ = sf.gate(_row(rmsd=constants.FINAL_MOTIF_RMSD_MAX + 0.01), parent_plddt=0.80)
    assert ok is True          # tighter final threshold must NOT apply here


def test_gate_propagates_an_upstream_failure_status():
    ok, status = sf.gate(_row(status="fold_missing"), parent_plddt=0.80)
    assert ok is False and status == "fold_missing"


def test_gate_rejects_a_nan_measurement_rather_than_passing_it():
    """A nan comparison is False in Python, so a naive `>` test would let a
    broken measurement through as a pass."""
    ok, status = sf.gate(_row(rmsd=float("nan")), parent_plddt=0.80)
    assert ok is False and status == "unmeasurable"


def test_columns_carry_the_cell_coordinates_and_the_gate_inputs():
    for col in ("design_id", "parent", "arm", "partial_t", "temperature", "bias",
                "plddt", "ptm", "motif_rmsd", "cleft_clearance", "status", "passed"):
        assert col in sf.COLUMNS
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_score_folds.py -q
```
Expected: FAIL on import.

- [ ] **Step 3: Write `score_folds.py`**

```python
"""Gate folded designs on confidence and active-site geometry.

Reuses Part 1's score_structure for every geometric measurement -- there is one
Kabsch implementation and one motif-RMSD implementation in this campaign, and
this module adds neither.

Three deliberate choices:
  - The pLDDT gate is RELATIVE to a parent folded in the same mode
    (reference_baseline), because reduced sampling depresses pLDDT for
    everything.
  - The screen uses SCREEN_MOTIF_RMSD_MAX (1.5 A), looser than the final gate's
    1.0 A, because reduced-sampling folds are noisier. Survivors are re-folded at
    full sampling and re-gated at the tighter threshold in Part 3b.
  - A `nan` measurement is a REJECTION with its own status, never a pass. Every
    comparison against nan is False in Python, so a naive `value > threshold`
    test silently admits broken measurements -- the exact shape of the defect
    that turned every Zn distance into nan in Part 2.

Honesty ceiling: pLDDT is model confidence and motif RMSD is geometry. Neither is
stability, solubility, or activity.
"""
import argparse
import json
import math
import os

from . import (constants, io, motif, provenance, reference_baseline,
               score_structure, substrate)

COLUMNS = ("design_id", "backbone", "cell", "parent", "arm", "partial_t",
           "temperature", "bias", "plddt", "ptm", "motif_rmsd",
           "cleft_clearance", "status", "passed")


def _is_bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def gate(row, parent_plddt):
    """(passed, status) for one scored design."""
    if row.get("status", "ok") != "ok":
        return False, row["status"]
    plddt, rmsd = row["plddt"], row["motif_rmsd"]
    if _is_bad(plddt) or _is_bad(rmsd):
        return False, "unmeasurable"
    if plddt < parent_plddt - constants.SCREEN_PLDDT_MARGIN:
        return False, "low_plddt"
    if rmsd > constants.SCREEN_MOTIF_RMSD_MAX:
        return False, "motif_drift"
    return True, "ok"


def score_one(cif_path, metrics_path, ref_atoms, residues, substrate_xyz):
    """Geometry + confidence for one folded design. Never raises."""
    out = {"plddt": float("nan"), "ptm": float("nan"),
           "motif_rmsd": float("nan"), "cleft_clearance": float("nan"),
           "status": "ok"}
    if not (os.path.exists(cif_path) and os.path.exists(metrics_path)):
        out["status"] = "fold_missing"
        return out
    try:
        m = json.load(open(metrics_path))
        out["plddt"] = float(m["plddt"])
        out["ptm"] = float(m.get("ptm", float("nan")))
    except (OSError, ValueError, KeyError) as exc:
        out["status"] = f"metrics_unreadable: {exc}"
        return out
    try:
        atoms = score_structure.heavy_atoms_from_cif(cif_path)
        out["motif_rmsd"] = score_structure.motif_rmsd(ref_atoms, atoms, residues)
        out["cleft_clearance"] = score_structure.cleft_clearance(
            ref_atoms, atoms, substrate_xyz)
    except Exception as exc:                 # noqa: BLE001 - one bad fold must
        out["status"] = f"unscorable: {exc}"  # not kill the stage
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(sub, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    designs = io.read_tsv(os.path.join(args.run_dir, "designs.tsv"))
    if not designs:
        raise SystemExit("[score_folds] designs.tsv is empty or missing")
    baseline = reference_baseline.read_baseline(
        args.run_dir, require=[(p, "screen") for p in constants.PARENTS])

    masks = motif.load_masks()
    az = substrate.substrate_xyz()
    refs = {p: score_structure.heavy_atoms_from_pdb(constants.RMSD_REFERENCE[p])
            for p in constants.PARENTS}
    shard_dir = os.path.join(args.run_dir, "fold_screen")
    out_path = os.path.join(args.run_dir, "fold_screen.tsv")
    if os.path.exists(out_path):
        os.unlink(out_path)

    n_pass = 0
    for d in designs:
        cif = os.path.join(shard_dir, f"{d['design_id']}.cif")
        metrics = os.path.join(shard_dir, f"{d['design_id']}.metrics.json")
        scored = score_one(cif, metrics, refs[d["parent"]],
                           motif.arm_residues(d["arm"], masks), az)
        passed, status = gate(scored, baseline[(d["parent"], "screen")])
        n_pass += passed
        io.append_row(out_path, {
            "design_id": d["design_id"], "backbone": d["backbone"],
            "cell": d["cell"], "parent": d["parent"], "arm": d["arm"],
            "partial_t": d["partial_t"], "temperature": d["temperature"],
            "bias": d["bias"],
            "plddt": round(scored["plddt"], 4), "ptm": round(scored["ptm"], 4),
            "motif_rmsd": round(scored["motif_rmsd"], 3),
            "cleft_clearance": round(scored["cleft_clearance"], 3),
            "status": status, "passed": str(bool(passed)),
        }, COLUMNS)

    print(f"[score_folds] {n_pass}/{len(designs)} passed -> {out_path}")
    final, degraded = provenance.output_path(out_path, len(designs), len(designs))
    provenance.write(args.run_dir, "score_folds", len(designs), len(designs),
                     extra={"n_passed": n_pass,
                            "pass_rate": round(n_pass / float(len(designs)), 4)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note the degraded gate compares rows WRITTEN to inputs, not pass rate — the same correction Part 2's final review required, for the same reason: a low pass rate is a measurement, not a stage failure.

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_score_folds.py -q
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/score_folds.py tada_redesign/tests/test_score_folds.py
git commit -F - <<'EOF'
feat: gate folded designs on confidence and active-site geometry

Reuses Part 1's score_structure for every measurement -- one Kabsch, one motif
RMSD in this campaign. The pLDDT gate is relative to a parent folded in the same
mode, and the screen uses the looser SCREEN_MOTIF_RMSD_MAX because
reduced-sampling folds are noisier; survivors are re-gated at the tighter
threshold in Part 3b.

A nan measurement is rejected with its own `unmeasurable` status rather than
passing: every comparison against nan is False in Python, so a naive
`value > threshold` test silently admits broken measurements -- the exact shape
of the Part 2 defect that turned every Zn distance into nan and would have
rejected all 512 backbones for a plausible-looking reason.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: Preflight additions and the debug shard

**Files:**
- Modify: `tada_redesign/preflight.py`, `tada_redesign/tests/test_preflight.py`
- Create: `docs/logs/20260806_tada_redesign_part3a.md` (in THIS repo)

- [ ] **Step 1: Add two checks and their tests**

Append to `preflight.py`, and register both in `run_checks` before the env probes:

```python
def _fold_many_check():
    """The batch folder must exist and expose --jobs; without it the screen pays
    the model load once per design (~660 GPU-hours at the measured rate)."""
    path = os.path.join(constants.MONOREPO, "tools/esmfold2/fold_many.py")
    if not os.path.exists(path):
        return Check("fold_many available", False, f"missing {path}")
    src = open(path).read()
    ok = "--jobs" in src and "ESMFold2Model.from_pretrained" in src
    return Check("fold_many available", ok, path)


def _designs_enriched_check():
    """designs.tsv must carry the spec's identity_to_parent and mutation_count."""
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(sub, "outputs", constants.RUN_DIR_NAME, "designs.tsv")
    if not os.path.exists(path):
        return Check("designs.tsv enriched", False, f"missing {path}")
    header = open(path).readline().rstrip("\n").split("\t")
    missing = [c for c in ("identity_to_parent", "mutation_count") if c not in header]
    return Check("designs.tsv enriched", not missing,
                 path if not missing else f"missing columns {missing}")
```

Append to `test_preflight.py`:

```python
def test_new_part3_checks_are_registered():
    names = {c.name for c in preflight.run_checks(with_env_probes=False)}
    assert "fold_many available" in names
    assert "designs.tsv enriched" in names
```

- [ ] **Step 2: Run the full suite and preflight**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -q -m "not slow"
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.preflight; echo "exit=$?"
```
Report the ACTUAL suite count (expect ~140) and the preflight result (expect 20/20).

- [ ] **Step 3: Run ONE debug shard end to end**

Set `SHARDS` so a shard is small (~20 designs), fold it, then score:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
sbatch --wait --array=1-1 --export=ALL,SHARDS=527 fold_screen.slurm
tail -12 logs/tada_fold_screen_*_1.out
cat outputs/20260806_tada_redesign_gen1/fold_screen/timing.json
```
Then score just what exists and report the distribution:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.score_folds 2>&1 | tail -3
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python3 -c "
import csv, collections, statistics as st
rows=[r for r in csv.DictReader(open('outputs/20260806_tada_redesign_gen1/fold_screen.tsv'), delimiter='\t') if r['status']!='fold_missing']
print('scored (non-missing):', len(rows))
print('status counts:', dict(collections.Counter(r['status'] for r in rows)))
v=[float(r['plddt']) for r in rows if r['plddt' ]not in ('nan','NA')]
if v: print(f'plddt median {st.median(v):.3f} range {min(v):.3f}-{max(v):.3f}')
w=[float(r['motif_rmsd']) for r in rows if r['motif_rmsd'] not in ('nan','NA')]
if w: print(f'motif_rmsd median {st.median(w):.3f} max {max(w):.3f}')
"
```

**Report, and stop rather than work around, if:** every design fails `unmeasurable` (the folded CIF's chain letter or numbering does not match what `score_structure` expects — the same class of defect as Part 2's relabelled Zn); or pLDDT values fall outside [0, 1] (the scale assumption is wrong); or the folded model has no chain F.

- [ ] **Step 4: Write the log and commit**

Create `docs/logs/20260806_tada_redesign_part3a.md` recording: what Part 3a built; the measured model-load and per-fold seconds; the chosen `FOLD_BATCH_SIZE`/`FOLD_SHARDS` with the arithmetic; the four baseline pLDDT values; the debug shard's status distribution; the suite count; and the honesty ceiling. State plainly that the full screen has NOT been submitted.

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/preflight.py tada_redesign/tests/test_preflight.py docs/logs/20260806_tada_redesign_part3a.md
git commit -F - <<'EOF'
feat: Part 3a preflight checks, debug shard, and measured fold costs

Two checks: fold_many must exist and expose --jobs (without it the screen pays
the model load per design rather than per process), and designs.tsv must carry
the spec's identity_to_parent and mutation_count.

The log records the measured model-load / per-fold split, the resulting batch
size and shard count with their arithmetic, the four parent baseline pLDDT
values, and the debug shard's status distribution. The full screen has not been
submitted.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git push origin main
```

---

## Self-Review

**Spec coverage (Part 3a scope).** Stage 3's ESMFold2 screen → Tasks 2, 3. The relative pLDDT gate and its baseline → Task 4. The geometric gates on folded models → Task 5. `identity_to_parent`/`mutation_count` → Task 1. The AF3-vs-ESMFold2 pLDDT scale hazard → recorded in constants (Part 2) and consumed correctly here; Part 3b must convert before comparing. Preflight coverage → Task 6.

**Deliberately deferred to Part 3b**, and honestly so — none can be sized until this plan's measurements exist: the full-sampling re-fold of survivors, `score_rosetta` (using `relax_core`'s `free_score`), `prep_af3` + AF3 inference, `correlate`, `rank`/`report`, per-backbone bias re-evaluation, and committing `fold.slurm`.

**Known open risks, stated rather than hidden:**
1. **Cost.** If the measured amortised cost puts 10,542 folds beyond ~300 GPU-hours even at batch 1000, Task 2 Step 5 stops and escalates rather than absorbing it. The spec's fallback (gate on MPNN score first, log what was dropped unfolded) is a human decision.
2. **The folded model's chain and numbering are unverified.** `score_folds` assumes ESMFold2 emits chain F with the input numbering. Part 2 was bitten by exactly this class of assumption when RFD3 relabelled the Zn's chain. Task 6's debug shard is where it gets checked, on ~20 designs rather than 10,542.
3. **`relax_core` with `mutations=()`** is assumed to relax an arbitrary input PDB — plausible from its signature but NOT verified. Part 3b's first task must verify it against a real folded model before any Rosetta array.

**Type consistency.** `shard_of(rows, shard, n_shards)` is 1-based in both the module and the SLURM script. `read_baseline(run_dir, require=None)` returns `{(parent, mode): plddt}` and is called with `require=[(p, "screen")]` in `score_folds`. `gate(row, parent_plddt)` takes a dict with `plddt`/`motif_rmsd`/`status` — the exact keys `score_one` returns. `out_paths(out_dir, design_id)` returns `(cif, metrics)` in that order at both call sites.
