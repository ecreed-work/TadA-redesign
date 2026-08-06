# TadA Redesign — Part 2 (Generation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 512 motif-frozen backbones by RFD3 partial diffusion and 10,752 LigandMPNN sequences on them, with the shared infrastructure and preflight coverage that make a batch of that size safe to submit.

**Architecture:** Four foundation modules first (TSV/provenance I/O, an upgraded preflight that actually gates, the substrate/DNA-context extractor, and the RMSD-reference decision), then the generation chain: `prep_rfd_inputs` → `rfd_partial.slurm` → `filter_backbones` → `prep_mpnn_inputs` → `run_ligandmpnn.slurm` → `collect_designs`. The last task is a cheap end-to-end debug gate that **measures** per-stage wall time; no full batch is submitted by this plan.

**Tech Stack:** Python 3.12, numpy, Biopython, PyYAML, pytest. RFdiffusion3 (`rfd3 design`, env `cas9-pam-design`), LigandMPNN (env `ligandmpnn-sc`), SLURM `gpu` partition (H100).

**Spec:** `docs/superpowers/specs/2026-08-05-tada-redesign-design.md`
**Part 1 (complete):** `docs/superpowers/plans/2026-08-05-tada-redesign-part1.md` — `constants.py`, `motif.py`, `score_structure.py`, `preflight.py`, 42 tests green.

## Global Constraints

- Residue numbering is **Met = 1** (UniProt P68398) everywhere. Never use literature TadA-8e mutation labels.
- Scaffold chain is **F**. The catalytic Zn is referenced in RFD3 specs by **CCD name** (`"ZN"`), never chain+resid.
- `masks.json`'s `FROZEN` key (36 residues) is **not** this campaign's motif. Always go through `motif.arm_residues()`.
- `PARTIAL_T` may never reach 8.0 Å — `partial_t=8` was measured (denovo_tada, 2026-08-04) to degrade the active site from ~1.1 Å to 2.5–6.75 Å RMSD, failing every design.
- Cross-repo paths resolve through `TADA_MONOREPO`. All work is in the submodule `tada-redesign/`; commit **there first**, then bump the parent pointer.
- Submit with `sbatch`, never `bsub`. Partitions: `gpu` (`gpu:h100:8`) and `hpcf_test` only. CPU-only stages run on `gpu` **without** `--gres`.
- Push over `ssh://git@ssh.github.com:443/...` — port 22 and HTTPS are blocked from this cluster.
- Every commit carries both trailers:
  `Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>` and
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- Tests run in `ligandmpnn_env`: `cd <submodule> && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -v`
- Shell state does NOT persist between command invocations. Absolute paths or a `cd` in the same command.
- **No batch job is submitted by this plan.** Task 8's debug gate is the only compute, and it is deliberately tiny.
- Honesty ceiling, repeated in every new module docstring: these metrics measure structural plausibility and energetic ranking, not function. No wet-lab validation.

## Verified facts this plan is built on

Measured on 2026-08-05; do not re-derive or "correct" these.

| Fact | Value |
|---|---|
| DNA context window | 6VPC chain **D only**, residues **23–29** (everything within 12 Å of the 8AZ). Chain C has **nothing** within 12 Å — it is excluded entirely, which also keeps RFD3's token budget small. |
| Chain D composition | 11–31 then 39–50 (gap 32–38); **8AZ at residue 26** |
| AtomWorks drops non-standard residues | So RFD3 never sees the 8AZ; the context contigs are **D23–25 and D27–29**, split around it |
| Hotspots | **D25 and D27** — the retained nucleotides flanking the dropped target base |
| RFD3 partial-diffusion spec shape | `{input, partial_t, is_non_loopy, select_fixed_atoms, ligand, select_hotspots, infer_ori_strategy}`, one per cell, `yaml.safe_dump`ed as a dict keyed by cell id (mirrors `denovo_tada/make_rfd_inputs_brace.py::build_partial_specs`) |
| `partial_t` units | **Ångströms of re-noise**, not timesteps (RFD3 recommends ≤ 15) |
| RFD3 accepts a `.pdb` input | `rfd_denovo.slurm`'s from-scratch mode requires `rfd_input.pdb` |
| RFD3 output | `<out_dir>/<name>.cif.gz` plus a sibling `<name>.json` |
| LigandMPNN bias flags exist | `--bias_AA` (`"A:1.0,G:-1.0"`), `--bias_AA_per_residue` (JSON `{"F12": {"L": -1.0}}`), `--omit_AA`, and the `*_multi` variants |
| LigandMPNN batching | `--pdb_path_multi`, `--fixed_residues_multi`, `--bias_AA_per_residue_multi` take JSON maps, so ONE process handles many backbones per temperature |
| LigandMPNN FASTA | first record is the input sequence (**no `id=`**); each design record is `>{name}, id={i}, T={t}, seed={s}, overall_confidence={c}, ligand_confidence={lc}, seq_rec={r}` |
| LigandMPNN output path | `<out_folder>/seqs/<name>.fa` |
| Relaxed parents and the crystal share the 6VPC frame | So DNA lifted from 6VPC can be appended to a relaxed parent without superposition |

## File Structure

All paths relative to the submodule `tada-redesign/`.

**Foundation (Tasks 1–3)**
- `tada_redesign/io.py` — TSV read/append/write with `#`-comment skipping and per-row flush. One responsibility: durable tabular I/O.
- `tada_redesign/provenance.py` — `<stage>.provenance.json` writer and the degraded-run gate.
- `tada_redesign/substrate.py` — 8AZ coordinates, the DNA context window, and its gap/non-standard-aware contigs.
- Modified: `tada_redesign/constants.py` (RMSD reference, pLDDT scales, bias magnitudes, DNA context), `tada_redesign/preflight.py` (all five env probes, `require_green`, RMSD-reference completeness).

**Generation (Tasks 4–7)**
- `tada_redesign/prep_rfd_inputs.py` — builds the per-cell RFD3 input PDB and `rfd_inputs.yaml`.
- `rfd_partial.slurm` — `rfd3 design` over the 16 cells.
- `tada_redesign/filter_backbones.py` — motif drift, chain breaks, length, Zn donors → `backbones.tsv`.
- `tada_redesign/prep_mpnn_inputs.py` — per-arm multi-JSONs (pdb paths, fixed residues, per-residue bias).
- `run_ligandmpnn.slurm` — one process per (arm, temperature), plus the zero-bias control.
- `tada_redesign/_run_ligandmpnn.py` — vendored numpy-compat wrapper (cited copy).
- `tada_redesign/collect_designs.py` — FASTA → `designs.tsv`.

**Tests:** one module per unit under `tada_redesign/tests/`.

Run dir: `outputs/20260805_tada_redesign/` (gitignored).

---

### Task 1: `io.py` and `provenance.py` — durable tabular I/O and the degraded gate

Every later stage is a sharded array whose output must survive a killed job and must refuse to look complete when it isn't. The spec requires incremental writes, `#`-comment tolerance, a provenance sidecar per stage, and a degraded-run refusal at `DEGRADED_FRACTION`. `constants.DEGRADED_FRACTION` has existed since Part 1 and is still unused — this task is what uses it.

**Files:**
- Create: `tada_redesign/io.py`, `tada_redesign/provenance.py`
- Test: `tada_redesign/tests/test_io.py`, `tada_redesign/tests/test_provenance.py`

**Interfaces:**
- Consumes: `constants.DEGRADED_FRACTION` (0.20), `constants.MONOREPO`.
- Produces:
  - `io.read_tsv(path) -> list[dict]` — skips blank and `#` lines, strips `\r`.
  - `io.write_tsv(path, rows, columns, header_comment=None) -> None`
  - `io.append_row(path, row, columns) -> None` — writes the header if the file is absent or empty, then one row, flushed and `os.fsync`'d.
  - `io.count_rows(path) -> int` — 0 for a missing file.
  - `provenance.write(stage_dir, stage, n_in, n_out, extra=None) -> str` (returns the JSON path)
  - `provenance.is_degraded(n_in, n_out, fraction=None) -> bool`
  - `provenance.output_path(path, n_in, n_out) -> tuple[str, bool]` — `(canonical, False)` or `(<stem>.degraded.tsv, True)`

- [ ] **Step 1: Write the failing tests**

`tada_redesign/tests/test_io.py`:

```python
"""Tabular I/O is load-bearing: a 10k-row array stage must survive being killed,
and a reader that chokes on a '#' header silently loses every row after it."""
import os

import pytest

from tada_redesign import io as tio

COLUMNS = ("design_id", "parent", "score")


def test_append_row_creates_the_header_then_appends(tmp_path):
    path = str(tmp_path / "out.tsv")
    tio.append_row(path, {"design_id": "d1", "parent": "TadA8e", "score": "1.5"}, COLUMNS)
    tio.append_row(path, {"design_id": "d2", "parent": "TadA9", "score": "2.5"}, COLUMNS)
    lines = open(path).read().splitlines()
    assert lines[0] == "design_id\tparent\tscore"
    assert len(lines) == 3
    rows = tio.read_tsv(path)
    assert [r["design_id"] for r in rows] == ["d1", "d2"]


def test_read_tsv_skips_comment_and_blank_lines(tmp_path):
    path = tmp_path / "c.tsv"
    path.write_text("# generated by something\n\ndesign_id\tparent\tscore\n"
                    "# a mid-file note\nd1\tTadA8e\t1.5\n")
    rows = tio.read_tsv(str(path))
    assert len(rows) == 1
    assert rows[0]["parent"] == "TadA8e"


def test_read_tsv_strips_carriage_returns(tmp_path):
    path = tmp_path / "crlf.tsv"
    path.write_bytes(b"design_id\tparent\tscore\r\nd1\tTadA8e\t1.5\r\n")
    rows = tio.read_tsv(str(path))
    assert rows[0]["score"] == "1.5"


def test_read_tsv_returns_empty_for_a_missing_file(tmp_path):
    assert tio.read_tsv(str(tmp_path / "nope.tsv")) == []


def test_count_rows_is_zero_for_missing_and_header_only(tmp_path):
    assert tio.count_rows(str(tmp_path / "nope.tsv")) == 0
    p = tmp_path / "hdr.tsv"
    p.write_text("design_id\tparent\tscore\n")
    assert tio.count_rows(str(p)) == 0


def test_append_row_rejects_an_unknown_column(tmp_path):
    """A typo'd key must fail loudly rather than being dropped from the row."""
    path = str(tmp_path / "out.tsv")
    with pytest.raises(ValueError):
        tio.append_row(path, {"design_id": "d1", "typo": "x"}, COLUMNS)


def test_append_row_fills_missing_columns_with_NA(tmp_path):
    path = str(tmp_path / "out.tsv")
    tio.append_row(path, {"design_id": "d1"}, COLUMNS)
    assert tio.read_tsv(path)[0]["score"] == "NA"


def test_write_tsv_emits_the_header_comment_first(tmp_path):
    path = str(tmp_path / "w.tsv")
    tio.write_tsv(path, [{"design_id": "d1", "parent": "p", "score": "1"}],
                  COLUMNS, header_comment="dropped 3 rows")
    text = open(path).read()
    assert text.startswith("# dropped 3 rows\n")
    assert tio.read_tsv(path)[0]["design_id"] == "d1"
```

`tada_redesign/tests/test_provenance.py`:

```python
"""A stage that lost most of its inputs must refuse the canonical output path."""
import json

from tada_redesign import constants, provenance


def test_is_degraded_at_the_threshold(tmp_path):
    assert provenance.is_degraded(100, 79) is True      # 21% lost
    assert provenance.is_degraded(100, 80) is False     # exactly 20% lost
    assert provenance.is_degraded(100, 100) is False


def test_is_degraded_treats_zero_inputs_as_not_degraded():
    """No inputs is an upstream problem, not this stage failing."""
    assert provenance.is_degraded(0, 0) is False


def test_output_path_diverts_a_degraded_run(tmp_path):
    canonical = str(tmp_path / "backbones.tsv")
    path, degraded = provenance.output_path(canonical, 100, 10)
    assert degraded is True
    assert path.endswith("backbones.degraded.tsv")
    path, degraded = provenance.output_path(canonical, 100, 95)
    assert (path, degraded) == (canonical, False)


def test_write_records_counts_and_thresholds(tmp_path):
    p = provenance.write(str(tmp_path), "filter_backbones", 512, 480,
                         extra={"cells": 16})
    doc = json.load(open(p))
    assert p.endswith("filter_backbones.provenance.json")
    assert doc["stage"] == "filter_backbones"
    assert doc["n_in"] == 512 and doc["n_out"] == 480
    assert doc["is_degraded"] is False          # always present, never absent
    assert doc["degraded_fraction"] == constants.DEGRADED_FRACTION
    assert doc["extra"]["cells"] == 16
    assert "submodule_sha" in doc


def test_write_marks_a_degraded_run(tmp_path):
    doc = json.load(open(provenance.write(str(tmp_path), "s", 100, 5)))
    assert doc["is_degraded"] is True
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_io.py tada_redesign/tests/test_provenance.py -v
```
Expected: FAIL — `ImportError: cannot import name 'io'` / `'provenance'`.

- [ ] **Step 3: Write `io.py`**

```python
"""Durable tabular I/O for the campaign's array stages.

Every stage writes incrementally -- header plus one flushed row at a time -- so a
killed SLURM array leaves a parsable partial TSV instead of nothing, and can be
resumed rather than restarted. Every reader skips `#` comment lines, because
several stages annotate their output with what they dropped, and a reader that
treats a comment as data silently loses the rest of the file.

Honesty ceiling: this module moves numbers around. It does not compute or
validate any biophysical quantity.
"""
import csv
import os

COMMENT = "#"
MISSING = "NA"


def read_tsv(path):
    """Rows as dicts. Missing file -> []. Skips blank and `#` lines, strips \\r."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        clean = (ln.replace("\r", "") for ln in fh
                 if ln.strip() and not ln.lstrip().startswith(COMMENT))
        return list(csv.DictReader(clean, delimiter="\t"))


def _validate(row, columns):
    unknown = set(row) - set(columns)
    if unknown:
        raise ValueError(f"unknown column(s) {sorted(unknown)}; expected {list(columns)}")
    return {c: row.get(c, MISSING) for c in columns}


def append_row(path, row, columns):
    """Append one row, writing the header first if the file is absent/empty.

    Flushed and fsync'd per row: an array task that dies mid-stage must leave
    every row it already computed on disk.
    """
    values = _validate(row, columns)
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                                lineterminator="\n")
        if need_header:
            writer.writeheader()
        writer.writerow(values)
        fh.flush()
        os.fsync(fh.fileno())


def write_tsv(path, rows, columns, header_comment=None):
    """Whole-file write. `header_comment` is emitted as a leading `#` line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        if header_comment:
            fh.write(f"{COMMENT} {header_comment}\n")
        writer = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_validate(row, columns))


def count_rows(path):
    return len(read_tsv(path))
```

- [ ] **Step 4: Write `provenance.py`**

```python
"""Per-stage provenance sidecars and the degraded-run refusal.

Two failure modes this guards. First, a run whose inputs or checkpoints cannot
later be identified is not reproducible, so every stage records what it read,
what it wrote, and which code produced it. Second, a stage that failed on most
of its inputs must not hand downstream a canonical-looking output file: past
`DEGRADED_FRACTION` it writes `<stem>.degraded.tsv` instead, so a later stage
reading the canonical path finds nothing rather than a quiet subset.

`is_degraded` is always written to the JSON, never omitted -- a consumer testing
`"is_degraded" in doc` must not read a clean run as indeterminate.
"""
import json
import os
import subprocess

from . import constants

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _submodule_sha():
    try:
        out = subprocess.run(["git", "-C", PACKAGE_DIR, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def is_degraded(n_in, n_out, fraction=None):
    """True when more than `fraction` of the inputs produced no output.

    Zero inputs is NOT degraded: that is an upstream failure to report, not this
    stage silently succeeding on nothing.
    """
    fraction = constants.DEGRADED_FRACTION if fraction is None else fraction
    if n_in <= 0:
        return False
    return (n_in - n_out) / float(n_in) > fraction


def output_path(canonical, n_in, n_out, fraction=None):
    if not is_degraded(n_in, n_out, fraction):
        return canonical, False
    stem, ext = os.path.splitext(canonical)
    return f"{stem}.degraded{ext}", True


def write(stage_dir, stage, n_in, n_out, extra=None):
    """Write `<stage_dir>/<stage>.provenance.json`; return its path."""
    doc = {
        "stage": stage,
        "n_in": n_in,
        "n_out": n_out,
        "is_degraded": is_degraded(n_in, n_out),
        "degraded_fraction": constants.DEGRADED_FRACTION,
        "submodule_sha": _submodule_sha(),
        "monorepo": constants.MONOREPO,
        "extra": extra or {},
    }
    os.makedirs(stage_dir, exist_ok=True)
    path = os.path.join(stage_dir, f"{stage}.provenance.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return path
```

- [ ] **Step 5: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_io.py tada_redesign/tests/test_provenance.py -v
```
Expected: 13 passed (8 io + 5 provenance).

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/io.py tada_redesign/provenance.py tada_redesign/tests/test_io.py tada_redesign/tests/test_provenance.py
git commit -F - <<'EOF'
feat: durable tabular I/O and the degraded-run refusal

Every array stage writes header-plus-one-flushed-row so a killed SLURM task
leaves a parsable partial TSV, and every reader skips `#` comments because
stages annotate what they dropped.

provenance.write() records stage, row counts, thresholds and the submodule SHA,
always emitting is_degraded so a consumer cannot misread a clean run as
indeterminate. Past DEGRADED_FRACTION (0.20, defined in Part 1 and until now
unused) output_path() diverts to <stem>.degraded.tsv, so a downstream stage
reading the canonical path finds nothing rather than a quiet subset.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: preflight that actually gates — five envs, RMSD reference, pLDDT scales

Three defects from Part 1's final review, all of which would waste GPU time. Preflight probes 2 of the 5 conda envs the campaign runs on, so it can report green and then 16 GPU tasks die on `import rfd3`. Nothing declares which structure is the authoritative RMSD reference, and the two candidates differ by **2.166 Å** over the FULL arm — twice `BACKBONE_MOTIF_RMSD_MAX` — while the crystal is missing nine FULL-arm sidechain atoms (Arg153, Asn157) so `motif_rmsd` raises `KeyError` against it. And nothing records that AF3 reports pLDDT on 0–100 while `SCREEN_PLDDT_MARGIN` assumes ESMFold2's 0–1, a 20× error waiting for Part 3.

**Files:**
- Modify: `tada_redesign/constants.py`, `tada_redesign/preflight.py`
- Test: `tada_redesign/tests/test_constants.py`, `tada_redesign/tests/test_preflight.py` (append)

**Interfaces:**
- Consumes: `motif.arm_residues`, `score_structure.heavy_atoms_from_pdb`, `constants.PARENT_PDB`.
- Produces:
  - `constants.RMSD_REFERENCE` — `dict` parent → relaxed PDB path (the authoritative reference).
  - `constants.ESMFOLD_PLDDT_SCALE = 1.0`, `constants.AF3_PLDDT_SCALE = 100.0`
  - `constants.ENV_MODULES` — tuple of `(env_name, module_to_import)` for all five envs.
  - `preflight.require_green(with_env_probes=True) -> None` — raises `SystemExit` naming failures.
  - `preflight._rmsd_reference_check()` — every FULL-arm heavy atom present in every reference.

- [ ] **Step 1: Write the failing tests**

Append to `tada_redesign/tests/test_constants.py`:

```python
def test_rmsd_reference_is_the_relaxed_parents_not_the_crystal():
    """Partial diffusion starts from the relaxed parent, so motif drift must be
    measured against that same coordinate set. Measured divergence between the
    two candidates is 2.166 A over the FULL arm -- 2x BACKBONE_MOTIF_RMSD_MAX --
    and the crystal is missing nine FULL-arm sidechain atoms."""
    assert constants.RMSD_REFERENCE == constants.PARENT_PDB
    assert constants.CHAINF_RAW not in constants.RMSD_REFERENCE.values()


def test_plddt_scales_are_recorded_for_both_folding_models():
    """ESMFold2 reports 0-1, AF3 reports 0-100. Reusing one margin across both
    would be a 20x error."""
    assert constants.ESMFOLD_PLDDT_SCALE == 1.0
    assert constants.AF3_PLDDT_SCALE == 100.0
    assert constants.SCREEN_PLDDT_MARGIN < constants.ESMFOLD_PLDDT_SCALE


def test_env_modules_covers_every_env_the_campaign_uses():
    envs = {e for e, _ in constants.ENV_MODULES}
    assert envs == {constants.ENV_TEST, constants.ENV_ROSETTA,
                    constants.ENV_RFD3, constants.ENV_MPNN, constants.ENV_ESM}
```

Append to `tada_redesign/tests/test_preflight.py`:

```python
def test_env_probes_cover_all_five_envs():
    """A gate that checks 2 of 5 envs lets a batch die on `import rfd3` after
    reporting green."""
    from tada_redesign import constants
    names = {c.name for c in preflight.run_checks(with_env_probes=True)}
    for env, module in constants.ENV_MODULES:
        assert f"conda env '{env}' has {module}" in names


def test_rmsd_reference_check_is_present_and_passes():
    checks = {c.name: c for c in preflight.run_checks(with_env_probes=False)}
    assert "RMSD reference completeness" in checks
    assert checks["RMSD reference completeness"].ok is True


def test_require_green_raises_when_a_check_fails(monkeypatch):
    monkeypatch.setattr(preflight, "run_checks",
                        lambda **kw: [preflight.Check("bad", False, "nope")])
    with pytest.raises(SystemExit) as excinfo:
        preflight.require_green(with_env_probes=False)
    assert "bad" in str(excinfo.value)


def test_require_green_returns_quietly_when_all_pass(monkeypatch):
    monkeypatch.setattr(preflight, "run_checks",
                        lambda **kw: [preflight.Check("fine", True, "ok")])
    assert preflight.require_green(with_env_probes=False) is None
```

Note: `test_env_probes_cover_all_five_envs` runs five real `conda run` probes and is therefore slow (tens of seconds each). Mark it so the fast suite can skip it:

```python
import pytest
pytestmark_slow = pytest.mark.slow
```
and decorate that one test with `@pytest.mark.slow`. Register the marker in `pytest.ini`:

```ini
[pytest]
testpaths = tada_redesign/tests
python_files = test_*.py
markers =
    slow: exercises real conda-run probes; deselect with -m "not slow"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_constants.py tada_redesign/tests/test_preflight.py -v -m "not slow"
```
Expected: FAIL — `AttributeError: module 'tada_redesign.constants' has no attribute 'RMSD_REFERENCE'` and `no attribute 'require_green'`.

- [ ] **Step 3: Extend `constants.py`**

Add after the existing `PARENT_PDB` / `CHAINF_RAW` block:

```python
# The authoritative reference for every motif RMSD is each parent's RELAXED
# structure, never the crystal. Partial diffusion starts FROM the relaxed
# parent, so drift must be measured against that same coordinate set. Measured
# 2026-08-05: FULL-arm heavy-atom RMSD crystal -> TadA8e.pdb is 2.166 A, twice
# BACKBONE_MOTIF_RMSD_MAX, and chainF_raw.pdb is missing nine FULL-arm sidechain
# atoms (Arg153, Asn157), so motif_rmsd raises KeyError against it. chainF_raw
# is used ONLY as check_zn_geometry's crystallographic Zn reference.
RMSD_REFERENCE = dict(PARENT_PDB)
```

Add next to `SCREEN_PLDDT_MARGIN`:

```python
# The two folding models do NOT share a pLDDT scale: ESMFold2 reports 0-1,
# AF3 reports 0-100. SCREEN_PLDDT_MARGIN is expressed on ESMFold2's scale;
# any AF3 comparison must scale by AF3_PLDDT_SCALE / ESMFOLD_PLDDT_SCALE first.
ESMFOLD_PLDDT_SCALE = 1.0
AF3_PLDDT_SCALE = 100.0
```

Add next to the `ENV_*` names:

```python
# (env, module) pairs preflight probes. All five are load-bearing: a batch that
# starts before its env is verified dies on an import after preflight said green.
ENV_MODULES = (
    (ENV_TEST, "Bio.PDB"),
    (ENV_ROSETTA, "pyrosetta"),
    (ENV_RFD3, "rfd3"),
    (ENV_MPNN, "prody"),
    (ENV_ESM, "transformers"),
)
```

Add the DNA-context and bias constants (used by Tasks 3, 4 and 6):

```python
# DNA context, measured 2026-08-05: chain D residues 23-29 are everything within
# 12 A of the 8AZ. Chain C has NOTHING within 12 A and is excluded entirely,
# which also keeps RFD3's fixed-context token budget small (an earlier campaign
# OOM'd an 80 GB A100 by including the whole Cas9 context).
DNA_CONTEXT_CUTOFF = 12.0
DNA_CONTEXT_RESIDS = (23, 24, 25, 26, 27, 28, 29)
# AtomWorks drops non-standard residues on load, so RFD3 never sees the 8AZ at
# D26; hotspots go on the retained nucleotides flanking it.
HOTSPOT_RESIDS = (25, 27)

# LigandMPNN solubility biasing, applied ONLY at exposed, designable positions.
# Magnitudes are logit offsets and are deliberately modest; the zero-bias control
# set exists to measure whether they help at all rather than assuming it.
HYDROPHOBIC_SET = "FILMVW"
POLAR_SET = "DEKNQRST"
HYDROPHOBIC_BIAS = -1.0
POLAR_BIAS = 0.3
```

- [ ] **Step 4: Extend `preflight.py`**

Replace the two hard-coded env probes in `run_checks` with a loop over `constants.ENV_MODULES`:

```python
    if with_env_probes:
        for env, module in constants.ENV_MODULES:
            checks.append(_conda_env_check(env, module))
    return checks
```

Add the RMSD-reference check and register it in `run_checks` (before the env probes):

```python
def _rmsd_reference_check():
    """Every FULL-arm heavy atom must exist in every RMSD reference.

    `motif_rmsd` raises KeyError on a missing measured atom -- correct behaviour,
    but if the REFERENCE is the incomplete structure then every design fails at
    once, after the folds are already paid for. The crystal chainF_raw.pdb is
    exactly such a structure (missing nine FULL-arm sidechain atoms at Arg153 and
    Asn157), which is why RMSD_REFERENCE is the relaxed parents.
    """
    from . import motif, score_structure
    try:
        masks = motif.load_masks()
    except (OSError, ValueError, KeyError) as exc:
        return Check("RMSD reference completeness", False, f"masks unreadable: {exc}")
    residues = set(motif.arm_residues(motif.ARM_FULL, masks))
    problems = []
    for parent, pdb in constants.RMSD_REFERENCE.items():
        if not os.path.exists(pdb):
            problems.append(f"{parent}: missing {pdb}")
            continue
        atoms = score_structure.heavy_atoms_from_pdb(pdb)
        present = {resnum for resnum, _ in atoms}
        absent = sorted(residues - present)
        if absent:
            problems.append(f"{parent}: no atoms for residues {absent}")
    return Check("RMSD reference completeness", not problems,
                 "all FULL-arm residues present in both references"
                 if not problems else "; ".join(problems))


def require_green(with_env_probes=True):
    """Raise SystemExit unless every check passes.

    Stages call this before doing any work, so a stage cannot run ungated --
    the spec requires preflight to "refuse to let any batch stage run", which a
    gate nobody calls does not do.
    """
    failed = [c for c in run_checks(with_env_probes=with_env_probes) if not c.ok]
    if failed:
        raise SystemExit(
            "preflight FAILED; refusing to run: "
            + "; ".join(f"{c.name} ({c.detail})" for c in failed))
```

- [ ] **Step 5: Run the fast suite, then the slow probe once**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -v -m "not slow"
```
Expected: **61 passed** (42 from Part 1 + 13 from Task 1 + 7 new, minus the 1 `slow`-marked test that `-m "not slow"` deselects). Report the actual number.

Then the five real env probes, once:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -v -m slow
```
Expected: 1 passed. **If any env probe FAILS, do not delete or weaken the probe** — report BLOCKED with which env lacks which module. A missing `rfd3` or `prody` is exactly the finding this task exists to surface, and it must be fixed in the environment, not in the test.

- [ ] **Step 6: Run preflight for real**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.preflight; echo "exit=$?"
```
Expected: 18 checks (12 non-env + 5 env + RMSD reference), all PASS except the LigandMPNN checkpoint if weights are absent. Record the verbatim output.

- [ ] **Step 7: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/constants.py tada_redesign/preflight.py pytest.ini tada_redesign/tests/test_constants.py tada_redesign/tests/test_preflight.py
git commit -F - <<'EOF'
fix: preflight gates all five envs, the RMSD reference, and require_green

Three defects from Part 1's final review, each of which would have wasted GPU
time:

- preflight probed 2 of the 5 conda envs the campaign runs on, so it could
  report green and then have 16 GPU tasks die on `import rfd3`. Now driven by
  constants.ENV_MODULES, covering all five.
- nothing declared which structure is the authoritative RMSD reference. It is
  the RELAXED parents: measured FULL-arm RMSD crystal -> TadA8e.pdb is 2.166 A
  (2x the gate), and the crystal is missing nine FULL-arm sidechain atoms at
  Arg153/Asn157 so motif_rmsd raises KeyError against it. A new check asserts
  the reference carries every FULL-arm heavy atom.
- require_green() lets a stage refuse to run ungated, which the spec requires
  and a gate nobody calls does not provide.

Also records that AF3 reports pLDDT on 0-100 while ESMFold2 uses 0-1 -- reusing
SCREEN_PLDDT_MARGIN across both would be a 20x error -- plus the measured DNA
context window and the LigandMPNN bias magnitudes.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: `substrate.py` — the 8AZ target base and the DNA context window

Three consumers need this and none should re-derive it: `prep_rfd_inputs` (context contigs and hotspots), `score_structure.cleft_clearance` (the substrate coordinates it measures against), and Part 3's scorers. Part 1's tests each built the 8AZ extraction inline — that duplication stops here.

**Files:**
- Create: `tada_redesign/substrate.py`
- Test: `tada_redesign/tests/test_substrate.py`

**Interfaces:**
- Consumes: `constants.PDB6VPC`, `constants.SUBSTRATE_CHAIN` (`"D"`), `constants.SUBSTRATE_RESID` (26), `constants.SUBSTRATE_RESNAME` (`"8AZ"`), `constants.DNA_CONTEXT_RESIDS`, `constants.HOTSPOT_RESIDS`.
- Produces:
  - `substrate.substrate_xyz(pdb=None) -> np.ndarray` — the 8AZ heavy atoms, shape (22, 3).
  - `substrate.context_residues(pdb=None, cutoff=None) -> tuple[int, ...]` — chain D residues within `cutoff` of the 8AZ, **including** 26.
  - `substrate.context_contigs(resids, drop=(26,)) -> tuple[tuple[int, int], ...]` — maximal contiguous runs after dropping non-standard residues.
  - `substrate.STANDARD_DNA = ("DA", "DC", "DG", "DT")`

- [ ] **Step 1: Write the failing test**

```python
"""The substrate is measured once, here. Two facts are load-bearing: AtomWorks
drops the 8AZ so RFD3's context must be split around it, and chain C has nothing
within 12 A so it is excluded entirely."""
import numpy as np
import pytest

from tada_redesign import constants, substrate


def test_substrate_xyz_returns_the_8az_heavy_atoms():
    xyz = substrate.substrate_xyz()
    assert xyz.shape == (22, 3)
    assert xyz.dtype == float


def test_context_residues_are_chain_d_23_to_29():
    """Measured 2026-08-05 at a 12 A cutoff."""
    assert substrate.context_residues() == (23, 24, 25, 26, 27, 28, 29)
    assert substrate.context_residues() == constants.DNA_CONTEXT_RESIDS


def test_context_contigs_split_around_the_dropped_8az():
    """RFD3's AtomWorks loader drops non-standard residues, and a labelled contig
    range spanning a residue it never loaded raises ComponentValidationError."""
    assert substrate.context_contigs((23, 24, 25, 26, 27, 28, 29)) == ((23, 25), (27, 29))


def test_context_contigs_split_around_a_numbering_gap():
    """Chain D really has a 32-38 gap; a range spanning it would also fail."""
    assert substrate.context_contigs((29, 30, 31, 39, 40)) == ((29, 31), (39, 40))


def test_context_contigs_drops_only_what_it_is_told_to():
    assert substrate.context_contigs((23, 24, 25), drop=()) == ((23, 25),)


def test_hotspots_are_the_retained_neighbours_of_the_target_base():
    resids = substrate.context_residues()
    for h in constants.HOTSPOT_RESIDS:
        assert h in resids
        assert h != constants.SUBSTRATE_RESID     # the base itself is dropped


def test_substrate_xyz_is_in_the_same_frame_as_the_reference_parents():
    """cleft_clearance treats these coordinates as living in the reference frame.
    Sanity: the 8AZ must sit within a few Angstrom of the relaxed parent's Zn,
    which is the catalytic geometry the whole campaign is built on."""
    from tada_redesign import score_structure as ss
    atoms = ss.heavy_atoms_from_pdb(constants.PARENT_PDB["TadA8e"])
    zn = next(v for (resnum, name), v in atoms.items() if name == "ZN")
    d = float(np.min(np.linalg.norm(substrate.substrate_xyz() - zn, axis=1)))
    assert 1.5 < d < 4.0, d
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_substrate.py -v
```
Expected: FAIL — `ImportError: cannot import name 'substrate'`.

- [ ] **Step 3: Write `substrate.py`**

```python
"""The 8-azanebularine target-base analogue and the ssDNA context window.

Measured once here so no consumer re-derives it. Two facts drive the shape of
this module:

  - RFD3's AtomWorks loader DROPS non-standard residues, and the 8AZ (chain D
    residue 26) is one. A labelled contig range spanning a residue the loader
    never produced raises ComponentValidationError, so the context must be
    emitted as maximal contiguous runs split around it -- and around chain D's
    real 32-38 numbering gap.
  - Measured 2026-08-05: chain D residues 23-29 are everything within 12 A of
    the 8AZ, and chain C has NOTHING within 12 A. Chain C is therefore excluded
    entirely, which also keeps RFD3's fixed-context token budget small.

Honesty ceiling: these are coordinates and residue numbers from a crystal
structure. Nothing here says a design binds the substrate.
"""
import numpy as np

from . import constants

STANDARD_DNA = ("DA", "DC", "DG", "DT")


def _model(pdb=None):
    from Bio.PDB import PDBParser
    return PDBParser(QUIET=True).get_structure(
        "ref", pdb or constants.PDB6VPC)[0]


def substrate_xyz(pdb=None):
    """Heavy-atom coordinates of the 8AZ target-base analogue, shape (N, 3).

    In the 6VPC frame, which the relaxed parents share -- so these can be used
    directly as `cleft_clearance`'s reference-frame substrate.
    """
    chain = _model(pdb)[constants.SUBSTRATE_CHAIN]
    res = next(r for r in chain
               if r.get_resname().strip() == constants.SUBSTRATE_RESNAME)
    return np.array([a.get_coord() for a in res], dtype=float)


def context_residues(pdb=None, cutoff=None):
    """Chain-D residue numbers within `cutoff` of the 8AZ, including the 8AZ."""
    cutoff = constants.DNA_CONTEXT_CUTOFF if cutoff is None else cutoff
    az = substrate_xyz(pdb)
    keep = []
    for res in _model(pdb)[constants.SUBSTRATE_CHAIN]:
        xyz = np.array([a.get_coord() for a in res], dtype=float)
        if float(np.min(np.linalg.norm(
                xyz[:, None, :] - az[None, :, :], axis=2))) < cutoff:
            keep.append(res.id[1])
    return tuple(sorted(keep))


def context_contigs(resids, drop=None):
    """Maximal contiguous runs over `resids` after removing `drop`.

    Default drops the 8AZ, which RFD3's loader never produces.
    """
    drop = (constants.SUBSTRATE_RESID,) if drop is None else drop
    kept = sorted(set(resids) - set(drop))
    runs = []
    for resid in kept:
        if runs and resid == runs[-1][1] + 1:
            runs[-1][1] = resid
        else:
            runs.append([resid, resid])
    return tuple((lo, hi) for lo, hi in runs)
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_substrate.py -v
```
Expected: 7 passed. If `test_substrate_xyz_is_in_the_same_frame_as_the_reference_parents` fails, STOP and report the measured distance — a frame mismatch would silently invalidate `cleft_clearance` for the whole campaign.

- [ ] **Step 5: Refactor Part 1's inline duplicate**

`tada_redesign/tests/test_score_structure.py` defines a local `_substrate_xyz()` helper. Replace its body with a delegation so there is one extractor:

```python
def _substrate_xyz():
    """Delegates to substrate.substrate_xyz -- one extractor, not two."""
    from tada_redesign import substrate
    return substrate.substrate_xyz()
```

Re-run that module: `conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_score_structure.py -v` — expected 12 passed, unchanged numbers (2.121 / 2.330 / 2.211 / 2.271).

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/substrate.py tada_redesign/tests/test_substrate.py tada_redesign/tests/test_score_structure.py
git commit -F - <<'EOF'
feat: single extractor for the 8AZ target base and the ssDNA context window

Measured once, consumed by prep_rfd_inputs, cleft_clearance and Part 3's
scorers. Two facts shape it: AtomWorks drops the 8AZ at chain D 26, so context
contigs must split around it (and around chain D's real 32-38 gap) or RFD3
raises ComponentValidationError; and chain D 23-29 is everything within 12 A of
the target base while chain C has nothing within 12 A, so chain C is excluded
entirely and the fixed-context token budget stays small.

Also collapses test_score_structure's inline 8AZ helper into a delegation, so
there is one extractor rather than two.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 4: `prep_rfd_inputs.py` and `rfd_partial.slurm` — the 16-cell partial-diffusion inputs

**Files:**
- Create: `tada_redesign/prep_rfd_inputs.py`, `rfd_partial.slurm`
- Test: `tada_redesign/tests/test_prep_rfd_inputs.py`

**Interfaces:**
- Consumes: `motif.load_masks`, `motif.rfd_select_fixed_atoms`, `substrate.context_residues`, `substrate.context_contigs`, `constants.{PARENTS,ARMS,PARTIAL_T,PARENT_PDB,PDB6VPC,SCAFFOLD_CHAIN,ZN_RESNAME,SUBSTRATE_CHAIN,HOTSPOT_RESIDS,RUN_DIR_NAME}`, `provenance.write`.
- Produces:
  - `prep_rfd_inputs.cell_id(parent, arm, partial_t) -> str` — e.g. `"TadA8e_FULL_pt1.0"`.
  - `prep_rfd_inputs.write_input_pdb(parent, out_path, pdb6vpc=None) -> str` — relaxed parent (chain F + Zn) plus the chain-D context residues, one file per parent.
  - `prep_rfd_inputs.build_spec(parent, arm, partial_t, masks, input_pdb) -> dict`
  - `prep_rfd_inputs.build_specs(masks, input_pdbs) -> dict[str, dict]` — 16 entries.
  - `prep_rfd_inputs.main(argv=None) -> int` — writes `<run>/rfd_in/<parent>.pdb` and `<run>/rfd_in/rfd_inputs.yaml`.

- [ ] **Step 1: Write the failing test**

```python
"""RFD3 spec semantics are unforgiving and every rule here was learned from a
real failure -- see the citations in prep_rfd_inputs' docstring."""
import os

import pytest
import yaml

from tada_redesign import constants, motif, prep_rfd_inputs as prep


@pytest.fixture
def masks():
    return motif.load_masks()


def test_cell_id_encodes_all_three_axes():
    assert prep.cell_id("TadA8e", "FULL", 1.0) == "TadA8e_FULL_pt1.0"
    assert prep.cell_id("TadA9", "MIN", 6.0) == "TadA9_MIN_pt6.0"


def test_build_specs_covers_the_sixteen_cells(masks):
    specs = prep.build_specs(masks, {p: f"/tmp/{p}.pdb" for p in constants.PARENTS})
    assert len(specs) == 16 == len(constants.PARENTS) * len(constants.ARMS) * len(constants.PARTIAL_T)
    assert prep.cell_id("TadA8e", "FULL", 1.0) in specs


def test_spec_is_partial_diffusion_with_no_contig(masks):
    """Partial diffusion re-noises the input in place; emitting a contig would
    make RFD3 build a new chain instead."""
    spec = prep.build_spec("TadA8e", "FULL", 2.0, masks, "/tmp/TadA8e.pdb")
    assert spec["input"] == "/tmp/TadA8e.pdb"
    assert spec["partial_t"] == 2.0
    assert "contig" not in spec
    assert spec["is_non_loopy"] is True


def test_spec_fixes_the_arm_motif_and_the_zn_by_ccd_name(masks):
    spec = prep.build_spec("TadA8e", "MIN", 1.0, masks, "/tmp/x.pdb")
    fixed = spec["select_fixed_atoms"]
    assert fixed == motif.rfd_select_fixed_atoms("MIN", masks)
    assert fixed[constants.ZN_RESNAME] == "ALL"
    assert "F201" not in fixed
    assert spec["ligand"] == constants.ZN_RESNAME


def test_spec_aims_hotspots_at_the_retained_flanking_nucleotides(masks):
    spec = prep.build_spec("TadA8e", "FULL", 1.0, masks, "/tmp/x.pdb")
    assert set(spec["select_hotspots"]) == {
        f"{constants.SUBSTRATE_CHAIN}{r}" for r in constants.HOTSPOT_RESIDS}
    assert spec["infer_ori_strategy"] == "hotspots"


def test_partial_t_never_reaches_the_measured_cliff(masks):
    """partial_t=8 was measured to degrade the active site to 2.5-6.75 A RMSD."""
    specs = prep.build_specs(masks, {p: f"/tmp/{p}.pdb" for p in constants.PARENTS})
    assert max(s["partial_t"] for s in specs.values()) < 8.0


def test_write_input_pdb_carries_chain_f_the_zn_and_the_dna_context(tmp_path):
    out = str(tmp_path / "TadA8e.pdb")
    prep.write_input_pdb("TadA8e", out)
    lines = [ln for ln in open(out) if ln.startswith(("ATOM", "HETATM"))]
    chains = {ln[21] for ln in lines}
    assert chains == {constants.SCAFFOLD_CHAIN, constants.SUBSTRATE_CHAIN}
    assert any(ln.startswith("HETATM") and ln[17:20].strip() == "ZN" for ln in lines)
    dna_resids = {int(ln[22:26]) for ln in lines if ln[21] == constants.SUBSTRATE_CHAIN}
    assert dna_resids == set(constants.DNA_CONTEXT_RESIDS)


def test_yaml_round_trips(tmp_path, masks):
    specs = prep.build_specs(masks, {p: f"/tmp/{p}.pdb" for p in constants.PARENTS})
    path = tmp_path / "rfd_inputs.yaml"
    path.write_text(yaml.safe_dump(specs, sort_keys=False))
    assert yaml.safe_load(path.read_text()) == specs
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_prep_rfd_inputs.py -v
```
Expected: FAIL — `ImportError: cannot import name 'prep_rfd_inputs'`.

- [ ] **Step 3: Write `prep_rfd_inputs.py`**

```python
"""RFD3 partial-diffusion inputs: one spec per (parent, arm, partial_t) cell.

RFD3 semantics used here, each learned from a real failure in
`domain-insertion/denovo_tada/` (read that package's make_rfd_inputs.py and
make_rfd_inputs_brace.py docstrings before touching this file):

  - PARTIAL diffusion re-noises the `input` structure IN PLACE by `partial_t`
    ANGSTROMS of injected noise (RFD3 recommends <= 15). No `contig` is emitted:
    a contig would make RFD3 build a new chain instead of perturbing this one,
    and residue numbering is preserved in place, which is what lets every
    downstream stage keep using chain F's 5-160 numbering.
  - `select_fixed_atoms` injects ZERO noise at the named atoms, so the motif's
    internal geometry is preserved while the group may still move as a rigid
    body. That is exactly the semantics this campaign wants for the active site.
  - The catalytic Zn is named by CCD NAME ("ZN"), never chain+resid. AtomWorks
    renames a hetero atom's chain when it shares a chain letter with protein
    residues, so an "F201" key raises
    `ValidationError: [component=F201] Residue F201 not found in atom array`.
  - `select_hotspots` + `infer_ori_strategy: hotspots` aim the pocket. The
    hotspots are chain D 25 and 27, the nucleotides FLANKING the target base --
    the 8AZ at D26 is a non-standard residue that AtomWorks drops, so it cannot
    be a hotspot.
  - The DNA context is chain D only. Chain C has nothing within 12 A of the
    target base (measured), and a smaller fixed context matters: an earlier
    campaign OOM'd an 80 GB A100 by handing RFD3 the whole Cas9 context.

Honesty ceiling: this module writes input files. It makes no claim that the
resulting backbones fold, bind, or catalyse anything.
"""
import argparse
import os

import yaml

from . import constants, motif, provenance, substrate


def cell_id(parent, arm, partial_t):
    return f"{parent}_{arm}_pt{partial_t}"


def write_input_pdb(parent, out_path, pdb6vpc=None):
    """Relaxed parent (chain F + Zn) plus the chain-D context residues.

    The relaxed parents share the 6VPC coordinate frame, so the DNA is copied
    across without superposition. Verified by test_substrate's frame check.
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    parser = PDBParser(QUIET=True)
    parent_model = parser.get_structure(parent, constants.PARENT_PDB[parent])[0]
    ref_model = parser.get_structure("ref", pdb6vpc or constants.PDB6VPC)[0]

    keep = set(substrate.context_residues(pdb6vpc))
    dna = ref_model[constants.SUBSTRATE_CHAIN].copy()
    for res in list(dna):
        if res.id[1] not in keep:
            dna.detach_child(res.id)
    parent_model.add(dna)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    io_writer = PDBIO()
    io_writer.set_structure(parent_model)
    io_writer.save(out_path)
    return out_path


def build_spec(parent, arm, partial_t, masks, input_pdb):
    """One RFD3 partial-diffusion InputSpecification."""
    return {
        "input": input_pdb,
        "partial_t": float(partial_t),
        "is_non_loopy": True,
        "select_fixed_atoms": motif.rfd_select_fixed_atoms(arm, masks),
        "ligand": constants.ZN_RESNAME,
        "select_hotspots": {f"{constants.SUBSTRATE_CHAIN}{r}": "ALL"
                            for r in constants.HOTSPOT_RESIDS},
        "infer_ori_strategy": "hotspots",
    }


def build_specs(masks, input_pdbs):
    specs = {}
    for parent in constants.PARENTS:
        for arm in constants.ARMS:
            for partial_t in constants.PARTIAL_T:
                specs[cell_id(parent, arm, partial_t)] = build_spec(
                    parent, arm, partial_t, masks, input_pdbs[parent])
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", constants.RUN_DIR_NAME))
    ap.add_argument("--skip-preflight", action="store_true",
                    help="only for unit-level smoke runs; never for a batch")
    args = ap.parse_args(argv)

    if not args.skip_preflight:
        from . import preflight
        preflight.require_green(with_env_probes=False)

    in_dir = os.path.join(args.run_dir, "rfd_in")
    masks = motif.load_masks()
    input_pdbs = {p: write_input_pdb(p, os.path.join(in_dir, f"{p}.pdb"))
                  for p in constants.PARENTS}
    specs = build_specs(masks, input_pdbs)

    yaml_path = os.path.join(in_dir, "rfd_inputs.yaml")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(specs, fh, sort_keys=False)
    provenance.write(in_dir, "prep_rfd_inputs", len(constants.PARENTS), len(specs),
                     extra={"cells": sorted(specs),
                            "partial_t": list(constants.PARTIAL_T)})
    print(f"[prep_rfd_inputs] wrote {len(specs)} specs -> {yaml_path}")
    for parent, pdb in input_pdbs.items():
        print(f"[prep_rfd_inputs] input {parent}: {pdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_prep_rfd_inputs.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Generate the real inputs (cheap, no GPU)**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.prep_rfd_inputs
head -30 outputs/20260805_tada_redesign/rfd_in/rfd_inputs.yaml
```
Expected: 16 specs, two input PDBs. Confirm by eye that `select_fixed_atoms` has 25 keys for a FULL cell (24 residues + `ZN`) and 5 for a MIN cell (4 + `ZN`).

- [ ] **Step 6: Write `rfd_partial.slurm` (do NOT submit)**

```bash
#!/bin/bash
# RFdiffusion3 PARTIAL diffusion for the TadA redesign campaign: re-noise each
# relaxed parent in place by partial_t Angstroms while holding the arm's motif
# and the catalytic Zn rigid, with hotspots aimed at the nucleotides flanking
# the target base. One array task per cell (16 cells).
#   MODE=debug  -> 1 batch x 2 = 2 backbones/cell, 20 timesteps (the gate)
#   MODE=run    -> 4 batches x 8 = 32 backbones/cell
# Structural design only -- no base-editing function is implied.
#SBATCH --job-name=tada_rfd_partial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/tada_rfd_partial_%A_%a.out
#SBATCH -e logs/tada_rfd_partial_%A_%a.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
SUB="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign"
RUN="${RUN:-$SUB/outputs/20260805_tada_redesign}"
INPUTS="$RUN/rfd_in/rfd_inputs.yaml"
OUT="$RUN/rfd"
RFD3_CKPT="${RFD3_CKPT:-/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/claude/foundry_ckpt/rfd3_latest.ckpt}"
MODE="${MODE:-debug}"

cd "$SUB"; mkdir -p logs "$OUT"
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" cas9-pam-design

[[ -s "$INPUTS" ]]    || { echo "[rfd_partial] ERROR: missing $INPUTS (run prep_rfd_inputs)" >&2; exit 1; }
[[ -s "$RFD3_CKPT" ]] || { echo "[rfd_partial] ERROR: missing ckpt $RFD3_CKPT" >&2; exit 1; }

# One cell per array task. Keys are sorted so the mapping is stable across runs.
CELL=$(python3 -c "
import sys, yaml
print(sorted(yaml.safe_load(open('$INPUTS')))[int(sys.argv[1]) - 1])
" "${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. --array=1-16}")
CELL_YAML="$RUN/rfd_in/cell_${CELL}.yaml"
python3 -c "
import yaml
specs = yaml.safe_load(open('$INPUTS'))
yaml.safe_dump({'$CELL': specs['$CELL']}, open('$CELL_YAML', 'w'), sort_keys=False)
"

echo "[rfd_partial] cell=$CELL mode=$MODE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
if [ "$MODE" = "debug" ]; then
  rfd3 design out_dir="$OUT" inputs="$CELL_YAML" ckpt_path="$RFD3_CKPT" \
    prevalidate_inputs=true n_batches=1 diffusion_batch_size=2 \
    inference_sampler.num_timesteps=20 skip_existing=true
else
  rfd3 design out_dir="$OUT" inputs="$CELL_YAML" ckpt_path="$RFD3_CKPT" \
    n_batches=4 diffusion_batch_size=8 skip_existing=true
fi
echo "[rfd_partial] done cell=$CELL ; backbones now: $(ls "$OUT"/*.cif.gz 2>/dev/null | wc -l)"
```

Do **not** submit it in this task. Task 8 runs the debug mode.

- [ ] **Step 7: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/prep_rfd_inputs.py tada_redesign/tests/test_prep_rfd_inputs.py rfd_partial.slurm
git commit -F - <<'EOF'
feat: RFD3 partial-diffusion inputs for the 16-cell sweep

One spec per (parent, arm, partial_t): re-noise the relaxed parent in place,
hold the arm's motif and the Zn rigid via select_fixed_atoms (zero injected
noise, so internal geometry survives while the group may still move rigid-body),
and aim the pocket with hotspots on chain D 25/27.

Every RFD3 rule here came from a real failure in denovo_tada and is cited in the
module docstring: no contig for partial diffusion, the Zn named by CCD name
because AtomWorks renames its chain, hotspots on the nucleotides FLANKING the
8AZ because AtomWorks drops the 8AZ itself, and chain-D-only context because
chain C has nothing within 12 A and an oversized fixed context has OOM'd an
80 GB A100 before.

rfd_partial.slurm written, not submitted.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 5: `filter_backbones.py` — reject what RFD3 got wrong before paying for sequences

RFD3 does not perfectly honour fixed atoms; verifying rather than assuming is the whole point. Each rejection reason is counted **per cell**, so a cell that silently produced nothing is visible instead of quietly absent.

**Files:**
- Create: `tada_redesign/filter_backbones.py`
- Test: `tada_redesign/tests/test_filter_backbones.py`

**Interfaces:**
- Consumes: `score_structure.{heavy_atoms_from_cif,heavy_atoms_from_pdb,ca_map,motif_rmsd}`, `motif.{load_masks,arm_residues}`, `constants.{RMSD_REFERENCE,BACKBONE_MOTIF_RMSD_MAX,CA_BREAK_MAX,LENGTH_RANGE,ZN_DONOR_RANGE,SCAFFOLD_CHAIN}`, `io.append_row`, `provenance`.
- Produces:
  - `filter_backbones.COLUMNS` — tuple of `backbones.tsv` column names.
  - `filter_backbones.max_ca_break(atoms) -> float`
  - `filter_backbones.zn_donor_distances(atoms) -> dict[str, float]`
  - `filter_backbones.evaluate(cif_path, ref_atoms, residues, cell) -> dict` — one row, always returns (never raises); `status` is `"ok"` or a reason.
  - `filter_backbones.main(argv=None) -> int` — writes `<run>/backbones.tsv`.

- [ ] **Step 1: Write the failing test**

```python
"""Every gate here is a cheap CPU check standing between RFD3's output and the
much more expensive sequence-design and folding stages."""
import numpy as np
import pytest

from tada_redesign import constants, filter_backbones as fb, motif, score_structure as ss

DONORS = ((57, "ND1"), (87, "SG"), (90, "SG"))


def _linear_backbone(n=160, start=5, step=3.8):
    """CA atoms on a straight line at ideal spacing, plus a Zn at ideal donor
    distance from three synthetic donor atoms."""
    atoms = {}
    for i in range(n):
        resnum = start + i
        atoms[(resnum, "CA")] = np.array([i * step, 0.0, 0.0])
    for resnum, name in DONORS:
        atoms[(resnum, name)] = np.array([0.0, 2.2, 0.0])
    atoms[(201, "ZN")] = np.array([0.0, 0.0, 0.0])
    return atoms


def test_max_ca_break_finds_the_largest_consecutive_gap():
    atoms = _linear_backbone(n=5)
    assert fb.max_ca_break(atoms) == pytest.approx(3.8)
    atoms[(9, "CA")] = np.array([100.0, 0.0, 0.0])       # residue 9 flung away
    assert fb.max_ca_break(atoms) > constants.CA_BREAK_MAX


def test_max_ca_break_ignores_numbering_gaps():
    """A missing residue is not a chain break: only CONSECUTIVE numbers count."""
    atoms = {(5, "CA"): np.array([0.0, 0.0, 0.0]),
             (6, "CA"): np.array([3.8, 0.0, 0.0]),
             (20, "CA"): np.array([200.0, 0.0, 0.0])}
    assert fb.max_ca_break(atoms) == pytest.approx(3.8)


def test_zn_donor_distances_measures_all_three_donors():
    d = fb.zn_donor_distances(_linear_backbone())
    assert set(d) == {"zn_57ND1", "zn_87SG", "zn_90SG"}
    assert all(v == pytest.approx(2.2) for v in d.values())


def test_zn_donor_distances_reports_nan_for_a_missing_donor():
    atoms = _linear_backbone()
    del atoms[(87, "SG")]
    assert np.isnan(fb.zn_donor_distances(atoms)["zn_87SG"])


def test_evaluate_rejects_a_length_outlier(tmp_path, monkeypatch):
    short = _linear_backbone(n=40)
    monkeypatch.setattr(fb, "load_backbone", lambda path: short)
    row = fb.evaluate("x.cif.gz", short, (5, 6, 7), "cell")
    assert row["status"] == "length_out_of_range"
    assert row["passed"] == "False"


def test_evaluate_rejects_a_chain_break(monkeypatch):
    broken = _linear_backbone(n=160)
    broken[(9, "CA")] = np.array([100.0, 0.0, 0.0])
    monkeypatch.setattr(fb, "load_backbone", lambda path: broken)
    row = fb.evaluate("x.cif.gz", broken, (5, 6, 7), "cell")
    assert row["status"] == "chain_break"


def test_evaluate_rejects_motif_drift(monkeypatch):
    """Displace a SIDECHAIN atom, not the CA: moving a CA far enough to drift the
    motif also opens a CA-CA gap, and `evaluate` checks chain_break BEFORE
    motif_drift, so a CA-based fixture would assert the wrong status for the
    right reason. Sidechain drift isolates the motif gate cleanly."""
    ref = _linear_backbone()
    ref[(6, "CB")] = ref[(6, "CA")] + np.array([1.5, 0.0, 0.0])
    moved = dict(ref)
    moved[(6, "CB")] = moved[(6, "CB")] + np.array([0.0, 0.0, 2.0])
    monkeypatch.setattr(fb, "load_backbone", lambda path: moved)
    row = fb.evaluate("x.cif.gz", ref, (6,), "cell")
    assert row["status"] == "motif_drift"
    assert float(row["motif_rmsd"]) > constants.BACKBONE_MOTIF_RMSD_MAX
    assert float(row["max_ca_break"]) <= constants.CA_BREAK_MAX   # not a chain break


def test_evaluate_rejects_a_displaced_zn(monkeypatch):
    ref = _linear_backbone()
    bad = dict(ref)
    bad[(201, "ZN")] = np.array([0.0, 8.0, 0.0])
    monkeypatch.setattr(fb, "load_backbone", lambda path: bad)
    row = fb.evaluate("x.cif.gz", ref, (5, 6, 7), "cell")
    assert row["status"] == "zn_displaced"


def test_evaluate_passes_a_good_backbone(monkeypatch):
    ref = _linear_backbone()
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(ref))
    row = fb.evaluate("x.cif.gz", ref, (5, 6, 7), "cell")
    assert row["status"] == "ok" and row["passed"] == "True"
    assert float(row["motif_rmsd"]) == pytest.approx(0.0, abs=1e-6)


def test_evaluate_returns_a_failed_row_instead_of_raising(monkeypatch):
    """A single unreadable file must not kill the shard."""
    def boom(path):
        raise OSError("corrupt cif")
    monkeypatch.setattr(fb, "load_backbone", boom)
    row = fb.evaluate("x.cif.gz", _linear_backbone(), (5,), "cell")
    assert row["passed"] == "False"
    assert "corrupt cif" in row["status"]


def test_columns_are_stable_and_include_the_cell():
    for col in ("backbone", "cell", "parent", "arm", "partial_t",
                "n_res", "max_ca_break", "motif_rmsd", "status", "passed"):
        assert col in fb.COLUMNS
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_filter_backbones.py -v
```
Expected: FAIL — `ImportError: cannot import name 'filter_backbones'`.

- [ ] **Step 3: Write `filter_backbones.py`**

```python
"""Reject RFD3 backbones before they cost sequence design and folding time.

RFD3 does not perfectly honour `select_fixed_atoms`, so motif geometry is
VERIFIED here rather than assumed. Four cheap CPU gates, each a documented
threshold from the design spec:

  - motif heavy-atom RMSD to the parent <= BACKBONE_MOTIF_RMSD_MAX (1.0 A),
    measured against the RELAXED parent (constants.RMSD_REFERENCE) because
    partial diffusion started from exactly that structure
  - no consecutive CA-CA distance > CA_BREAK_MAX (4.2 A)
  - length within LENGTH_RANGE (150-175 residues)
  - Zn within ZN_DONOR_RANGE (2.0-2.6 A) of all three donors

Rejection counts are printed PER CELL. A cell whose every backbone failed must
be visible as a zero, not merely absent from the results -- that is how a broken
partial_t level or arm gets noticed before the next stage runs on a silently
truncated set.

Honesty ceiling: passing these gates means the backbone is geometrically sane
and still carries its active-site motif. It says nothing about function.
"""
import argparse
import collections
import glob
import os

import numpy as np

from . import constants, io, motif, provenance, score_structure

COLUMNS = ("backbone", "cell", "parent", "arm", "partial_t", "path",
           "n_res", "max_ca_break", "motif_rmsd",
           "zn_57ND1", "zn_87SG", "zn_90SG", "status", "passed")

ZN_DONORS = ((57, "ND1"), (87, "SG"), (90, "SG"))


def load_backbone(path):
    """Chain-F heavy atoms from an RFD3 `.cif.gz` (or a plain `.cif`/`.pdb`)."""
    if path.endswith((".pdb", ".ent")):
        return score_structure.heavy_atoms_from_pdb(path)
    if path.endswith(".gz"):
        import gzip
        import shutil
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            with gzip.open(path, "rb") as src:
                shutil.copyfileobj(src, tmp)
            tmp_path = tmp.name
        try:
            return score_structure.heavy_atoms_from_cif(tmp_path)
        finally:
            os.unlink(tmp_path)
    return score_structure.heavy_atoms_from_cif(path)


def max_ca_break(atoms):
    """Largest CA-CA distance between CONSECUTIVELY NUMBERED residues.

    A numbering gap is unresolved density, not a chain break, so
    non-consecutive pairs are skipped -- counting them would reject every
    backbone built from a gapped input.
    """
    ca = score_structure.ca_map(atoms)
    worst = 0.0
    for resnum in sorted(ca):
        nxt = resnum + 1
        if nxt in ca:
            worst = max(worst, float(np.linalg.norm(ca[nxt] - ca[resnum])))
    return worst


def zn_donor_distances(atoms):
    """{"zn_57ND1": distance, ...}; nan for a donor or Zn that is absent."""
    zn = next((v for (resnum, name), v in atoms.items() if name == "ZN"), None)
    out = {}
    for resnum, name in ZN_DONORS:
        key = f"zn_{resnum}{name}"
        atom = atoms.get((resnum, name))
        out[key] = (float(np.linalg.norm(atom - zn))
                    if zn is not None and atom is not None else float("nan"))
    return out


def _row(backbone, cell, path, **kw):
    parts = cell.split("_")
    base = {"backbone": backbone, "cell": cell, "path": path,
            "parent": parts[0] if parts else "NA",
            "arm": parts[1] if len(parts) > 1 else "NA",
            "partial_t": parts[2][2:] if len(parts) > 2 else "NA"}
    base.update(kw)
    return base


def evaluate(cif_path, ref_atoms, residues, cell):
    """One `backbones.tsv` row. Never raises: an unreadable file becomes a
    failed row so a single bad file cannot kill the shard."""
    backbone = os.path.basename(cif_path).split(".")[0]
    try:
        atoms = load_backbone(cif_path)
    except Exception as exc:                      # noqa: BLE001 - deliberate
        return _row(backbone, cell, cif_path, n_res="NA", max_ca_break="NA",
                    motif_rmsd="NA", status=f"load_failed: {exc}", passed="False")

    ca = score_structure.ca_map(atoms)
    n_res = len(ca)
    zn = zn_donor_distances(atoms)
    break_max = max_ca_break(atoms)
    lo, hi = constants.LENGTH_RANGE
    zlo, zhi = constants.ZN_DONOR_RANGE

    try:
        rmsd = score_structure.motif_rmsd(ref_atoms, atoms, residues)
    except (KeyError, ValueError) as exc:
        return _row(backbone, cell, cif_path, n_res=n_res,
                    max_ca_break=round(break_max, 3), motif_rmsd="NA",
                    status=f"motif_unmeasurable: {exc}", passed="False", **zn)

    if not (lo <= n_res <= hi):
        status = "length_out_of_range"
    elif break_max > constants.CA_BREAK_MAX:
        status = "chain_break"
    elif rmsd > constants.BACKBONE_MOTIF_RMSD_MAX:
        status = "motif_drift"
    elif any(np.isnan(v) or not (zlo <= v <= zhi) for v in zn.values()):
        status = "zn_displaced"
    else:
        status = "ok"

    return _row(backbone, cell, cif_path, n_res=n_res,
                max_ca_break=round(break_max, 3), motif_rmsd=round(rmsd, 3),
                status=status, passed=str(status == "ok"), **zn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(
        sub_dir, "outputs", constants.RUN_DIR_NAME))
    ap.add_argument("--rfd-subdir", default="rfd")
    args = ap.parse_args(argv)

    masks = motif.load_masks()
    out_path = os.path.join(args.run_dir, "backbones.tsv")
    if os.path.exists(out_path):
        os.unlink(out_path)

    ref_cache = {}
    counts = collections.Counter()
    per_cell = collections.defaultdict(collections.Counter)
    paths = sorted(glob.glob(os.path.join(args.run_dir, args.rfd_subdir, "*.cif.gz")))

    for path in paths:
        cell = os.path.basename(path).split(".")[0].rsplit("_", 1)[0]
        parent, arm = cell.split("_")[0], cell.split("_")[1]
        if parent not in ref_cache:
            ref_cache[parent] = score_structure.heavy_atoms_from_pdb(
                constants.RMSD_REFERENCE[parent])
        residues = motif.arm_residues(arm, masks)
        row = evaluate(path, ref_cache[parent], residues, cell)
        io.append_row(out_path, row, COLUMNS)
        counts[row["status"]] += 1
        per_cell[cell][row["status"]] += 1

    n_ok = counts["ok"]
    for cell in sorted(per_cell):
        detail = ", ".join(f"{k}={v}" for k, v in sorted(per_cell[cell].items()))
        print(f"[filter_backbones] {cell}: {detail}")
        if not per_cell[cell]["ok"]:
            print(f"[filter_backbones] WARNING {cell}: ZERO backbones passed")
    print(f"[filter_backbones] {n_ok}/{len(paths)} passed -> {out_path}")

    final, degraded = provenance.output_path(out_path, len(paths), n_ok)
    if degraded:
        os.rename(out_path, final)
        print(f"[filter_backbones] DEGRADED: >{constants.DEGRADED_FRACTION:.0%} "
              f"of backbones failed; wrote {final} instead of the canonical path")
    provenance.write(args.run_dir, "filter_backbones", len(paths), n_ok,
                     extra={"per_cell": {c: dict(v) for c, v in per_cell.items()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_filter_backbones.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/filter_backbones.py tada_redesign/tests/test_filter_backbones.py
git commit -F - <<'EOF'
feat: backbone filter -- verify RFD3 honoured the motif before paying for sequences

Four cheap CPU gates between diffusion and the expensive stages: motif
heavy-atom RMSD against the RELAXED parent (the structure partial diffusion
actually started from), consecutive-CA chain breaks, length, and Zn donor
distances. RFD3 does not perfectly honour select_fixed_atoms, so this verifies
rather than assumes.

Rejections are counted PER CELL and a cell with zero survivors prints a warning,
so a broken partial_t level is visible as a zero rather than quietly absent. A
single unreadable file becomes a failed row instead of killing the shard, and a
run losing more than DEGRADED_FRACTION diverts to backbones.degraded.tsv.

max_ca_break deliberately skips non-consecutive residue numbers: a numbering gap
is unresolved density, not a chain break.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: `prep_mpnn_inputs.py`, the vendored wrapper, and `run_ligandmpnn.slurm`

LigandMPNN's `*_multi` flags let ONE process design many backbones at one temperature, so this stage is 10 invocations (2 arms × (4 temperatures + 1 zero-bias control)) rather than 2,560 model loads.

**Files:**
- Create: `tada_redesign/prep_mpnn_inputs.py`, `tada_redesign/_run_ligandmpnn.py`, `run_ligandmpnn.slurm`
- Test: `tada_redesign/tests/test_prep_mpnn_inputs.py`

**Interfaces:**
- Consumes: `io.read_tsv`, `motif.{load_masks,arm_residues,mpnn_fixed_residues}`, `constants.{ARMS,MPNN_TEMPS,SEQS_PER_TEMP,CONTROL_TEMP,HYDROPHOBIC_SET,POLAR_SET,HYDROPHOBIC_BIAS,POLAR_BIAS,SCAFFOLD_CHAIN}`.
- Produces:
  - `prep_mpnn_inputs.bias_positions(arm, masks) -> tuple[int, ...]` — `EXPOSED ∩ MODELED − arm_residues`.
  - `prep_mpnn_inputs.bias_json(arm, masks, chain=None) -> dict` — `{"F12": {"L": -1.0, ...}}`.
  - `prep_mpnn_inputs.convert_to_pdb(cif_gz, out_pdb) -> str`
  - `prep_mpnn_inputs.main(argv=None) -> int` — writes, per arm, `pdb_paths.json`, `fixed_residues.json`, `bias.json`, and `mpnn_manifest.tsv`.

- [ ] **Step 1: Write the failing test**

```python
"""Solubility bias is an assumption, so it is applied narrowly and measured
against a zero-bias control -- never applied to a buried or frozen position."""
import json

import pytest

from tada_redesign import constants, motif, prep_mpnn_inputs as prep


@pytest.fixture
def masks():
    return motif.load_masks()


def test_bias_positions_are_exposed_designable_and_never_frozen(masks):
    for arm in constants.ARMS:
        positions = set(prep.bias_positions(arm, masks))
        frozen = set(motif.arm_residues(arm, masks))
        exposed = set(masks["EXPOSED"]) & set(masks["MODELED"])
        assert positions == exposed - frozen
        assert not (positions & frozen)


def test_bias_never_touches_a_buried_position(masks):
    """Biasing a core position would trade away the stability we are buying."""
    buried = set(masks["BURIED"])
    for arm in constants.ARMS:
        assert not (set(prep.bias_positions(arm, masks)) & buried)


def test_bias_json_penalises_hydrophobics_and_rewards_polars(masks):
    bias = prep.bias_json("FULL", masks)
    key = f"{constants.SCAFFOLD_CHAIN}{prep.bias_positions('FULL', masks)[0]}"
    assert bias[key]["L"] == constants.HYDROPHOBIC_BIAS < 0
    assert bias[key]["E"] == constants.POLAR_BIAS > 0
    assert set(bias[key]) == set(constants.HYDROPHOBIC_SET + constants.POLAR_SET)


def test_bias_json_keys_are_chain_prefixed_residues(masks):
    for key in prep.bias_json("MIN", masks):
        assert key[0] == constants.SCAFFOLD_CHAIN
        assert key[1:].isdigit()


def test_bias_json_is_json_serialisable(masks):
    assert json.loads(json.dumps(prep.bias_json("FULL", masks)))


def test_control_temperature_is_one_of_the_swept_temperatures():
    """The control must be comparable to a biased cell, not an orphan point."""
    assert constants.CONTROL_TEMP in constants.MPNN_TEMPS
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_prep_mpnn_inputs.py -v
```
Expected: FAIL — `ImportError: cannot import name 'prep_mpnn_inputs'`.

- [ ] **Step 3: Write `prep_mpnn_inputs.py`**

```python
"""LigandMPNN inputs: one multi-JSON set per arm, plus the solubility bias.

LigandMPNN accepts `--pdb_path_multi`, `--fixed_residues_multi` and
`--bias_AA_per_residue_multi` as JSON maps keyed by PDB path, so ONE process
designs every backbone of an arm at one temperature. That turns this stage from
~2,560 model loads into 10 invocations (2 arms x (4 temperatures + 1 control)).

The Zn and the ssDNA context travel inside each backbone PDB: LigandMPNN reads
non-protein atoms as ligand context, which is the whole reason this campaign
uses LigandMPNN rather than plain ProteinMPNN (which has no ligand channel and
would design the metal site as if it were empty).

Solubility bias is applied ONLY at positions that are exposed AND designable
(`EXPOSED & MODELED - arm_residues`). Buried positions are never biased --
pushing polar residues into the core would trade away the very stability the
campaign is buying -- and frozen positions cannot change identity at all. The
magnitudes are assumptions, which is why a zero-bias control set is designed
alongside and carried through every scoring stage.

Honesty ceiling: a bias makes a residue more likely to be chosen. It does not
make the protein soluble, and nothing here measures solubility.
"""
import argparse
import json
import os

from . import constants, io, motif, provenance


def bias_positions(arm, masks):
    """Exposed, designable positions: `EXPOSED & MODELED - arm_residues`."""
    exposed = set(masks["EXPOSED"]) & set(masks["MODELED"])
    return tuple(sorted(exposed - set(motif.arm_residues(arm, masks))))


def bias_json(arm, masks, chain=None):
    """`{"F12": {"L": -1.0, ..., "E": 0.3, ...}}` for LigandMPNN."""
    chain = chain or constants.SCAFFOLD_CHAIN
    per_residue = {aa: constants.HYDROPHOBIC_BIAS
                   for aa in constants.HYDROPHOBIC_SET}
    per_residue.update({aa: constants.POLAR_BIAS for aa in constants.POLAR_SET})
    return {f"{chain}{resnum}": dict(per_residue)
            for resnum in bias_positions(arm, masks)}


def convert_to_pdb(cif_gz, out_pdb):
    """LigandMPNN reads PDB, RFD3 writes .cif.gz."""
    import gzip
    import shutil
    import tempfile

    from Bio.PDB import MMCIFParser, PDBIO

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        with gzip.open(cif_gz, "rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp_path = tmp.name
    try:
        structure = MMCIFParser(QUIET=True).get_structure("bb", tmp_path)
    finally:
        os.unlink(tmp_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(out_pdb)
    return out_pdb


MANIFEST_COLUMNS = ("backbone", "cell", "parent", "arm", "pdb_path")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(
        sub_dir, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    masks = motif.load_masks()
    rows = [r for r in io.read_tsv(os.path.join(args.run_dir, "backbones.tsv"))
            if r["passed"] == "True"]
    if not rows:
        raise SystemExit("[prep_mpnn_inputs] no passing backbones in backbones.tsv")

    out_dir = os.path.join(args.run_dir, "mpnn_in")
    manifest = []
    per_arm = {arm: [] for arm in constants.ARMS}
    for row in rows:
        pdb = convert_to_pdb(row["path"], os.path.join(
            out_dir, "pdb", f"{row['backbone']}.pdb"))
        per_arm[row["arm"]].append(pdb)
        manifest.append({"backbone": row["backbone"], "cell": row["cell"],
                         "parent": row["parent"], "arm": row["arm"],
                         "pdb_path": pdb})

    for arm, pdbs in per_arm.items():
        arm_dir = os.path.join(out_dir, arm)
        os.makedirs(arm_dir, exist_ok=True)
        fixed = motif.mpnn_fixed_residues(arm, masks)
        bias = bias_json(arm, masks)
        json.dump({p: "" for p in pdbs},
                  open(os.path.join(arm_dir, "pdb_paths.json"), "w"), indent=1)
        json.dump({p: fixed for p in pdbs},
                  open(os.path.join(arm_dir, "fixed_residues.json"), "w"), indent=1)
        json.dump({p: bias for p in pdbs},
                  open(os.path.join(arm_dir, "bias.json"), "w"), indent=1)
        print(f"[prep_mpnn_inputs] {arm}: {len(pdbs)} backbones, "
              f"{len(bias)} biased positions, fixed='{fixed[:40]}...'")

    io.write_tsv(os.path.join(out_dir, "mpnn_manifest.tsv"), manifest,
                 MANIFEST_COLUMNS)
    provenance.write(out_dir, "prep_mpnn_inputs", len(rows), len(manifest),
                     extra={"per_arm": {a: len(p) for a, p in per_arm.items()},
                            "hydrophobic_bias": constants.HYDROPHOBIC_BIAS,
                            "polar_bias": constants.POLAR_BIAS,
                            "hydrophobic_set": constants.HYDROPHOBIC_SET,
                            "polar_set": constants.POLAR_SET})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Vendor the LigandMPNN wrapper**

Copy `domain-insertion/denovo_tada/_run_ligandmpnn.py` to `tada_redesign/_run_ligandmpnn.py` verbatim, then prepend this note to its docstring (keeping everything already there):

```
VENDORED from domain-insertion/denovo_tada/_run_ligandmpnn.py (2026-08-05).
Copied rather than imported: that package is Cas9/RuvC-coupled throughout, and
this campaign must not depend on its module layout. The numpy-alias monkeypatch
below is the whole point -- see the original for the two environment blockers it
works around.
```

Verify it still runs its own help:

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn-sc python tada_redesign/_run_ligandmpnn.py --help 2>&1 | head -5
```
Expected: LigandMPNN's argparse help, not a traceback.

- [ ] **Step 5: Run the unit tests**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_prep_mpnn_inputs.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Write `run_ligandmpnn.slurm` (do NOT submit)**

```bash
#!/bin/bash
# LigandMPNN sequence design for the TadA redesign campaign. One array task per
# (arm, temperature) plus the zero-bias control, using LigandMPNN's *_multi JSON
# flags so a single process designs every backbone of that arm.
#   --array=1-10   (2 arms x (4 temperatures + 1 control))
# The Zn and ssDNA context ride inside each backbone PDB; LigandMPNN reads
# non-protein atoms as ligand context.
# Structural design only -- no base-editing function is implied.
#SBATCH --job-name=tada_lmpnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/tada_lmpnn_%A_%a.out
#SBATCH -e logs/tada_lmpnn_%A_%a.err
set -euo pipefail

CONDA_BASE="/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/common/miniforge3"
R="/research/rgs01/home/clusterHome/ecreed/claude-proteindesign"
SUB="$R/tada-redesign"
RUN="${RUN:-$SUB/outputs/20260805_tada_redesign}"
CKPT="$R/design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"
NBATCH="${NBATCH:-5}"          # sequences per backbone at this temperature
SEED="${SEED:-111}"

cd "$SUB"; mkdir -p logs
unset PYTHONPATH
source "${CONDA_BASE}/bin/activate" ligandmpnn-sc
[[ -s "$CKPT" ]] || { echo "[lmpnn] ERROR: missing checkpoint $CKPT" >&2; exit 1; }

# Task -> (arm, temperature, biased?). Arms x 4 temps = 8, then 2 controls.
ARMS=(FULL MIN); TEMPS=(0.1 0.15 0.2 0.3); CONTROL_TEMP=0.15
IDX=$(( ${SLURM_ARRAY_TASK_ID:?run as an array job, e.g. --array=1-10} - 1 ))
if [ "$IDX" -lt 8 ]; then
  ARM="${ARMS[$(( IDX / 4 ))]}"; T="${TEMPS[$(( IDX % 4 ))]}"; BIASED=1
  TAG="T${T}"
else
  ARM="${ARMS[$(( IDX - 8 ))]}"; T="$CONTROL_TEMP"; BIASED=0
  TAG="control"
fi

IN="$RUN/mpnn_in/$ARM"
OUT="$RUN/lmpnn/$ARM/$TAG"
[[ -s "$IN/pdb_paths.json" ]] || { echo "[lmpnn] ERROR: missing $IN/pdb_paths.json (run prep_mpnn_inputs)" >&2; exit 1; }

BIAS_ARGS=()
if [ "$BIASED" = "1" ]; then
  BIAS_ARGS=(--bias_AA_per_residue_multi "$IN/bias.json")
fi

echo "[lmpnn] arm=$ARM tag=$TAG T=$T biased=$BIASED nbatch=$NBATCH host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
python3 "$SUB/tada_redesign/_run_ligandmpnn.py" \
  --model_type ligand_mpnn \
  --checkpoint_ligand_mpnn "$CKPT" \
  --pdb_path_multi "$IN/pdb_paths.json" \
  --fixed_residues_multi "$IN/fixed_residues.json" \
  "${BIAS_ARGS[@]}" \
  --out_folder "$OUT" \
  --temperature "$T" \
  --batch_size 1 \
  --number_of_batches "$NBATCH" \
  --seed "$SEED" \
  --save_stats 1 \
  --verbose 0
echo "[lmpnn] done -> $OUT/seqs ; fastas: $(ls "$OUT"/seqs/*.fa 2>/dev/null | wc -l)"
```

- [ ] **Step 7: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/prep_mpnn_inputs.py tada_redesign/_run_ligandmpnn.py tada_redesign/tests/test_prep_mpnn_inputs.py run_ligandmpnn.slurm
git commit -F - <<'EOF'
feat: LigandMPNN inputs with narrow solubility bias, batched by arm

Uses LigandMPNN's *_multi JSON flags so one process designs every backbone of an
arm at one temperature -- 10 invocations instead of ~2,560 model loads.

Solubility bias is applied ONLY at exposed, designable positions
(EXPOSED & MODELED - arm_residues), never buried and never frozen: biasing a
core position would trade away the stability the campaign is buying. Tests
assert the bias set never intersects BURIED or the frozen motif. The magnitudes
are assumptions, so a zero-bias control is designed alongside at the same
temperature and carried through every scoring stage.

_run_ligandmpnn.py is vendored from denovo_tada with a provenance note rather
than imported -- that package is Cas9-coupled throughout.

run_ligandmpnn.slurm written, not submitted.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 7: `collect_designs.py` — LigandMPNN FASTA → `designs.tsv`

**Files:**
- Create: `tada_redesign/collect_designs.py`
- Test: `tada_redesign/tests/test_collect_designs.py`

**Interfaces:**
- Consumes: `io.{read_tsv,append_row}`, `provenance`, `constants.PARENTS`.
- Produces:
  - `collect_designs.COLUMNS`
  - `collect_designs.parse_fasta(path) -> list[dict]` — one dict per DESIGN record; the leading input record (no `id=`) is skipped.
  - `collect_designs.design_id(backbone, tag, idx) -> str`
  - `collect_designs.main(argv=None) -> int` — writes `<run>/designs.tsv`.

- [ ] **Step 1: Write the failing test**

```python
"""The FASTA's FIRST record is the input sequence, not a design. Counting it
would inflate the design set by one per backbone and put a non-design into
scoring."""
import pytest

from tada_redesign import collect_designs as cd

FASTA = (
    ">bb1, T=0.15, seed=111, num_res=156, num_ligand_res=8, "
    "use_ligand_context=True, ligand_cutoff_distance=8.0, batch_size=1, "
    "number_of_batches=2, model_path=/x/ckpt.pt\n"
    "MSEVEFSHEYWMRHAL\n"
    ">bb1, id=1, T=0.15, seed=111, overall_confidence=0.4123, "
    "ligand_confidence=0.5231, seq_rec=0.7812\n"
    "MSEVEFSHEYWMRHAA\n"
    ">bb1, id=2, T=0.15, seed=111, overall_confidence=0.3011, "
    "ligand_confidence=0.4410, seq_rec=0.7011\n"
    "MSEVEFSHEYWMRHAG\n"
)


def test_parse_fasta_skips_the_input_record(tmp_path):
    path = tmp_path / "bb1.fa"
    path.write_text(FASTA)
    records = cd.parse_fasta(str(path))
    assert len(records) == 2
    assert [r["id"] for r in records] == ["1", "2"]
    assert records[0]["sequence"] == "MSEVEFSHEYWMRHAA"


def test_parse_fasta_extracts_every_confidence_field(tmp_path):
    path = tmp_path / "bb1.fa"
    path.write_text(FASTA)
    r = cd.parse_fasta(str(path))[0]
    assert r["temperature"] == "0.15"
    assert r["seed"] == "111"
    assert r["overall_confidence"] == "0.4123"
    assert r["ligand_confidence"] == "0.5231"
    assert r["seq_rec"] == "0.7812"


def test_parse_fasta_returns_empty_for_an_input_only_fasta(tmp_path):
    path = tmp_path / "empty.fa"
    path.write_text(FASTA.split(">bb1, id=1")[0])
    assert cd.parse_fasta(str(path)) == []


def test_parse_fasta_raises_on_a_multi_chain_sequence(tmp_path):
    """LigandMPNN joins chains with ':'. This campaign designs ONE protein
    chain, so a ':' means the DNA context was parsed as a designed chain --
    which would silently corrupt every downstream length and RMSD measurement."""
    path = tmp_path / "multi.fa"
    path.write_text(FASTA.replace("MSEVEFSHEYWMRHAA", "MSEVEF:ACGT"))
    with pytest.raises(ValueError):
        cd.parse_fasta(str(path))


def test_design_id_is_stable_and_encodes_its_provenance():
    assert cd.design_id("TadA8e_FULL_pt1.0_0", "T0.15", "3") == \
        "TadA8e_FULL_pt1.0_0__T0.15__3"


def test_columns_carry_the_full_cell_coordinates():
    for col in ("design_id", "backbone", "cell", "parent", "arm", "partial_t",
                "temperature", "bias", "sequence", "seq_len",
                "overall_confidence", "ligand_confidence", "seq_rec"):
        assert col in cd.COLUMNS
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_collect_designs.py -v
```
Expected: FAIL — `ImportError: cannot import name 'collect_designs'`.

- [ ] **Step 3: Write `collect_designs.py`**

```python
"""LigandMPNN FASTA output -> designs.tsv, one row per design.

Two format facts, confirmed by reading `design/LigandMPNN/run.py`:

  - The FIRST record in each `.fa` is the INPUT sequence and has no `id=` field.
    It is not a design. Counting it would add one phantom design per backbone
    and push a non-design into the folding stages.
  - Design records are
    `>{name}, id={i}, T={t}, seed={s}, overall_confidence={c},
      ligand_confidence={lc}, seq_rec={r}`

A sequence containing `:` means LigandMPNN emitted more than one designed chain
-- i.e. the DNA context was parsed as protein rather than as ligand context.
That would corrupt every downstream length and RMSD measurement, so it raises
rather than being silently accepted.

Honesty ceiling: a confidence value here is LigandMPNN's own sequence score. It
is not a stability measurement and not evidence of function.
"""
import argparse
import glob
import os

from . import constants, io, provenance

COLUMNS = ("design_id", "backbone", "cell", "parent", "arm", "partial_t",
           "temperature", "bias", "seed", "mpnn_id", "sequence", "seq_len",
           "overall_confidence", "ligand_confidence", "seq_rec")

_FIELDS = ("id", "T", "seed", "overall_confidence", "ligand_confidence", "seq_rec")


def _parse_header(header):
    out = {}
    for chunk in header.split(","):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        out[key.strip()] = value.strip()
    return out


def parse_fasta(path):
    """Design records only; the leading input record is skipped."""
    records, header, seq = [], None, []

    def flush():
        if header is None:
            return
        fields = _parse_header(header)
        if "id" not in fields:            # the input record, not a design
            return
        sequence = "".join(seq)
        if ":" in sequence:
            raise ValueError(
                f"{path}: design sequence has multiple chains ({sequence[:40]}...); "
                "the DNA context was parsed as a designed chain, not ligand context")
        records.append({
            "id": fields["id"],
            "temperature": fields.get("T", "NA"),
            "seed": fields.get("seed", "NA"),
            "overall_confidence": fields.get("overall_confidence", "NA"),
            "ligand_confidence": fields.get("ligand_confidence", "NA"),
            "seq_rec": fields.get("seq_rec", "NA"),
            "sequence": sequence,
        })

    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header, seq = line[1:], []
        else:
            seq.append(line)
    flush()
    return records


def design_id(backbone, tag, mpnn_id):
    return f"{backbone}__{tag}__{mpnn_id}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(
        sub_dir, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    manifest = {r["backbone"]: r for r in io.read_tsv(
        os.path.join(args.run_dir, "mpnn_in", "mpnn_manifest.tsv"))}
    out_path = os.path.join(args.run_dir, "designs.tsv")
    if os.path.exists(out_path):
        os.unlink(out_path)

    fastas = sorted(glob.glob(os.path.join(
        args.run_dir, "lmpnn", "*", "*", "seqs", "*.fa")))
    n_rows, skipped = 0, []
    for fasta in fastas:
        tag = os.path.basename(os.path.dirname(os.path.dirname(fasta)))
        backbone = os.path.basename(fasta)[:-3]
        meta = manifest.get(backbone)
        if meta is None:
            skipped.append(fasta)
            continue
        for rec in parse_fasta(fasta):
            io.append_row(out_path, {
                "design_id": design_id(backbone, tag, rec["id"]),
                "backbone": backbone,
                "cell": meta["cell"],
                "parent": meta["parent"],
                "arm": meta["arm"],
                "partial_t": meta["cell"].split("_")[2][2:],
                "temperature": rec["temperature"],
                "bias": "none" if tag == "control" else "solubility",
                "seed": rec["seed"],
                "mpnn_id": rec["id"],
                "sequence": rec["sequence"],
                "seq_len": len(rec["sequence"]),
                "overall_confidence": rec["overall_confidence"],
                "ligand_confidence": rec["ligand_confidence"],
                "seq_rec": rec["seq_rec"],
            }, COLUMNS)
            n_rows += 1

    for fasta in skipped:
        print(f"[collect_designs] WARNING no manifest row for {fasta}; skipped")
    print(f"[collect_designs] {n_rows} designs from {len(fastas)} fastas -> {out_path}")
    provenance.write(args.run_dir, "collect_designs", len(fastas), n_rows,
                     extra={"skipped_fastas": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests/test_collect_designs.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m pytest tada_redesign/tests -v -m "not slow"
```
Expected: **99 passed** — 42 Part 1 + 13 io/provenance + 7 preflight/constants + 7 substrate + 8 rfd inputs + 11 filter + 6 mpnn + 6 collect = 100 collected, minus the 1 `slow`-marked env-probe test that `-m "not slow"` deselects. Report the ACTUAL number rather than adjusting a test to reach it.

- [ ] **Step 6: Commit**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git add tada_redesign/collect_designs.py tada_redesign/tests/test_collect_designs.py
git commit -F - <<'EOF'
feat: collect LigandMPNN designs into designs.tsv

Parses LigandMPNN's FASTA, skipping the leading INPUT record (which has no
`id=` field and is not a design -- counting it would add one phantom design per
backbone and push a non-design into folding).

Raises on a sequence containing ':', which means LigandMPNN emitted more than
one designed chain, i.e. the DNA context was parsed as protein rather than
ligand context. That would corrupt every downstream length and RMSD
measurement, so it fails loudly instead.

Each row carries its full cell coordinates plus the bias condition, so "what did
more noise / a hotter temperature / the solubility bias actually buy" is a query
rather than a re-run.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 8: The debug gate — one cell, end to end, with measured wall times

CLAUDE.md forbids submitting a batch before a cheap run has exercised the whole path. This task runs **one** cell at debug scale and records the per-stage cost that Part 3's shard widths depend on. It does **not** launch the 16-cell or 10-task arrays.

**Files:**
- Create: `docs/logs/20260805_tada_redesign_part2.md` (in the monorepo)
- Modify: none

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: measured per-design wall times, recorded in the log.

- [ ] **Step 1: Confirm preflight is green**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.preflight; echo "exit=$?"
```
Expected: exit 0. **If any check fails, STOP** — that is the gate doing its job. Report which check and its detail.

- [ ] **Step 2: Generate the real RFD3 inputs**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.prep_rfd_inputs
python3 -c "
import yaml; s = yaml.safe_load(open('outputs/20260805_tada_redesign/rfd_in/rfd_inputs.yaml'))
print(len(s), 'cells'); k = 'TadA8e_FULL_pt1.0'
print(k, '->', len(s[k]['select_fixed_atoms']), 'fixed keys, partial_t', s[k]['partial_t'])
"
```
Expected: 16 cells; 25 fixed keys for the FULL cell (24 residues + `ZN`).

- [ ] **Step 3: Run ONE cell of RFD3 at debug scale, timed**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
sbatch --wait --array=1-1 --export=ALL,MODE=debug rfd_partial.slurm
tail -20 logs/tada_rfd_partial_*_1.out
ls -l outputs/20260805_tada_redesign/rfd/*.cif.gz | head
sacct -n -P -j <JOBID> --format=JobID,State,Elapsed | head -2
```
Expected: 2 backbones (`n_batches=1 × diffusion_batch_size=2`), exit 0. **Record the elapsed time — this is the number the 16-cell array's `-t` is sized from.**

If RFD3 rejects the input, capture the verbatim `ValidationError`/`ComponentValidationError` and STOP. Do not guess at spec-key fixes; the semantics are cited in `prep_rfd_inputs`' docstring and any real deviation from them is a finding worth reporting.

- [ ] **Step 4: Filter those two backbones**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.filter_backbones
column -t -s $'\t' outputs/20260805_tada_redesign/backbones.tsv
```
Expected: 2 rows with per-cell counts printed. Record `motif_rmsd` for both — at `partial_t=1.0` it should be well under 1.0 Å; if it is not, that is a genuine finding about the noise ladder and must be reported, not filtered around.

- [ ] **Step 5: Prepare MPNN inputs and design 2 sequences, timed**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.prep_mpnn_inputs
sbatch --wait --array=1-1 --export=ALL,NBATCH=2 run_ligandmpnn.slurm
tail -20 logs/tada_lmpnn_*_1.out
head -6 outputs/20260805_tada_redesign/lmpnn/FULL/T0.1/seqs/*.fa
```
Expected: a `.fa` per backbone, each with the input record plus 2 design records. **Confirm no design sequence contains `:`** — that would mean the DNA context was parsed as a designed chain. Record the elapsed time.

- [ ] **Step 6: Collect and sanity-check**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign && conda run -n ligandmpnn_env python -m tada_redesign.collect_designs
column -t -s $'\t' outputs/20260805_tada_redesign/designs.tsv | head
python3 -c "
import csv
rows = list(csv.DictReader(open('outputs/20260805_tada_redesign/designs.tsv'), delimiter='\t'))
print(len(rows), 'designs; lengths', sorted({r['seq_len'] for r in rows}))
print('frozen positions unchanged check is Part 3 work; here just confirm lengths are sane')
"
```
Expected: 4 designs (2 backbones × 2 sequences), all `seq_len` inside `LENGTH_RANGE`.

- [ ] **Step 7: Write the log with the measured numbers**

Create `docs/logs/20260805_tada_redesign_part2.md` in the monorepo recording: what Part 2 built; the suite total; the real preflight output; the **measured** elapsed time per stage from Steps 3, 5; the debug `motif_rmsd` values; the projected cost of the full 16-cell and 10-task arrays derived from those measurements; and the honesty ceiling. State plainly that no batch array has been submitted.

- [ ] **Step 8: Commit, push, bump the pointer**

```bash
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign/tada-redesign
git push origin main
cd /research/rgs01/home/clusterHome/ecreed/claude-proteindesign
git add docs/logs/20260805_tada_redesign_part2.md tada-redesign
git diff --cached --stat
git commit -F - <<'EOF'
docs(tada-redesign): Part 2 generation stages + measured debug gate

Foundation (io, provenance, substrate, preflight covering all five envs and the
RMSD reference) plus the generation chain: RFD3 partial-diffusion inputs for the
16-cell sweep, the backbone filter, LigandMPNN inputs with narrow solubility
bias, and design collection.

The debug gate ran ONE cell end to end at debug scale and the log records the
measured per-stage wall times, which is what the full arrays' resource requests
are sized from. No batch array submitted.

Co-Authored-By: Ethan Creed <ethan.creed@stjude.org>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

## Self-Review

**Spec coverage.** Design matrix (16 cells × 32 backbones) → Tasks 4, 8. Fixed motif in both senses → Task 4 (`select_fixed_atoms`) and Task 6 (`fixed_residues`). Zn + ssDNA context in both generative stages → Tasks 3, 4, 6. Backbone filter's four gates → Task 5. Solubility biasing at design time plus the zero-bias control → Task 6. `designs.tsv` with cell coordinates → Task 7. Incremental writes, degraded refusal, provenance → Task 1, used by Tasks 4–7. Preflight as a real gate → Task 2. Debug gate before any batch → Task 8.

**Part 1's four deferred findings**: env probes → Task 2; RMSD reference → Task 2; shared infrastructure (TSV, provenance, degraded gate, `require_green`, 8AZ extractor) → Tasks 1–3; AF3 pLDDT scale → Task 2 (recorded as constants; Part 3 consumes them).

**Deliberately deferred to Part 3**, not gaps: the ESMFold2 screen and full re-fold, `reference_baseline`, the Rosetta stage, AF3, `correlate`, `rank`/`report`, and the CIF path into `check_zn_geometry` (Part 3 is where a CIF model first meets that check). `fold.slurm`'s commit-vs-gitignore decision also stays open, and Part 3's plan must resolve it before its batch — an uncommitted job script makes the folding runs unreproducible.

**Type consistency.** `io.append_row(path, row, columns)` is called with that argument order in Tasks 5 and 7. `provenance.write(stage_dir, stage, n_in, n_out, extra=None)` and `provenance.output_path(canonical, n_in, n_out)` match their call sites. `motif.rfd_select_fixed_atoms(arm, masks)` and `motif.mpnn_fixed_residues(arm, masks)` match Part 1's committed signatures. `score_structure.motif_rmsd(ref_atoms, pred_atoms, residues)` is called positionally in Task 5 exactly as Part 1 defines it. `substrate.context_contigs(resids, drop=None)` defaults to dropping `SUBSTRATE_RESID`, and Task 3's test passes `drop=()` explicitly to check that. `cell_id` produces `"{parent}_{arm}_pt{partial_t}"` and Tasks 5 and 7 both parse it by splitting on `_` and stripping the `pt` prefix — the same convention in both.

**One known limitation, stated rather than hidden.** `filter_backbones.main` derives a backbone's cell by stripping the last `_`-separated field from the filename, which assumes RFD3 names outputs `<spec_key>_<n>.cif.gz`. Task 8's Step 4 is what confirms that against real output; if RFD3 names them differently, the fix is a one-line change to that derivation, and the debug gate exists precisely so it is found on 2 files rather than 512.
