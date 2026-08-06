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


def backbone_id(path):
    """Filename minus its compression/format suffix.

    NOT `split(".")[0]`: cell ids embed a float (e.g. `pt1.0`), so splitting on
    the first dot truncates mid-name and makes every model in a cell collide on
    one id.
    """
    name = os.path.basename(path)
    for suffix in (".cif.gz", ".cif", ".pdb", ".ent"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


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


def zn_donor_distances(atoms, zn_xyz=None):
    """{"zn_57ND1": distance, ...}; nan for a donor or Zn that is absent.

    `zn_xyz` is supplied by the caller when the metal was located
    chain-agnostically (RFD3 relabels its chain); otherwise the metal is looked
    up in `atoms`, which is how synthetic fixtures supply it.
    """
    zn = zn_xyz if zn_xyz is not None else next(
        (v for (resnum, name), v in atoms.items() if name == "ZN"), None)
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
    backbone = backbone_id(cif_path)
    try:
        atoms = load_backbone(cif_path)
    except Exception as exc:                      # noqa: BLE001 - deliberate
        return _row(backbone, cell, cif_path, n_res="NA", max_ca_break="NA",
                    motif_rmsd="NA", status=f"load_failed: {exc}", passed="False")

    ca = score_structure.ca_map(atoms)
    n_res = len(ca)
    try:
        zn_xyz = score_structure.metal_xyz(cif_path)
    except (OSError, ValueError):
        zn_xyz = None
    zn = zn_donor_distances(atoms, zn_xyz)
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
    paths = sorted(glob.glob(os.path.join(args.run_dir, args.rfd_subdir, "*", "*.cif.gz")))

    for path in paths:
        # RFD3 names outputs <yaml-stem>_<group>_<batch>_model_<n>.cif.gz and cell
        # ids embed a float, so the cell comes from the per-cell output DIRECTORY,
        # never from parsing the filename (measured 2026-08-05).
        cell = os.path.basename(os.path.dirname(path))
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
