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
    # Every design produced a row (score_one never raises), so the degraded gate
    # compares rows WRITTEN to inputs and cannot trip here. A low PASS rate is a
    # measurement, not a stage failure -- the correction an earlier review
    # required, kept deliberately in place.
    provenance.write(args.run_dir, "score_folds", len(designs), len(designs),
                     extra={"n_passed": n_pass,
                            "pass_rate": round(n_pass / float(len(designs)), 4)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
