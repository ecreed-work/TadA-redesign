"""Gate folded designs on confidence and active-site geometry.

Reuses Part 1's score_structure for every geometric measurement -- there is one
Kabsch implementation and one motif-RMSD implementation in this campaign, and
this module adds neither.

Three deliberate choices:
  - The pLDDT gate is RELATIVE to a parent folded in the same mode
    (reference_baseline), because sampling depth can shift pLDDT for
    everything at once.
  - Motif RMSD is measured against motif.CORE_MOTIF, not the design's own
    frozen arm: the arm is what was FROZEN during design, CORE is what gets
    MEASURED here. The two-tier screen/full split was retired 2026-08-06 on
    measured evidence (see constants.MOTIF_RMSD_MAX) in favour of a single
    full-sampling fold gated at one threshold.
  - A `nan` measurement is a REJECTION with its own status, never a pass. Every
    comparison against nan is False in Python, so a naive `value > threshold`
    test silently admits broken measurements -- the exact shape of the defect
    that turned every Zn distance into nan in Part 2.

CAVEAT ON THE PASS RATE THIS MODULE PRINTS/RECORDS (2026-08-09 correction,
superseding the 2026-08-08 ruling below): the 2026-08-08 claim that
`MOTIF_RMSD_MAX` is a "gross-failure catch, not a ranking metric" and that "a
near-100% pass rate is expected" was inferred from 21 probe designs that all
came from a single cell (`TadA8e_FULL_pt1.0`, the easiest of sixteen). The full
10,542-design screen falsifies it: the gate discriminates strongly and by
design (dose-response over re-noising, FULL-arm pt1.0 65.6% down to pt6.0
0.7% on the all-heavy-atom run; 58.6% down to 0.1% on the current
backbone-only run). Do not predict a pass rate here -- state the one this run
measured. See docs/logs/20260809_backbone_core_metric.md for the full
sampling-error post-mortem and the corrected numbers.

[SUPERSEDED 2026-08-08 text, kept for the record -- refuted by the full run
above] "with the entire 21-probe CORE distribution sitting inside the
parent's own fold-to-fold jitter band (see constants.MOTIF_RMSD_MAX),
`MOTIF_RMSD_MAX` is a GROSS-FAILURE catch, not a ranking metric. A pass rate
near 100% is therefore an expected consequence of that re-role, not evidence
the designs are good -- it reads a design's core as 'not obviously
collapsed,' nothing stronger."

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
    if plddt < parent_plddt - constants.PLDDT_MARGIN:
        return False, "low_plddt"
    if rmsd > constants.MOTIF_RMSD_MAX:
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
        atoms = score_structure.align_numbering(ref_atoms, atoms)
        # BACKBONE atoms only. Measuring all heavy atoms made every MIN-arm
        # design unscorable: MIN freezes 4 of CORE's 17 residues, so LigandMPNN
        # redesigned the other 13 and the reference's sidechain atoms do not
        # exist in the prediction -- `motif_rmsd` then raises (correctly; a
        # silently shrunk measured set would report a falsely good number).
        # 5,166 of 10,542 designs died that way on the 2026-08-09 run. N/CA/C/O
        # are identity-independent, so both arms score and compare directly.
        out["motif_rmsd"] = score_structure.motif_rmsd(
            ref_atoms, atoms, residues, atom_names=constants.BACKBONE_ATOMS)
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
        args.run_dir, require=list(constants.PARENTS))

    masks = motif.load_masks()
    az = substrate.substrate_xyz()
    refs = {p: score_structure.heavy_atoms_from_pdb(constants.RMSD_REFERENCE[p])
            for p in constants.PARENTS}
    # d["arm"] is what was FROZEN during design (FULL or MIN); CORE_MOTIF is the
    # fixed, arm-independent set that gets MEASURED here -- see motif.py.
    core_residues = motif.arm_residues(motif.CORE_MOTIF, masks)
    shard_dir = os.path.join(args.run_dir, "fold_screen")
    out_path = os.path.join(args.run_dir, "fold_screen.tsv")
    if os.path.exists(out_path):
        os.unlink(out_path)

    n_pass = 0
    for d in designs:
        cif = os.path.join(shard_dir, f"{d['design_id']}.cif")
        metrics = os.path.join(shard_dir, f"{d['design_id']}.metrics.json")
        scored = score_one(cif, metrics, refs[d["parent"]], core_residues, az)
        passed, status = gate(scored, baseline[d["parent"]])
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

    print(f"[score_folds] {n_pass}/{len(designs)} passed "
          f"({n_pass / float(len(designs)):.1%}) -> {out_path} "
          f"(MOTIF_RMSD_MAX discriminates -- rate is measured per run, not "
          f"predicted; see docs/logs/20260809_backbone_core_metric.md)")
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
