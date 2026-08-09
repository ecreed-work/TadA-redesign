"""Fold both parents through the IDENTICAL path the designs take.

The gate's pLDDT check is relative: `plddt >= parent_plddt - PLDDT_MARGIN`.
That only means anything if the parent was folded the same way as the design
being gated, so both parents are folded once, at the single full-sampling
setting every design now uses.

The earlier two-tier design folded each parent in a `screen` mode and a `full`
mode, on the belief that reduced sampling substantially depresses pLDDT (from
a single uncontrolled 78-mer fold on 2026-08-04 that returned 0.45). A
controlled comparison on 2026-08-06 (job 234208) found the opposite for
confidence: TadA8e full 0.8968 / screen 0.9005, TadA9 full 0.8882 / screen
0.8900 -- pLDDT was flat within 0.004 across sampling depth. But the SAME
measurement found the core-motif RMSD noise floor was 3.6x worse under
reduced sampling (2.020 A vs 0.563 A median parent-vs-parent jitter) --
confidence and structural reproducibility are different quantities, and a
flat pLDDT hid a real structural problem. The two-tier screen is retired on
that evidence; there is exactly one fold setting now, `constants.ESMFOLD_SETTINGS`.

Honesty ceiling: this is a confidence reference, not a stability reference. The
energetic baseline is Part 3b's Rosetta stage. Motif RMSD (below) measures
geometry, not activity.

Also scores each parent's own CORE-motif RMSD against `constants.RMSD_REFERENCE`
(via `score_structure.motif_rmsd`, after `align_numbering`) -- the SAME path
`score_folds.py` uses for every design, so the campaign's headline parent
numbers (TadA8e 1.354 A, TadA9 1.357 A, all heavy atoms) are reproducible from
this committed module rather than from an unversioned side-script.
`--score-only` scores whatever `<parent>__fold.cif` already exists in
`--run-dir/baseline` without invoking `fold_many.py` at all -- no fold, no
GPU, no SLURM.

Reports BOTH the all-heavy-atom RMSD and the backbone-only (N/CA/C/O,
`constants.BACKBONE_ATOMS`) RMSD, side by side (TadA8e 0.7348 A / TadA9
0.6464 A backbone-only, measured 2026-08-09). The backbone-only figure is
identity-independent, so it is comparable across the FULL and MIN design arms
-- see docs/plans/2026-08-09-backbone-core-metric.md, which also derives
`constants.MOTIF_RMSD_MAX` from this same number.
"""
import argparse
import json
import os
import subprocess

from . import constants, io, motif, provenance, score_structure


def baseline_id(parent):
    return f"{parent}__fold"


def baseline_jobs():
    return [{"parent": parent, "design_id": baseline_id(parent),
             "sequence": constants.PARENT_SEQUENCE[parent]}
            for parent in constants.PARENTS]


def read_baseline(run_dir, require=None):
    """{parent: plddt}. Raises if any `require`d parent is absent."""
    out = {}
    d = os.path.join(run_dir, "baseline")
    for parent in constants.PARENTS:
        path = os.path.join(d, f"{baseline_id(parent)}.metrics.json")
        if os.path.exists(path):
            out[parent] = float(json.load(open(path))["plddt"])
    for parent in (require or []):
        if parent not in out:
            raise FileNotFoundError(
                f"missing baseline fold for {parent}; gating designs against an "
                f"absent baseline would pass or fail all of them")
    return out


def core_motif_residues():
    """Residue numbers of `motif.CORE_MOTIF`, intersected with MODELED --
    identical to what `score_folds.py` measures for every design."""
    return motif.arm_residues(motif.CORE_MOTIF, motif.load_masks())


def score_baseline_rmsd(run_dir, parents=None, residues=None, atom_names=None):
    """{parent: CORE-motif RMSD (A) vs `constants.RMSD_REFERENCE`}.

    Scores whatever `<parent>__fold.cif` already exists under
    `run_dir/baseline` -- never folds, never invokes `fold_many.py`. Uses the
    exact same `heavy_atoms_from_cif` -> `align_numbering` -> `motif_rmsd`
    sequence `score_folds.score_one` uses for a design, so a parent's number
    is measured through the identical code path as everything it gates. A
    parent with no fold on disk yet is simply absent from the returned dict,
    not raised on: `read_baseline`'s `require=` is the place that enforces
    presence when it matters.

    `atom_names` is forwarded to `score_structure.motif_rmsd` unchanged;
    default `None` measures every heavy atom, matching every existing caller.
    Pass `constants.BACKBONE_ATOMS` for the identity-independent measurement
    (see `main`, which reports both side by side).
    """
    residues = core_motif_residues() if residues is None else residues
    out = {}
    d = os.path.join(run_dir, "baseline")
    for parent in (parents or constants.PARENTS):
        cif = os.path.join(d, f"{baseline_id(parent)}.cif")
        if not os.path.exists(cif):
            continue
        ref_atoms = score_structure.heavy_atoms_from_pdb(constants.RMSD_REFERENCE[parent])
        pred_atoms = score_structure.heavy_atoms_from_cif(cif)
        pred_atoms = score_structure.align_numbering(ref_atoms, pred_atoms)
        out[parent] = score_structure.motif_rmsd(ref_atoms, pred_atoms, residues,
                                                  atom_names=atom_names)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(sub, "outputs", constants.RUN_DIR_NAME))
    ap.add_argument("--score-only", action="store_true",
                     help="Skip folding entirely; score CORE-motif RMSD (and "
                          "read existing pLDDT) from whatever "
                          "<parent>__fold.cif already exists under "
                          "--run-dir/baseline. Never invokes fold_many.py, "
                          "so no GPU and no SLURM job are used.")
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.run_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    jobs = baseline_jobs()

    if not args.score_only:
        settings = constants.ESMFOLD_SETTINGS
        jobs_tsv = os.path.join(out_dir, "jobs_fold.tsv")
        io.write_tsv(jobs_tsv, [{"design_id": j["design_id"], "sequence": j["sequence"]}
                                for j in jobs], ("design_id", "sequence"))
        cmd = ["python", os.path.join(constants.MONOREPO, "tools/esmfold2/fold_many.py"),
               "--jobs", jobs_tsv, "--out-dir", out_dir,
               "--ligand-ccd", constants.ZN_RESNAME,
               "--num-loops", str(settings["num_loops"]),
               "--num-sampling-steps", str(settings["num_sampling_steps"]),
               "--skip-existing"]
        print(f"[reference_baseline] fold: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False)

    got = read_baseline(args.run_dir)
    rmsd = score_baseline_rmsd(args.run_dir)
    rmsd_bb = score_baseline_rmsd(args.run_dir, atom_names=constants.BACKBONE_ATOMS)
    for parent in constants.PARENTS:
        plddt_s = f"{got[parent]:.4f}" if parent in got else "MISSING"
        rmsd_s = f"{rmsd[parent]:.4f} A" if parent in rmsd else "MISSING"
        rmsd_bb_s = f"{rmsd_bb[parent]:.4f} A" if parent in rmsd_bb else "MISSING"
        print(f"[reference_baseline] {parent}: pLDDT {plddt_s}  "
              f"CORE motif RMSD (all heavy atoms) {rmsd_s}  "
              f"CORE motif RMSD (backbone only) {rmsd_bb_s}")

    io.write_tsv(
        os.path.join(out_dir, "baseline_summary.tsv"),
        [{"parent": p,
          "plddt": round(got[p], 4) if p in got else io.MISSING,
          "core_motif_rmsd": round(rmsd[p], 4) if p in rmsd else io.MISSING,
          "core_motif_rmsd_backbone": round(rmsd_bb[p], 4) if p in rmsd_bb else io.MISSING}
         for p in constants.PARENTS],
        ("parent", "plddt", "core_motif_rmsd", "core_motif_rmsd_backbone"),
        header_comment="CORE motif RMSD is geometry (Kabsch-superposed RMSD vs "
                        "constants.RMSD_REFERENCE), not stability or activity. "
                        "core_motif_rmsd is all heavy atoms (identity-dependent); "
                        "core_motif_rmsd_backbone is N/CA/C/O only "
                        "(constants.BACKBONE_ATOMS, identity-independent -- see "
                        "docs/plans/2026-08-09-backbone-core-metric.md).")

    extra = dict(got)
    extra["core_motif_rmsd"] = rmsd
    extra["core_motif_rmsd_backbone"] = rmsd_bb
    provenance.write(out_dir, "reference_baseline", len(jobs), len(got), extra=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
