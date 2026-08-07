"""Fold both parents through the IDENTICAL path the designs take.

The screen's pLDDT gate is relative: `plddt >= parent_plddt - SCREEN_PLDDT_MARGIN`.
That only means anything if the parent was folded the same way as the design
being gated, so both parents are folded in BOTH modes: `screen` gates the
screen, `full` gates the full-sampling re-fold of survivors.

MEASURED 2026-08-06 (job 234208): TadA8e full 0.8968 / screen 0.9005; TadA9
full 0.8882 / screen 0.8900. Sampling depth changes pLDDT by under 0.004 for
these 156-residue proteins -- the two modes are equivalent within noise, and
`screen` was in fact marginally HIGHER for both parents. The earlier belief
that reduced sampling substantially depresses pLDDT came from a single
uncontrolled 78-mer fold on 2026-08-04 (which returned 0.45) and does not
survive this controlled comparison; that low value reflected the difficulty
of that particular peptide, not the sampling settings. The relative baseline
is therefore kept for correctness and symmetry -- so a screen-mode design is
always gated against a screen-mode parent -- not because a large penalty
needs correcting.

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
