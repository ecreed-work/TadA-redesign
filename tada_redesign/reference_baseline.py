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
that evidence; MODES now holds exactly one mode.

Honesty ceiling: this is a confidence reference, not a stability reference. The
energetic baseline is Part 3b's Rosetta stage.
"""
import argparse
import json
import os
import subprocess

from . import constants, io, provenance

MODES = {"fold": constants.ESMFOLD_SETTINGS}


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(sub, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    out_dir = os.path.join(args.run_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    jobs = baseline_jobs()
    settings = MODES["fold"]

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
    for parent, plddt in sorted(got.items()):
        print(f"[reference_baseline] {parent}: pLDDT {plddt:.4f}")
    provenance.write(out_dir, "reference_baseline", len(jobs), len(got), extra=got)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
