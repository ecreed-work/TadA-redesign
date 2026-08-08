"""Shard designs.tsv across a SLURM array and hand each shard to fold_many.

Sharding is a contiguous, deterministic split of the row order. Contiguity keeps
a shard's designs adjacent in designs.tsv, which makes a partial run easy to
reason about; determinism means a resubmitted shard folds exactly the same
designs, so `--skip-existing` composes correctly.

Folds at constants.ESMFOLD_SETTINGS -- the single full-sampling setting every
design uses since the two-tier screen was retired 2026-08-06. The gate is
still RELATIVE to a parent folded in the identical mode -- see
reference_baseline.

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
           "--num-loops", str(constants.ESMFOLD_SETTINGS["num_loops"]),
           "--num-sampling-steps", str(constants.ESMFOLD_SETTINGS["num_sampling_steps"]),
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
