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
