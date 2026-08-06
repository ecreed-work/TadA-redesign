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
