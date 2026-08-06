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
