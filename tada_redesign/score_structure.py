"""Shared geometric scorer for both folding stages.

One implementation of superposition and RMSD, called by the ESMFold2 screen,
the full-sampling re-fold, and the AF3 stage. Two independent forward passes of
any structure predictor share no global frame, so EVERY comparison here
superposes before measuring. The predecessor campaign shipped a `pocket_rmsd`
that summed raw coordinate differences with no superposition at all; at a 1.0-A
threshold that silently fails every design after the GPU spend is already
gone (docs/logs/20260728_tada_stabilization.md, fix round 2).

Heavy-atom rather than CA-only RMSD is tractable here precisely because motif
identity is LOCKED by `motif.py`, so atom names match one-to-one between
reference and design.

Honesty ceiling: an intact motif geometry in a predicted model is not evidence
of catalytic activity. Nothing in this module measures function.
"""
import os

import numpy as np

from . import constants

_HYDROGEN = {"H", "D"}


def kabsch(P, Q):
    """Rotation and centroids such that `(Q - Q_mean) @ R.T + P_mean`
    least-squares superposes `Q` onto `P`.

    Mean-centred SVD with a reflection correction, mirroring
    `tada_stability.gate_fold._kabsch` and
    `denovo_tada/rf3_gate.py::ca_rmsd_over_resids` rather than introducing a
    third implementation of the same arithmetic.
    """
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
    if len(P) != len(Q):
        raise ValueError(f"point count mismatch: {len(P)} vs {len(Q)}")
    if len(P) < 3:
        raise ValueError(f"Kabsch is ill-posed on {len(P)} points; need >= 3")
    P_mean, Q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - P_mean, Q - Q_mean
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return R, P_mean, Q_mean


def apply_transform(X, R, P_mean, Q_mean):
    return (np.asarray(X, dtype=float) - Q_mean) @ R.T + P_mean


def _element_of(atom_name, type_symbol=None):
    if type_symbol and type_symbol.strip():
        return type_symbol.strip().upper()
    return atom_name.strip()[0].upper()


def heavy_atoms_from_pdb(path, chain=None):
    """{(resnum, atom_name): xyz} for one chain's non-hydrogen atoms.

    Hetero atoms (the Zn) are kept: the metal's position is a measured quantity
    for this campaign, not a decoration.
    """
    chain = chain or constants.SCAFFOLD_CHAIN
    atoms = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21] != chain:
                continue
            atom_name = line[12:16].strip()
            element = _element_of(atom_name, line[76:78] if len(line) > 77 else "")
            if element in _HYDROGEN:
                continue
            resnum = int(line[22:26])
            atoms[(resnum, atom_name)] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return atoms


def heavy_atoms_from_cif(path, chain=None):
    """Same mapping, from an mmCIF `_atom_site` loop.

    ESMFold2 and AF3 both emit mmCIF. Uses Biopython's MMCIF2Dict (as
    `denovo_tada/filter_backbones.py::_load_atom_site` does) but requires only
    the standard columns -- RFD3's custom `is_motif_atom_with_fixed_seq` column
    is absent from folding-model output.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    chain = chain or constants.SCAFFOLD_CHAIN
    d = MMCIF2Dict(path)

    def col(*candidates):
        for name in candidates:
            key = f"_atom_site.{name}"
            if key in d:
                return d[key]
        raise KeyError(f"mmCIF {path} has none of {candidates}")

    chains = col("auth_asym_id", "label_asym_id")
    resids = col("auth_seq_id", "label_seq_id")
    names = col("auth_atom_id", "label_atom_id")
    elements = col("type_symbol")
    xs, ys, zs = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")

    atoms = {}
    for i in range(len(chains)):
        if chains[i] != chain:
            continue
        if elements[i].strip().upper() in _HYDROGEN:
            continue
        atoms[(int(resids[i]), names[i].strip())] = np.array(
            [float(xs[i]), float(ys[i]), float(zs[i])])
    return atoms


def metal_xyz(path, resname="ZN"):
    """Coordinates of the single metal ion, found by IDENTITY not by chain.

    RFD3's AtomWorks loader relabels a hetero atom's chain when it shared a
    chain letter with protein residues on input -- measured 2026-08-05: the Zn
    comes back on asym id `B` while the protein is elsewhere. Every chain-scoped
    reader therefore misses it, which silently turns all Zn-donor distances into
    nan. Returns None when absent; raises ValueError on more than one match, so
    an unexpectedly multi-metal input fails loudly instead of picking one.
    """
    hits = []
    if path.endswith((".pdb", ".ent")):
        for line in open(path):
            if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == resname:
                hits.append(np.array([float(line[30:38]), float(line[38:46]),
                                      float(line[46:54])]))
    else:
        opener = __import__("gzip").open if path.endswith(".gz") else open
        import tempfile
        if path.endswith(".gz"):
            import shutil
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                with opener(path, "rb") as src:
                    shutil.copyfileobj(src, tmp)
                cif_path = tmp.name
        else:
            cif_path = path
        try:
            from Bio.PDB.MMCIF2Dict import MMCIF2Dict
            d = MMCIF2Dict(cif_path)
            comp = d.get("_atom_site.auth_comp_id") or d.get("_atom_site.label_comp_id")
            xs, ys, zs = (d["_atom_site.Cartn_x"], d["_atom_site.Cartn_y"],
                          d["_atom_site.Cartn_z"])
            for i, name in enumerate(comp):
                if name.strip() == resname:
                    hits.append(np.array([float(xs[i]), float(ys[i]), float(zs[i])]))
        finally:
            if path.endswith(".gz"):
                os.unlink(cif_path)
    if len(hits) > 1:
        raise ValueError(f"{path}: expected one {resname}, found {len(hits)}")
    return hits[0] if hits else None


def ca_map(atoms):
    """{resnum: xyz} over CA atoms only."""
    return {resnum: xyz for (resnum, name), xyz in atoms.items() if name == "CA"}


def _anchor_arrays(ref_atoms, pred_atoms, anchor_residues):
    """Paired CA coordinate arrays for the superposition anchor.

    Default anchor is EVERY CA shared by the two structures -- the full
    modelled backbone, deliberately NOT the (much smaller) set being measured.
    Fitting on the measured points would trivially shrink the very quantity
    being reported.
    """
    ref_ca, pred_ca = ca_map(ref_atoms), ca_map(pred_atoms)
    shared = sorted(set(ref_ca) & set(pred_ca))
    if anchor_residues is not None:
        shared = [r for r in sorted(anchor_residues) if r in ref_ca and r in pred_ca]
    if len(shared) < 3:
        raise ValueError(
            f"superposition anchor has {len(shared)} shared CA; need >= 3")
    P = np.array([ref_ca[r] for r in shared])
    Q = np.array([pred_ca[r] for r in shared])
    return P, Q


def motif_rmsd(ref_atoms, pred_atoms, residues, anchor_residues=None):
    """Heavy-atom RMSD (A) over `residues`, after CA superposition.

    Raises `KeyError` if any atom of a measured residue is missing from either
    structure -- a silently shrunk measured set would report a falsely good
    number on a broken design.
    """
    P, Q = _anchor_arrays(ref_atoms, pred_atoms, anchor_residues)
    R, P_mean, Q_mean = kabsch(P, Q)

    keys = sorted(k for k in ref_atoms if k[0] in set(residues))
    if not keys:
        raise KeyError(f"no reference atoms for residues {tuple(residues)}")
    missing = [k for k in keys if k not in pred_atoms]
    if missing:
        raise KeyError(f"prediction is missing measured atoms: {missing}")

    ref_xyz = np.array([ref_atoms[k] for k in keys])
    pred_xyz = apply_transform(np.array([pred_atoms[k] for k in keys]),
                               R, P_mean, Q_mean)
    return float(np.sqrt(np.mean(np.sum((ref_xyz - pred_xyz) ** 2, axis=1))))


def cleft_clearance(ref_atoms, pred_atoms, substrate_xyz, anchor_residues=None,
                    exclude_atom_names=(constants.ZN_RESNAME,)):
    """Minimum distance (A) from any substrate atom to any design heavy atom.

    `substrate_xyz` is in the REFERENCE frame (the crystal 8AZ coordinates).
    The prediction is superposed onto the reference, so the substrate's
    crystallographic position is effectively mapped into the design's frame; a
    small value means the design's own atoms now occupy the space the target
    base must sit in, i.e. the substrate cleft closed. This is the specific
    failure the tetrad-only MIN arm is exposed to, since nothing but the
    substrate context holds that cleft open during design.

    `exclude_atom_names` defaults to excluding the catalytic Zn. The metal
    coordinates the target base at ~2.12 A as a matter of correct catalytic
    geometry, not a clash -- counting it makes a correctly-placed metal read
    as a collapsed cleft and rewards designs that lose it. Measured on the
    committed references (8AZ = 6VPC chain D residue 26): the crystal
    `chainF_raw.pdb` measured against itself gives 2.121 A with the Zn
    counted (closest atom the Zn itself), versus 2.330 A for the closest
    protein atom, Arg111:NH1, once the Zn is excluded.
    """
    P, Q = _anchor_arrays(ref_atoms, pred_atoms, anchor_residues)
    R, P_mean, Q_mean = kabsch(P, Q)

    keys = sorted(k for k in pred_atoms if k[1] not in set(exclude_atom_names))
    if not keys:
        raise ValueError("no design atoms left to measure after exclusions")
    pred_xyz = apply_transform(np.array([pred_atoms[k] for k in keys]),
                               R, P_mean, Q_mean)
    sub = np.atleast_2d(np.asarray(substrate_xyz, dtype=float))
    d = np.linalg.norm(pred_xyz[None, :, :] - sub[:, None, :], axis=2)
    return float(d.min())
