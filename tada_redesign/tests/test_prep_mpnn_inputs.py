"""Solubility bias is an assumption, so it is applied narrowly and measured
against a zero-bias control -- never applied to a buried or frozen position."""
import json

import numpy as np
import pytest

from tada_redesign import (constants, motif, prep_mpnn_inputs as prep,
                            score_structure as ss, substrate)

# Locked literals, independent of bias_positions' own set expression.
FULL_MOTIF = (28, 30, 46, 54, 57, 58, 59, 84, 85, 86, 87, 88, 90,
              108, 109, 110, 111, 148, 149, 152, 153, 154, 156, 157)
TETRAD = (57, 59, 87, 90)


@pytest.fixture
def masks():
    return motif.load_masks()


def test_full_arm_bias_never_touches_a_frozen_motif_position(masks):
    """Asserted against a literal residue list, not against the same set
    expression bias_positions uses -- otherwise a bug in that expression would
    be duplicated on both sides of the comparison and never caught."""
    positions = set(prep.bias_positions("FULL", masks))
    assert not (positions & set(FULL_MOTIF))
    assert positions <= set(masks["EXPOSED"])
    assert positions <= set(masks["MODELED"])
    assert len(positions) == 67


def test_min_arm_biases_the_pocket_but_never_the_catalytic_tetrad(masks):
    """The MIN arm freezes only the tetrad, so pocket and DNA-face residues ARE
    designable and biasable here -- that difference from the FULL arm is the
    whole point of running two arms, so it is asserted rather than assumed."""
    positions = set(prep.bias_positions("MIN", masks))
    assert not (positions & set(TETRAD))
    assert positions & set(FULL_MOTIF)          # pocket/DNA-face ARE biasable
    assert len(positions) == 81
    # 82 exposed+modelled positions minus the one exposed tetrad member (His57)
    assert len(set(masks["EXPOSED"]) & set(masks["MODELED"])) == 82
    assert set(masks["EXPOSED"]) & set(TETRAD) == {57}


def test_bias_never_touches_a_buried_position(masks):
    """Biasing a core position would trade away the stability we are buying."""
    buried = set(masks["BURIED"])
    for arm in constants.ARMS:
        assert not (set(prep.bias_positions(arm, masks)) & buried)


def test_bias_json_penalises_hydrophobics_and_rewards_polars(masks):
    bias = prep.bias_json("FULL", masks)
    key = f"{constants.SCAFFOLD_CHAIN}{prep.bias_positions('FULL', masks)[0]}"
    assert bias[key]["L"] == constants.HYDROPHOBIC_BIAS < 0
    assert bias[key]["E"] == constants.POLAR_BIAS > 0
    assert set(bias[key]) == set(constants.HYDROPHOBIC_SET + constants.POLAR_SET)


def test_bias_json_keys_are_chain_prefixed_residues(masks):
    for key in prep.bias_json("MIN", masks):
        assert key[0] == constants.SCAFFOLD_CHAIN
        assert key[1:].isdigit()


def test_bias_json_is_json_serialisable(masks):
    assert json.loads(json.dumps(prep.bias_json("FULL", masks)))


def test_control_temperature_is_one_of_the_swept_temperatures():
    """The control must be comparable to a biased cell, not an orphan point."""
    assert constants.CONTROL_TEMP in constants.MPNN_TEMPS


# ---------------------------------------------------------------------------
# graft_substrate: RFD3 consumes chain D for hotspot orientation but does not
# emit it (measured; prep_rfd_inputs.py). LigandMPNN needs it present as
# ligand context, so it is grafted back on here by superposition.
# ---------------------------------------------------------------------------

TETRAD = (57, 59, 87, 90)


def test_graft_substrate_reproduces_the_crystal_dna_when_designed_is_the_reference(tmp_path):
    """Grafting onto the reference parent ITSELF is a pure identity transform
    (the anchor's `P` and `Q` come from the same file, so the fitted rotation
    must be ~I and the translation ~0) -- an inverted or transposed transform
    would place the substrate on the wrong side of the backbone instead, and
    this is the cheapest check that catches that."""
    parent = "TadA8e"
    ref_pdb = constants.RMSD_REFERENCE[parent]
    designed_atoms = ss.heavy_atoms_from_pdb(ref_pdb)
    out = str(tmp_path / "grafted.pdb")
    prep.graft_substrate(designed_atoms, parent, ref_pdb, out)

    grafted_dna = ss.heavy_atoms_from_pdb(out, chain=constants.SUBSTRATE_CHAIN)
    from Bio.PDB import PDBParser
    ref_model = PDBParser(QUIET=True).get_structure("ref", constants.PDB6VPC)[0]
    crystal = {(res.id[1], atom.get_name()): atom.get_coord()
               for res in ref_model[constants.SUBSTRATE_CHAIN] for atom in res}
    diffs = [float(np.linalg.norm(v - crystal[k]))
             for k, v in grafted_dna.items() if k in crystal]
    assert len(diffs) == len(grafted_dna) > 0
    assert max(diffs) < 1e-3


def test_graft_substrate_writes_chain_d_with_exactly_the_context_residues(tmp_path):
    parent = "TadA8e"
    ref_pdb = constants.RMSD_REFERENCE[parent]
    designed_atoms = ss.heavy_atoms_from_pdb(ref_pdb)
    out = str(tmp_path / "grafted.pdb")
    prep.graft_substrate(designed_atoms, parent, ref_pdb, out)

    lines = [ln for ln in open(out) if ln.startswith(("ATOM", "HETATM"))]
    chains = {ln[21] for ln in lines}
    assert constants.SCAFFOLD_CHAIN in chains
    assert constants.SUBSTRATE_CHAIN in chains
    assert any(ln[17:20].strip() == constants.ZN_RESNAME for ln in lines)
    dna_resids = {int(ln[22:26]) for ln in lines if ln[21] == constants.SUBSTRATE_CHAIN}
    assert dna_resids == set(substrate.context_residues())
    assert constants.SUBSTRATE_RESID in dna_resids     # the 8AZ itself


def test_graft_substrate_places_the_substrate_in_the_pocket(tmp_path):
    """The substrate must sit near the catalytic tetrad, not merely somewhere
    in the file: the 8AZ-to-tetrad distance in the grafted output must match
    the reference's own distance within a small tolerance."""
    parent = "TadA8e"
    ref_pdb = constants.RMSD_REFERENCE[parent]
    ref_atoms = ss.heavy_atoms_from_pdb(ref_pdb)
    ref_az = substrate.substrate_xyz()
    ref_tetrad_xyz = np.array([v for (r, n), v in ref_atoms.items() if r in TETRAD])
    ref_dist = float(np.min(np.linalg.norm(
        ref_tetrad_xyz[:, None, :] - ref_az[None, :, :], axis=2)))

    designed_atoms = ss.heavy_atoms_from_pdb(ref_pdb)
    out = str(tmp_path / "grafted.pdb")
    prep.graft_substrate(designed_atoms, parent, ref_pdb, out)
    grafted = ss.heavy_atoms_from_pdb(out)
    grafted_dna = ss.heavy_atoms_from_pdb(out, chain=constants.SUBSTRATE_CHAIN)
    grafted_az = np.array([v for (r, n), v in grafted_dna.items()
                          if r == constants.SUBSTRATE_RESID])
    grafted_tetrad_xyz = np.array([v for (r, n), v in grafted.items() if r in TETRAD])
    grafted_dist = float(np.min(np.linalg.norm(
        grafted_tetrad_xyz[:, None, :] - grafted_az[None, :, :], axis=2)))
    assert abs(grafted_dist - ref_dist) < 0.05
