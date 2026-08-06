"""The substrate is measured once, here. Two facts are load-bearing: AtomWorks
drops the 8AZ so RFD3's context must be split around it, and chain C has nothing
within 12 A so it is excluded entirely."""
import numpy as np
import pytest

from tada_redesign import constants, substrate


def test_substrate_xyz_returns_the_8az_heavy_atoms():
    xyz = substrate.substrate_xyz()
    assert xyz.shape == (22, 3)
    assert xyz.dtype == float


def test_context_residues_are_chain_d_23_to_29():
    """Measured 2026-08-05 at a 12 A cutoff."""
    assert substrate.context_residues() == (23, 24, 25, 26, 27, 28, 29)
    assert substrate.context_residues() == constants.DNA_CONTEXT_RESIDS


def test_context_contigs_split_around_the_dropped_8az():
    """RFD3's AtomWorks loader drops non-standard residues, and a labelled contig
    range spanning a residue it never loaded raises ComponentValidationError."""
    assert substrate.context_contigs((23, 24, 25, 26, 27, 28, 29)) == ((23, 25), (27, 29))


def test_context_contigs_split_around_a_numbering_gap():
    """Chain D really has a 32-38 gap; a range spanning it would also fail."""
    assert substrate.context_contigs((29, 30, 31, 39, 40)) == ((29, 31), (39, 40))


def test_context_contigs_drops_only_what_it_is_told_to():
    assert substrate.context_contigs((23, 24, 25), drop=()) == ((23, 25),)


def test_hotspots_are_the_retained_neighbours_of_the_target_base():
    resids = substrate.context_residues()
    for h in constants.HOTSPOT_RESIDS:
        assert h in resids
        assert h != constants.SUBSTRATE_RESID     # the base itself is dropped


def test_substrate_xyz_is_in_the_same_frame_as_the_reference_parents():
    """cleft_clearance treats these coordinates as living in the reference frame.
    Sanity: the 8AZ must sit within a few Angstrom of the relaxed parent's Zn,
    which is the catalytic geometry the whole campaign is built on."""
    from tada_redesign import score_structure as ss
    atoms = ss.heavy_atoms_from_pdb(constants.PARENT_PDB["TadA8e"])
    zn = next(v for (resnum, name), v in atoms.items() if name == "ZN")
    d = float(np.min(np.linalg.norm(substrate.substrate_xyz() - zn, axis=1)))
    assert 1.5 < d < 4.0, d
