"""The motif set is the campaign's single source of truth. Its predecessor's
worst defects came from a generator and a gate disagreeing about one residue
set, so these tests assert the three renderers cannot diverge."""
import pytest

from tada_redesign import constants, motif

# Locked 2026-08-05 by computing (CATALYTIC | POCKET | DNA_FACE) & MODELED
# against the on-disk masks.json.
FULL_EXPECTED = (28, 30, 46, 54, 57, 58, 59, 84, 85, 86, 87, 88, 90,
                 108, 109, 110, 111, 148, 149, 152, 153, 154, 156, 157)
TETRAD = (57, 59, 87, 90)


@pytest.fixture
def masks():
    return motif.load_masks()


def test_full_arm_is_the_locked_24_residues(masks):
    assert motif.arm_residues(motif.ARM_FULL, masks) == FULL_EXPECTED
    assert len(FULL_EXPECTED) == 24


def test_min_arm_is_the_catalytic_tetrad(masks):
    assert motif.arm_residues(motif.ARM_MIN, masks) == TETRAD


def test_full_arm_is_not_masks_json_frozen(masks):
    """masks.json's own FROZEN key is 36 residues and includes ZN_PROXIMITY --
    it exists to keep cartesian_ddg away from the metal site, a failure mode
    diffusion + LigandMPNN do not exhibit. Reading it here would silently
    freeze 12 extra designable positions."""
    assert len(masks["FROZEN"]) == 36
    assert set(motif.arm_residues(motif.ARM_FULL, masks)) != set(masks["FROZEN"])


def test_all_three_renderers_agree_on_one_residue_set(masks):
    for arm in constants.ARMS:
        residues = set(motif.arm_residues(arm, masks))
        rfd = motif.rfd_select_fixed_atoms(arm, masks)
        rfd_residues = {int(k[1:]) for k in rfd if k != constants.ZN_RESNAME}
        mpnn_residues = {int(tok[1:])
                         for tok in motif.mpnn_fixed_residues(arm, masks).split()}
        assert rfd_residues == residues
        assert mpnn_residues == residues
        assert set(motif.measured_residues(arm, masks)) == residues


def test_zn_is_keyed_by_ccd_name_not_chain_resid(masks):
    """AtomWorks renames a hetero atom's chain when it shares a chain letter
    with protein residues, so a "F201" key can never resolve post-load -- a
    real RFD3 ValidationError, not a precaution."""
    rfd = motif.rfd_select_fixed_atoms(motif.ARM_FULL, masks)
    assert rfd[constants.ZN_RESNAME] == motif.FIXED_ATOM_KEYWORD
    assert "F201" not in rfd


def test_every_motif_residue_is_fixed_backbone_and_sidechain(masks):
    """ALL, not TIP: tada-stability commit 132509f only passed
    check_zn_geometry once the donors' BACKBONE was frozen too."""
    rfd = motif.rfd_select_fixed_atoms(motif.ARM_FULL, masks)
    assert set(rfd.values()) == {"ALL"}


def test_mpnn_fixed_residues_uses_the_scaffold_chain(masks):
    tokens = motif.mpnn_fixed_residues(motif.ARM_MIN, masks).split()
    assert tokens == [f"{constants.SCAFFOLD_CHAIN}{r}" for r in TETRAD]


def test_unknown_arm_raises(masks):
    with pytest.raises(ValueError):
        motif.arm_residues("EVERYTHING", masks)


def test_arm_residues_are_all_modeled(masks):
    modeled = set(masks["MODELED"])
    for arm in constants.ARMS:
        assert set(motif.arm_residues(arm, masks)) <= modeled
