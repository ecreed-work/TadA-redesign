"""Single source of truth for the frozen active-site motif.

Emits the SAME residue set in the three formats its consumers need: RFD3
`select_fixed_atoms`, LigandMPNN `--fixed_residues`, and the structural
scorer's measured-residue list. The predecessor campaign's worst defects came
from a generator and a gate disagreeing about one set (see
docs/logs/20260728_tada_stabilization.md, fix rounds 1-2), so there is exactly
one definition here and three renderers over it.

TRAP -- masks.json's own "FROZEN" key is NOT this campaign's motif. There it
means CATALYTIC | POCKET | DNA_FACE | ZN_PROXIMITY (36 residues) and exists to
keep `cartesian_ddg` away from the metal site. Diffusion + LigandMPNN do not
exhibit that failure, so ZN_PROXIMITY is deliberately designable here and the
FULL arm is CATALYTIC | POCKET | DNA_FACE intersected with MODELED (24).

Honesty ceiling: freezing a residue set constrains geometry and identity. It
does not make the resulting protein catalytically active, and nothing in this
module measures function.

NOTE: ARM_FULL and ARM_MIN describe what gets FROZEN during design; CORE_MOTIF
describes what gets MEASURED when scoring a predicted structure. These are
different jobs and must not be conflated.
"""
import json

from . import constants

ARM_FULL = "FULL"
ARM_MIN = "MIN"

# The MEASURED subset, used for scoring rather than for freezing. The FULL arm's
# DNA-face residues sit near the chain terminus, where a predicted structure and
# a relaxed crystal-derived one diverge freely: residue 156's ring atoms alone
# deviated 19-23 A, dominating a 202-atom average and pushing the UNMODIFIED
# parent to 7.673 A against a 1.5 A gate. Restricted to the catalytic machinery
# and the substrate pocket, the same parent measures 1.414 A. Measured
# 2026-08-06; see docs/plans/2026-08-06-tada-redesign-part3a-gatefix.md.
CORE_MOTIF = "CORE"

# Every motif residue is fixed with the ALL keyword (backbone + sidechain),
# never TIP. Freeing the Zn donors' backbone is MEASURED to let the metal
# migrate: tada-stability commit 132509f ("freeze donor backbone too -- both
# parents pass check_zn_geometry") only went green once the donor backbone was
# frozen as well.
FIXED_ATOM_KEYWORD = "ALL"

_ARM_MASKS = {
    ARM_FULL: ("CATALYTIC", "POCKET", "DNA_FACE"),
    ARM_MIN: ("CATALYTIC",),
    CORE_MOTIF: ("CATALYTIC", "POCKET"),
}


def load_masks(path=None):
    """The `masks` block of masks.json: {mask_name: [resnum, ...]}."""
    with open(path or constants.MASKS_JSON) as fh:
        return json.load(fh)["masks"]


def arm_residues(arm, masks):
    """Sorted residue numbers frozen by `arm`, intersected with MODELED.

    The intersection matters: EVOLVED positions 166/167 and the disordered
    termini are outside chain F's modelled 5-160 span and can never be frozen
    or designed.
    """
    try:
        names = _ARM_MASKS[arm]
    except KeyError:
        raise ValueError(
            f"unknown arm {arm!r}; expected one of {tuple(_ARM_MASKS)}") from None
    selected = set()
    for name in names:
        selected |= set(masks[name])
    return tuple(sorted(selected & set(masks["MODELED"])))


def rfd_select_fixed_atoms(arm, masks, chain=None, ligand=None):
    """RFD3 `select_fixed_atoms` mapping: {"F57": "ALL", ..., "ZN": "ALL"}.

    The ion is keyed by CCD NAME, not chain+resid. RFD3's AtomWorks loader
    renames a hetero atom's chain when it shares a chain letter with protein
    residues (our Zn is `HETATM ZN F 201`), so a "F201" key raises
    `ValidationError: [component=F201] Residue F201 not found in atom array` --
    confirmed by a real RFD3 run. `fetch_mask_from_name` matches res_name
    chain-independently and is tried first, so a CCD key resolves regardless.
    """
    chain = chain or constants.SCAFFOLD_CHAIN
    ligand = ligand or constants.ZN_RESNAME
    fixed = {f"{chain}{resnum}": FIXED_ATOM_KEYWORD
             for resnum in arm_residues(arm, masks)}
    fixed[ligand] = FIXED_ATOM_KEYWORD
    return fixed


def mpnn_fixed_residues(arm, masks, chain=None):
    """LigandMPNN `--fixed_residues` string: space-separated "F57 F59 ..."."""
    chain = chain or constants.SCAFFOLD_CHAIN
    return " ".join(f"{chain}{resnum}"
                    for resnum in arm_residues(arm, masks))


def measured_residues(arm, masks):
    """Residues whose heavy atoms the structural scorer measures.

    Identical to the frozen set by construction: the gate must measure exactly
    what the generators were told to hold, or a divergence between the two
    becomes invisible.
    """
    return arm_residues(arm, masks)
