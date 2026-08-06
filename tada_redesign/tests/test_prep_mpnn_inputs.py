"""Solubility bias is an assumption, so it is applied narrowly and measured
against a zero-bias control -- never applied to a buried or frozen position."""
import json

import pytest

from tada_redesign import constants, motif, prep_mpnn_inputs as prep

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
