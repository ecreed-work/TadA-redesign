"""Solubility bias is an assumption, so it is applied narrowly and measured
against a zero-bias control -- never applied to a buried or frozen position."""
import json

import pytest

from tada_redesign import constants, motif, prep_mpnn_inputs as prep


@pytest.fixture
def masks():
    return motif.load_masks()


def test_bias_positions_are_exposed_designable_and_never_frozen(masks):
    for arm in constants.ARMS:
        positions = set(prep.bias_positions(arm, masks))
        frozen = set(motif.arm_residues(arm, masks))
        exposed = set(masks["EXPOSED"]) & set(masks["MODELED"])
        assert positions == exposed - frozen
        assert not (positions & frozen)


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
