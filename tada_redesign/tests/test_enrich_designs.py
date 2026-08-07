"""identity_to_parent and mutation_count are spec-required and were missing from
Part 2's designs.tsv, so Part 3 would have had to recompute them ad hoc."""
import pytest

from tada_redesign import constants, enrich_designs as en, io as tio


def test_mutation_count_counts_substitutions():
    assert en.mutation_count("AAAA", "AAAA") == 0
    assert en.mutation_count("AAAA", "AAAC") == 1
    assert en.mutation_count("ACDE", "WWWW") == 4


def test_identity_is_the_complement_of_mutation_rate():
    assert en.identity_to_parent("AAAA", "AAAA") == pytest.approx(1.0)
    assert en.identity_to_parent("AAAA", "AAAC") == pytest.approx(0.75)


def test_length_mismatch_raises_rather_than_truncating():
    """zip() would silently compare only the overlap and report a falsely high
    identity for a truncated design."""
    with pytest.raises(ValueError):
        en.mutation_count("AAA", "AAAA")


def test_a_real_parent_sequence_is_self_identical():
    seq = constants.PARENT_SEQUENCE["TadA8e"]
    assert en.identity_to_parent(seq, seq) == pytest.approx(1.0)
    assert en.mutation_count(seq, seq) == 0


def test_main_appends_both_columns_and_preserves_every_row(tmp_path):
    rows = [{"design_id": "d1", "parent": "TadA8e",
             "sequence": constants.PARENT_SEQUENCE["TadA8e"]},
            {"design_id": "d2", "parent": "TadA9",
             "sequence": constants.PARENT_SEQUENCE["TadA9"]}]
    src = tmp_path / "designs.tsv"
    tio.write_tsv(str(src), rows, ("design_id", "parent", "sequence"))
    assert en.main(["--designs", str(src)]) == 0
    out = tio.read_tsv(str(src))
    assert len(out) == 2
    assert out[0]["mutation_count"] == "0"
    assert float(out[0]["identity_to_parent"]) == pytest.approx(1.0)
    assert "sequence" in out[0]          # original columns retained


def test_main_compares_each_design_against_ITS_OWN_parent(tmp_path):
    """A TadA9 design scored against TadA8e would read as 2 spurious mutations."""
    rows = [{"design_id": "d1", "parent": "TadA9",
             "sequence": constants.PARENT_SEQUENCE["TadA9"]}]
    src = tmp_path / "d.tsv"
    tio.write_tsv(str(src), rows, ("design_id", "parent", "sequence"))
    en.main(["--designs", str(src)])
    assert tio.read_tsv(str(src))[0]["mutation_count"] == "0"
