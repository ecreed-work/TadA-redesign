"""The FASTA's FIRST record is the input sequence, not a design. Counting it
would inflate the design set by one per backbone and put a non-design into
scoring."""
import pytest

from tada_redesign import collect_designs as cd

FASTA = (
    ">bb1, T=0.15, seed=111, num_res=156, num_ligand_res=8, "
    "use_ligand_context=True, ligand_cutoff_distance=8.0, batch_size=1, "
    "number_of_batches=2, model_path=/x/ckpt.pt\n"
    "MSEVEFSHEYWMRHAL\n"
    ">bb1, id=1, T=0.15, seed=111, overall_confidence=0.4123, "
    "ligand_confidence=0.5231, seq_rec=0.7812\n"
    "MSEVEFSHEYWMRHAA\n"
    ">bb1, id=2, T=0.15, seed=111, overall_confidence=0.3011, "
    "ligand_confidence=0.4410, seq_rec=0.7011\n"
    "MSEVEFSHEYWMRHAG\n"
)


def test_parse_fasta_skips_the_input_record(tmp_path):
    path = tmp_path / "bb1.fa"
    path.write_text(FASTA)
    records = cd.parse_fasta(str(path))
    assert len(records) == 2
    assert [r["id"] for r in records] == ["1", "2"]
    assert records[0]["sequence"] == "MSEVEFSHEYWMRHAA"


def test_parse_fasta_extracts_every_confidence_field(tmp_path):
    path = tmp_path / "bb1.fa"
    path.write_text(FASTA)
    r = cd.parse_fasta(str(path))[0]
    assert r["temperature"] == "0.15"
    assert r["seed"] == "111"
    assert r["overall_confidence"] == "0.4123"
    assert r["ligand_confidence"] == "0.5231"
    assert r["seq_rec"] == "0.7812"


def test_parse_fasta_returns_empty_for_an_input_only_fasta(tmp_path):
    path = tmp_path / "empty.fa"
    path.write_text(FASTA.split(">bb1, id=1")[0])
    assert cd.parse_fasta(str(path)) == []


def test_parse_fasta_raises_on_a_multi_chain_sequence(tmp_path):
    """LigandMPNN joins chains with ':'. This campaign designs ONE protein
    chain, so a ':' means the DNA context was parsed as a designed chain --
    which would silently corrupt every downstream length and RMSD measurement."""
    path = tmp_path / "multi.fa"
    path.write_text(FASTA.replace("MSEVEFSHEYWMRHAA", "MSEVEF:ACGT"))
    with pytest.raises(ValueError):
        cd.parse_fasta(str(path))


def test_design_id_is_stable_and_encodes_its_provenance():
    assert cd.design_id("TadA8e_FULL_pt1.0_0", "T0.15", "3") == \
        "TadA8e_FULL_pt1.0_0__T0.15__3"


def test_columns_carry_the_full_cell_coordinates():
    for col in ("design_id", "backbone", "cell", "parent", "arm", "partial_t",
                "temperature", "bias", "sequence", "seq_len",
                "overall_confidence", "ligand_confidence", "seq_rec"):
        assert col in cd.COLUMNS
