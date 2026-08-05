"""Constants are load-bearing: several encode a measured cluster or tool fact
that a plausible-looking edit would silently break."""
import importlib
import os

from tada_redesign import constants


def test_monorepo_defaults_to_the_known_root():
    assert constants.MONOREPO.endswith("claude-proteindesign")


def test_monorepo_honours_the_env_override(monkeypatch):
    monkeypatch.setenv("TADA_MONOREPO", "/somewhere/else")
    reloaded = importlib.reload(constants)
    try:
        assert reloaded.MONOREPO == "/somewhere/else"
        assert reloaded.MASKS_JSON.startswith("/somewhere/else")
    finally:
        monkeypatch.delenv("TADA_MONOREPO")
        importlib.reload(constants)


def test_both_parents_have_a_reference_pdb():
    assert set(constants.PARENT_PDB) == set(constants.PARENTS)
    for path in constants.PARENT_PDB.values():
        assert path.endswith(".pdb")


def test_partial_t_ladder_stays_below_the_measured_cliff():
    # denovo_tada's 2026-08-04 debug gate measured partial_t=8 A degrading the
    # TadA active site to 2.5-6.75 A RMSD (from ~1.1 A seeds), failing a 1.5 A
    # gate on every design. Nothing in this ladder may reach 8.
    assert max(constants.PARTIAL_T) < 8.0
    assert min(constants.PARTIAL_T) > 0.0


def test_design_counts_match_the_spec():
    assert constants.n_designs() == 10240
    assert constants.n_control_designs() == 512
    assert constants.n_designs() + constants.n_control_designs() == 10752


def test_rfd_batch_product_equals_backbones_per_cell():
    assert constants.RFD_N_BATCHES * constants.RFD_BATCH_SIZE == \
        constants.BACKBONES_PER_CELL


def test_screen_rmsd_gate_is_looser_than_the_final_gate():
    # Reduced-sampling folds are noisier; a screen gate tighter than the final
    # one would discard designs the full-sampling fold would have kept.
    assert constants.SCREEN_MOTIF_RMSD_MAX > constants.FINAL_MOTIF_RMSD_MAX


def test_esmfold_screen_is_cheaper_than_full():
    assert constants.ESMFOLD_SCREEN["num_loops"] < constants.ESMFOLD_FULL["num_loops"]
    assert constants.ESMFOLD_SCREEN["num_sampling_steps"] < \
        constants.ESMFOLD_FULL["num_sampling_steps"]


def test_known_bad_pdb_is_recorded_so_preflight_can_forbid_it():
    assert "6vpc_dCas9_TadA8e.pdb" in os.path.basename(constants.KNOWN_BAD_PDB)
