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


def test_known_bad_pdb_is_recorded_so_preflight_can_forbid_it():
    assert "6vpc_dCas9_TadA8e.pdb" in os.path.basename(constants.KNOWN_BAD_PDB)


def test_rmsd_reference_is_the_relaxed_parents_not_the_crystal():
    """Partial diffusion starts from the relaxed parent, so motif drift must be
    measured against that same coordinate set. Measured divergence between the
    two candidates is 2.166 A over the FULL arm -- 2x BACKBONE_MOTIF_RMSD_MAX --
    and the crystal is missing nine FULL-arm sidechain atoms."""
    assert constants.RMSD_REFERENCE == constants.PARENT_PDB
    assert constants.CHAINF_RAW not in constants.RMSD_REFERENCE.values()


def test_plddt_scales_are_recorded_for_both_folding_models():
    """ESMFold2 reports 0-1, AF3 reports 0-100. Reusing one margin across both
    would be a 20x error."""
    assert constants.ESMFOLD_PLDDT_SCALE == 1.0
    assert constants.AF3_PLDDT_SCALE == 100.0
    assert constants.PLDDT_MARGIN < constants.ESMFOLD_PLDDT_SCALE


def test_env_modules_covers_every_env_the_campaign_uses():
    envs = {e for e, _ in constants.ENV_MODULES}
    assert envs == {constants.ENV_TEST, constants.ENV_ROSETTA,
                    constants.ENV_RFD3, constants.ENV_MPNN, constants.ENV_ESM}


def test_run_dir_points_at_the_production_run():
    """Part 2's production generation wrote 20260806_tada_redesign_gen1; the old
    default was the debug dir and would silently score the wrong run."""
    assert constants.RUN_DIR_NAME == "20260806_tada_redesign_gen1"


def test_parent_sequences_are_present_and_the_right_length():
    assert set(constants.PARENT_SEQUENCE) == set(constants.PARENTS)
    for parent, seq in constants.PARENT_SEQUENCE.items():
        assert len(seq) == 156, (parent, len(seq))
        assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_tada9_differs_from_tada8e_at_exactly_its_two_defining_positions():
    """TadA-9 = TadA-8e + N108Q + L145T. Chain F starts at residue 5, so
    sequence index = resnum - 5."""
    a, b = constants.PARENT_SEQUENCE["TadA8e"], constants.PARENT_SEQUENCE["TadA9"]
    diffs = {i + 5 for i, (x, y) in enumerate(zip(a, b)) if x != y}
    assert diffs == {108, 145}, diffs
    assert (a[108 - 5], b[108 - 5]) == ("N", "Q")
    assert (a[145 - 5], b[145 - 5]) == ("L", "T")


def test_parent_sequences_come_from_contiguous_residues_5_to_160():
    """len == 156 is not enough: a gapped PDB extending past 160 would give the
    same length with shifted indexing, silently corrupting every mutation count
    and the N108Q/L145T positions that depend on resnum - 5."""
    import sys
    sys.path.insert(0, constants.TADA_STABILITY)
    from tada_stability import prep_scaffolds
    for parent, pdb in constants.PARENT_PDB.items():
        by_resnum = prep_scaffolds.parent_sequence(pdb)
        assert sorted(by_resnum) == list(range(5, 161)), parent
        joined = "".join(by_resnum[r] for r in sorted(by_resnum))
        assert joined == constants.PARENT_SEQUENCE[parent], parent


def test_single_sampling_setting_at_full_depth():
    """The two-tier screen is retired. Measured 2026-08-06: reduced sampling has
    a 2.020 A parent-vs-parent noise floor at the core motif, vs 0.563 A at full --
    3.6x worse, and above any usable threshold. pLDDT was flat between the modes
    (0.899-0.908 vs 0.835-0.902), so confidence alone could not reveal this."""
    assert constants.ESMFOLD_SETTINGS == {"num_loops": 20, "num_sampling_steps": 100}
    assert not hasattr(constants, "ESMFOLD_SCREEN")
    assert not hasattr(constants, "ESMFOLD_FULL")


def test_shard_count_reflects_load_dominated_cost():
    """Each shard pays the model load once (22.4 s quiet, ~2400 s contended), so
    shard COUNT multiplies that overhead while fold cost stays fixed."""
    assert constants.FOLD_BATCH_SIZE == 1000
    assert constants.FOLD_SHARDS == 11
    assert constants.FOLD_SHARDS * constants.FOLD_BATCH_SIZE >= 10542


def test_motif_threshold_exceeds_the_parents_own_offset_and_jitter():
    """SUPERSEDED 2026-08-08 (Task 3b): the 1.468 A figure this test used to cite
    came from a side-script that folded with a non-default `--seed`, not from
    `reference_baseline.py`'s production path -- it was never reproducible
    through the code the campaign actually runs. Re-measured through the
    fixed, iteratively-refined anchor (`constants.ANCHOR_OUTLIER_CUTOFF`) via
    the actual production path (baseline job 238437): parent vs crystal, CORE
    = TadA8e 1.354 A, TadA9 1.357 A. Floor = max(1.354, 1.357) + 0.563 (given
    fold-to-fold jitter, not re-derived here) = 1.920 A. A threshold below that
    floor rejects the unmodified parent and is meaningless.
    MOTIF_RMSD_MAX = 2.0 sits one tick above the floor -- per the repo owner's
    2026-08-08 ruling this gate is now a gross-failure catch, not a ranking
    metric: the entire 21-probe distribution (min 1.296, max 1.713) sits
    inside the parent's own fold-to-fold jitter band, so no cutoff above the
    floor could discriminate among them."""
    floor = max(1.354, 1.357) + 0.563
    assert constants.MOTIF_RMSD_MAX > floor
    # Floor-bound alone lets MOTIF_RMSD_MAX drift arbitrarily high (e.g. 50.0)
    # while still passing -- and the unbounded direction is exactly the
    # flattering one for a gate whose job is to REJECT gross failures. Pin the
    # measured, derived value exactly; if it ever needs to change, this test
    # must be edited deliberately, not silently satisfied by a loose bound.
    assert constants.MOTIF_RMSD_MAX == 2.0


def test_anchor_constants_are_pinned_to_their_measured_values():
    """Floor/relationship tests alone do not stop these three from drifting to
    a value that reproduces the original 3.5 A anchor defect (constants.py
    itself records ANCHOR_OUTLIER_CUTOFF=50.0 -- i.e. no filtering at all --
    as reproducing exactly that) while the rest of the suite stays green. Pin
    all three exactly, mirroring the MOTIF_RMSD_MAX pin above."""
    assert constants.ANCHOR_OUTLIER_CUTOFF == 5.0
    assert constants.ANCHOR_MAX_ITER == 10
    assert constants.ANCHOR_MIN_RETAINED_FRAC == 0.60
