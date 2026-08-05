"""Preflight must fail loudly and specifically. A check that reports success
when its subject is absent is worse than no check.

Every test here passes `with_env_probes=False`. The env probes shell out to
`conda run` twice, which takes tens of seconds each; four tests calling
run_checks() with them enabled would turn a unit suite into a multi-minute one.
The probes are exercised directly in test_conda_env_check_* against a
monkeypatched subprocess, and for real by the Step 5 live run.
"""
import subprocess

from tada_redesign import preflight


def test_run_checks_returns_named_checks():
    checks = preflight.run_checks(with_env_probes=False)
    assert checks
    for c in checks:
        assert isinstance(c.name, str) and c.name
        assert isinstance(c.ok, bool)
        assert isinstance(c.detail, str)


def test_env_probes_are_appended_only_when_requested():
    without = {c.name for c in preflight.run_checks(with_env_probes=False)}
    assert not any(n.startswith("conda env") for n in without)


def test_conda_env_check_passes_on_returncode_zero(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    c = preflight._conda_env_check("someenv", "numpy")
    assert c.ok is True
    assert c.name == "conda env 'someenv' has numpy"


def test_conda_env_check_fails_on_nonzero_returncode(monkeypatch):
    """The predecessor's cartesian_ddg check never inspected returncode and so
    reported success while singularity was never running the binary."""
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "ModuleNotFoundError"))
    c = preflight._conda_env_check("someenv", "nope")
    assert c.ok is False
    assert "ModuleNotFoundError" in c.detail


def test_every_expected_check_is_present():
    names = {c.name for c in preflight.run_checks(with_env_probes=False)}
    for expected in ("TADA_MONOREPO", "masks.json", "reference parents",
                     "Zn coordination geometry", "RFD3 checkpoint",
                     "LigandMPNN checkpoint", "AF3 SIF", "AF3 weights",
                     "ESMFold2 HF cache", "ESMFold2 ligand support",
                     "known-bad PDB not referenced"):
        assert expected in names, f"missing check: {expected}"


def test_missing_path_check_reports_not_ok(tmp_path):
    c = preflight._path_check("nonexistent thing", str(tmp_path / "nope"))
    assert c.ok is False
    assert "nope" in c.detail


def test_present_path_check_reports_ok(tmp_path):
    f = tmp_path / "here"
    f.write_text("x")
    assert preflight._path_check("present thing", str(f)).ok is True


def test_known_bad_pdb_is_not_referenced_in_this_package():
    checks = {c.name: c for c in preflight.run_checks(with_env_probes=False)}
    assert checks["known-bad PDB not referenced"].ok is True


def test_main_returns_nonzero_when_a_check_fails(monkeypatch):
    monkeypatch.setattr(
        preflight, "run_checks",
        lambda: [preflight.Check("fake", False, "deliberately failing")])
    assert preflight.main() == 1


def test_main_returns_zero_when_all_checks_pass(monkeypatch):
    monkeypatch.setattr(
        preflight, "run_checks",
        lambda: [preflight.Check("fake", True, "fine")])
    assert preflight.main() == 0
