"""Sharding must be deterministic and lossless: every design folded exactly once
across the array, or the screen silently drops designs."""
import pytest

from tada_redesign import fold_screen as fs, io as tio


def _rows(n):
    return [{"design_id": f"d{i}", "sequence": "MKV", "parent": "TadA8e"} for i in range(n)]


def test_shards_partition_every_design_exactly_once():
    rows = _rows(1000)
    seen = []
    for s in range(1, 8):
        seen += [r["design_id"] for r in fs.shard_of(rows, s, 7)]
    assert sorted(seen) == sorted(r["design_id"] for r in rows)
    assert len(seen) == len(set(seen))       # no design folded twice


def test_shards_are_balanced_within_one():
    rows = _rows(1000)
    sizes = [len(fs.shard_of(rows, s, 7)) for s in range(1, 8)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_is_deterministic():
    rows = _rows(100)
    assert fs.shard_of(rows, 3, 7) == fs.shard_of(rows, 3, 7)


def test_shard_index_is_one_based_and_validated():
    rows = _rows(10)
    with pytest.raises(ValueError):
        fs.shard_of(rows, 0, 4)
    with pytest.raises(ValueError):
        fs.shard_of(rows, 5, 4)


def test_write_shard_jobs_emits_only_the_two_needed_columns(tmp_path):
    path = fs.write_shard_jobs(_rows(3), str(tmp_path / "jobs.tsv"))
    rows = tio.read_tsv(path)
    assert set(rows[0]) == {"design_id", "sequence"}
    assert len(rows) == 3
