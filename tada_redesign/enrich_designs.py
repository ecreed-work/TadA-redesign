"""Add the two spec-required per-design columns Part 2 shipped without.

`identity_to_parent` and `mutation_count` are named in the design spec's Stage 2
but were never emitted, so every later stage would have recomputed them ad hoc --
and `constants` carried no parent sequence to compute them from.

Each design is compared against ITS OWN parent. TadA-9 differs from TadA-8e at
two positions (N108Q, L145T), so comparing a TadA9 design against TadA8e would
report two mutations that are not the designer's doing.

Honesty ceiling: sequence identity is a similarity measure. It says nothing about
whether a design folds, is stable, or is active.
"""
import argparse
import os

from . import constants, io, provenance


def mutation_count(seq, parent_seq):
    """Substitutions between equal-length sequences.

    Raises on a length mismatch: `zip` would silently compare only the overlap
    and report a falsely high identity for a truncated design.
    """
    if len(seq) != len(parent_seq):
        raise ValueError(
            f"length mismatch: design {len(seq)} vs parent {len(parent_seq)}")
    return sum(1 for a, b in zip(seq, parent_seq) if a != b)


def identity_to_parent(seq, parent_seq):
    return 1.0 - mutation_count(seq, parent_seq) / float(len(parent_seq))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--designs", default=os.path.join(
        sub, "outputs", constants.RUN_DIR_NAME, "designs.tsv"))
    args = ap.parse_args(argv)

    rows = io.read_tsv(args.designs)
    if not rows:
        raise SystemExit(f"[enrich_designs] no rows in {args.designs}")

    failed = []
    for row in rows:
        parent_seq = constants.PARENT_SEQUENCE[row["parent"]]
        try:
            row["mutation_count"] = str(mutation_count(row["sequence"], parent_seq))
            row["identity_to_parent"] = f"{identity_to_parent(row['sequence'], parent_seq):.4f}"
        except (ValueError, KeyError) as exc:
            row["mutation_count"] = io.MISSING
            row["identity_to_parent"] = io.MISSING
            failed.append((row["design_id"], str(exc)))

    columns = tuple(rows[0].keys())
    io.write_tsv(args.designs, rows, columns)
    for design_id, why in failed[:10]:
        print(f"[enrich_designs] WARNING {design_id}: {why}")
    print(f"[enrich_designs] enriched {len(rows) - len(failed)}/{len(rows)} rows "
          f"-> {args.designs}")
    provenance.write(os.path.dirname(args.designs), "enrich_designs",
                     len(rows), len(rows) - len(failed),
                     extra={"failed": failed[:50]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
