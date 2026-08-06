"""Durable tabular I/O for the campaign's array stages.

Every stage writes incrementally -- header plus one flushed row at a time -- so a
killed SLURM array leaves a parsable partial TSV instead of nothing, and can be
resumed rather than restarted. Every reader skips `#` comment lines, because
several stages annotate their output with what they dropped, and a reader that
treats a comment as data silently loses the rest of the file.

Honesty ceiling: this module moves numbers around. It does not compute or
validate any biophysical quantity.
"""
import csv
import os

COMMENT = "#"
MISSING = "NA"


def read_tsv(path):
    """Rows as dicts. Missing file -> []. Skips blank and `#` lines, strips \\r."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        clean = (ln.replace("\r", "") for ln in fh
                 if ln.strip() and not ln.lstrip().startswith(COMMENT))
        return list(csv.DictReader(clean, delimiter="\t"))


def _validate(row, columns):
    unknown = set(row) - set(columns)
    if unknown:
        raise ValueError(f"unknown column(s) {sorted(unknown)}; expected {list(columns)}")
    return {c: row.get(c, MISSING) for c in columns}


def append_row(path, row, columns):
    """Append one row, writing the header first if the file is absent/empty.

    Flushed and fsync'd per row: an array task that dies mid-stage must leave
    every row it already computed on disk.
    """
    values = _validate(row, columns)
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                                lineterminator="\n")
        if need_header:
            writer.writeheader()
        writer.writerow(values)
        fh.flush()
        os.fsync(fh.fileno())


def write_tsv(path, rows, columns, header_comment=None):
    """Whole-file write. `header_comment` is emitted as a leading `#` line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        if header_comment:
            fh.write(f"{COMMENT} {header_comment}\n")
        writer = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(_validate(row, columns))


def count_rows(path):
    return len(read_tsv(path))
