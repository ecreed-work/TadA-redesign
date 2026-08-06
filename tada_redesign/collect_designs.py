"""LigandMPNN FASTA output -> designs.tsv, one row per design.

Two format facts, confirmed by reading `design/LigandMPNN/run.py`:

  - The FIRST record in each `.fa` is the INPUT sequence and has no `id=` field.
    It is not a design. Counting it would add one phantom design per backbone
    and push a non-design into the folding stages.
  - Design records are
    `>{name}, id={i}, T={t}, seed={s}, overall_confidence={c},
      ligand_confidence={lc}, seq_rec={r}`

A sequence containing `:` means LigandMPNN emitted more than one designed chain
-- i.e. the DNA context was parsed as protein rather than as ligand context.
That would corrupt every downstream length and RMSD measurement, so it raises
rather than being silently accepted.

Honesty ceiling: a confidence value here is LigandMPNN's own sequence score. It
is not a stability measurement and not evidence of function.
"""
import argparse
import glob
import os

from . import constants, io, provenance

COLUMNS = ("design_id", "backbone", "cell", "parent", "arm", "partial_t",
           "temperature", "bias", "seed", "mpnn_id", "sequence", "seq_len",
           "overall_confidence", "ligand_confidence", "seq_rec")

_FIELDS = ("id", "T", "seed", "overall_confidence", "ligand_confidence", "seq_rec")


def _parse_header(header):
    out = {}
    for chunk in header.split(","):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        out[key.strip()] = value.strip()
    return out


def parse_fasta(path):
    """Design records only; the leading input record is skipped."""
    records, header, seq = [], None, []

    def flush():
        if header is None:
            return
        fields = _parse_header(header)
        if "id" not in fields:            # the input record, not a design
            return
        sequence = "".join(seq)
        if ":" in sequence:
            raise ValueError(
                f"{path}: design sequence has multiple chains ({sequence[:40]}...); "
                "the DNA context was parsed as a designed chain, not ligand context")
        records.append({
            "id": fields["id"],
            "temperature": fields.get("T", "NA"),
            "seed": fields.get("seed", "NA"),
            "overall_confidence": fields.get("overall_confidence", "NA"),
            "ligand_confidence": fields.get("ligand_confidence", "NA"),
            "seq_rec": fields.get("seq_rec", "NA"),
            "sequence": sequence,
        })

    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header, seq = line[1:], []
        else:
            seq.append(line)
    flush()
    return records


def design_id(backbone, tag, mpnn_id):
    return f"{backbone}__{tag}__{mpnn_id}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(
        sub_dir, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    manifest = {r["backbone"]: r for r in io.read_tsv(
        os.path.join(args.run_dir, "mpnn_in", "mpnn_manifest.tsv"))}
    out_path = os.path.join(args.run_dir, "designs.tsv")
    if os.path.exists(out_path):
        os.unlink(out_path)

    fastas = sorted(glob.glob(os.path.join(
        args.run_dir, "lmpnn", "*", "*", "seqs", "*.fa")))
    n_rows, skipped = 0, []
    for fasta in fastas:
        tag = os.path.basename(os.path.dirname(os.path.dirname(fasta)))
        backbone = os.path.basename(fasta)[:-3]
        meta = manifest.get(backbone)
        if meta is None:
            skipped.append(fasta)
            continue
        for rec in parse_fasta(fasta):
            io.append_row(out_path, {
                "design_id": design_id(backbone, tag, rec["id"]),
                "backbone": backbone,
                "cell": meta["cell"],
                "parent": meta["parent"],
                "arm": meta["arm"],
                "partial_t": meta["cell"].split("_")[2][2:],
                "temperature": rec["temperature"],
                "bias": "none" if tag == "control" else "solubility",
                "seed": rec["seed"],
                "mpnn_id": rec["id"],
                "sequence": rec["sequence"],
                "seq_len": len(rec["sequence"]),
                "overall_confidence": rec["overall_confidence"],
                "ligand_confidence": rec["ligand_confidence"],
                "seq_rec": rec["seq_rec"],
            }, COLUMNS)
            n_rows += 1

    for fasta in skipped:
        print(f"[collect_designs] WARNING no manifest row for {fasta}; skipped")
    print(f"[collect_designs] {n_rows} designs from {len(fastas)} fastas -> {out_path}")
    provenance.write(args.run_dir, "collect_designs", len(fastas), n_rows,
                     extra={"skipped_fastas": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
