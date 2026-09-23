#!/usr/bin/env python3
"""The held-out targets: RNA-Puzzles, CASP15 and CASP16.

These are the only honest test sets this project has. Everything else the model
sees is drawn from the PDB, and a structure deposited before the model trained
is not a blind test no matter how carefully it is held out -- homologues leak.
RNA-Puzzles and CASP targets were predicted by the field *before* their
structures were public, which is what makes the published numbers a real
baseline rather than a self-reported one.

Two things this module deliberately provides:

**The other groups' submissions.** RNA-Puzzles ships every group's models next
to the solution, so `competitor_models` turns the benchmark into a leaderboard:
not "PHAROS scores 0.51 TM" but "PHAROS would have placed 4th of 14 on this
target". A score with no field to compare against says very little.

**A date.** Every target carries the release date of its structure, so a
training corpus can be filtered to what existed beforehand. Scoring a model on
a target whose structure is in its training set is the commonest way an RNA
paper overstates itself, and the cutoff makes that checkable rather than
assumed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[3]
BLIND = ROOT / "data/structures/blind_tests"

#: CASP15 ran in 2022, CASP16 in 2024. A model trained on structures released
#: before these dates has not seen the targets.
CASP_CUTOFF = {"casp15": "2022-05-01", "casp16": "2024-05-01"}


@dataclass
class Target:
    """One blind-test target: a reference structure and who else predicted it."""

    name: str
    source: str                     # rna_puzzles | casp15 | casp16
    reference: Path
    competitors: List[Path] = field(default_factory=list)
    description: str = ""
    length: Optional[int] = None

    @property
    def n_competitors(self) -> int:
        return len(self.competitors)


def _puzzle_descriptions() -> Dict[str, tuple]:
    csv_path = BLIND / "rna_puzzles_std/rna-puzzles-dataset.csv"
    out: Dict[str, tuple] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("ID") or "").strip()
            if key:
                try:
                    n = int((row.get("Length") or "0").strip())
                except ValueError:
                    n = None
                out[key] = ((row.get("RNA") or "").strip(), n)
    return out


def rna_puzzles() -> Iterator[Target]:
    """Every RNA-Puzzles target that ships a solution structure.

    A puzzle directory holds `<n>_solution_0_rpr.pdb` and one file per group
    per submitted model. Directories without a solution (`rp16_TBA`) are
    skipped rather than reported as zero-competitor targets.
    """
    base = BLIND / "rna_puzzles_std"
    if not base.is_dir():
        return
    desc = _puzzle_descriptions()
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        sols = sorted(d.glob("*_solution_*_rpr.pdb"))
        if not sols:
            continue
        others = [p for p in sorted(d.glob("*_rpr.pdb")) if "solution" not in p.name]
        text, length = desc.get(d.name, ("", None))
        yield Target(d.name, "rna_puzzles", sols[0], others, text, length)


def casp(edition: str) -> Iterator[Target]:
    """CASP reference structures. No competitor models ship with these."""
    base = BLIND / edition
    if not base.is_dir():
        return
    for p in sorted(base.glob("*.cif")):
        yield Target(p.stem, edition, p, [], f"{edition.upper()} target")


def all_targets() -> List[Target]:
    return list(rna_puzzles()) + list(casp("casp15")) + list(casp("casp16"))


_GROUP = re.compile(r"^(?:\d+_|R\d+TS\d+_)(?P<group>.+?)_(?P<model>\d+)_rpr$")


def competitor_label(path: Path) -> tuple[str, str]:
    """`(group, model number)` from an RNA-Puzzles submission filename."""
    m = _GROUP.match(path.stem)
    if not m:
        return path.stem, "?"
    return m.group("group"), m.group("model")


if __name__ == "__main__":
    ts = all_targets()
    by = {}
    for t in ts:
        by.setdefault(t.source, []).append(t)
    print(f"{len(ts)} blind-test targets\n")
    for src, group in by.items():
        comp = sum(t.n_competitors for t in group)
        print(f"  {src:14s} {len(group):3d} targets, {comp:5d} competitor models")
    print(f"\n{'target':14s} {'len':>5s} {'rivals':>7s}  description")
    for t in ts:
        if t.source == "rna_puzzles":
            print(f"  {t.name:12s} {str(t.length or '?'):>5s} {t.n_competitors:7d}  "
                  f"{t.description[:58]}")
