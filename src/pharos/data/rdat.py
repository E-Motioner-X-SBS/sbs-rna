#!/usr/bin/env python3
"""RDAT parsing — the ionic-response channel (§6.4, head 7).

An RDAT holds one construct, its sequence and reference secondary structure, a
set of experimental conditions common to the whole file, and then one
`REACTIVITY` row per data condition with its own `ANNOTATION_DATA`. A titration
is exactly that: the same construct and probe, read out at many magnesium
concentrations.

This is the supervision the PDB structurally cannot provide. Deposited Mg²⁺
spans 5–15 mM because unfolded RNA is not crystallised, so the *response* to
changing ionic strength has no signal there at any corpus size. The 24 ladders
acquired here all cross the sub-millimolar regime, which is where the folding
transition happens.

What this module deliberately does not do
-----------------------------------------
It does not average a titration into one profile. The quantity of interest is
the **derivative** — how reactivity at each position changes with [Mg²⁺] — and
averaging destroys precisely that. Each condition is kept as its own example,
tagged with its concentration, so a model conditioned on ionic strength can be
asked to reproduce the ladder rather than its mean.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_CONC = re.compile(r"chemical:([A-Za-z0-9_+\-]+):\s*([0-9.]+)\s*(mM|uM|nM|M)\b", re.I)
_SCALE = {"m": 1.0, "u": 1e-3, "n": 1e-6}
_PROBES = ("DMS", "SHAPE", "1M7", "CMC", "NMIA", "2A3", "BzCN", "MCA")


def to_mM(v: float, unit: str) -> float:
    u = unit.lower()
    return v * 1000.0 if u == "m" else v * _SCALE.get(u[0], 1.0)


@dataclass
class RdatFile:
    """One RDAT: a construct plus its conditions and their reactivity rows."""
    path: Path
    name: str = ""
    sequence: str = ""
    structure: str = ""
    offset: int = 0
    seqpos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    common: Dict[str, float] = field(default_factory=dict)
    common_raw: List[str] = field(default_factory=list)
    conditions: List[Dict] = field(default_factory=list)   # per row
    reactivity: List[np.ndarray] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.reactivity)

    def probe(self) -> str:
        for tok in self.common_raw:
            if tok.lower().startswith("modifier:"):
                return tok.split(":", 1)[1]
        stem = self.path.stem.upper()
        for p in _PROBES:
            if p in stem:
                return p
        return "?"

    def ion_series(self, ion: str = "MgCl2") -> Optional[np.ndarray]:
        """Concentration of `ion` for each row, or None if it never varies."""
        vals = [c.get(ion) for c in self.conditions]
        if any(v is None for v in vals):
            return None
        arr = np.asarray(vals, dtype=np.float64)
        return arr if len(set(np.round(arr, 6))) > 1 else None


def parse_rdat(path: Path) -> Optional[RdatFile]:
    """Parse an RDAT. Returns None if it carries no reactivity rows."""
    r = RdatFile(path=Path(path))
    rows: Dict[int, np.ndarray] = {}
    ann: Dict[int, List[str]] = {}
    try:
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        if not line.strip():
            continue
        # RDAT 0.34 separates fields with tabs; 0.24 uses spaces, and splitting
        # only on tab makes every line of a 0.24 file one unrecognised key, so
        # the file parses to nothing at all rather than to an error.
        # SRPDIV_DMS_0001 is one: 16 Mg levels that silently vanished.
        # Splitting the KEY on any whitespace handles both, and concentrations
        # are recovered by scanning the remainder rather than tokenising it --
        # annotation values contain spaces ("MgCl2:0.04 mM") and tokenising
        # would split them apart.
        parts = line.split(None, 1)
        head = parts[0].strip()
        rest = parts[1] if len(parts) > 1 else ""
        if head == "NAME":
            r.name = rest.strip()
        elif head == "SEQUENCE":
            r.sequence = rest.strip().replace(" ", "")
        elif head == "STRUCTURE":
            r.structure = rest.strip().replace(" ", "")
        elif head == "OFFSET":
            try:
                r.offset = int(float(rest.strip() or 0))
            except ValueError:
                pass
        elif head == "SEQPOS":
            toks = rest.split()
            nums = []
            for t in toks:
                m = re.search(r"(-?\d+)$", t)
                if m:
                    nums.append(int(m.group(1)))
            r.seqpos = np.asarray(nums, dtype=np.int32)
        elif head == "ANNOTATION":
            r.common_raw = [t.strip() for t in re.split(r"\t+", rest) if t.strip()]
            for m in _CONC.finditer(rest):
                r.common[m.group(1)] = to_mM(float(m.group(2)), m.group(3))
        elif head.startswith("ANNOTATION_DATA:"):
            try:
                i = int(head.split(":", 1)[1])
            except ValueError:
                continue
            ann[i] = [rest]          # scanned by regex, not tokenised
        elif head.startswith("REACTIVITY:"):
            try:
                i = int(head.split(":", 1)[1])
            except ValueError:
                continue
            vals = []
            for t in rest.split():
                try:
                    vals.append(float(t))
                except ValueError:
                    vals.append(np.nan)
            rows[i] = np.asarray(vals, dtype=np.float32)

    if not rows:
        return None
    for i in sorted(rows):
        cond = dict(r.common)
        for tok in ann.get(i, []):
            for m in _CONC.finditer(tok):
                cond[m.group(1)] = to_mM(float(m.group(2)), m.group(3))
        r.conditions.append(cond)
        r.reactivity.append(rows[i])
    return r


def titration_examples(path: Path, ion: str = "MgCl2",
                       min_levels: int = 3) -> List[Dict]:
    """One example per (construct, concentration), or `[]` if not a titration.

    Reactivity is aligned to the construct's own sequence through `SEQPOS`,
    which is not always 1..L: probing reads out an interior window, and the
    positions it covers are named explicitly. Ignoring it and assuming a
    contiguous run silently shifts every label by the offset.
    """
    r = parse_rdat(path)
    if r is None or not r.sequence:
        return []
    series = r.ion_series(ion)
    if series is None or len(set(np.round(series, 6))) < min_levels:
        return []

    L = len(r.sequence)
    out: List[Dict] = []
    for row, conc in zip(r.reactivity, series):
        prof = np.full(L, np.nan, dtype=np.float32)
        mask = np.zeros(L, dtype=bool)
        if r.seqpos.size and r.seqpos.size == row.size:
            idx = r.seqpos - r.offset - 1
            ok = (idx >= 0) & (idx < L)
            prof[idx[ok]] = row[ok]
            mask[idx[ok]] = True
        else:
            n = min(L, row.size)
            prof[:n] = row[:n]
            mask[:n] = True
        mask &= np.isfinite(prof)
        out.append({
            "rmdb_id": r.path.stem, "name": r.name, "probe": r.probe(),
            "sequence": r.sequence, "structure": r.structure,
            "ion": ion, "conc_mM": float(conc),
            "reactivity": prof, "valid": mask,
            "conditions": {k: float(v) for k, v in r.conditions[0].items()},
        })
    return out
