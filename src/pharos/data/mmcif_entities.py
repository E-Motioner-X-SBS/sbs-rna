#!/usr/bin/env python3
"""Canonical RNA-chain resolution from mmCIF. ONE definition, shared.

Defect #22 (cycle 6): every analysis script rolled its own notion of "an RNA
residue" and they disagreed on **72 of 180 structures**, in both directions:

  * `analyze_ions_motifs.py`  -- a hardcoded list of 16 residue names. Misses any
    modification not on the list (7PKT: 767 residues short).
  * `audit_generalization.py` -- "any component <= 3 chars that is not water,
    group=ATOM". Over-inclusive, and drops HETATM-coded RNA (8JDJ: 156 short the
    other way).

Neither is authoritative. mmCIF *declares* polymer type in `_entity_poly.type`:
`polyribonucleotide` / `polydeoxyribonucleotide` / `polypeptide(L)`, with
`pdbx_strand_id` naming the chains. That declaration is the ground truth, so a
residue is RNA iff it sits in a chain whose entity is a polyribonucleotide --
regardless of how exotic its modification is.

Hybrid chains (`polydeoxyribonucleotide/polyribonucleotide hybrid`) are resolved
per residue by the presence of an `O2'` atom -- ribose has a 2'-hydroxyl,
deoxyribose does not. That is a structural test taken from the coordinates, so it
settles hybrids without reintroducing the curated-list problem above. Cycle 6
excluded hybrids wholesale after measuring the loss at 6 residues; cycle 7
resolves them instead, which closes the sample to **179 of 180** structures. The
one remaining (7PU7) has a single nucleic entity, declared hybrid and modelled
entirely as deoxyribonucleotide -- there is no RNA in its coordinates to count.

Defect #25 (cycle 7): mmCIF serialises a category in two forms, and only the
`loop_` form was handled. The key-value form (`_entity_poly.tag  value`) is what
the PDB writes when a category has exactly one row -- i.e. a structure with a
single polymer entity -- so all 16 such structures in the sample silently parsed
to zero RNA. They are not a random 16: a lone polymer entity means a small
isolated RNA, which is exactly the population G2 makes a claim about, and their
absence inflated "RNA residues in complexes" from 98.95% to a spurious 99.70%.
"""
from __future__ import annotations
import gzip
import re
from pathlib import Path

RNA_STD = {"A", "C", "G", "U"}


def _open(path: Path):
    return gzip.open(path, "rt", errors="ignore") if str(path).endswith(".gz") else open(path, errors="ignore")


def entity_poly_types(path: Path) -> dict[str, str]:
    """auth chain id -> entity_poly.type, from the declaration itself.

    mmCIF serialises a category in TWO forms and both occur in the PDB archive:

      loop form          `loop_` / `_entity_poly.tag` headers / one line per row
      key-value form     `_entity_poly.tag   value` -- used when the category has
                         exactly ONE row, i.e. a structure with a single polymer
                         entity

    Defect #25 (cycle 7): only the loop form was handled, so every
    single-polymer-entity structure parsed to `{}` and silently reported zero RNA.
    That hit **16 of 180** sampled structures, and they are not a random 16 --
    a lone polymer entity means a small isolated RNA, which is exactly the
    population G2 ("99.7% of RNA residues are in complex") makes a claim about.

    Both forms also allow `;`-delimited multi-line values, which the sequence
    fields routinely use, so tokens are accumulated until a row is full.
    """
    cols: list[str] = []
    rows: list[list[str]] = []
    kv: dict[str, str] = {}
    in_hdr = in_loop = False
    buf: list[str] = []
    semi = False
    semi_val: list[str] = []
    semi_tag: str | None = None

    def toks(s: str) -> list[str]:
        return re.findall(r"'[^']*'|\"[^\"]*\"|\S+", s)

    with _open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")

            # a `;`-block continues until a lone `;` -- in either form
            if semi:
                if s.startswith(";"):
                    semi = False
                    val = "".join(semi_val)
                    semi_val = []
                    if semi_tag is not None:
                        kv[semi_tag] = val
                        semi_tag = None
                    else:
                        buf.append(val)
                        while len(buf) >= len(cols) and cols:
                            rows.append(buf[:len(cols)]); buf = buf[len(cols):]
                else:
                    semi_val.append(s)
                continue

            if s.startswith("_entity_poly."):
                tag, _, rest = s.strip().partition(" ")
                tag = tag.split(".", 1)[1]
                rest = rest.strip()
                if rest:
                    # KEY-VALUE form: the value is on this line
                    kv[tag] = rest.strip("'\"")
                else:
                    # loop header, or a key whose value is the next `;` block
                    cols.append(tag)
                in_hdr = True
                continue

            if not in_hdr:
                continue

            if s.startswith(";"):
                # a value introduced by the previous bare `_entity_poly.tag`
                semi = True
                semi_val = [s[1:]]
                semi_tag = cols.pop() if (cols and not in_loop and not rows) else None
                continue

            if s.startswith("#") or s.startswith("loop_"):
                break
            if s.startswith("_"):
                break
            if not s.strip():
                continue

            # a data line: we are in the loop form
            in_loop = True
            if not cols:
                break
            buf.extend(toks(s))
            while len(buf) >= len(cols):
                rows.append(buf[:len(cols)]); buf = buf[len(cols):]

    if kv and not rows:
        # single-row key-value form
        t = kv.get("type", "").strip("'\"")
        strand = kv.get("pdbx_strand_id", "").strip("'\"")
        out: dict[str, str] = {}
        for ch in strand.split(","):
            ch = ch.strip()
            if ch and ch != "?":
                out[ch] = t
        return out

    if not cols or not rows:
        return {}
    try:
        i_type = cols.index("type")
        i_strand = cols.index("pdbx_strand_id")
    except ValueError:
        return {}
    out = {}
    for r in rows:
        t = r[i_type].strip("'\"")
        for ch in r[i_strand].strip("'\"").split(","):
            ch = ch.strip()
            if ch and ch != "?":
                out[ch] = t
    return out


def rna_residues(path: Path):
    """Return (rna_chain -> set of residue ids, composition counter, chain types).

    A residue counts as RNA iff:

      * its AUTH chain is declared `polyribonucleotide`, **or** the chain is
        declared a DNA/RNA hybrid and the residue carries an `O2'` atom; and
      * it occupies a polymer position (`label_seq_id` assigned).

    The `O2'` test is structural, not a name list -- ribose has a 2'-hydroxyl and
    deoxyribose does not -- so it resolves hybrids without reintroducing the
    curated-list problem that defect #22 was about. Verified on all three hybrid
    entries in the sample: it splits 7S3B chain B into 6 ribo (U,C,G) and 2 deoxy
    (DU, BRU -- 5-bromo-deoxyuridine, correctly classed as DNA), and finds 7PU7
    and 8DFA's hybrid chains to be entirely deoxy in the modelled coordinates.
    """
    from collections import Counter, defaultdict
    types = entity_poly_types(path)
    rna_chains = {c for c, t in types.items() if t == "polyribonucleotide"}
    hyb_chains = {c for c, t in types.items()
                  if t.startswith("polydeoxyribonucleotide/polyribonucleotide")}
    if not rna_chains and not hyb_chains:
        return {}, Counter(), types
    want = rna_chains | hyb_chains
    cols: list[str] = []
    in_loop = header = False
    chains: dict[str, set] = defaultdict(set)
    comp = Counter()
    hyb_names: dict[tuple, str] = {}
    hyb_ribo: set = set()
    with _open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1].strip())
                header = True
                continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False
                    continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                ch = r.get("auth_asym_id", r.get("label_asym_id", "?"))
                if ch not in want:
                    continue
                # POLYMER test: mmCIF assigns label_seq_id only to polymer
                # positions. Water, ions and ligands sitting in the same auth
                # chain carry '.', and counting them was inflating the total by
                # ~24% (HOH 37,857 and MG 9,997 in the first 60 files alone).
                if r.get("label_seq_id", ".") in (".", "?"):
                    continue
                key = (ch, r.get("auth_seq_id", r.get("label_seq_id", "?")),
                       r.get("pdbx_PDB_ins_code", "?"))
                if ch in hyb_chains:
                    hyb_names[key] = r["label_comp_id"].strip('"')
                    if r.get("label_atom_id", "").strip('"') == "O2'":
                        hyb_ribo.add(key)
                    continue
                if key not in chains[ch]:
                    chains[ch].add(key)
                    comp[r["label_comp_id"].strip('"')] += 1
    for key in hyb_ribo:
        chains[key[0]].add(key)
        comp[hyb_names[key]] += 1
    return chains, comp, types


def rna_chain_coords(path: Path, drop_hydrogens: bool = True):
    """Canonical per-chain, per-residue heavy-atom coordinates.

    The single entry point for anything that needs RNA *geometry* rather than
    counts. Returns `{chain: [(residue_key, comp_id, [(x,y,z), ...]), ...]}`
    with residues in polymer order.

    C13/C14 (cycle 11): the analysis scripts each selected residues with
    `label_comp_id in {A,C,G,U}`. That is not merely a narrower label set -- it
    DELETES modified residues from the middle of a chain, so the residues either
    side become adjacent and every downstream index shifts: chain length,
    sequence separation |i-j|, which pairs clear the SEQ_SEP guard, and the block
    indices i//b that block occupancy is computed from.

    Measured over the sample (`audit_definition_sensitivity.py`): **34.1% of
    chains change length**, by a median of +0.87% and up to **+27.9%** (7VNV,
    L 61 -> 78). Corpus medians barely move (contacts/nt +1.37%, effective c
    +1.19% on the affected chains) and the c=20 sparse-track budget is unaffected
    -- max effective c is 19.04 canonical against 19.03 published, with no chain
    in the sample breaching 20 under either definition.

    So the published corpus statistics stand, but a training pipeline must use
    this function rather than an ACGU filter: a 28% length error on an individual
    chain is invisible in a median and unacceptable as a model input.
    """
    from collections import defaultdict
    types = entity_poly_types(path)
    pure = {c for c, t in types.items() if t == "polyribonucleotide"}
    hyb = {c for c, t in types.items()
           if t.startswith("polydeoxyribonucleotide/polyribonucleotide")}
    want = pure | hyb
    if not want:
        return {}
    cols: list[str] = []
    in_loop = header = False
    atoms: dict[tuple, list] = defaultdict(list)
    names: dict[tuple, str] = {}
    ribo: set = set()
    model: str | None = None
    with _open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1].strip())
                header = True
                continue
            if header and in_loop:
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False
                    continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                # C16: FIRST MODEL ONLY. An NMR entry deposits an ensemble --
                # 20 models is the convention -- and every model repeats every
                # atom of every residue. Keyed only by (chain, seq, ins), the
                # ensemble collapsed into one residue holding 20 superposed
                # copies: 1ARJ came back with **424 atoms per residue** against
                # a nucleotide's real ~21. Geometry built from that is the
                # UNION of the ensemble's contacts, so two residues count as
                # touching if they touch in any single model, and both the
                # contact count and `effective_c` inflate for every NMR chain.
                # It also made the raw-corpus scan allocate 28 GB in one worker.
                # The four chains that set the published maxima (7PAS, 6Q95,
                # 1VY7, 6XHV) are all single-model cryo-EM or X-ray, so the
                # published tail statistics are unaffected by this.
                mn = r.get("pdbx_PDB_model_num")
                if mn is not None:
                    if model is None:
                        model = mn
                    elif mn != model:
                        continue
                ch = r.get("auth_asym_id", r.get("label_asym_id", "?"))
                if ch not in want:
                    continue
                if r.get("label_seq_id", ".") in (".", "?"):
                    continue
                if drop_hydrogens and r.get("type_symbol") == "H":
                    continue
                try:
                    seq = int(r["label_seq_id"])
                    xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
                except (KeyError, ValueError):
                    continue
                key = (ch, seq, r.get("pdbx_PDB_ins_code", "?"))
                atoms[key].append(xyz)
                names[key] = r["label_comp_id"].strip('"')
                if r.get("label_atom_id", "").strip('"') == "O2'":
                    ribo.add(key)
    out: dict[str, list] = defaultdict(list)
    for key in sorted(atoms, key=lambda k: (k[0], k[1])):
        if key[0] in hyb and key not in ribo:
            continue            # deoxyribonucleotide inside a hybrid chain
        out[key[0]].append((key, names[key], atoms[key]))
    return dict(out)


def entry_composition(path: Path) -> dict:
    """Whole-entry polymer residue counts, by declared entity type.

    The entry-level counterpart of `rna_chain_coords`, for the claims that are
    about *entries* rather than chains -- G1 (RNA residues per entry), G2 (what
    fraction sit in entries containing protein), G3 (what fraction come from
    ribosome-like entries). None of the three is computable on RNA3DB or
    RNASolo, which distribute per-chain extracts with the protein stripped out,
    so they can only be measured on raw PDB entries.

    C15: this function exists because defect #22 recurred. Two analysis scripts
    had each grown a private `entry_composition` that keyed `_atom_site` rows by
    **`label_asym_id`** while `entity_poly_types` returns **auth** chain ids.
    Where an entry's two labellings differ -- overwhelmingly the older entries,
    e.g. 1ARJ, whose sole RNA chain is `label A` / `auth N` -- every row missed
    its entity and the entry reported zero polymer residues of any kind. It hit
    **3,254 of 10,527** raw entries, including 686 that reported no polymer at
    all, and it silently deflated G2 and G3. Counting rules therefore live here,
    once, beside the resolver whose keys they have to match.

    Rules, identical to `rna_chain_coords` so the two never disagree:

      * chain identity is `auth_asym_id`, falling back to `label_asym_id`;
      * a residue is counted only at a polymer position (`label_seq_id` set),
        which excludes water, ions and ligands sharing the auth chain;
      * residues are deduplicated on (chain, label_seq_id, ins_code), so the
        20 models of an NMR entry collapse to one copy;
      * a hybrid chain is split per residue on the `O2'` test -- ribose has a
        2'-hydroxyl, deoxyribose does not.
    """
    types = entity_poly_types(path)
    prot = {c for c, t in types.items() if "polypeptide" in t}
    pure = {c for c, t in types.items() if t == "polyribonucleotide"}
    hyb = {c for c, t in types.items()
           if t.startswith("polydeoxyribonucleotide/polyribonucleotide")}
    dna = {c for c, t in types.items()
           if t.startswith("polydeoxyribonucleotide")} - hyb
    want = prot | pure | hyb | dna

    per_rna: dict[str, int] = {}
    n_prot = n_dna = 0
    seen: set = set()
    hyb_ribo: set = set()
    cols: list[str] = []
    in_loop = header = False
    if want:
        with _open(path) as fh:
            for line in fh:
                s = line.rstrip("\n")
                if s.startswith("_atom_site."):
                    if not in_loop:
                        in_loop, cols = True, []
                    cols.append(s.split(".", 1)[1].strip())
                    header = True
                    continue
                if not (header and in_loop):
                    continue
                if s.startswith(("#", "_", "loop_")):
                    in_loop = header = False
                    continue
                p = s.split()
                if len(p) < len(cols):
                    continue
                r = dict(zip(cols, p))
                ch = r.get("auth_asym_id", r.get("label_asym_id", "?"))
                if ch not in want:
                    continue
                sq = r.get("label_seq_id", ".")
                if sq in (".", "?"):
                    continue
                key = (ch, sq, r.get("pdbx_PDB_ins_code", "?"))
                if ch in hyb:
                    # decided after the pass, once every atom of the residue
                    # has been seen -- the O2' may come after other atoms
                    if r.get("label_atom_id", "").strip('"') == "O2'":
                        hyb_ribo.add(key)
                    continue
                if key in seen:
                    continue
                seen.add(key)
                if ch in prot:
                    n_prot += 1
                elif ch in pure:
                    per_rna[ch] = per_rna.get(ch, 0) + 1
                else:
                    n_dna += 1
    for ch, _, _ in hyb_ribo:
        per_rna[ch] = per_rna.get(ch, 0) + 1
    n_rna = sum(per_rna.values())

    return {
        "pdb": Path(path).stem.replace(".cif", ""),
        "n_protein_res": n_prot,
        "n_rna_res": n_rna,
        "n_dna_res": n_dna,
        "n_protein_chains": len(prot),
        "n_rna_chains": len(per_rna),
        "longest_rna_chain": max(per_rna.values(), default=0),
        "has_protein": len(prot) > 0,
        "has_dna": n_dna > 0,
        "entity_types": sorted(set(types.values())),
    }


def residue_labels(path: Path, ion_cutoff: float = 3.0,
                   ions: tuple = ("MG",)) -> dict:
    """The per-residue supervision that is already inside every deposited file.

    Four of ARCHITECTURE v0.2's heads are trained on labels nobody has to
    produce, because the depositor already did. This extracts them in one pass:

    * **head 5, Mg2+ sites** -- a residue is positive if any of its atoms lies
      within `ion_cutoff` of a magnesium. 3.0 A is inner-sphere coordination;
      the corpus holds 816,270 Mg sites and 77.9% of their inner-sphere contacts
      are to OP1/OP2.
    * **head 6, rigidity** -- the mean B-factor of a residue's atoms, and the
      experimental method, because **D12 restricts this head to X-ray**. The
      Mg-rigidity gradient is monotonic on 1,535 X-ray structures and *not*
      monotonic on cryo-EM, where per-atom B is a fitted display parameter
      rather than a measured one. The method travels with the label so a
      training loop cannot mix them by accident.
    * **head 10, base identity** -- which residues are `N_struct`: identity
      unassigned but ribose modelled, so their geometry trains normally and
      recovering the base is free supervision.
    * **head 11, disorder** -- `_pdbx_unobs_or_zero_occ_residues` names every
      residue too mobile or disordered to model. It is a direct per-residue
      flexibility label, present in every deposited structure, used by no RNA
      structure predictor. It is returned as `unobserved_seq_id` rather than as
      a mask over the modelled residues, because an unobserved residue has no
      atoms and so is absent from them -- a disorder mask aligned to the
      coordinates is necessarily all zeros.

    Returns `{chain: {...}}` keyed the same way as `rna_chain_coords`, so the
    arrays line up position for position with the coordinates.
    """
    from collections import defaultdict
    import numpy as _np

    types = entity_poly_types(path)
    pure = {c for c, t in types.items() if t == "polyribonucleotide"}
    hyb = {c for c, t in types.items()
           if t.startswith("polydeoxyribonucleotide/polyribonucleotide")}
    want = pure | hyb
    if not want:
        return {}

    method = "?"
    unobs: set = set()
    cols: list[str] = []
    in_loop = header = False
    model: str | None = None
    bsum: dict = defaultdict(float)
    bcnt: dict = defaultdict(int)
    names: dict = {}
    order: dict = defaultdict(list)
    ion_xyz: list = []
    res_xyz: dict = defaultdict(list)

    # `_pdbx_unobs_or_zero_occ_residues` is its own loop and has to be read
    # before the coordinates, since unobserved residues have none
    ucols: list[str] = []
    uin = uhdr = False

    with _open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            st = s.strip()

            if st.startswith("_exptl.method"):
                # exact token: `_exptl.method_details` is a different field, and
                # a prefix match on it silently zeroed the cryo-EM stratum once
                tag, _, rest = st.partition(" ")
                if tag == "_exptl.method" and rest.strip():
                    method = rest.strip().strip("'\"")
                continue

            if st.startswith("_pdbx_unobs_or_zero_occ_residues."):
                if not uin:
                    uin, ucols = True, []
                ucols.append(st.split(".", 1)[1])
                uhdr = True
                continue
            if uhdr and uin:
                if st.startswith(("#", "_", "loop_")):
                    uin = uhdr = False
                else:
                    q = st.split()
                    if len(q) >= len(ucols):
                        r = dict(zip(ucols, q))
                        ch = r.get("auth_asym_id", r.get("label_asym_id", "?"))
                        sq = r.get("label_seq_id", ".")
                        if sq not in (".", "?"):
                            unobs.add((ch, sq))
                    continue

            if s.startswith("_atom_site."):
                if not in_loop:
                    in_loop, cols = True, []
                cols.append(s.split(".", 1)[1].strip())
                header = True
                continue
            if not (header and in_loop):
                continue
            if s.startswith(("#", "_", "loop_")):
                in_loop = header = False
                continue
            q = s.split()
            if len(q) < len(cols):
                continue
            r = dict(zip(cols, q))
            mn = r.get("pdbx_PDB_model_num")
            if mn is not None:
                if model is None:
                    model = mn
                elif mn != model:
                    continue
            comp = r.get("label_comp_id", "").strip('"').upper()
            try:
                xyz = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
            except (KeyError, ValueError):
                continue
            if comp in ions:
                ion_xyz.append(xyz)
                continue
            ch = r.get("auth_asym_id", r.get("label_asym_id", "?"))
            if ch not in want:
                continue
            sq = r.get("label_seq_id", ".")
            if sq in (".", "?"):
                continue
            if r.get("type_symbol") == "H":
                continue
            key = (ch, int(sq), r.get("pdbx_PDB_ins_code", "?"))
            if key not in names:
                names[key] = comp
                order[ch].append(key)
            res_xyz[key].append(xyz)
            try:
                bsum[key] += float(r.get("B_iso_or_equiv", "0") or 0.0)
                bcnt[key] += 1
            except ValueError:
                pass

    out: dict = {}
    ions_arr = _np.asarray(ion_xyz, dtype=float) if ion_xyz else None
    tree = None
    if ions_arr is not None and len(ions_arr):
        from scipy.spatial import cKDTree
        tree = cKDTree(ions_arr)

    for ch, keys in order.items():
        keys = sorted(keys, key=lambda k: k[1])
        n = len(keys)
        bf = _np.zeros(n, dtype=_np.float32)
        mg = _np.zeros(n, dtype=_np.uint8)
        unk = _np.zeros(n, dtype=_np.uint8)
        for i, k in enumerate(keys):
            if bcnt[k]:
                bf[i] = bsum[k] / bcnt[k]
            if tree is not None and res_xyz[k]:
                if tree.query_ball_point(_np.asarray(res_xyz[k]), ion_cutoff,
                                         return_length=True).sum():
                    mg[i] = 1
            if names[k] in ("N", "UNK"):
                unk[i] = 1
        # B-factors are only comparable within a structure, so normalise per
        # chain; the raw mean travels too, because the scale itself differs
        # between X-ray and cryo-EM and D12 needs that distinction visible
        obs = bf[bf > 0]
        norm = ((bf - obs.mean()) / obs.std()) if obs.size > 1 and obs.std() > 0 \
            else _np.zeros_like(bf)
        # Positions named by `_pdbx_unobs_or_zero_occ_residues` for this chain.
        # These are NOT in the arrays above and cannot be: an unobserved
        # residue has no atoms, so it never appears in `_atom_site`. Keeping
        # `disordered` aligned to the modelled residues therefore makes it
        # vacuously zero, which is what a first version returned. The label is
        # a statement about the polymer, not about the coordinates, so the
        # missing positions travel separately and the dataset builder places
        # them against the full sequence.
        missing = sorted(int(q) for (c, q) in unobs if c == ch and q.isdigit())
        modelled = [k[1] for k in keys]
        n_poly = max(max(modelled, default=0), max(missing, default=0))
        out[ch] = {"b_factor": bf, "b_factor_z": norm.astype(_np.float32),
                   "mg_site": mg, "unknown_base": unk,
                   "modelled_seq_id": _np.asarray(modelled, dtype=_np.int32),
                   "unobserved_seq_id": _np.asarray(missing, dtype=_np.int32),
                   "n_polymer": int(n_poly),
                   "frac_unobserved": (len(missing) / n_poly) if n_poly else 0.0,
                   "method": method,
                   "rigidity_valid": bool("X-RAY" in method.upper())}
    # a chain whose residues are ALL unobserved has no coordinates and so no
    # entry above; it is still a real disorder observation, recorded as such
    for (c, q) in unobs:
        if c in want and c not in out:
            out[c] = {"b_factor": _np.zeros(0, dtype=_np.float32),
                      "b_factor_z": _np.zeros(0, dtype=_np.float32),
                      "mg_site": _np.zeros(0, dtype=_np.uint8),
                      "unknown_base": _np.zeros(0, dtype=_np.uint8),
                      "modelled_seq_id": _np.zeros(0, dtype=_np.int32),
                      "unobserved_seq_id": _np.asarray(
                          sorted(int(x) for (cc, x) in unobs
                                 if cc == c and x.isdigit()), dtype=_np.int32),
                      "n_polymer": 0, "frac_unobserved": 1.0,
                      "method": method,
                      "rigidity_valid": bool("X-RAY" in method.upper())}
    return out


def longest_rna_chain(path: Path):
    """The longest canonical RNA chain, as a list of per-residue atom lists."""
    chains = rna_chain_coords(path)
    if not chains:
        return None
    ch = max(chains, key=lambda c: len(chains[c]))
    return [a for _, _, a in chains[ch]]


if __name__ == "__main__":
    import sys
    from collections import Counter
    root = Path(__file__).resolve().parents[3] / "data/samples/structures"
    files = sorted(root.glob("*.cif.gz"))
    tot = nonstd = 0
    per_type = Counter()
    n_rna_struct = 0
    for f in files:
        ch, comp, types = rna_residues(f)
        per_type.update(types.values())
        n = sum(len(v) for v in ch.values())
        if n:
            n_rna_struct += 1
        tot += n
        nonstd += sum(c for k, c in comp.items() if k not in RNA_STD)
    print(f"structures with a declared RNA entity : {n_rna_struct}/{len(files)}")
    print(f"RNA residues (canonical)              : {tot:,}")
    print(f"  of which outside A/C/G/U            : {nonstd:,} = {100*nonstd/max(tot,1):.2f}%")
    print(f"declared entity types seen            : {dict(per_type.most_common())}")
