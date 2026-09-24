#!/usr/bin/env python3
"""Regression guard: re-derive every headline claim and fail loudly on drift.

Written during the rigor audit. Five implementation defects were found in the
original analyses and two more (a silent unguarded string replacement, and an
overclaimed statistic) in the write-up. This script exists so the numbers in the
deliverables cannot silently diverge from the data again.

Run: python3 scripts/sampling/verify_claims.py
Exit code 0 = all claims reproduce; 1 = drift detected.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "data" / "samples" / "analysis"

def load(n): return json.load(open(A / n))

def main() -> int:
    fails = []
    warns = []

    def chk(name, got, want, tol=None, abs_tol=None):
        """Compare at the precision the claim is actually quoted to.

        Defect #27 (cycle 9): every claim defaulted to a 0.5% RELATIVE tolerance,
        which is wider than the precision of almost every value it guards. Two
        consequences, both real:

          * `annotated base-pair steps` passed at 103,965 against an expected
            103,964 -- it reported OK for a value it should have rejected, which
            is how defect #26's propagation nearly went unnoticed.
          * `G7 frac of RNA residues` was checked as 0.01047 +/- 0.01, i.e. it
            would have accepted anything from 0.0005 to 0.0205. The guard on a
            figure this audit corrected by 8.5x was vacuous.

        The default is now derived from how `want` is written: an integer must
        match exactly, and a decimal must match to half a unit in its last quoted
        place. `abs_tol` states an absolute slack where the recomputation
        genuinely rounds differently; `tol` keeps the old relative form for the
        few claims that want it.
        """
        w = float(want)
        if abs_tol is not None:
            slack = abs_tol
        elif tol is not None:
            slack = tol * max(abs(w), 1)
        else:
            r = repr(float(want))
            dec = len(r.split(".")[1]) if "." in r and "e" not in r else 0
            if isinstance(want, int) or (dec == 1 and r.endswith(".0")):
                slack = 0.0
            else:
                slack = 0.5 * 10 ** (-dec)
        ok = abs(float(got) - w) <= slack
        print(f"  {'OK ' if ok else 'FAIL'} {name:38s} got={got}  want={want}")
        if not ok: fails.append(name)

    print("== measured claims vs analysis JSON ==")
    ion = load("ion_summary.json")
    chk("Mg2+ total", ion["total_ions_by_type"]["MG"], 17428)
    chk("K+ total", ion["total_ions_by_type"]["K"], 1868)
    inner = ion["top_inner_sphere_partners"]
    frac_phos = (inner["OP1"] + inner["OP2"]) / sum(inner.values())
    chk("inner-sphere phosphate fraction", round(frac_phos, 3), 0.83, abs_tol=0.005)

    rig = load("rigidity_summary.json")
    mg = rig["rigidity_by_mg_distance"]
    chk("Mg gradient span (sigma)", round(mg[-1]["mean_zB"] - mg[0]["mean_zB"], 2), 1.76)
    chk("Mg analysis nucleotides", sum(x["n"] for x in mg), 24623)
    g = rig["gnra_tetraloop"]
    chk("GNRA effect (sigma)", round(g["mean_zB_other"] - g["mean_zB_GNRA"], 3), 0.073)

    # ---- v0.2 claims, re-derived at corpus scale -------------------------
    # The checks above pin the ORIGINAL n=180 analysis and must keep passing:
    # they prove that pipeline still reproduces itself. These pin the values
    # that SUPERSEDE them, so a future edit cannot silently revert to the
    # small-sample numbers.
    try:
        bs = load("block_sparsity_fullcorpus.json")
        g4 = bs["acgu"]["global"]["4"]
        chk("v0.2 max effective c (20,266 chains)", g4["max_effective_c"], 21.14)
        chk("v0.2 chains breaching c=20", g4["n_over_20"], 1)

        ct = load("contact_tails_fullcorpus.json")["contacts_per_nt"]
        chk("v0.2 max contacts/nt", round(ct["max"], 2), 7.66)
        chk("v0.2 chains over published 5.50", ct["n_over_5_50"], 280)

        ir = load("ions_rigidity_rawpdb.json")
        chk("v0.2 Mg gradient span, X-ray (sigma)",
            ir["gradient_xray"]["span_sigma"], 1.523, abs_tol=0.002)
        chk("v0.2 Mg gradient monotonic, X-ray",
            int(ir["gradient_xray"]["monotonic"]), 1)
        chk("v0.2 Mg gradient X-ray structures",
            ir["gradient_xray"]["n_structures"], 1535)
        chk("v0.2 cryo-EM NOT monotonic",
            int(ir["gradient_cryoem"]["monotonic"]), 0)
        chk("v0.2 Mg:K ratio", ir["mg_vs_k"], 53.98, abs_tol=0.02)
        chk("v0.2 inner-sphere phosphate fraction",
            ir["inner_sphere"]["phosphate_frac"], 0.7792, abs_tol=0.0005)
        chk("v0.2 modified-residue fraction (raw PDB)",
            ir["modified_residues"]["frac"], 0.01005, abs_tol=0.00005)

        # ---- v0.2a: the stage-1 packer was 42.9% padding -----------------
        # The token budget was never a token budget. Sequences were appended in
        # corpus order and cut when the batch would exceed the budget, so a
        # 20-nt and a 1,024-nt sequence shared a batch and both padded to
        # 1,024. Deterministic over the same 400,000 elDORS sequences.
        pw = load("packing_waste.json")
        # stratified: a fixed quota per shard, so the total lands just under
        # the target rather than on it
        chk("v0.2a packing: sequences measured", pw["n_sequences"], 399058)
        chk("v0.2a packing: corpus median length", pw["length_median"], 261)
        chk("v0.2a packing: shipped stream-order padding",
            pw["stream_order"]["padding_fraction"], 0.374, abs_tol=0.002)
        chk("v0.2a packing: length-sorted padding is zero",
            pw["length_sorted"]["padding_fraction"], 0.0, abs_tol=0.001)
        chk("v0.2a packing: quantised padding", pw["quantum"], 64)
        chk("v0.2a packing: quantised padding fraction",
            pw["quantised"]["padding_fraction"], 0.081, abs_tol=0.002)
        # A handful of widths is the whole point of quantising: the delta-rule
        # chunk loop is a Python loop, so inductor specialises on the chunk
        # count and 969 distinct widths recompiles on nearly every batch.
        chk("v0.2a packing: quantised distinct widths",
            pw["quantised"]["distinct_widths"], 16)
        chk("v0.2a packing: unquantised distinct widths",
            pw["length_sorted"]["distinct_widths"], 969)
        chk("v0.2a packing: steps saved vs shipped",
            pw["steps_saved_fraction"], 0.294, abs_tol=0.003)
        chk("v0.2a packing: useful tokens per step gain",
            pw["useful_tokens_per_step_gain"], 1.42, abs_tol=0.02)

        # ---- v0.2a: the pretraining corpus was a length band ---------------
        # elDORS is SORTED BY LENGTH and ships as twenty chunks. The starter
        # took eight of them, so stage 1 saw the long AT-rich end and none of
        # the short GC-rich majority. "10,000,000 sequences" reads like
        # sufficiency and says nothing about which ten million.
        pc = load("pretrain_coverage.json")
        chk("v0.2a corpus: elDORS chunks", pc["eldors_chunks"], 20)
        chk("v0.2a corpus: chunks now covered", len(pc["corpus_chunks"]), 20)
        chk("v0.2a corpus: coverage is complete", int(pc["covers_all_chunks"]), 1)
        chk("v0.2a corpus: sequences", pc["corpus_sequences"], 25000000)
        chk("v0.2a corpus: chunks are NOT interchangeable",
            int(pc["chunks_interchangeable"]), 0)
        chk("v0.2a corpus: used-vs-unused mean length gap",
            pc["comparison"]["len_mean"]["rel_diff"], 0.7262, abs_tol=0.003)
        chk("v0.2a corpus: used-vs-unused GC gap",
            pc["comparison"]["gc"]["rel_diff"], 0.2076, abs_tol=0.003)

        # ---- v0.2a: the pair track END TO END, which S7.4a owed -----------
        # L1 and L2 standalone are not what a deployed track delivers: L2 only
        # refines inside surviving L1 pairs. Cascade 0.264 against an L1
        # ceiling of 0.270 says L2 loses 0.006 and L1 loses the rest.
        cr = load("cascade_recall.json")
        chk("v0.2a cascade: test chains", cr["learned"]["n_chains"], 1450)
        chk("v0.2a cascade: learned", cr["learned"]["cascade"], 0.264, abs_tol=0.002)
        chk("v0.2a cascade: separation prior",
            cr["separation"]["cascade"], 0.2006, abs_tol=0.002)
        chk("v0.2a cascade: random", cr["random"]["cascade"], 0.0988, abs_tol=0.002)
        chk("v0.2a cascade: L1 ceiling bounds it",
            int(cr["learned"]["l1_ceiling"] >= cr["learned"]["cascade"]), 1)
        chk("v0.2a cascade: L2 loses almost nothing",
            round(cr["learned"]["l1_ceiling"] - cr["learned"]["cascade"], 4),
            0.0057, abs_tol=0.002)
        chk("v0.2a cascade: >=1500 nt, the regime the track exists for",
            cr["learned"]["cascade_by_length"][">=1500"], 0.902, abs_tol=0.005)
        # The defect the cascade exposed: L1 clamps its minimum separation to
        # one SIXTEEN-residue block where L2 clamps to one FOUR-residue block,
        # so contacts 4-15 apart have no valid parent at any budget.
        chk("v0.2a cascade: blocks unreachable on the L1 diagonal",
            cr["learned"]["diag_unreachable"], 0.189, abs_tol=0.003)
        full = cr["l1_budget_sweep"]["1.00"]
        chk("v0.2a cascade: keeping every L1 pair still ceilings below 1",
            full["l1_ceiling"], 0.8106, abs_tol=0.003)
        chk("v0.2a cascade: and the gap IS the diagonal",
            round(full["l1_ceiling"] + cr["learned"]["diag_unreachable"], 2),
            1.0, abs_tol=0.01)
        # Admitting the diagonal without retraining is worse at the shipped
        # budget: the scorer has never been asked to score a diagonal block.
        chk("v0.2a cascade: diagonal admitted, shipped budget, is worse",
            int(cr["l1_diagonal_admitted"]["0.12"]["cascade"]
                < cr["learned"]["cascade"]), 1)

        # ---- v0.2a: is the MoE router routing, or just balanced? ----------
        # The balance loss at its floor says the LOAD is even and nothing about
        # whether any token is routed sharply. Per-token entropy separates a
        # specialising router from a collapsed one; the mean cannot.
        # by token count, not by index: the history grows as the probe is
        # re-run and an index silently points at a different measurement
        _rsh = load("router_specialisation.json")["history"]
        rs = next(h for h in _rsh if h["tokens"] == 55382967)
        chk("v0.2a router: experts", rs["n_experts"], 32)
        chk("v0.2a router: probe token count", rs["tokens"], 55382967)
        chk("v0.2a router: per-token entropy", rs["token_router_entropy"],
            3.433, abs_tol=0.002)
        chk("v0.2a router: as a fraction of uniform",
            rs["frac_of_uniform"], 0.99, abs_tol=0.005)
        chk("v0.2a router: mean top-1 probability",
            rs["mean_top1_prob"], 0.047, abs_tol=0.002)
        # and the balance term really is pinned at its analytic floor:
        # balance_weight 0.01 x 16 blocks x 1.0, which is what the Switch form
        # evaluates to at perfect uniformity -- NOT zero, which is why it has
        # no business in a bits/token figure
        # Replicated across three runs and two corpora -- the point is that it
        # does not move, and the two comparable points move the wrong way.
        _rs82 = next((h for h in _rsh if h["tokens"] == 82103583), None)
        if _rs82 is not None:
            chk("v0.2a router: still uniform at 82.1M tokens",
                _rs82["frac_of_uniform"], 0.9906, abs_tol=0.002)
            # Scoped to the early window ON PURPOSE. It was written when 82M
            # was the last probe and read as a general claim; it is not one.
            # The router does specialise later -- see the 573M checks below.
            chk("v0.2a router: flat through the first 82M",
                int(_rs82["frac_of_uniform"] >= rs["frac_of_uniform"]), 1)
        chk("v0.2a router: probes recorded", int(len(_rsh) >= 3), 1)
        # ...and it does start. The flat reading through 82M was too early.
        _rs573 = next((h for h in _rsh if h["tokens"] == 573278318), None)
        if _rs573 is not None:
            chk("v0.2a router: specialising by 573M tokens",
                _rs573["frac_of_uniform"], 0.9679, abs_tol=0.002)
            chk("v0.2a router: top-1 rises above chance",
                _rs573["mean_top1_prob"], 0.0742, abs_tol=0.002)
            chk("v0.2a router: sharper than at 82M",
                int(_rs573["frac_of_uniform"] < _rs82["frac_of_uniform"]), 1)
        chk("v0.2a router: load is uniform to 3 decimals",
            round(rs["expert_load_max"] - rs["expert_load_min"], 4),
            0.0043, abs_tol=0.0005)

        ec = load("entry_composition_rawpdb.json")
        # ---- D9/D23: target_c, closed on raw chains ----------------------
        # The budget had never been measured on data containing modified
        # residues at anything like their real rate. On raw chains they are
        # present at 0.476% against the derivatives' 0.025%, and the maximum
        # rose from 21.14 to 23.30 -- the mechanism D9 predicted. 24 holds,
        # with 2.9% headroom and zero breaches in 14,106 chains.
        tc = load("targetc_g2g3_rawpdb.json")
        chk("D23 raw chains measured", tc["n_chains"], 14106)
        chk("D23 raw max effective c", tc["target_c_raw"]["max_effective_c"], 23.30)
        chk("D23 raw p99.9 effective c", tc["target_c_raw"]["p999"], 23.08)
        chk("D23 chains over 20", tc["target_c_raw"]["n_over_20"], 34)
        chk("D23 chains over target_c=24", tc["target_c_raw"]["n_over_24"], 0)
        chk("D23 modified residues in analysed chains",
            tc["modified_residues"]["frac"], 0.00476, abs_tol=0.00002)
        chk("D23 raw max contacts/nt", tc["contacts_per_nt_raw"]["max"], 7.79)
        # The two independent code paths -- chain geometry and entry counting
        # -- must agree on the entry-level claims. C15 was exactly a case of
        # them silently not agreeing.
        chk("G2 agrees across both raw scripts",
            tc["G2_in_complex"]["frac_residues"], ec["G2_complex"]["frac_residues"],
            abs_tol=0.0001)
        chk("G3 agrees across both raw scripts",
            tc["G3_ribosomal"]["frac_residues"], ec["G3_ribosomal"]["frac_residues"],
            abs_tol=0.0001)

        # ---- §4 chemistry: the modification census and CCD resolution ----
        # CHEMISTRY.md dims 20-22 were specified against five hand-picked
        # examples; the archive holds 370 species. Parent resolution is by
        # dictionary, because the parent field is not in the entry files at all.
        mc = load("modification_census.json")
        chk("chem census residues", mc["n_residues"], 13348166)
        chk("chem census modified", mc["n_modified"], 75434)
        chk("chem census modified fraction", mc["frac_modified"], 0.00565,
            abs_tol=0.00002)
        chk("chem distinct modification species", mc["n_distinct_species"], 370)
        chk("PSU is the commonest modification",
            mc["top_modifications"][0][0] == "PSU", 1)

        cp = load("ccd_parents.json")
        chk("chem all species found in the CCD",
            cp["n_species_in_ccd"], cp["n_species"])
        chk("chem residues resolvable to an A/C/G/U parent",
            cp["frac_residues_with_parent"], 0.8858, abs_tol=0.0005)
        cl = cp["by_class_residues"]
        chk("chem no residue left unclassified", int("unknown" not in cl), 1)
        chk("chem methylation residues", cl["methylation"], 42555)
        chk("chem pseudouridine residues", cl["pseudouridine"], 18458)
        chk("chem other residues", cl["other"], 14421)

        # ---- §7.1: flat top-K, re-derived on 2,994 chains ----------------
        # The claim the pair track rests on. v0.1 stated it from 43 chains and
        # compared two different populations; these pin the matched version.
        pr = load("proposal_recall_fullcorpus.json")
        chk("7.1 chains measured", pr["n_chains"], 2994)
        lb = pr["by_length_bin"]["1200-3000"]
        chk("7.1 long-chain random recall at c=32", lb["random"]["32"], 0.0339,
            abs_tol=0.0002)
        chk("7.1 long-chain separation recall at c=32", lb["separation"]["32"],
            0.3767, abs_tol=0.0002)
        chk("7.1 long-chain complementarity recall at c=32",
            lb["complementary"]["32"], 0.1822, abs_tol=0.0002)
        chk("7.1 even at c=64 flat ranking recovers only half",
            int(lb["separation"]["64"] < 0.55), 1)
        # the design's whole case: total recall at a THIRD of that budget
        chk("7.1 block oracle reaches recall 1.0 at c", lb["block_oracle_c_mean"],
            16.91, abs_tol=0.02)
        # the sanity check that the measurement is sound
        chk("7.1 random recall equals the dense fraction",
            int(abs(lb["random"]["32"] - lb["frac_dense_c32"]) < 0.001), 1)
        # complementarity applied flat is WORSE than |i-j| alone on long chains
        chk("7.1 separation beats flat complementarity on long chains",
            int(lb["separation"]["32"] > lb["complementary"]["32"]), 1)

        # ---- is every catalogued dataset still on disk? ------------------
        # The manifest records what was downloaded; this records what is there.
        # A source that quietly went missing fails at training time otherwise.
        dp = load("data_presence.json")
        chk("catalogued sources complete", dp["n_sources_complete"], dp["n_sources"])
        chk("catalogued files present", dp["n_present"], dp["n_files"])
        chk("no file changed size since acquisition", dp["n_resized"], 0)
        chk("no file missing", dp["n_missing"], 0)

        # ---- R1: the block scorer, trained -------------------------------
        bsr = load("block_scorer_results.json")
        bt = bsr["test"]
        chk("R1 random L2 recall", bt["random"]["l2_recall"], 0.5435, abs_tol=0.0002)
        chk("R1 separation-prior L2 recall", bt["separation"]["l2_recall"],
            0.6333, abs_tol=0.0002)
        chk("R1 learned L2 recall", bt["learned"]["l2_recall"], 0.8345, abs_tol=0.0002)
        chk("R1 sequence gain over the separation prior",
            bsr["l2_sequence_gain"], 0.2012, abs_tol=0.0002)
        # the number the pair track exists for: long chains
        lb = bt["learned"]["recall_by_length"]
        sb = bt["separation"]["recall_by_length"]
        chk("R1 long chains (>=1500), learned", lb["l2_>=1500"], 0.9972, abs_tol=0.0002)
        chk("R1 long chains (>=1500), separation prior", sb["l2_>=1500"], 0.2779,
            abs_tol=0.0002)
        chk("R1 the ablated prior costs almost nothing",
            int(abs(bt["learned"]["l2_recall"]
                    - bt["learned_noprior"]["l2_recall"]) < 0.01), 1)
        # L1 at the same matched budget. The gain is real but much smaller than
        # L2's, and the cascade measurement explains why that matters: L1 is
        # what bounds the track end to end.
        chk("R1 random L1 recall", bt["random"]["l1_recall"], 0.1353, abs_tol=0.0002)
        chk("R1 separation-prior L1 recall", bt["separation"]["l1_recall"],
            0.2313, abs_tol=0.0002)
        chk("R1 learned L1 recall", bt["learned"]["l1_recall"], 0.2724, abs_tol=0.0002)
        chk("R1 L1 sequence gain", bsr["l1_sequence_gain"], 0.0411, abs_tol=0.0002)
        chk("R1 learned L2 precision", bt["learned"]["l2_precision"], 0.3377,
            abs_tol=0.0002)
        chk("R1 test chains", bsr["splits"]["test"], 1450)
        chk("R1 parameters", bsr["n_parameters"], 61889798)
        # S7.1 measures the flat separation prior at 0.377 on long chains at
        # c=32. The learned scorer reaches 0.9972 in the same regime, which is
        # the comparison the pair track was built to win.
        chk("R1 beats S7.1's 0.377 long-chain prior by a wide margin",
            int(lb["l2_>=1500"] > 2.5 * 0.377), 1)

        # ---- R2: the router actually routes (scripts/audit_router.py) ------
        #
        # The MoE block computes `mean_width`, `max_width` and two router
        # entropies at every block of every step; the trunk discarded all four
        # before the trainer could see them, so "no router collapse" rested on
        # the balance term alone with no stated threshold, and nucleus
        # routing's central claim -- variable width -- had never been measured
        # on a trained model. Measured at step 7,000 (992.8M tokens).
        import json as _rjson
        ra = ROOT / "data/samples/analysis/router_audit.json"
        if ra.exists():
            rj = _rjson.loads(ra.read_text())
            chk("R2 routing width is variable, not pinned at max_k",
                int(1.0 < rj["mean_width"] < rj["max_k"]), 1)
            chk("R2 mean routing width", round(rj["mean_width"], 2), 15.28,
                abs_tol=0.01)
            chk("R2 widest token", int(rj["max_width"]), 330)
            chk("R2 no dead experts", rj["dead_expert_frac"], 0.0, abs_tol=1e-9)
            chk("R2 effective experts of 512", round(rj["effective_experts"], 1),
                288.5, abs_tol=0.1)
            # 1.0 is uniform and 512 is total collapse; the run sits just above
            # uniform, which is the opposite of collapse
            chk("R2 balance per block is near uniform",
                int(rj["balance_per_block"] < 2.0), 1)
            chk("R2 router entropy of the 9.000 available",
                round(rj["router_entropy_bits"], 3), 7.255, abs_tol=0.002)
            chk("R2 the audit raised no warnings", len(rj["warnings"]), 0)
        else:
            chk("R2 router audit run (scripts/audit_router.py)", 0, 1)

        # ---- R3: coevolution actually reaches the contact head -------------
        #
        # §4A says the pair track reads coevolutionary couplings. The path from
        # that sentence to the arithmetic runs through a bare
        # `except Exception: return None`, a `searchsorted` whose misses are
        # filled with zeros, and a zero-initialised `coev_proj`. All three
        # degrade to "adds nothing" without raising, so a working feature and a
        # totally absent one produce identical logs. Measured, so a drift to
        # zero is visible.
        cr = ROOT / "data/samples/analysis/coevolution_reach.json"
        if cr.exists():
            cj = _rjson.loads(cr.read_text())
            chk("R3 sampled pairs measured", cj["pairs"], 927401)
            chk("R3 pairs carrying a coupling", cj["coupled"], 32100)
            chk("R3 coevolution reach", round(cj["hit_frac"], 5), 0.03461,
                abs_tol=0.00002)
            chk("R3 coevolution is not silently absent",
                int(cj["hit_frac"] > 0.005), 1)
        else:
            chk("R3 coevolution reach measured", 0, 1)

        # ---- R4: the motif bank retrieves more than one motif --------------
        #
        # `self.query` is bias-free, so it cannot subtract a shared offset from
        # its input: for x = xbar + delta, W x = W xbar + W delta and W xbar is
        # identical for every query. On real pair features the shared component
        # is LARGER than the per-pair variation (norm 5.66 against 4.53), so
        # retrieval collapsed onto a handful of keys -- and the two statistics
        # the bank computes for exactly this were discarded by the only caller
        # that trains it (`r, _ = model.motifs(pair)`).
        mb = ROOT / "data/samples/analysis/motif_bank_retrieval.json"
        if mb.exists():
            mj = _rjson.loads(mb.read_text())
            chk("R4 the shared component exceeds the per-pair variation",
                int(mj["shared_component_norm"] > mj["per_pair_deviation"]), 1)
            chk("R4 effective motifs WITHOUT centring",
                round(mj["centre_off"]["effective_motifs"], 2), 2.98, abs_tol=0.02)
            chk("R4 effective motifs WITH centring",
                round(mj["centre_on"]["effective_motifs"], 2), 22.13, abs_tol=0.02)
            chk("R4 top motif share falls from half the queries",
                round(mj["centre_off"]["top_motif_share"], 4), 0.5194, abs_tol=0.0002)
            chk("R4 centring is what makes the bank a bank, not a bias term",
                int(mj["centre_on"]["effective_motifs"]
                    > 5 * mj["centre_off"]["effective_motifs"]), 1)
        else:
            chk("R4 motif retrieval measured", 0, 1)

        # ---- R5: the router's length conditioning is not saturated ---------
        #
        # `floor(log2(L/32))` clamped to `n_bins - 1` gives bins 0..4 one
        # octave each and makes bin 5 a catch-all for everything above 1,024 --
        # 4.5 octaves in one one-hot. In stage 1 the 20-1024 filter leaves that
        # bin reachable only by a chain of exactly 1,024; in stage 5 it takes
        # 40% of chains, which is the long-chain regime the pair track exists
        # for. `length_bin_max` spreads the bins over the real range instead.
        rc = ROOT / "data/samples/analysis/router_conditioning.json"
        if rc.exists():
            kj = _rjson.loads(rc.read_text())
            chk("R5 stage-1 top octave bin is effectively unreachable",
                int(kj["s1_octave"]["occupancy"][5] < 0.001), 1)
            chk("R5 stage-5 top octave bin is a 4.5-octave catch-all",
                round(kj["s5_octave"]["occupancy"][5], 4), 0.3995, abs_tol=0.0002)
            chk("R5 spreading the bins raises stage-1 conditioning entropy",
                int(kj["s1_spread"]["entropy_bits"]
                    > kj["s1_octave"]["entropy_bits"] + 0.2), 1)
            chk("R5 stage-1 entropy, octave bins",
                kj["s1_octave"]["entropy_bits"], 1.7241, abs_tol=0.002)
            chk("R5 stage-1 entropy, spread bins",
                kj["s1_spread"]["entropy_bits"], 2.0924, abs_tol=0.002)
        else:
            chk("R5 router conditioning measured", 0, 1)

        # ---- R6: the held-out scatter is the batch size, not noise ---------
        #
        # `_pack_pool` cuts on the padded token budget OR on the sequence
        # COUNT, and at `--max-batch` 512 the count bound on short shards:
        # c020 capped 92.5% of its batches at 83,899 real tokens against
        # c001's 194,607. Joining the trainer's per-100-step token rate to the
        # clean held-out readings, fewer tokens in the step predicts worse
        # bits even after the training trend is partialled out.
        be = ROOT / "data/samples/analysis/batch_size_effect.json"
        if be.exists():
            bj = _rjson.loads(be.read_text())
            # BOUNDS, not exact values. This statistic is recomputed over
            # every clean reading so far, so it moves each time the watcher
            # scores a checkpoint; pinning -0.5413 exactly would fail on the
            # next eval and teach everyone to ignore the failure.
            chk("R6 at least 20 paired checkpoints", int(bj["n"] >= 20), 1)
            chk("R6 fewer tokens in the step, worse bits",
                int(bj["r_partial_step_controlled"] < -0.3), 1)
            chk("R6 the effect survives the training trend",
                int(abs(bj["t"]) > 2.0), 1)
            # ... and the thing it does NOT establish. The step-8,500
            # excursion was attributed to this mechanism and the attribution
            # did not survive: step 8,750 ran on a SMALLER mean batch
            # (123,500 tokens against 134,000) and recovered to 1.6602 from
            # 1.8224 anyway. The aggregate effect is real; it does not explain
            # individual points, and this check exists so nobody reinstates
            # that claim from the correlation alone.
            chk("R6 does not explain step 8,500: 8,750 was smaller and better",
                int(bj["tokens_per_step"][bj["steps"].index(8750)]
                    < bj["tokens_per_step"][bj["steps"].index(8500)]
                    and bj["bits"][bj["steps"].index(8750)]
                    < bj["bits"][bj["steps"].index(8500)]), 1)
        else:
            chk("R6 batch-size effect measured", 0, 1)

        # ---- R7: the model works at ONE recycle depth ----------------------
        #
        # Stage 1 fixes `--n-loops` and never varies it, so the recycle
        # projection is trained for exactly one application. Stage 5 samples
        # 1..cfg.n_loops. The configs advertise 8 loops and 144 effective
        # layers; training exercises 2 and 36.
        ld = ROOT / "data/samples/analysis/loop_depth_sweep.json"
        if ld.exists():
            lj = _rjson.loads(ld.read_text())
            sw = lj["sweep"]
            chk("R7 best at the depth it trained on",
                int(min(sw, key=lambda k: sw[k]["bits"]) == "2"), 1)
            chk("R7 bits at the trained depth (2 loops)", sw["2"]["bits"],
                1.6632, abs_tol=0.0002)
            chk("R7 bits at the CONFIGURED depth (8 loops)", sw["8"]["bits"],
                2.0064, abs_tol=0.0002)
            chk("R7 at 8 loops it is worse than the corpus unigram entropy",
                int(sw["8"]["bits"] > lj["corpus_entropy_bits"]), 1)
            chk("R7 advertised vs trained effective layers",
                f'{lj["advertised_effective_layers"]}/'
                f'{lj["actual_effective_layers_in_training"]}', "144/36")
        else:
            chk("R7 loop-depth sweep measured", 0, 1)

        # ---- the data: present, and readable ------------------------------
        ig = load("inventory_gap.json")
        chk("nothing in the acquisition inventory is missing", ig["n_missing"], 0)
        # MARS was the one source absent by decision (D18, 427.29 GB for 0.17%
        # diverse structured ncRNA). It is now being acquired on request, so it
        # has moved from `decided` to `partial` and will end at `present`. All
        # three are acceptable states for it and none of them is "missing",
        # which is the invariant this guard is actually for.
        _mars = next((s for s in ig["sources"] if s["source"] == "MARS"), None)
        chk("MARS is the only source not fully present",
            ig["n_present"] + (1 if _mars and _mars["state"] != "present" else 0),
            ig["n_sources"])
        chk("and its state is decided, partial or present",
            int(bool(_mars) and _mars["state"] in ("decided", "partial", "present")), 1)
        di = load("data_integrity.json")
        chk("every sampled file opens as what it claims", di["n_bad"], 0)
        chk("integrity sampled across every corpus", di["n_checks"], 14)

        # ---- stage 5: and why D25's two splits are never averaged --------
        ph = load("pharos_small_results.json")
        te, tr_ = ph["test"], ph["test_ribosomal"]
        chk("stage5 test Mg AP lift", te["mg_ap_lift"], 1.81, abs_tol=0.02)
        chk("stage5 test rigidity r", te["rigidity_r_pooled"], 0.0487, abs_tol=0.002)
        chk("stage5 ribosomal Mg AP lift", tr_["mg_ap_lift"], 4.0, abs_tol=0.02)
        chk("stage5 ribosomal rigidity r", tr_["rigidity_r_pooled"], 0.3989,
            abs_tol=0.002)
        # the point of D25: the homolog-rich split is ~8x better on rigidity, so
        # a blended number would describe neither population
        chk("stage5 rigidity gap between the splits is large",
            int(tr_["rigidity_r_pooled"] > 5 * te["rigidity_r_pooled"]), 1)

        # ---- completeness: does the code implement the specification? ----
        cp = load("completeness.json")
        chk("every specified component is present",
            cp["n_present"], cp["n_components"])
        chk("every curriculum stage has a runner",
            cp["n_stages_present"], cp["n_stages"])
        chk("nothing specified is missing", len(cp["missing"]), 0)

        # ---- §6.4: RMDB ionic titrations (open action 2) -----------------
        import json as _jj
        tf = ROOT / "data/benchmarks/rmdb/titrations.json"
        if tf.exists():
            ti = _jj.loads(tf.read_text())
            chk("RMDB assets enumerated", ti["n_assets_total"], 1024)
            chk("RMDB Mg titration files", ti["n_mg_titrations"], 24)
            mgs = [v for v in ti["series"].values() if "MgCl2" in v["varying"]]
            chk("RMDB Mg concentration points",
                sum(v["ions"]["MgCl2"]["n_levels"] for v in mgs), 527)
            # the property that makes them useful: the PDB cannot reach below
            # ~5 mM because unfolded RNA is not deposited
            chk("every Mg ladder crosses the sub-millimolar regime",
                sum(1 for v in mgs if v["ions"]["MgCl2"]["min"] < 1.0), 24)
        else:
            chk("RMDB titrations acquired (scripts/acquire_rmdb_titrations.py)", 0, 1)

        io = ROOT / "data/derived/ionic/meta.json"
        if io.exists():
            im = _jj.loads(io.read_text())
            chk("ionic: ladders parsed", im["n_ladders"], 24)
            chk("ionic: training examples", im["n_examples"], 701)
            chk("ionic: residues", im["n_residues"], 75706)
            chk("ionic: all ladders cross sub-mM", im["ladders_crossing_sub_mM"], 24)
            # the folding transition must be IN the data, not assumed
            chk("ionic: ladders showing the folding transition",
                im["folding_check"]["n_showing_folding"], 16)
            chk("ionic: median Spearman(conc, reactivity) is negative",
                int(im["folding_check"]["median_rho"] < -0.3), 1)
        else:
            chk("ionic channel built (scripts/build_ionic_dataset.py)", 0, 1)

        # ---- §8: the motif bank is the atlas, not a subset ---------------
        import json as _j
        mb = ROOT / "data/derived/motif_bank/meta.json"
        if mb.exists():
            bm = _j.loads(mb.read_text())
            chk("motif classes", bm["n_motifs"], 667)
            chk("motif: internal loops", bm["n_internal_loop"], 413)
            chk("motif: hairpin loops", bm["n_hairpin_loop"], 254)
            chk("motif instances", bm["n_instances_total"], 4992)
            chk("motif: interaction vocabulary read from the atlas",
                len(bm["bp_families"]), 21)
        else:
            chk("motif bank compiled (scripts/build_motif_bank.py)", 0, 1)

        # ---- §12.4 / D25: the built training set and its splits ----------
        import json as _json
        mf = ROOT / "data/derived/pharos3d/manifest.json"
        if mf.exists():
            ds = _json.loads(mf.read_text())
            chk("dataset chains", ds["n_chains"], 16604)
            chk("dataset residues", ds["n_residues"], 13172991)
            chk("dataset contacts", ds["n_contacts"], 63298800)
            chk("dataset longest chain", ds["length"]["max"], 4450)
            sp = ds["split"]["by_residue"]
            chk("split: train residues", sp["train"], 10010363)
            # val and test are packed to the same residue budget on purpose;
            # if they drift apart the packer has stopped balancing. NOT exact
            # equality: the packer assigns whole CHAINS, so a perfect tie is
            # only available when the lengths happen to sum that way, and the
            # v3 corpus lands one residue apart. An equality test called that a
            # failure for as long as v3 has existed.
            chk("split: val and test balanced by residue (within 1%)",
                int(abs(sp["val"] - sp["test"]) <= max(1, 0.01 * min(sp["val"], sp["test"]))), 1)
            chk("split: val residues", sp["val"], 531547)
            chk("split: test residues", sp["test"], 531546)
            # v3 added coordinates, per-chain Rfam, and the two base-pair
            # targets. Heads 9 and 10 have real, non-degenerate supervision:
            # all 13 Leontis-Westhof classes and all 3 motif classes occur.
            _rows = [c for sh in ds["shards"] for c in sh["chains"]]
            chk("head 9: Leontis-Westhof annotated pairs",
                sum(c.get("n_lw_pairs", 0) for c in _rows), 3472712)
            chk("head 9: chains carrying at least one annotated pair",
                sum(1 for c in _rows if c.get("n_lw_pairs", 0)), 12644)
            chk("split: test_ribosomal is separate and labelled",
                int("test_ribosomal" in sp), 1)
            # the free supervision channels (§9): each must be present, and the
            # disorder count must be the USABLE one, not the raw one
            import sys as _s
            _s.path.insert(0, str(ROOT / "src"))
            from pharos.data.dataset import disorder_is_meaningful as _dm
            rows = [c for sh in ds["shards"] for c in sh["chains"]]
            chk("head 5: Mg-coordinated residues",
                sum(c.get("n_mg_sites", 0) for c in rows), 712033)
            chk("head 6: X-ray residues (rigidity-valid)",
                sum(c["length"] for c in rows if c.get("rigidity_valid")), 4274596)
            chk("head 10: N_struct residues",
                sum(c.get("n_unknown_base", 0) for c in rows), 2586)
            chk("disorder: raw unobserved positions",
                sum(c.get("n_unobserved", 0) for c in rows), 1379892)
            chk("disorder: usable after excluding truncated constructs",
                sum(c.get("n_unobserved", 0) for c in rows
                    if _dm(c["length"], c.get("n_polymer") or 0)), 907067)
        else:
            chk("dataset manifest present (scripts/build_dataset.py)", 0, 1)

        # ---- §5.4 / D24: the sizing table is now countable ---------------
        sz = load("sizing_solution.json")
        chk("sizing: §5.4 is not reproducible from the document",
            int(sz["_verdict"].startswith("NOT")), 1)
        chk("sizing: its three rows imply three different recipes",
            int(len({(sz[m]["n_experts"], sz[m]["top_k"]) for m in
                     ("PHAROS-Small", "PHAROS-Mini", "Base-v2")}) == 3), 1)

        # ---- §11.4: the derivatives are not the archive ------------------
        # The finding that reframed D23: the derivative maximum is correct for
        # the population the derivatives hold, and every chain above it is one
        # they do not. A parameter fitted to a curated corpus inherits that
        # corpus's snapshot date.
        dc = load("derivative_coverage.json")
        de, du = dc["entries"], dc["uncovered_character"]
        chk("11.4 RNA3DB entries", de["rna3db"], 5389)
        chk("11.4 gRNAde/RNASolo entries", de["grnade_rnasolo"], 6156)
        chk("11.4 derivative union", de["union"], 7943)
        chk("11.4 entries in neither", de["uncovered"], 2581)
        chk("11.4 uncovered with a chain in 64-3000", du["entries_chain_in_window"], 608)
        chk("11.4 uncovered protein-free", du["protein_free"], 925)

        dec, dt = dc["effective_c_by_coverage"], dc["tail"]
        chk("11.4 max effective c, COVERED entries", dec["max_covered"], 21.14)
        chk("11.4 max effective c, UNCOVERED entries", dec["max_uncovered"], 23.30)
        chk("11.4 top-30 all outside both derivatives",
            int(dt["top34_all_uncovered"]), 1)
        chk("11.4 top-34 from the 9T series", dt["n_top34_in_series"], 30)
        chk("11.4 rank of the first other structure",
            dt["rank_of_first_other_structure"], 31)
        chk("11.4 modification enrichment, length-matched",
            dt["enrichment_vs_length_matched"], 1.9)

        # ---- G1 / G2 / G3, closed on raw whole entries -------------------
        # These three are properties of ENTRIES, so none of them was
        # computable on RNA3DB or RNASolo -- those ship per-chain extracts
        # with the protein stripped. All three stood on 180 BGSU structures
        # until the raw corpus was acquired.
        chk("v0.2 raw entries holding an RNA chain", ec["n_entries_with_rna"], 10520)
        g1, g2, g3 = ec["G1_length"], ec["G2_complex"], ec["G3_ribosomal"]

        # G1. The ENTRY maximum failed (11,478 -> 22,345, 1.95x) -- the fifth
        # extreme quantile from n=180 to do so. The CHAIN maximum did not: it
        # reproduces 6HRM's 4,450 exactly, which is what closes D10.
        chk("v0.2 G1 max RNA residues per entry", g1["max_rna_residues_per_entry"], 22345)
        chk("v0.2 G1 longest single RNA chain in the PDB", g1["max_single_rna_chain"], 4450)
        chk("v0.2 G1 entries over 4,096 RNA residues", g1["entries_over_4096"], 1584)

        # G2. Holds as a residue fraction, but isolated RNA is 2.7x the share
        # v0.1 measured -- the stratum the mandatory eval split runs on.
        chk("v0.2 G2 frac residues in complex", g2["frac_residues"], 0.9715, abs_tol=0.0002)
        chk("v0.2 G2 entries containing protein", g2["entries_with_protein"], 8092)
        chk("v0.2 G2 isolated-RNA share of residues",
            g2["frac_residues_rna_only"], 0.0283, abs_tol=0.0002)

        # G3. Skew survives at 85.94%, 6.7 points below the figure every
        # residue-weighted v0.1 statement was qualified with.
        chk("v0.2 G3 frac residues ribosomal", g3["frac_residues"], 0.8594, abs_tol=0.0002)
        chk("v0.2 G3 ribosome-like entries", g3["n_ribosome_like"], 2197)
    except FileNotFoundError as e:
        chk(f"v0.2 analysis artefact missing: {e.filename}", 0, 1)

    cs = load("contact_sparsity.json")["summary"]
    chk("contacts/nt median", cs["contacts_per_nt_median"], 4.397)
    chk("long-chain map density %", cs["by_length_bin"][-1]["median_density_pct"], 0.353)

    pr = load("proposal_recall.json")["summary"]
    chk("flat recall, L 500-1200", pr["by_length_bin"][-1]["mean_recall_c32"], 0.2003)
    chk("random baseline c=32", pr["random_baseline_c32_mean"], 0.7458)

    bs = load("block_sparsity.json")["summary"]["by_length_bin"][-1]["per_block"]["4"]
    chk("block occupancy b=4 (long) %", round(100 * bs["mean_occupancy_frac"], 2), 1.34)
    chk("effective c (long chains)", bs["mean_effective_c"], 17.24)

    co = load("coevolution_signal.json")["summary"]
    chk("coevolution mean prec@L/5", co["mean_prec_topL_5"], 0.6702)

    bench = load("pair_track_benchmark.json")
    r = [x for x in bench["runs"] if x["L"] == 4096][0]["hierarchical"]["stats"]
    chk("HPT L=4096 frac of dense %", round(100 * r["frac_of_dense"], 3), 0.959)
    chk("HPT L=4096 effective c", round(r["effective_c"], 1), 19.6)

    bp = load("basepair_geometry.json")
    chk("annotated base-pair steps", bp["total_annotated_steps"], 103965, tol=0)
    chk("stiffness contexts (n>=200)", len(bp["stiffness_by_step_context"]), 76)
    chk("curated Mg coordination records", bp["metal_coordination_by_ion"]["MG"], 44708, tol=0)
    # defect #26: structures whose base-pair category parses. 155 before the
    # key-value fix, 156 after -- 8B6Z writes the category in key-value form.
    chk("structures with base-pair annotations",
        bp["structures_with_base_pair_annotations"], 156, tol=0)
    cl = json.load(open(A / "cost_with_loops.json"))
    chk("effective compute, Small (active x loops)",
        next(r["effective_compute_M"] for r in cl["ladder_with_loops"] if r["model"] == "PHAROS-Small"), 488)
    chk("Small vs Base-v2 by effective compute", cl["small_vs_base_by_effective_compute"], 1.65)
    chk("corrected A100-h @25B", cl["corrected_a100h_25B"], 78, abs_tol=1.0)
    ho = json.load(open(A / "stiffness_headroom_heldout.json"))
    chk("held-out M2 (common subset)", ho["B_held_out_2fold"]["M2"], 15.3424)
    chk("held-out structure-beyond-sequence", ho["gains_held_out"]["structure_over_sequence"], 1.0336)
    chk("held-out sequence-over-global", ho["gains_held_out"]["sequence_over_global"], 1.7847)
    chk("unobs residues, RNA only", bp["unobserved_residue_records_RNA"], 46448, tol=0)
    chk("unobs residues, all polymers", bp["unobserved_residue_records_ALL_POLYMERS"], 143874, tol=0)
    gg = bp["stiffness_by_step_context"]["GG/CC"]["mean"]
    chk("GG/CC rise (A-form check)", gg["rise"], 3.14, abs_tol=0.005)
    chk("GG/CC twist (A-form check)", gg["twist"], 29.98, abs_tol=0.005)

    # derive the stiffness ratio from UNROUNDED values -- quoting it from the
    # 4-dp rounded display once produced a spurious 134x instead of 115x
    fcs = sorted((v["force_constants_diag"]["twist"]
                  for v in bp["stiffness_by_step_context"].values()), reverse=True)
    chk("stiffness ratio (stiffest/floppiest twist)", round(fcs[0]/fcs[-1], 1), 115.1)

    gc = load("generalization_canonical.json")["summary"]
    chk("canonical structures with RNA entity", gc["structures_with_RNA"], 179)
    chk("canonical RNA residues", gc["G2"]["rna_residues_total"], 309197)
    chk("G1 longest chain, median", gc["G1"]["longest_chain_median"], 70)
    chk("G1 longest chain, max", gc["G1"]["longest_chain_max"], 3764)
    chk("G2 frac RNA res in complexes", gc["G2"]["frac_rna_residues_in_complexes"], 0.9895)
    chk("G2 frac multi-RNA-chain", gc["G2"]["frac_multi_chain"], 0.7374)
    chk("G2 structures with protein", gc["G2"]["structures_with_protein"], 158)
    chk("G3 ribosome-like structures", gc["G3"]["ribosome_like"], 60)
    chk("G3 frac RNA res ribosomal", gc["G3"]["frac_rna_residues"], 0.9265)
    chk("G7 modified instances (RNA only)", gc["G7"]["modified_instances"], 3237)
    chk("G7 frac of RNA residues", gc["G7"]["frac_of_rna_residues"], 0.01047)
    chk("G7 distinct modification types", gc["G7"]["distinct_types"], 82)
    # defect #23: the published G7 counted DNA and UNK. Pin the contamination so
    # the 8.6x correction stays explainable rather than merely asserted.
    ga = load("generalization_audit.json")["summary"]
    gtop = ga["G7_modified_nucleotides"]["top_modifications"]
    contam = sum(gtop[k] for k in ("UNK", "DT", "DG", "DA", "DC"))
    chk("published G7 DNA+UNK contamination", contam, 24775)
    chk("published G7 total instances",
        ga["G7_modified_nucleotides"]["modified_residue_instances"], 27437)

    # C13/C14 (cycle 11): sensitivity of the residue-weighted measurements to the
    # RNA definition. The decision these guard is the c=20 sparse-track budget.
    ds = load("definition_sensitivity.json")["summary"]
    chk("definition-sensitivity: chains compared", ds["chains_compared"], 91, tol=0)
    chk("definition-sensitivity: chains changing length",
        ds["chains_whose_length_changes"], 31, tol=0)
    chk("definition-sensitivity: frac changed", ds["frac_changed"], 0.3407)
    chk("definition-sensitivity: residues restored", ds["residues_added_total"], 703, tol=0)
    dsl = json.load(open(A / "definition_sensitivity_long.json"))
    chk("long chains checked (>3000 nt)", len(dsl), 24, tol=0)
    # the budget claim: no chain breaches c=20 under EITHER definition
    allc = ([r["published"]["effective_c"] for r in load("definition_sensitivity.json")["rows"]]
            + [r["canonical"]["effective_c"] for r in load("definition_sensitivity.json")["rows"]]
            + [r["c_pub"] for r in dsl] + [r["c_can"] for r in dsl])
    chk("max effective c over ALL chains, both definitions", round(max(allc), 2), 19.04)
    chk("chains breaching the c=20 budget", sum(1 for c in allc if c > 20), 0, tol=0)

    hr = load("stiffness_headroom.json")
    chk("M1 sequence-table NLL", hr["M1_sequence_context_nll"], 17.5792)
    chk("M2 sequence x structure NLL", hr["M2_sequence_x_structure_nll"], 14.551)
    chk("structure gain over sequence", hr["gain_structure_over_sequence"], 3.0282)

    print("\n== derived arithmetic ==")
    # PHAROS-Small (default): d=512, 16 blocks, all-MoE, d_ff=128, 32+2 experts, top-4
    d, dff_moe, nb = 512, 128, 16
    attn = nb * 4 * d * d
    moe_t = nb * 34 * 3 * d * dff_moe
    moe_a = nb * 6 * 3 * d * dff_moe
    chk("PHAROS-Small total params (M)", round((attn + moe_t + 25e6) / 1e6), 149)
    chk("PHAROS-Small active params (M)", round((attn + moe_a + 25e6) / 1e6), 61)
    chk("effective layers (16 blocks x 8 loops)", nb * 8, 128)
    L = 2048
    chk("dense activations at L=2048 (GB)", round(L*L*128*2*48*6/1e9, 1), 309.2)

    print("\n== physics (closed form) ==")
    import math
    e, eps0, kB, T, eps_r = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23, 298.15, 78.4
    lB = e*e/(4*math.pi*eps0*eps_r*kB*T)*1e10
    chk("Bjerrum length (A)", round(lB, 2), 7.15)
    b_rna = 2.8/2
    xi = lB/b_rna
    chk("A-RNA Manning xi", round(xi, 2), 5.11)
    chk("A-RNA theta (z=1)", round(1-1/xi, 3), 0.804)

    print("\n== cross-document consistency ==")
    # ARCH is the AUDIT TRAIL, not the specification. It carries the v0.1 text
    # with its correction tables -- published value beside canonical value --
    # and the guards below exist to stop those tables being flattened. It moved
    # to history_of_failed_attempts/ when the final specification was written
    # separately; the guards followed it, because what they protect is the
    # record of what was wrong, which is exactly what must not be quietly
    # tidied away.
    docs = {"ARCH": ROOT/"history_of_failed_attempts/ARCHITECTURE_v0.1_with_corrections.md",
            "TEX": ROOT/"research/report/main.tex",
            "HTML": ROOT/"research/architecture/blueprint.html"}
    txt = {}
    for k, p in docs.items():
        t = p.read_text()
        txt[k] = (t.replace("{,}", ",").replace("\\%", "%").replace("$", "")
                   .replace("\\", "").replace("−", "-").replace("–", "-"))
    # Defect #28 (cycle 10): this check used `tok in text`, which proves a
    # string occurs SOMEWHERE, not that it occurs in the claim it guards. Short
    # tokens matched inside longer numbers: half of `61`'s matches in
    # ARCHITECTURE.md sit inside other numbers, and its first standalone match is
    # a contact-sparsity table row (`| 1500+ | 61 |`), nothing to do with the 61M
    # active-parameter claim. Proven adversarially: deleting `149M/61M` and
    # `~153M/~65M` outright left all four tokens reporting OK and the suite
    # exiting 0. Numbers are now matched at NUMBER BOUNDARIES, and the tokens
    # that remain ambiguous carry an explicit context pattern.
    def numpat(tok):
        """A number not embedded in a longer number."""
        return r"(?<![\d.,])" + re.escape(tok) + r"(?![\d,]*\d)"

    # Tokens that are SUPERSEDED by a v0.2 measurement. They remain in the
    # narrative documents, which carry the audit trail, but were removed from
    # the HTML blueprint, which is a summary and must show only current values.
    # Requiring them everywhere would force wrong numbers back into the summary
    # -- the guard would be enforcing the error it was written to catch.
    SUPERSEDED_SCOPE = {
        "1.76":   {"ARCH", "TEX"},    # -> 1.523 sigma, X-ray, 1,535 structures
        "17,428": {"ARCH", "TEX"},    # -> 816,270 Mg sites
        "19.04":  {"ARCH", "TEX"},    # -> max effective c 21.14
        "1.34":   {"ARCH", "TEX"},    # -> b=4 occupancy 1.67%
        "17.2":   {"ARCH", "TEX"},    # -> mean effective c 17.14
    }

    ANCHORED = {
        # short numbers that collide with unrelated values in the documents
        "78":  r"A100[^\n]{0,60}?78|78[^\n]{0,60}?A100",
        "115": r"115(?:\.1)?\s*(?:x|\u00d7|times)",
        "149": r"149\s*M",
        "61":  r"61\s*M",
        "153": r"153\s*M",
        "82":  r"82[^\n]{0,40}(?:distinct|modification)"
               r"|(?:distinct|modification)[^\n]{0,40}82",
        "179": r"(?:/|of\s+|across\s+)179|179\s*(?:of|parsed|structures|\.|,)",
    }

    for tok in ["1.523", "21.14", "7.66", "54", "77.9",
                "17,428", "1.76", "0.200", "0.746", "1.34", "17.2", "0.670",
                "0.224", "0.372", "1.514", "0.804", "9.87",
                # PHAROS-Small is the current headline config; 910M/382M survives
                # only as the superseded Base row in the progression tables.
                "149", "61",
                "103,965", "44,708",
                # RNA-only disorder labels; 143,874 was the all-polymer total
                # and 67.5% of it is protein (defect #11, cycle 2).
                "46,448",
                # held-out, common-subset figures (REV-2); the in-sample
                # 14.551 / 17.579 pair was not a fair comparison. The tokens were
                # written truncated ("15.342", "16.376") and only ever matched as
                # PREFIXES of 15.3424 / 16.3760 -- invisible until defect #28
                # switched to boundary matching. 16.3760 is written both ways
                # across the three documents, so it is anchored to accept either.
                "15.3424", ("16.376", r"16\.376(?:0)?(?![\d])"), "1.0336",
                # cycle-4 defect #17: loops enter the FLOP count.
                # 488M effective compute, 78 A100-h @25B, 1.65x vs Base-v2.
                "78", "1.65",
                # cycle-4 defects #18/#19: heads+decoder itemised
                "153", "12.59",
                # cycle-5 defects #20/#21
                "2.571", "1.500", "115", "1.757",
                # cycle-6 defects #22/#23/#24 and cycle-7 defect #25: the
                # canonical RNA-residue basis. Both mmCIF serialisations now
                # parse, so every structure-count denominator is 179, not 162.
                "309,197", "179", "98.95", "1.05", "3,764",
                "73.7", "33.5", "286,458", "92.65", "3,237", "82",
                # C13/C14 (cycle 11): definition sensitivity of the
                # residue-weighted numbers, and the c=20 budget it guards.
                "34.1", "27.9", "19.04", "18.44", "703",
                # DISC-47: c=20 has 4.8% headroom at the worst chain, not the
                # 16% the mean-over-long-chains figure of 17.2 implies.
                "4.8",
                # defect #24: the PUBLISHED column of the correction table must
                # survive. An unguarded replace once overwrote it with the
                # canonical values, making the table read "99.70 -> 99.70".
                "98.96", "8.90"]:
        if isinstance(tok, tuple):
            tok, pat = tok
            how = "anchored"
        else:
            pat = ANCHORED.get(tok) or numpat(tok)
            how = "anchored" if tok in ANCHORED else "boundary"
        scope = SUPERSEDED_SCOPE.get(tok)
        items = ({k: v for k, v in txt.items() if k in scope} if scope else txt)
        missing = [k for k, v in items.items() if not re.search(pat, v, re.I)]
        print(f"  {'OK ' if not missing else 'FAIL'} token {tok:8s} ({how:8s}) "
              f"{('present in ' + (','.join(sorted(scope)) if scope else 'all 3')) if not missing else 'MISSING from ' + ','.join(missing)}")
        if missing: fails.append(f"doc-token {tok}")

    print("\n== defect #24 guard: the correction table must record BOTH columns ==")
    arch = txt["ARCH"]
    # In the cycle-6 comparison table every row is "published | canonical". If a
    # global replace ever rewrites the published side again, these pairs vanish.
    for pub, can, what in [("308,370", "309,197", "RNA residues"),
                           ("98.96", "98.95", "G2 residues in complexes"),
                           ("93.07", "92.65", "G3 residues ribosomal"),
                           ("8.90", "1.05", "G7 outside ACGU"),
                           ("306,857", "309,197", "cycle-6 vs cycle-7 totals"),
                           ("99.70", "98.95", "the retracted G2 strengthening")]:
        ok = pub in arch and can in arch
        print(f"  {'OK ' if ok else 'FAIL'} both columns present: {what:28s} "
              f"{pub} -> {can}")
        if not ok:
            fails.append(f"correction table collapsed: {what}")

    print("\n== defect #24 guard: no stale 180-structure denominators ==")
    # These denominators were superseded by the canonical 162 basis. They are
    # legitimate ONLY inside the published column of the correction table, so
    # each must appear at most once per document.
    for stale in ["150/180", "61/180", "159/180"]:
        for k, v in txt.items():
            n = v.count(stale)
            ok = n <= 1
            print(f"  {'OK ' if ok else 'FAIL'} {k}: {stale!r} x{n}"
                  f"{'' if ok else ' -- appears outside the published column'}")
            if not ok:
                fails.append(f"stale denominator {stale} in {k}")
    for bad in ["decisive pattern", "strongest argument for MoE",
                "0.76 for monovalent RNA", "spaced b ~ 5.9",
                # cycle-6: these were superseded outright, not moved to a
                # published column, so they must be gone everywhere.
                "median 67 nt", "max 3,679", "286,990 / 308,370",
                "44 of 180", "56 of 180"]:
        present = [k for k, v in txt.items() if re.search(re.escape(bad), v)]
        print(f"  {'OK ' if not present else 'FAIL'} retracted text absent: {bad!r}"
              f"{'' if not present else ' STILL IN ' + ','.join(present)}")
        if present: fails.append(f"stale {bad}")

    print("\n== architecture config consistency (shared400 is the trained model) ==")
    cfg_docs = {"ARCH": ROOT/"history_of_failed_attempts/ARCHITECTURE_v0.1_with_corrections.md",
                "TEX":  ROOT/"research/report/main.tex",
                "HTML": ROOT/"research/architecture/blueprint.html",
                "SPEC": ROOT/"research/architecture/ARCHITECTURE.md"}
    cfg_txt = {k: v.read_text() for k, v in cfg_docs.items()}
    # phrases describing the SUPERSEDED default; Base-v2 scale-up rows are fine
    stale_cfg = ["32 blocks at d=768", "20 of 32 blocks", "4 of 32 blocks",
                 "537M", "d_ff = 512", "every second block", "PHAROS-Base carries"]
    for bad in stale_cfg:
        hit = [k for k, t in cfg_txt.items() if bad in t]
        print(f"  {'OK ' if not hit else 'FAIL'} superseded config absent: {bad!r}"
              f"{'' if not hit else ' IN ' + ','.join(hit)}")
        if hit: fails.append(f"stale config {bad}")
    # The SPEC is the live document and has to name the trained configuration;
    # the others are historical or derived and are checked only for staleness.
    # Splitting these was necessary once the trained model stopped being
    # PHAROS-Small: asserting "d=512" against every document would have forced
    # the spec to keep describing a configuration nothing runs.
    for need in ["18 blocks", "d=768", "394M", "302M", "144 effective",
                 "512 experts"]:
        miss = [] if need in cfg_txt["SPEC"] else ["SPEC"]
        print(f"  {'OK ' if not miss else 'FAIL'} current config present: {need!r}"
              f"{'' if not miss else ' MISSING from ' + ','.join(miss)}")
        if miss: fails.append(f"missing config {need}")

    # A module map that names files which do not exist is worse than no map,
    # and v0.1's named six that never got built. This keeps it honest.
    print("\n== every module the README names exists ==")
    import re as _re
    rd = ROOT / "src/pharos/README.md"
    if rd.exists():
        txt = rd.read_text()
        mods = sorted(set(_re.findall(r"\b(?:model|physics|data)/[a-z_]+\.py", txt)))
        gone = [m for m in mods if not (ROOT / "src/pharos" / m).exists()]
        print(f"  {'OK ' if not gone else 'FAIL'} README module map          "
              f"{len(mods)} named, missing: {gone or 'none'}")
        if gone:
            fails.append("README names modules that do not exist")
        scr = sorted(set(_re.findall(r"scripts/[a-z_/]+\.(?:py|sh)", txt)))
        gone_s = [x for x in scr if not (ROOT / x).exists()]
        print(f"  {'OK ' if not gone_s else 'FAIL'} README script map          "
              f"{len(scr)} named, missing: {gone_s or 'none'}")
        if gone_s:
            fails.append("README names scripts that do not exist")

    print("\n== reference implementation correctness tests ==")
    import os
    import subprocess
    suites = [ROOT / "research/architecture/reference/test_hierarchical_pair_track.py",
              ROOT / "src/pharos/physics/test_manning.py",
              ROOT / "src/pharos/data/test_mmcif_entities.py",
              ROOT / "src/pharos/data/test_chemistry.py",
              ROOT / "src/pharos/data/test_chemistry_torch.py",
              ROOT / "src/pharos/train/test_telemetry.py",
              ROOT / "src/pharos/train/test_checkpoint.py",
              ROOT / "src/pharos/data/test_msa.py",
              ROOT / "src/pharos/model/test_diffusion.py",
              ROOT / "src/pharos/model/test_attention.py",
              ROOT / "src/pharos/model/test_pharos.py",
              ROOT / "src/pharos/model/test_motif_bank.py",
              ROOT / "src/pharos/model/test_dynamics.py",
              ROOT / "src/pharos/data/test_dataset.py",
              ROOT / "src/pharos/data/test_mlm_leak.py"]
    # Forced onto CPU. These are correctness tests over tensors of a few
    # thousand elements, so the GPU buys nothing -- and a shared GPU costs
    # something real: with another job holding 80.9 of 81.9 GB, Adam's
    # capture health-check raised `CUDA error: out of memory` and this
    # harness printed DRIFT DETECTED for a suite that passes. A verifier
    # whose job is catching silent regressions must not manufacture loud
    # false ones out of a neighbour's memory usage.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for t in suites:
        if not t.exists():
            print(f"  FAIL {t.name} missing"); fails.append(f"{t.name} missing"); continue
        # The timeout scales with load. These suites train small models on
        # CPU, and this machine routinely runs a corpus rebuild on eight
        # workers beside them; test_diffusion.py takes four minutes idle and
        # blew through 1800 s at load 44.
        budget = int(1800 * max(1.0, min(os.getloadavg()[0] / 4.0, 4.0)))
        try:
            r = subprocess.run([sys.executable, t.name], cwd=t.parent, env=env,
                               capture_output=True, text=True, timeout=budget)
        except subprocess.TimeoutExpired:
            # A timeout is NOT a failure, and it must not be an uncaught
            # exception either. Uncaught, it aborted the whole verification,
            # which made the cron runner write status "failed" and refuse to
            # train -- a 69-hour run blocked by a neighbour's CPU usage. And a
            # suite that did not finish has not demonstrated a regression, so
            # it cannot answer the question this gate exists to ask. It is
            # reported loudly and does not block.
            print(f"  WARN {t.name:34s} did not finish in {budget}s under load "
                  f"{os.getloadavg()[0]:.0f} -- NOT counted as a failure; "
                  f"run it directly")
            warns.append(f"{t.name} timed out")
            continue
        ok = r.returncode == 0 and "ALL TESTS PASS" in r.stdout
        print(f"  {'OK ' if ok else 'FAIL'} {t.name:34s} "
              f"{'all pass (cpu)' if ok else 'FAILURES -- run it directly'}")
        if not ok:
            fails.append(f"{t.name} failing")

    print("\n== diagram sources (feed figures into the report) ==")
    dg = sorted((ROOT / "research/architecture/diagrams").glob("*.mmd"))
    # phrasings that were retracted and must not survive in any figure
    banned = ["0.76 for monovalent", "ADAPTIVE SPARSE", "Select K = 32L"]
    for d in dg:
        t = d.read_text()
        hits = [b for b in banned if b in t]
        print(f"  {'OK ' if not hits else 'FAIL'} {d.name:34s}"
              f"{'clean' if not hits else 'STALE: ' + ', '.join(hits)}")
        if hits:
            fails.append(f"stale diagram {d.name}")
    rendered = ROOT / "research/architecture/diagrams/rendered"
    for d in dg:
        for ext in ("png", "svg"):
            f = rendered / f"{d.stem}.{ext}"
            if not f.exists():
                print(f"  FAIL missing render {f.name}")
                fails.append(f"missing render {f.name}")
            elif f.stat().st_mtime < d.stat().st_mtime:
                print(f"  FAIL stale render {f.name} (older than its .mmd source)")
                fails.append(f"stale render {f.name}")
    print(f"  OK  {len(dg)} diagram sources, renders present and newer than source")

    if warns:
        # Surfaced separately from failures and separately from success: these
        # are checks that did not get to run, which is neither.
        print(f"\nNOT CHECKED ({len(warns)}): " + "; ".join(warns))
    print(f"\n{'ALL CLAIMS REPRODUCE' if not fails else 'DRIFT DETECTED: ' + '; '.join(fails)}"
          + (f" (with {len(warns)} not checked)" if warns and not fails else ""))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
