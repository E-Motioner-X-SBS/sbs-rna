#!/usr/bin/env python3
"""Sequence -> 3D backbone, written as PDB. The link between model and benchmark.

`eval_blind_tests.py` scores `<target>.pdb` files and had nothing to score:
the model could compute a loss but could not produce a structure a scorer
would accept. This closes that loop, so a checkpoint can be put through the
RNA-Puzzles field without a human in the middle.

    python scripts/predict_structure.py --ckpt data/derived/checkpoints/pharos.pt \
        --targets rna_puzzles --out data/samples/analysis/predictions
    python scripts/eval_blind_tests.py --pred data/samples/analysis/predictions

**Sampling, not argmax.** Head 3 is a denoiser, so a prediction is a *draw*
from a distribution over structures, and one draw is not a prediction of the
mode. `--n-samples` draws several and `--select` picks between them; the
default picks the sample with the fewest self-clashes, which is the only
quality signal available without the answer. RNA-Puzzles itself allows five
submissions per target for exactly this reason, and `--write-all` emits them
in that form.

**A prediction is written even when it is bad.** A sampler that has not
learned will produce a tangle, and the honest thing is to write it and let the
metric say so, rather than to filter and report a flattering subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharos.data.chemistry import chain_chemistry                   # noqa: E402
from pharos.data.vocab import encode_chain                          # noqa: E402
from pharos.eval.blind_tests import all_targets                     # noqa: E402
from pharos.model.diffusion import BOND_C4_N, BOND_P_P               # noqa: E402
from pharos.model.moe import LENGTH_BIN_MAX                         # noqa: E402
from pharos.eval.metrics import clash_score                         # noqa: E402
from pharos.eval.structure import read_structure                    # noqa: E402
from pharos.model.diffusion import N_ATOM                           # noqa: E402
from pharos.model.moe import RouterFeatures                         # noqa: E402
from pharos.model.pharos import Pharos, PharosConfig                # noqa: E402

#: PDB atom names in the model's own order. The third is the GLYCOSIDIC
#: nitrogen, which is N9 on a purine and N1 on a pyrimidine -- writing N9 for
#: every residue makes the file unreadable for pyrimidines, and
#: `read_structure` silently reports them as missing atoms. The first version
#: of this did exactly that and round-tripped only 43% of residues, which is
#: the purine fraction.
ATOM_NAMES = ("P", "C4'", None)
_PURINE = {"A", "G"}


def _glycosidic(base: str) -> str:
    return "N9" if base.upper() in _PURINE else "N1"


def write_pdb(path: Path, coords: np.ndarray, seq: str,
              mask: Optional[np.ndarray] = None) -> None:
    """Write `(L, 3, 3)` backbone coordinates as a minimal PDB.

    Only the three atoms the model predicts are written. A scorer that expects
    a full-atom model will see a backbone trace, which is what this is -- the
    alternative, inventing the other eighteen atoms per residue from ideal
    geometry, would make the file look like a complete prediction and would be
    scored as one.
    """
    lines: List[str] = []
    n = 0
    for i, base in enumerate(seq):
        if mask is not None and not bool(mask[i]):
            continue
        for k, name in enumerate(ATOM_NAMES):
            name = name or _glycosidic(base)
            x, y, z = (float(v) for v in coords[i, k])
            n += 1
            lines.append(
                f"ATOM  {n:5d} {name:<4s}{base:>3s} A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                f"{name[0]:>2s}")
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def predict(model: Pharos, seq: str, device, *, n_samples: int = 5,
            n_steps: int = 50, n_loops: Optional[int] = None,
            seed: int = 0) -> tuple[np.ndarray, List[np.ndarray], List[float],
                                    List[tuple]]:
    """Draw `n_samples` backbones for one sequence; return the chosen one too."""
    comps = list(seq)
    tokens, mods = encode_chain(comps)
    chem = chain_chemistry(comps)
    L = len(seq)

    tok = torch.as_tensor(tokens, dtype=torch.long, device=device)[None]
    mod = torch.as_tensor(mods, dtype=torch.long, device=device)[None]
    ch = torch.as_tensor(chem, dtype=torch.float32, device=device)[None]
    msk = torch.ones(1, L, dtype=torch.bool, device=device)

    with torch.autocast(device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        # The router conditioning must be the one TRAINING built, or the
        # experts are selected from a vector the model never saw. Stage 5
        # sets length, in_complex and the pooled chemistry; this passed
        # length alone, leaving 6 of the 14 conditioning dims at zero and
        # moving the representation by 35% of signal scale on an untrained
        # model. `chem` is already computed four lines up, so the omission
        # bought nothing. `in_complex` is genuinely unknown for a bare
        # sequence and stays absent -- that one is a real unknown, not an
        # oversight, and it reads as zero exactly as §5.3 specifies.
        out = model(tok, mod, ch, msk, feats=RouterFeatures(
            length=msk.sum(1).float(),
            chem_summary=(ch.sum(1) / msk.sum(1, keepdim=True).clamp(min=1))[:, :5],
            # the SAME binning stage 5 trained with. Left at the default,
            # 800, 1,200 and 2,000 nt chains each land one bin off.
            length_bin_max=LENGTH_BIN_MAX,
        ), n_loops=n_loops, mlm=False)
    single = out["hidden"].float()

    head = model.heads.structure
    # The SAME pair features stage 5 trains against -- `train_pharos.py`
    # builds `model.diff_pair(hidden, cv_dense)` and passes it to
    # `structure.loss`. Coevolution is left at zero here because a blind
    # target has no MSA in this path and `diff_pair.coev` is zero-init, so
    # absent and zero are the same tensor; the relative-position embedding,
    # which is the part the backbone needs, is built from the length.
    with torch.no_grad():
        # `msk` explicitly, though this path is batch-1 and unpadded.
        # The triangle update (finding 109) sums over every k, so a
        # pad left unmasked there contributes to every real pair --
        # and passing the mask that is already in scope costs nothing.
        pair = model.diff_pair(out["hidden"], None, msk)
    draws, clashes, bonds = [], [], []
    for s in range(n_samples):
        g = torch.Generator(device="cpu").manual_seed(seed + s)
        noise = torch.randn(1, L, N_ATOM, 3, generator=g).to(device)
        x = _sample_with(head, single, pair, msk, n_steps, noise)
        c = x[0].float().cpu().numpy()
        draws.append(c)
        clashes.append(clash_score(c))
        bonds.append(bond_violation(c))
    # Selecting on clash ALONE is inert exactly when it matters: an exploded
    # draw has no non-bonded neighbours to clash with, so every draw scores
    # 0.000 and `argmin` returns draw 0 whatever the geometry. Rank on the
    # bonded violation first, which an exploded draw cannot win.
    best = int(min(range(len(draws)), key=lambda i: (bonds[i][0], clashes[i])))
    return draws[best], draws, clashes, bonds


def bond_violation(c: np.ndarray) -> tuple[float, float, float]:
    """`(violation fraction, median C4'-N, median consecutive P-P)` in angstrom.

    `clash_score` excludes pairs within one residue of each other, which is
    right for a non-bonded clash and leaves it blind to the BONDED geometry.
    A draw from an undertrained denoiser came back `clash 0.000` with a
    consecutive P-P median of 40.35 A, a 118x122x109 A bounding box and two
    atoms 0.59 A apart: a structure that cannot exist, reported as clean,
    because every impossible distance was one the clash metric skips.

    The bands are the ones the TRAINING loss already penalises
    (`BOND_C4_N`, `BOND_P_P`), so inference is checked against the same
    physics it was fitted to rather than against a second opinion.
    """
    cn_mu, cn_tol = BOND_C4_N
    pp_mu, pp_tol = BOND_P_P
    cn = np.linalg.norm(c[:, 1] - c[:, 2], axis=-1)          # C4' - glycosidic N
    pp = np.linalg.norm(c[1:, 0] - c[:-1, 0], axis=-1)       # P(i) - P(i+1)
    bad = int((np.abs(cn - cn_mu) > cn_tol).sum() + (np.abs(pp - pp_mu) > pp_tol).sum())
    return (bad / max(len(cn) + len(pp), 1),
            float(np.median(cn)), float(np.median(pp)))


def _sample_with(head, single, pair, mask, n_steps, noise):
    """`head.sample` seeded from a caller-supplied noise draw.

    The head seeds itself from the global RNG, which makes a prediction run
    unreproducible and makes several draws correlated with whatever ran before
    them. Threading the initial noise in is the only way to get five
    independent, reproducible samples.

    **`pair` IS NOT OPTIONAL and this function used to hardcode `None`.**
    This is a second implementation of `head.sample`, written to control the
    seed, and it dropped an argument the original threads. The decoder is
    trained with `diff_pair(hidden, coev)`, whose whole reason for existing
    is the RELATIVE SEQUENCE POSITION embedding -- its own docstring says
    "without it the decoder has to infer every geometric relationship from
    per-residue embeddings". Sampling with `None` therefore asked the model
    to build a chain without telling it which residues are adjacent, and it
    produced exactly that: consecutive phosphates at 16-19 A against a true
    5.95, on all seventeen RNA-Puzzles targets, while the training-time bond
    violation looked fine because training had the features.

    Two implementations of one thing, and the public one was the broken one
    -- the same shape as the `pair_index` defect in `Pharos.forward`.
    """
    cfg = head.cfg
    steps = head.sigmas(n_steps or cfg.n_steps, single.device, torch.float32)
    x = noise * steps[0]
    for i in range(len(steps) - 1):
        s, s_next = steps[i], steps[i + 1]
        d = head.denoise(x, s.expand(x.shape[0]), single, pair, mask)
        deriv = (x - d) / s.clamp(min=1e-8)
        x_next = x + (s_next - s) * deriv
        if s_next > 0:                       # Heun's second-order correction
            d2 = head.denoise(x_next, s_next.expand(x.shape[0]), single,
                              pair, mask)
            deriv2 = (x_next - d2) / s_next.clamp(min=1e-8)
            x_next = x + (s_next - s) * 0.5 * (deriv + deriv2)
        x = x_next
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--size", default="shared400")
    ap.add_argument("--targets", default="rna_puzzles",
                    choices=["rna_puzzles", "casp15", "casp16", "all"])
    ap.add_argument("--target", default=None, help="one target by name")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/samples/analysis/predictions")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=1024,
                    help="skip longer targets; sampling is O(L^2) per step")
    ap.add_argument("--write-all", action="store_true",
                    help="also write <target>_m<k>.pdb for every draw, the "
                         "form RNA-Puzzles accepts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-unexpected", action="store_true",
                    help="sample even though the checkpoint carries tensors "
                         "this model has no slot for. Off by default because "
                         "that silently turns an architecture A/B into a "
                         "baseline run scored under the arm's name.")
    args = ap.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)

    # THE CHECKPOINT'S OWN CONFIG, not `--size`'s defaults. This used to
    # build `getattr(PharosConfig, args.size)()` and nothing else, which is
    # fine while every run shares one architecture and silently fatal the
    # moment one does not. An arm trained with `--triangle-layers 2` or
    # `--bidirectional-gdn` writes weights this model would have no slot
    # for; `strict=False` drops them into `unexpected`, and the line below
    # used to count `unexpected` and warn only about `missing`. The arm
    # would then be SAMPLED WITH THE BASELINE ARCHITECTURE and scored as
    # "no effect" for a change that was never switched on at inference.
    #
    # Same shape as finding 90, in the same file: a second implementation
    # of model construction that dropped something the first one threads.
    cfg = getattr(PharosConfig, args.size)()
    _saved = state.get("cfg")
    if isinstance(_saved, dict):
        _c = PharosConfig(**{k: v for k, v in _saved.items()
                             if k in PharosConfig.__dataclass_fields__})
        for k, v in _saved.items():
            if not hasattr(_c, k):
                setattr(_c, k, v)
        _diff = {k: (getattr(cfg, k, None), v) for k, v in _c.__dict__.items()
                 if getattr(cfg, k, None) != v}
        if _diff:
            print(f"[predict] architecture from the checkpoint, not --size: "
                  + ", ".join(f"{k} {a}->{b}" for k, (a, b) in
                              sorted(_diff.items())), flush=True)
        cfg = _c

    model = Pharos(cfg).to(device).eval()
    sd = state.get("model", state)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[predict] {args.ckpt.name}: {len(sd):,} tensors, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        # head 3 untrained would silently emit noise that still scores
        print(f"[predict] WARNING missing: {missing[:4]}"
              f"{' ...' if len(missing) > 4 else ''}")
    if unexpected:
        # An unexpected key is a weight the checkpoint HAS and this model has
        # nowhere to put. For an architecture A/B that is not a warning, it
        # is the experiment silently not happening.
        print(f"[predict] FATAL {len(unexpected)} unexpected tensors, i.e. "
              f"the checkpoint carries weights this model has no slot for: "
              f"{unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}",
              flush=True)
        if not args.allow_unexpected:
            print("[predict] refusing to sample a different architecture "
                  "than the one that was trained; pass --allow-unexpected "
                  "only if you know the extra tensors are inert.", flush=True)
            return 1

    targets = all_targets()
    if args.targets != "all":
        targets = [t for t in targets if t.source == args.targets]
    if args.target:
        targets = [t for t in targets if t.name == args.target]

    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for t in targets:
        try:
            ref = read_structure(t.reference)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {t.name}: unreadable reference ({e})"); skipped += 1; continue
        seq = ref.seq.replace("N", "A")       # the model has no N token
        if len(seq) > args.max_length:
            print(f"  {t.name}: {len(seq)} nt > --max-length, skipped")
            skipped += 1
            continue
        best, draws, clashes, bonds = predict(model, seq, device,
                                       n_samples=args.n_samples,
                                       n_steps=args.n_steps,
                                       n_loops=args.n_loops)
        write_pdb(args.out / f"{t.name}.pdb", best, seq)
        if args.write_all:
            for k, d in enumerate(draws, 1):
                write_pdb(args.out / f"{t.name}_m{k}.pdb", d, seq)
        bv, cn, pp = bonds[int(min(range(len(draws)),
                                   key=lambda i: (bonds[i][0], clashes[i])))]
        flag = "" if bv < 0.5 else "   <-- BACKBONE NOT CONNECTED"
        print(f"  {t.name}: L={len(seq):4d} {len(draws)} draws, "
              f"clash {min(clashes):.3f}-{max(clashes):.3f}, "
              f"bond-violation {bv:.3f} (C4'-N {cn:.2f} A, P-P {pp:.2f} A), "
              f"wrote best{flag}")
        done += 1

    # relative_to raises when --out is outside the repo, which is the normal
    # case for a scratch directory; a summary line must not be able to kill a
    # run whose real work has already succeeded
    def _rel(q: Path) -> str:
        try:
            return str(q.relative_to(ROOT))
        except ValueError:
            return str(q)
    print(f"\n[predict] {done} written, {skipped} skipped -> {_rel(args.out)}")
    print(f"[predict] score with: python scripts/eval_blind_tests.py "
          f"--pred {_rel(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
