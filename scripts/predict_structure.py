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
            seed: int = 0) -> tuple[np.ndarray, List[np.ndarray], List[float]]:
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
        out = model(tok, mod, ch, msk, feats=RouterFeatures(
            length=msk.sum(1).float()), n_loops=n_loops, mlm=False)
    single = out["hidden"].float()

    head = model.heads.structure
    draws, clashes = [], []
    for s in range(n_samples):
        g = torch.Generator(device="cpu").manual_seed(seed + s)
        noise = torch.randn(1, L, N_ATOM, 3, generator=g).to(device)
        x = head.sample(single, None, msk, n_steps=n_steps,
                        generator=None) if noise is None else _sample_with(
            head, single, msk, n_steps, noise)
        c = x[0].float().cpu().numpy()
        draws.append(c)
        clashes.append(clash_score(c))
    best = int(np.argmin(clashes))
    return draws[best], draws, clashes


def _sample_with(head, single, mask, n_steps, noise):
    """`head.sample` seeded from a caller-supplied noise draw.

    The head seeds itself from the global RNG, which makes a prediction run
    unreproducible and makes several draws correlated with whatever ran before
    them. Threading the initial noise in is the only way to get five
    independent, reproducible samples.
    """
    cfg = head.cfg
    steps = head.sigmas(n_steps or cfg.n_steps, single.device, torch.float32)
    x = noise * steps[0]
    for i in range(len(steps) - 1):
        s, s_next = steps[i], steps[i + 1]
        d = head.denoise(x, s.expand(x.shape[0]), single, None, mask)
        deriv = (x - d) / s.clamp(min=1e-8)
        x_next = x + (s_next - s) * deriv
        if s_next > 0:                       # Heun's second-order correction
            d2 = head.denoise(x_next, s_next.expand(x.shape[0]), single,
                              None, mask)
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
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = getattr(PharosConfig, args.size)()
    model = Pharos(cfg).to(device).eval()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = state.get("model", state)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[predict] {args.ckpt.name}: {len(sd):,} tensors, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        # head 3 untrained would silently emit noise that still scores
        print(f"[predict] WARNING missing: {missing[:4]}"
              f"{' ...' if len(missing) > 4 else ''}")

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
        best, draws, clashes = predict(model, seq, device,
                                       n_samples=args.n_samples,
                                       n_steps=args.n_steps,
                                       n_loops=args.n_loops)
        write_pdb(args.out / f"{t.name}.pdb", best, seq)
        if args.write_all:
            for k, d in enumerate(draws, 1):
                write_pdb(args.out / f"{t.name}_m{k}.pdb", d, seq)
        print(f"  {t.name}: L={len(seq):4d} {len(draws)} draws, "
              f"clash {min(clashes):.3f}-{max(clashes):.3f}, wrote best")
        done += 1

    print(f"\n[predict] {done} written, {skipped} skipped -> "
          f"{args.out.relative_to(ROOT)}")
    print(f"[predict] score with: python scripts/eval_blind_tests.py "
          f"--pred {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
