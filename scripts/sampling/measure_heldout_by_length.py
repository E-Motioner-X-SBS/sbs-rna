"""Score three checkpoints per LENGTH BAND on a corpus-wide sample.

The clean held-out sample is 1,024 sequences drawn from eldors_c020_shard0004 --
one shard of one length band, mean 185 nt. Its readings have swung 1.650 ->
1.789 -> 1.953 bits over 500 steps while training CE alternated between ~0.35
and ~1.25 with the shard. If the swing is the model drifting toward and away
from c020's distribution rather than getting better and worse, a sample that
spans the corpus's length range will not swing with it.
"""
import sys, itertools, numpy as np, torch, json
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from pathlib import Path
import importlib
pm = importlib.import_module('pretrain_mlm')
ev = importlib.import_module('eval_mlm_checkpoint')
from pharos.model.pharos import PharosConfig, Pharos
dev = torch.device('cuda')

corp=['data/derived/parquet_starter','data/derived/parquet_mars']
res_files = pm.heldout_files(corp)
# a length-stratified sample over the RESERVED shards
pool=[]
for s in itertools.islice(pm.iter_sequences(res_files,20,1024,None,
                          np.random.default_rng(3), n_interleave=8), 200000):
    pool.append(s)
bands=[(20,80),(80,160),(160,320),(320,640),(640,1025)]
rng=np.random.default_rng(1234)
strat={}
for lo,hi in bands:
    c=[s for s in pool if lo<=len(s)<hi]
    if len(c)<64: continue
    idx=rng.choice(len(c), min(256,len(c)), replace=False)
    strat[f"{lo}-{hi-1}"]=[c[int(i)] for i in idx]
print("band sizes:", {k:len(v) for k,v in strat.items()}, flush=True)

cfg=PharosConfig.shared400(); m=Pharos(cfg).to(dev).eval()
out={}
for step in (9000, 9250, 9500):
    ck=f'data/derived/checkpoints/heldout/step{step}.pt'
    st=torch.load(ck, map_location=dev, weights_only=False)
    m.load_state_dict({k.replace('_orig_mod.',''):v for k,v in st['model'].items()}, strict=False)
    row={}
    allseq=[]
    for band, seqs in strat.items():
        r=ev.score(m, pm, seqs, dev, 8192, 2, 1234)
        row[band]=round(float(r['bits']),4); allseq+=seqs
    r=ev.score(m, pm, allseq, dev, 8192, 2, 1234)
    row['ALL']=round(float(r['bits']),4)
    out[step]=row
    print(f"step {step}: " + "  ".join(f"{k} {v:.4f}" for k,v in row.items()), flush=True)
Path('data/samples/analysis/heldout_by_length.json').write_text(json.dumps(out, indent=1))
print("\nswing per band (9500 - 9000):")
for band in list(strat)+['ALL']:
    print(f"  {band:10s} {out[9500][band]-out[9000][band]:+.4f} bits")
