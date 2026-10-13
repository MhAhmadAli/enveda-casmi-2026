"""Pool-window retrieval of one or several FPNet checkpoints (logits averaged) on the held-out, NP-panel and training
groups of a fold-safe split. Same groups and metric as the evaluation inside train_fpnet.py.

usage: python eval_ckpt.py --code <casmi code dir> --run <run dir with work/> --ckpt a.pt[,b.pt] [--eval_groups 3000] [--out file.json]
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--run', required=True); ap.add_argument('--ckpt', required=True)
ap.add_argument('--eval_groups', type=int, default=3000); ap.add_argument('--holdout', default='hold,np'); ap.add_argument('--out', default='')
args = ap.parse_args()
sys.path.insert(0, args.code)
ckpts = [os.path.abspath(p) for p in args.ckpt.split(',')]
os.chdir(args.run)
import torch
from casmi import fpnet, traindata
from casmi.spectra import merge_spectra


def eval_groups(D, fold_names, n, rng):
    ok = np.isin(D.fold[D.sid], fold_names) & (D.sid2pid[D.sid] >= 0)
    groups = {}
    for i in np.where(ok)[0]:
        groups.setdefault((int(D.sid[i]), int(D.mode[i])), []).append(int(i))
    keys = sorted(groups)
    if len(keys) > n:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), n, replace=False))]
    return [groups[k][:8] for k in keys]


@torch.no_grad()
def window_eval(models, D, PFP, entries, dtype, bs=128):
    ranks, sizes = [], []
    for s in range(0, len(entries), bs):
        chunk = entries[s:s + bs]; items = []
        for idxs in chunk:
            i = idxs[0]
            if len(idxs) == 1:
                mz = D.mz[D.off[i]:D.off[i + 1]]; it = D.it[D.off[i]:D.off[i + 1]]; ce = D.ce[i]; nm = max(1, D.ce_n[i])
            else:
                mz, it = merge_spectra([(D.mz[D.off[j]:D.off[j + 1]], D.it[D.off[j]:D.off[j + 1]]) for j in idxs])
                ces = D.ce[idxs]; ces = ces[~np.isnan(ces)]; ce = ces.mean() if len(ces) else np.nan
                nm = int(sum(max(1, D.ce_n[j]) for j in idxs))
            pm, pi = fpnet.prep_peaks(mz, it, D.prec[i])
            items.append(dict(mz=pm, it=pi, prec=float(D.prec[i]), adduct_ix=int(D.adduct[i]), ce=0.0 if np.isnan(ce) else float(ce),
                              ce_known=0.0 if np.isnan(ce) else 1.0, n_merged=min(nm, 8), mode=float(D.mode[i])))
        inp = fpnet.collate(items, 'cuda')
        with torch.autocast('cuda', dtype=dtype):
            z = torch.stack([m(*inp)[0].float() for m in models]).mean(0)
        for r, idxs in enumerate(chunk):
            pid = D.sid2pid[D.sid[idxs[0]]]; m = D.pmass[pid]; tol = m * 10e-6
            lo = np.searchsorted(D.pmass, m - tol, 'left'); hi = np.searchsorted(D.pmass, m + tol, 'right')
            sc = fpnet.unpack_gpu(PFP[lo:hi], D.nbits) @ z[r]
            ranks.append(int((sc > sc[pid - lo]).sum().item()) + 1); sizes.append(int(hi - lo))
    ranks = np.array(ranks)
    return dict(n=len(ranks), mrr25=float(np.where(ranks <= 25, 1.0 / ranks, 0).mean()), top1=float((ranks == 1).mean()),
                top5=float((ranks <= 5).mean()), top25=float((ranks <= 25).mean()), median_window=float(np.median(sizes)))


D = traindata.Data('cpu', seed=0, holdout_folds=tuple(args.holdout.split(',')))
rng = np.random.default_rng(12345)
sets = {name: eval_groups(D, names, args.eval_groups, rng) for name, names in (('hold', ['hold']), ('np', ['np']), ('train', ['train']))}
PFP = D.PFP.to('cuda')
models = [fpnet.load_model(p, 'cuda') for p in ckpts]
assert all(m.nbits == D.nbits for m in models), [m.nbits for m in models]
dtype = fpnet.autocast_dtype('cuda')
res = {name: window_eval(models, D, PFP, ent, dtype) for name, ent in sets.items()}
res['ckpt'] = ckpts
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
