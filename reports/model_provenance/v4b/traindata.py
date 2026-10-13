"""Training data access shared by the fingerprint model and the cross-encoder."""
import os, sys, time, math
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from . import fpnet, chem
from .spectra import merge_spectra

ELEMS = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']


class Data:
    def __init__(self, device, K=63, ppm=10.0, seed=0, holdout_folds=('np', 'nplib', 'syn', 'plusk', 'twin')):
        self.device = device; self.K = K; self.ppm = ppm
        self.rng = np.random.default_rng(seed)
        self.off = np.load('work/spec_off.npy'); self.mz = np.load('work/spec_mz.npy'); self.it = np.load('work/spec_it.npy')
        meta = pd.read_parquet('work/spec_meta.parquet', columns=['sid', 'adduct_ix', 'mode', 'precursor_mz',
                                                                  'ce_mean', 'ce_n', 'lib'])
        self.sid = meta.sid.values.astype(np.int64); self.adduct = meta.adduct_ix.values.astype(np.int64)
        self.mode = meta['mode'].values.astype(np.int64); self.prec = meta.precursor_mz.values.astype(np.float64)
        self.ce = meta.ce_mean.values.astype(np.float64); self.ce_n = meta.ce_n.values.astype(np.int64)
        self.lib = meta.lib.values.astype(np.int64)
        split = pd.read_parquet('work/split.parquet')
        st = pd.read_parquet('work/train_structs.parquet', columns=['sid', 'key'])
        pool = pd.read_parquet('work/pool_meta.parquet', columns=['key', 'mass', 'formula'])
        k2p = dict(zip(pool.key.values, np.arange(len(pool))))
        self.sid2pid = np.array([k2p.get(k, -1) for k in st.key.values], dtype=np.int64)
        self.pmass = pool.mass.values.astype(np.float64)
        pfp = np.load('work/pool_fp.npy')
        self.nbits = int(np.load('work/fp_bits.npy').shape[0])
        self.PFP = torch.from_numpy(pfp).to(device)          # packed uint8 on GPU
        # element targets per pool entry
        el = np.zeros((len(pool), len(ELEMS)), np.float32)
        for i, f in enumerate(pool.formula.values):
            c = chem.parse_formula(f)
            if c:
                for j, e in enumerate(ELEMS):
                    el[i, j] = c.get(e, 0)
        self.elem = torch.from_numpy(np.log1p(el)).to(device)
        fold = np.array(['train'] * len(st), dtype=object)
        fold[split.sid.values] = split.fold.values
        self.fold = fold
        is_hold = np.isin(fold, holdout_folds)
        ok = (~is_hold[self.sid]) & (self.sid2pid[self.sid] >= 0)
        self.tr = np.where(ok)[0]
        # structure groups (sid, mode) -> spectra, for merge augmentation and structure-uniform sampling
        key = self.sid * 2 + (self.mode > 0)
        order = np.argsort(key, kind='mergesort'); ks = key[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(ks)]])
        self.grp_order = order; self.grp_start = {}; self.grp_end = {}
        gk = ks[starts]
        self.gk = gk; self.gs = starts; self.ge = ends
        self.key2g = dict(zip(gk.tolist(), range(len(gk))))
        tr_groups = np.unique(key[self.tr])
        self.tr_groups = np.array([self.key2g[k] for k in tr_groups])
        print(f'[Data] train spectra {len(self.tr):,} groups {len(self.tr_groups):,} pool {len(self.pmass):,} nbits {self.nbits}', flush=True)

    def peers(self, i, kmax=4):
        k = self.sid[i] * 2 + (self.mode[i] > 0)
        g = self.key2g[k]
        members = self.grp_order[self.gs[g]:self.ge[g]]
        if len(members) <= 1:
            return [i]
        n = int(self.rng.integers(2, min(kmax, len(members)) + 1))
        pick = self.rng.choice(members, size=min(n, len(members)), replace=False)
        if i not in pick:
            pick[0] = i
        return list(pick)

    def item(self, i, merge_p=0.0, aug=False):
        idxs = [i]
        if merge_p > 0 and self.rng.random() < merge_p:
            idxs = self.peers(i)
        if len(idxs) == 1:
            mz = self.mz[self.off[i]:self.off[i + 1]].astype(np.float64); it = self.it[self.off[i]:self.off[i + 1]].astype(np.float64)
            ce = self.ce[i]; ce_known = 0.0 if np.isnan(ce) else 1.0
            n_merged = max(1, self.ce_n[i])
        else:
            mz, it = merge_spectra([(self.mz[self.off[j]:self.off[j + 1]], self.it[self.off[j]:self.off[j + 1]]) for j in idxs])
            ces = self.ce[idxs]; ces = ces[~np.isnan(ces)]
            ce = float(ces.mean()) if len(ces) else np.nan; ce_known = 1.0 if len(ces) else 0.0
            n_merged = int(sum(max(1, self.ce_n[j]) for j in idxs))
        prec = float(self.prec[i])
        if aug:
            # peak dropout, intensity jitter, ppm jitter
            if len(mz) > 4:
                keep = self.rng.random(len(mz)) > self.rng.uniform(0.0, 0.3)
                if keep.sum() >= 3:
                    mz, it = mz[keep], it[keep]
            it = it * np.exp(self.rng.normal(0.0, 0.3, size=len(it)))
            mz = mz * (1.0 + self.rng.normal(0.0, 4e-6, size=len(mz)))
            if not np.isnan(ce) and self.rng.random() < 0.1:
                ce_known = 0.0
        pm, pi = fpnet.prep_peaks(mz, it, prec)
        return dict(mz=pm, it=pi, prec=prec, adduct_ix=int(self.adduct[i]), ce=0.0 if np.isnan(ce) else float(ce),
                    ce_known=ce_known, n_merged=min(n_merged, 8), mode=float(self.mode[i]))

    def sample_ids(self, B):
        h = B // 2
        a = self.rng.choice(self.tr, size=h, replace=False)
        gs = self.rng.choice(self.tr_groups, size=B - h, replace=False)
        b = np.array([self.rng.choice(self.grp_order[self.gs[g]:self.ge[g]]) for g in gs])
        return np.concatenate([a, b])

    def negatives(self, pos):
        m = self.pmass[pos]; tol = m * self.ppm * 1e-6
        lo = np.searchsorted(self.pmass, m - tol, 'left'); hi = np.searchsorted(self.pmass, m + tol, 'right')
        out = np.empty((len(pos), self.K), np.int64)
        for r in range(len(pos)):
            n = hi[r] - lo[r]
            if n <= 1:
                out[r] = self.rng.integers(0, len(self.pmass), self.K)
            else:
                c = self.rng.integers(lo[r], hi[r], self.K)
                bad = c == pos[r]
                c[bad] = np.where(c[bad] + 1 < hi[r], c[bad] + 1, lo[r])
                if n < 8:   # tiny windows: mix in random decoys so the softmax is not trivial
                    k = self.K // 2
                    c[k:] = self.rng.integers(0, len(self.pmass), self.K - k)
                    c[k:][c[k:] == pos[r]] = (pos[r] + 1) % len(self.pmass)
                out[r] = c
        return out

    def batch(self, ids, merge_p, aug):
        items = [self.item(int(i), merge_p, aug) for i in ids]
        inp = fpnet.collate(items, self.device)
        pos = self.sid2pid[self.sid[ids]]
        negs = self.negatives(pos)
        cand = torch.as_tensor(np.concatenate([pos[:, None], negs], 1), device=self.device)
        return inp, torch.as_tensor(pos, device=self.device), cand


def window_mrr(model, D, spec_ids, device, merged=False, bs=128, K=25):
    """MRR@K of the true structure inside its +-10ppm pool window, scored by f.z."""
    model.eval()
    rr = []
    with torch.no_grad():
        if merged:
            # group by (sid, mode): merge all spectra of the group
            groups = {}
            for i in spec_ids:
                groups.setdefault((D.sid[i], D.mode[i]), []).append(i)
            entries = list(groups.values())
        else:
            entries = [[i] for i in spec_ids]
        for s in range(0, len(entries), bs):
            chunk = entries[s:s + bs]
            items = []
            for idxs in chunk:
                i = idxs[0]
                if len(idxs) == 1:
                    mz = D.mz[D.off[i]:D.off[i + 1]]; it = D.it[D.off[i]:D.off[i + 1]]
                    ce = D.ce[i]; nm = max(1, D.ce_n[i])
                else:
                    mz, it = merge_spectra([(D.mz[D.off[j]:D.off[j + 1]], D.it[D.off[j]:D.off[j + 1]]) for j in idxs])
                    ces = D.ce[idxs]; ces = ces[~np.isnan(ces)]; ce = ces.mean() if len(ces) else np.nan
                    nm = int(sum(max(1, D.ce_n[j]) for j in idxs))
                pm, pi = fpnet.prep_peaks(mz, it, D.prec[i])
                items.append(dict(mz=pm, it=pi, prec=float(D.prec[i]), adduct_ix=int(D.adduct[i]),
                                  ce=0.0 if np.isnan(ce) else float(ce), ce_known=0.0 if np.isnan(ce) else 1.0,
                                  n_merged=min(nm, 8), mode=float(D.mode[i])))
            inp = fpnet.collate(items, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(*inp)[0].float()
            for r, idxs in enumerate(chunk):
                pid = D.sid2pid[D.sid[idxs[0]]]
                m = D.pmass[pid]; tol = m * 10e-6
                lo = np.searchsorted(D.pmass, m - tol, 'left'); hi = np.searchsorted(D.pmass, m + tol, 'right')
                cand = torch.arange(lo, hi, device=device)
                fp = fpnet.unpack_gpu(D.PFP[cand], D.nbits)
                sc = fp @ z[r]
                rank = int((sc > sc[pid - lo]).sum().item()) + 1
                rr.append(1.0 / rank if rank <= K else 0.0)
    model.train()
    return float(np.mean(rr)), len(rr)


