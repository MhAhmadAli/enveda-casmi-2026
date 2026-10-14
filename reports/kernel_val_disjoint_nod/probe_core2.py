"""P_pubchem probe core (lever 3). Embedded verbatim into the Kaggle notebook (written to /kaggle/working and imported,
so worker processes can import it) and imported directly by the local smoke test.

Per molecule:
  1. f.z logits of the A+B FPNet ensemble, exactly like Engine.model_logits: z = 0.5 * (mean per-spectrum logits +
     mean per-polarity merged-spectrum logits); models averaged inside ModelBank.
  2. PubChem window: +-10 ppm around the target neutral mass (median over the molecule's spectra), no subsampling.
  3. pass 1 (src/casmi/pubchem.py tier): partial f.z on the ECFP4 block of the selected bits, keep top PC_N1 = 1000.
  4. pass 2: full selected-bit fingerprints, full f.z, sort descending.
  5. walk down the f.z order, compute the metric score key (tautomer-canonical InChIKey14); keep a structure only if
     its key is NOT in the candidate pool (train U COCONUT) and not already listed; stop at 25.
  An empty list becomes the single 'CCO' fallback (never padded).
Reading: sum of reciprocal ranks of PubChem-only truths / ~130 public molecules.
"""
from __future__ import annotations

import os
import time

import numpy as np

PC_N1 = 5000
PPM = 10.0
K = 25

_W = {}


# ------------------------------------------------------------------------------------------ model part (main process)
def molecule_logits(bank, spectra):
    """spectra: list of dict(mz, it, mode, adduct, prec, ce, ce_n). Returns z_avg (nbits,) float32 or None."""
    from casmi import chem, fpnet
    from casmi.spectra import merge_spectra
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec'])
        ce_ok = s['ce'] is not None and not np.isnan(s['ce'])
        items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']),
                            ce=float(s['ce']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(int(s.get('ce_n', 1)), 8), mode=float(s['mode'])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
        prec = float(np.median([s['prec'] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
        adducts = [s['adduct'] for s in grp]
        ad = max(adducts, key=adducts.count)          # deterministic tie-break (first occurrence)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    if not items_s:
        return None
    zs, _ = bank.logits_el(items_s)
    zm, _ = bank.logits_el(items_m)
    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)


# ------------------------------------------------------------------------------------------ PubChem part (workers)
def init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):
    import sys
    if code_dir and code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import pandas as pd
    from casmi import chem
    _W['chem'] = chem
    _W['mass'] = np.load(os.path.join(pc_dir, 'pc_mass.npy'), mmap_mode='r')
    _W['off'] = np.load(os.path.join(pc_dir, 'pc_off.npy'), mmap_mode='r')
    _W['buf'] = np.load(os.path.join(pc_dir, 'pc_smiles.npy'), mmap_mode='r')
    bits = np.load(bits_path)
    _W['bits'] = bits
    _W['sel_e'] = np.where(bits < 4096)[0]
    _W['raw_e'] = bits[_W['sel_e']]
    _W['pool_keys'] = set(pd.read_parquet(pool_meta_path, columns=['key']).key.values.tolist())
    r = chem._rdkit()
    _W['ecfp4'] = r['gen'].GetMorganGenerator(radius=2, fpSize=4096)
    _W['Chem'] = r['Chem']


def _window(target):
    tol = target * PPM * 1e-6
    a = int(np.searchsorted(_W['mass'], target - tol, 'left')); b = int(np.searchsorted(_W['mass'], target + tol, 'right'))
    off = np.asarray(_W['off'][a:b + 1]).astype(np.int64)
    if b <= a:
        return []
    raw = bytes(_W['buf'][off[0]:off[-1]])
    base = off[0]
    return [raw[off[i] - base:off[i + 1] - base].decode('ascii') for i in range(b - a)]


def probe_one(task):
    """task = (molecule_id, target_mass, z (nbits,) float32). Returns (molecule_id, [smiles], [f.z], [score keys], diag dict)."""
    mid, target, z = task
    t0 = time.time()
    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']
    smis = _window(float(target))
    n = len(smis)
    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0)
    if n == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)
    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True
    sc1 = E @ z_e
    sc1[~ok] = -np.inf
    n1 = min(PC_N1, int(ok.sum()))
    diag['n_pass1'] = n1
    if n1 == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]
    bits = _W['bits']
    fz, idx = [], []
    for i in top1:
        fp = chem.raw_fingerprint(smis[i])
        if fp is None:
            continue
        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))
    diag['n_pass2'] = len(idx)
    order = np.argsort(-np.asarray(fz), kind='stable')
    out, out_fz, out_k, seen = [], [], [], set()
    for o in order:
        s = smis[idx[o]]
        k = chem.score_key(s)
        diag['n_keyed'] += 1
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in _W['pool_keys']:
            diag['n_in_pool'] += 1
            continue
        out.append(s); out_fz.append(float(fz[o])); out_k.append(k)
        if len(out) >= K:
            break
    diag['n_listed'] = len(out)
    diag['secs'] = time.time() - t0
    return mid, out, out_fz, out_k, diag


# ---- v17 patch: popularity prior (log1p SID + log1p PMID from PubChem, mapped onto this store by stereo-stripped SMILES) ----
_POP_DIR = os.environ.get('CASMI_POP_DIR'); _POP_LAM = float(os.environ.get('CASMI_POP_LAM', '0')); _POP_UNION = int(os.environ.get('CASMI_POP_UNION', '0'))
_init_worker0 = init_worker


def init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):
    _init_worker0(pc_dir, bits_path, pool_meta_path, code_dir)
    if _POP_DIR:
        _W['ls'] = np.load(os.path.join(_POP_DIR, 'pc_lsid.npy'), mmap_mode='r')
        _W['lp'] = np.load(os.path.join(_POP_DIR, 'pc_lpmid.npy'), mmap_mode='r')


def probe_one(task):
    mid, target, z = task
    t0 = time.time()
    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']
    tol = float(target) * PPM * 1e-6
    a0 = int(np.searchsorted(_W['mass'], float(target) - tol, 'left'))
    smis = _window(float(target))
    n = len(smis)
    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0, fz_top=None)
    if n == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)
    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True
    sc1 = E @ z_e
    sc1[~ok] = -np.inf
    n1 = min(PC_N1, int(ok.sum()))
    diag['n_pass1'] = n1
    if n1 == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]
    use_pop = ('ls' in _W) and _POP_LAM > 0
    pop = (np.asarray(_W['ls'][a0:a0 + n], np.float32) + np.asarray(_W['lp'][a0:a0 + n], np.float32)) if use_pop else np.zeros(n, np.float32)
    if use_pop and _POP_UNION > 0:
        pp = np.where(ok, pop, -np.inf); ku = min(_POP_UNION, int(ok.sum()))
        top1 = np.union1d(top1, np.argpartition(-pp, ku - 1)[:ku])
    bits = _W['bits']
    fz, idx = [], []
    for i in top1:
        fp = chem.raw_fingerprint(smis[i])
        if fp is None:
            continue
        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))
    diag['n_pass2'] = len(idx)
    if not idx:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    fz = np.asarray(fz, np.float64); idx = np.asarray(idx)
    keyc = {}
    def key(j):
        if j not in keyc:
            keyc[j] = chem.score_key(smis[idx[j]]); diag['n_keyed'] += 1
        return keyc[j]
    for j in np.argsort(-fz, kind='stable'):        # gate quantity = v16's pc_fz[0] (best non-pool structure by raw f.z)
        k = key(j)
        if k is not None and k not in _W['pool_keys']:
            diag['fz_top'] = float(fz[j]); break
    zf = (fz - fz.mean()) / (fz.std() + 1e-9)
    sc = zf + _POP_LAM * pop[idx]
    order = np.argsort(-sc, kind='stable')
    out, out_s, out_k, out_p, out_sc, seen = [], [], [], [], [], set()
    for j in order:
        k = key(j)
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in _W['pool_keys']:
            diag['n_in_pool'] += 1
            continue
        out.append(smis[idx[j]]); out_s.append(float(sc[j])); out_k.append(k); out_p.append(float(pop[idx[j]])); out_sc.append(float(sc[j]))
        if len(out) >= K:
            break
    diag['n_listed'] = len(out)
    diag['S'] = float(out_sc[0]) if out_sc else 0.0
    diag['top_pop'] = float(out_p[0]) if out_p else 0.0
    diag['secs'] = time.time() - t0
    return mid, out, out_s, out_k, diag
