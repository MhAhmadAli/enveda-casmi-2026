"""prvsiyan analog-propagation pipeline components (public notebook, version of 2026-09-16), copied
verbatim where possible so local simulations and the Kaggle notebook share one source of truth.
Additions are marked `# ADDED`.
"""
import math
import numpy as np
from numba import njit, prange


class CFG:
    PPM_WIN = 10.0
    PPM_FALLBACK = 30.0
    INT_FLOOR = 0.002
    MAX_PEAKS = 256
    MZ_TOL = 0.01
    INT_POWER = 1.0
    ENT_WEIGHT = True
    ANALOG_WIN = 200.0
    N_ANALOG = 80
    SIM_POWER = 3.0
    W1_PRIORS = (0.30, 0.60)
    SEEDS = (0, 1, 2, 3)
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)
    TOPN = 25


# ----------------------------------------------------------------------------------------------
# similarity kernels
# ----------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx; c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64); j = 0
    for i in range(n):
        if it[i] >= thr: idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort(); idx = k2; c = topk
    om = np.empty(c, np.float32); oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]; v = it[idx[i]] ** power; oi[i] = v; s += v
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S; s2 = 0.0
            for i in range(c): oi[i] = oi[i] ** w; s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    SAB = 0.0; tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf[b] = qp[i]; i += 1; b += 1
        elif d > tol: buf[b] = cp[j]; j += 1; b += 1
        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out


@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if shift > -0.001 and shift < 0.001: return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b


# ADDED: search over a pre-cleaned store (reps cleaned once instead of on every call)
@njit(cache=True, fastmath=True, parallel=True)
def search_shift_pre(qmz, qp, lo, hi, roff, rmz, rit, tol, shift):
    out = np.zeros(hi - lo, np.float32)
    for k in prange(hi - lo):
        a = roff[lo + k]; b = roff[lo + k + 1]
        if b <= a: continue
        out[k] = entropy_sim_shift(qmz, qp, rmz[a:b], rit[a:b], tol, shift[k])
    return out


# ADDED: library search where each reference is shifted by (query precursor - reference precursor):
# the same molecule measured as a different adduct keeps its neutral losses (modified-cosine idea)
@njit(cache=True, fastmath=True, parallel=True)
def search_shift_rows(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])
    return out


# ADDED: clean a list of spectra into a flat store
@njit(cache=True)
def clean_store(rows, off, allmz, allin, floor, topk, power, ent_weight):
    n = len(rows)
    lens = np.zeros(n + 1, np.int64)
    tmp_m = []
    tmp_i = []
    for k in range(n):
        r = rows[k]
        cm, cp = _clean(allmz[off[r]:off[r + 1]], allin[off[r]:off[r + 1]], floor, topk, power, ent_weight)
        tmp_m.append(cm); tmp_i.append(cp)
        lens[k + 1] = lens[k] + len(cm)
    om = np.empty(lens[n], np.float32); oi = np.empty(lens[n], np.float32)
    for k in range(n):
        om[lens[k]:lens[k + 1]] = tmp_m[k]; oi[lens[k]:lens[k + 1]] = tmp_i[k]
    return lens, om, oi


# ----------------------------------------------------------------------------------------------
# adducts
# ----------------------------------------------------------------------------------------------
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990; PROTON = MASS['H'] - E; H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']; FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS['Na'] - E),
    "[M+K]+": (1, 1, MASS['K'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E), "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS['K'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}


def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan); ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any(): out[m] = (mz[m] * z - d) / n
    return out


# ----------------------------------------------------------------------------------------------
# ranking features (31)
# ----------------------------------------------------------------------------------------------
N_ANALOG = 80
P_SIM = 3.0
N_FEAT = 31


def _rank_norm(x):
    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def rank_features(cand_fp, cand_lib, analog_fp, analog_sim, model_logits=None, frag=None):
    nc = cand_fp.shape[0]
    cf = cand_fp.astype(np.float32); cs = cf.sum(1)
    lv = np.asarray(cand_lib, np.float32)
    lvmax = float(lv.max()) if nc else 0.0
    if analog_fp is not None and len(analog_sim):
        af = analog_fp.astype(np.float32); asum = af.sum(1)
        inter = cf @ af.T
        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)
        w = np.clip(np.asarray(analog_sim, np.float32), 0, None)
        ap = (tan * (w ** P_SIM)[None, :]).max(1)
        a1 = (tan * w[None, :]).max(1)
        best_tan = tan.max(1); top_tan = tan[:, 0]; top_sim = float(w[0])
        mean_tan = (tan * (w ** P_SIM)[None, :]).sum(1) / ((w ** P_SIM).sum() + 1e-9)
    else:
        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32); top_sim = 0.0
    apmax = float(ap.max()) if nc else 0.0
    if model_logits is not None:
        raw = cf @ np.asarray(model_logits, np.float32)
        nrm = raw / np.sqrt(np.maximum(cs, 1.0))
        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),
                 (raw == raw.max()).astype(np.float32)]
    else:
        mfeat = [np.zeros(nc, np.float32)] * 6
    if model_logits is not None and nc:
        mr = _rank_norm(cf @ np.asarray(model_logits, np.float32))
        lbest = int(np.argmax(lv)) if lvmax > 0 else -1
        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0
        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),
                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]
    else:
        xfeat = [np.zeros(nc, np.float32)] * 6
    if frag is not None:
        fr = np.asarray(frag, np.float32)
        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]
    else:
        ffeat = [np.zeros(nc, np.float32)] * 4
    return np.column_stack([
        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(float),
        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,
        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),
        np.full(nc, np.log(max(nc, 1))),
        *mfeat, *ffeat, *xfeat,
    ]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# MetFrag-lite
# ----------------------------------------------------------------------------------------------
AMU = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196, 'P': 30.97376163,
       'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473,
       'Na': 22.9897692809, 'K': 38.96370668, 'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213}
H = AMU['H']


def mol_graph(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n


def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps


def fragment_masses(smi, max_breaks=2, max_bonds=34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))


def frag_masses_safe(smi):
    try: return fragment_masses(smi)
    except Exception: return np.zeros(0)


# ADDED: charge carriers a fragment may keep, per precursor adduct (prvsiyan only used +-proton)
CARRIERS = {
    "[M+H]+": (PROTON,), "[M-H2O+H]+": (PROTON,), "[M-2H2O+H]+": (PROTON,),
    "[M+NH4]+": (PROTON, NH4 - E), "[M+Na]+": (PROTON, MASS['Na'] - E), "[M+K]+": (PROTON, MASS['K'] - E),
    "[M-H]-": (-PROTON,), "[M-H2O-H]-": (-PROTON,), "[M+CH2O2-H]-": (-PROTON, FORMATE - PROTON),
    "[M+Cl]-": (-PROTON, MASS['Cl'] + E),
}


def explain_score_adduct(frag_mass, peak_mz, peak_int, adduct, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    car = CARRIERS.get(adduct, (PROTON,) if "+" in adduct[-2:] else (-PROTON,))
    ion = np.sort(np.concatenate([frag_mass + dh * H + c for dh in h_shifts for c in car]))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)


def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    ion = []
    for dh in h_shifts:
        ion.append(frag_mass + dh * H + (PROTON if mode > 0 else -PROTON))
    ion = np.sort(np.concatenate(ion))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)