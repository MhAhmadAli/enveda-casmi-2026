"""frag2: FRAGMENTATION 2.0 -- discriminative peak explanation, depth-aware explanation, cleavage coverage (precision) and
neutral-loss diagnostics.

DECISION LOG (kept here so a restarted session can continue; newest last)
  * attempt 1 (killed by a network outage): discriminative explanation on the engine's depth-2 fragment set (df / idf /
    unique / top-5 / formula-group features) + diagnostic losses vs fragcaps capabilities; pool_cap.npy precomputed.
  * attempt 2 review: explanation kernel checked against frag.explain_peaks (same criterion |peak - (f + ion_delta + s*H)|
    <= max(0.01 Da, 15 ppm), |s| <= 2; nearest-fragment search == any-fragment search); peak prep == engine F block.
    Fixes: per-candidate sums now all run in numba loops (row-independent accumulation -> exact order invariance, BLAS
    sums are position dependent), ion deltas up to 4 (was 3), generated fragments cast to float32 like the pool tables.
  * exploration (scratch, 250 sim queries, within-formula-group truth MRR, single features): fr_int 0.34, depth-2 disc
    0.35-0.47 (sample dependent), depth-1 alone ~= depth-2, depth-mixed explanation (depth-1 = 1, depth-2-only = 0.5)
    0.39 (best recall-type), cleavage coverage (share of the candidate's observable 1-bond cleavages whose fragment ion is
    observed, |H shift| <= 1, intensity weighted) 0.37 with rr-correlation 0.64 to fr_int (complementary: precision vs
    recall). -> new pool table res/frag2/pool_f1_{off,mass}.npy (depth-1 fragments, res_frag2.py --step f1).
  * diagnostic losses (500 queries): observed-loss support truth vs same-formula decoys: H2O .85/.64, CO2 .71/.42,
    CH2O2 .57/.31, CH3 .87/.49, NH3 .63/.39, C2H2O .27/.04, hex .64/.40, CO .90/.91 (useless but harmless); the full
    pair cross product was mostly chance matches -> curated PAIRS in fragcaps. 'expected but not observed' counts
    (dl_missing / dl_present of attempt 1) dropped: they are mostly a structural prior (does the candidate carry a
    glycoside / amine ...), i.e. the kind of per-structure prior that can encode train-likeness.
  * generated candidates: depth-2 enumeration with the engine's edge-scan labelling cost up to 3 s/query (80-atom
    products) -> frag_masses_fast (CSR adjacency, bit-exact: 3328 pool + generated molecules identical, 12x faster;
    self-check in prepare()). Family cost ~30-45 ms/query on 1 core.
  * gate_fam 120 queries 0 failures; extra gate on 40 large queries with the frag-set cap lowered to 100: 0 failures.
  * frag2_v1 (26 feats, 4448 groups): ALL +0.0015 [-0.0040,+0.0071], C2FAM +0.0010, np ALL +0.0079 [-0.0086,+0.0234];
    provenance: new gains -0.0003 vs train decoys, -0.0028 vs COCONUT decoys (no provenance signature). Gain importance
    6.7% (base fr_* 2.4%), led by mix_rank, cov_rank, cov_grp_gap, covw1, grp_rank. Per-feature diagnostic (rank of
    train vs COCONUT decoys): every frag feature (and base fr_int, fz) ranks train decoys slightly better; covw1 is the
    most neutral (0.010 vs fr_int 0.032, fz 0.062), group gaps ~0.05-0.07 (like fz).
  * fragment plausibility tiers (labile hetero bond / C-C / ring opening / other 2-bond) tested offline: no better than
    the depth mix (ring-opening fragments hurt) -> not added.
  * lambdarank scores rows independently -> added lead-relative features (excl_gain / excl_loss / *_vs_lead) so the
    ranker can see "this candidate explains peaks the f.z favourite cannot" directly.
  * partial-replay evals (5340 groups, same group set for v2-v4; d [lo90, hi90]):
      v2 all 31 feats   ALL +0.0030 [-0.0021,+0.0077] C2FAM +0.0007 C2FAM_wt -0.0070 C3FAM +0.0217 [+0.0099,+0.0351]
                        c1 +0.0060 [+0.0009,+0.0110]; np ALL -0.0054, np C2FAM -0.0176 [-0.0341,-0.0005]  (dilution?)
      v3 compact 13     ALL +0.0054 [+0.0006,+0.0103] C2FAM +0.0041 [-0.0019,+0.0105] C2FAM_wt +0.0010 C3FAM +0.0174
                        [+0.0017,+0.0332] c1 +0.0063 [+0.0011,+0.0117]; np ALL +0.0065, np C2FAM -0.0023 (n.s.)
      v4 compact - lead ALL +0.0047 [-0.0001,+0.0093] C2FAM +0.0020 C2FAM_wt -0.0053 C3FAM +0.0194 c1 +0.0103
      paired v3 - v2: C2FAM_wt +0.0080 [+0.0006,+0.0161], np ALL +0.0119 [+0.0000,+0.0242]; v3 - v4: n.s. (C2FAM
      +0.0022, C2FAM_wt +0.0063, c1 -0.0040). Provenance gains (new - base) tr/co: v2 -0.0008/-0.0010, v3
      -0.0007/-0.0018, v4 -0.0003/-0.0022 (no COCONUT-specific gain anywhere).
    -> RECOMMENDED = v3 compact: RECOMMENDED list below (evalfe --only). The other features stay computed (cheap) for
       reference but are not recommended for the ranker (spread-thin gain, v2 < v3 on np / C2FAM_wt).

Peaks / explanation == engine frag_features: spectra.prep_query(mz, it, 0.002, 128, 1.0, False) per prepared query
spectrum, weight w = sqrt(it / max), fragment masses P.frags(pid) (pool, depth 2) / res/frag2 depth-1 table (pool) or
frag.fragment_masses on the fly (generated, cached by smiles, SAME function and float32 rounding as the pool tables),
ion deltas frag.ion_deltas(adduct), tolerance max(0.01 Da, 15 ppm), H shifts |s| <= 2 (strict: <= 1).
Explanation / coverage use the engine's frag set: top FRAG_MAX_CAND (600) by the A+B f.z, tie-inclusive; other
candidates get 0 (the engine gives them fr_* = 0 too). Loss features cover every candidate.

Features (per candidate; "mean"/"max" = over query spectra, intensities normalised by the spectrum's total weight):
  disc_max / disc_mean   sum over explained peaks of w * (1 - df), df = share of frag-set candidates explaining the peak
  disc1_mean             same with |H shift| <= 1 (df of the strict explanation)
  idf_frac, idf_rank     idf-weighted explained share (idf = log((N+1)/(cnt+1)) / log(N+1)); its rank in the query
  uniq_int               explained intensity of peaks explained by <= 10% of the frag set (max)
  top5_expl, top5_unexpl explained share of the 5 strongest peaks (mean); unexplained intensity of them (max)
  grp_disc, grp_miss     formula-group df: explained w*(1-df_g); missed w*df_g (peaks the candidate's isomers explain)
  grp_rank, grp_gap      idf_frac rank / gap to the best within the candidate's formula group
  d1_int                 explained share with 1-bond-break fragments only
  mix_disc, mix_rank, mix_grp_gap  depth-weighted (1 / 0.5) explanation * (1 - df_mix); rank; formula-group gap
  llr1                   log-likelihood ratio of the depth-1 explanation pattern (truth explains a peak w.p. 0.9 vs the
                         frag-set rate), explained AND unexplained peaks contribute
  cov1, covw1            share (and max-peak-weighted share) of observable 1-bond cleavages (either side's ion inside
                         [max(50, min peak - 0.5), precursor - 2]) with an observed ion (|H shift| <= 1)
  cov_rank, cov_grp_gap  covw1 rank in the query / gap to the best in the formula group
  excl_gain, excl_loss   vs the LEAD (frag-set candidates with the maximal A+B f.z; tied leads averaged): depth-mixed
                         explained intensity the candidate has and the lead lacks / the lead has and the candidate lacks
  mix_vs_lead, cov_vs_lead, llr_vs_lead   mix_disc / covw1 / llr1 minus the lead's value
  dl_sup_spec / dl_unsup_spec  intensity of peaks whose precursor loss matches a SPECIFIC diagnostic loss that the
                         candidate can / cannot produce (capabilities: fragcaps SMARTS, intrinsic chemistry) (max)
  dl_sup_gen / dl_unsup_gen    same for generic losses (H2O, CO, CO2, CH2O2, CH3, CH4O)
  dl_sup_gap             (dl_sup_spec + dl_sup_gen) - query max

Order invariance: df counts are integers; every per-candidate sum is a numba loop over peaks in a fixed order
(row-independent); ranks use _rank_norm (average ties); group maxima are exact; frag-set selection is tie-inclusive.
No provenance: pid is only a lookup key (fragment / capability tables built from SMILES); no labels / sid / regime.
"""
import os
import math

import numpy as np
from numba import njit

from ctxlib import _rank_norm, COL
from casmi import frag as fragmod
from casmi.spectra import prep_query
import fragcaps

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'res', 'frag2')
FRAG_MAX_CAND = 600
TOL_DA, TOL_PPM, HSHIFT = 0.01, 15.0, 2
COV_HSHIFT = 1
LOSS_TOL_DA, LOSS_TOL_PPM = 0.005, 10.0
UNIQ_DF = 0.10               # "unique" peaks: explained by <= 10% of the frag-set candidates (at least 1)
LLR_Q = 0.9
TOPN = 5
MAXD = 4                     # max ion deltas per adduct
H_MASS = fragmod.H_MASS

_R = {}
_GF = {}                     # generated-candidate fragments (depth 2, depth 1) by smiles (per process)
_GC = {}                     # generated-candidate capabilities by smiles

NAMES = ['disc_max', 'disc_mean', 'disc1_mean', 'idf_frac', 'idf_rank', 'uniq_int', 'top5_expl', 'top5_unexpl',
         'grp_disc', 'grp_miss', 'grp_rank', 'grp_gap',
         'd1_int', 'mix_disc', 'mix_rank', 'mix_grp_gap', 'llr1',
         'cov1', 'covw1', 'cov_rank', 'cov_grp_gap',
         'excl_gain', 'excl_loss', 'mix_vs_lead', 'cov_vs_lead', 'llr_vs_lead',
         'dl_sup_spec', 'dl_unsup_spec', 'dl_sup_gen', 'dl_unsup_gen', 'dl_sup_gap']
RECOMMENDED = ['mix_rank', 'cov_rank', 'mix_vs_lead', 'cov_grp_gap', 'covw1', 'grp_gap', 'cov_vs_lead', 'excl_loss',
               'grp_rank', 'mix_grp_gap', 'llr_vs_lead', 'excl_gain', 'dl_sup_gap']


def prepare(res):
    _R['cap'] = np.load(os.path.join(RES, 'pool_cap.npy'), mmap_mode='r')
    with open(os.path.join(RES, 'caps.txt')) as fh:
        assert fh.read().split('\n') == fragcaps.CAPS, 'pool_cap.npy was built with another capability list'
    _R['f1_off'] = np.load(os.path.join(RES, 'pool_f1_off.npy'), mmap_mode='r')
    _R['f1_mass'] = np.load(os.path.join(RES, 'pool_f1_mass.npy'), mmap_mode='r')
    assert len(_R['f1_off']) == len(res.P.mass) + 1 and len(_R['cap']) == len(res.P.mass)
    names, lm, lspec, req = fragcaps.loss_table()
    _R['lmass'] = lm; _R['lspec'] = lspec; _R['req'] = req
    _R['P'] = res.P
    for s in ('OC1OC(CO)C(O)C(O)C1Oc1ccc2c(c1)oc(-c1ccc(O)cc1)cc2=O', 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)O)OC(=O)CCC'):
        g = fragmod.mol_graph(s)                        # self-check: fast fragmenter == engine fragmenter
        for dd in (1, 2):
            assert np.array_equal(frag_masses_fast(g[0], g[1], dd, 200000), fragmod.fragment_masses(g[0], g[1], dd, 200000))


# ------------------------------------------------------------------------------------------------ numba kernels
@njit(cache=True)
def _label(n, start, nbr, eid, d1, d2, comp, stack):
    """== frag._components (components numbered by their smallest atom index) with CSR adjacency: O(n + E)."""
    for i in range(n):
        comp[i] = -1
    ncomp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        top = 0
        stack[top] = s; top += 1
        comp[s] = ncomp
        while top > 0:
            top -= 1
            u = stack[top]
            for q in range(start[u], start[u + 1]):
                e = eid[q]
                if e == d1 or e == d2:
                    continue
                x = nbr[q]
                if comp[x] < 0:
                    comp[x] = ncomp
                    stack[top] = x; top += 1
        ncomp += 1
    return ncomp


@njit(cache=True)
def frag_masses_fast(w, bonds, max_depth, max_out):
    """Bit-exact re-implementation of frag.fragment_masses (same component masses summed in the same atom order, same
    rounding / sort / unique, same max_out truncation order) with an adjacency list instead of an edge scan per atom
    (the engine's version is O(n*E) per bond pair: ~40 ms for an 80-atom product). Verified equal in prepare()."""
    n = len(w); E = bonds.shape[0]
    start = np.zeros(n + 1, np.int64)
    for e in range(E):
        start[bonds[e, 0] + 1] += 1; start[bonds[e, 1] + 1] += 1
    for i in range(n):
        start[i + 1] += start[i]
    fill = start[:n].copy()
    nbr = np.empty(2 * E, np.int64); eid = np.empty(2 * E, np.int64)
    for e in range(E):
        a = bonds[e, 0]; b = bonds[e, 1]
        nbr[fill[a]] = b; eid[fill[a]] = e; fill[a] += 1
        nbr[fill[b]] = a; eid[fill[b]] = e; fill[b] += 1
    out = np.empty(max_out, np.float64)
    k = 0
    tot = 0.0
    for i in range(n):
        tot += w[i]
    out[k] = tot; k += 1
    comp = np.empty(n, np.int32)
    cm = np.empty(n, np.float64)
    stack = np.empty(n, np.int64)
    if E > 0:
        for e1 in range(E):
            nc = _label(n, start, nbr, eid, e1, -1, comp, stack)
            if nc > 1:
                for c in range(nc):
                    cm[c] = 0.0
                for i in range(n):
                    cm[comp[i]] += w[i]
                for c in range(nc):
                    if k < max_out:
                        out[k] = cm[c]; k += 1
            if max_depth >= 2:
                for e2 in range(e1 + 1, E):
                    nc2 = _label(n, start, nbr, eid, e1, e2, comp, stack)
                    if nc2 > 1:
                        for c in range(nc2):
                            cm[c] = 0.0
                        for i in range(n):
                            cm[comp[i]] += w[i]
                        for c in range(nc2):
                            if k < max_out:
                                out[k] = cm[c]; k += 1
    res = out[:k]
    r = np.empty(k, np.float64)
    for i in range(k):
        r[i] = np.floor(res[i] * 1000.0 + 0.5) / 1000.0
    r.sort()
    u = np.empty(k, np.float64)
    m = 0
    for i in range(k):
        if i == 0 or r[i] != r[i - 1]:
            u[m] = r[i]; m += 1
    return u[:m]


@njit(cache=True)
def _explain(foff, fm, pk_mz, pk_spec, sp_del, sp_nd, hmax, tol_da, tol_ppm, h):
    """(n_cand, K) uint8: minimal |H shift| that explains peak k with a fragment of candidate i (255 = unexplained).
    Same criterion as frag.explain_peaks: |peak - (f + ion_delta + s*H)| <= max(tol_da, peak*tol_ppm) for |s| <= hmax."""
    n = len(foff) - 1
    K = len(pk_mz)
    out = np.full((n, K), 255, np.uint8)
    for i in range(n):
        a = foff[i]; b = foff[i + 1]
        if b <= a:
            continue
        f = fm[a:b]
        m = b - a
        for k in range(K):
            p = pk_mz[k]
            t = p * tol_ppm * 1e-6
            if t < tol_da:
                t = tol_da
            s = pk_spec[k]
            best = 255
            for d in range(sp_nd[s]):
                base = p - sp_del[s, d]
                lo = base - hmax * h - t
                j = np.searchsorted(f, lo)
                while j < m and f[j] <= base + hmax * h + t:
                    for sh in range(-hmax, hmax + 1):
                        ash = sh if sh >= 0 else -sh
                        if ash < best and abs(f[j] + sh * h - base) <= t:
                            best = ash
                    j += 1
            out[i, k] = best
    return out


@njit(cache=True)
def _acc(V, W, spec, S):
    """out[i, s, j] = sum over peaks k of spectrum s of V[i, k] * W[k, j] (fixed loop order -> row-independent)."""
    n, K = V.shape
    J = W.shape[1]
    out = np.zeros((n, S, J))
    for i in range(n):
        for k in range(K):
            x = V[i, k]
            if x != 0.0:
                s = spec[k]
                for j in range(J):
                    out[i, s, j] += x * W[k, j]
    return out


@njit(cache=True)
def _grp(E, w, spec, S, gid, cG, nG):
    """Formula-group relative: explained w*(1-df_g) and missed w*df_g per candidate and spectrum."""
    n, K = E.shape
    gexp = np.zeros((n, S))
    gmiss = np.zeros((n, S))
    for i in range(n):
        g = gid[i]
        ng = nG[g]
        for k in range(K):
            s = spec[k]
            dfg = cG[g, k] / ng
            if E[i, k] > 0.0:
                gexp[i, s] += w[k] * (1.0 - dfg)
            else:
                gmiss[i, s] += w[k] * dfg
    return gexp, gmiss


@njit(cache=True)
def _excl(E, EL, w, spec, S):
    """Exclusive explained intensity vs the lead profile EL (K,): gain = sum w * max(E - EL, 0), loss = sum w *
    max(EL - E, 0), per candidate and spectrum (row-independent)."""
    n, K = E.shape
    g = np.zeros((n, S))
    l = np.zeros((n, S))
    for i in range(n):
        for k in range(K):
            d = E[i, k] - EL[k]
            if d > 0.0:
                g[i, spec[k]] += w[k] * d
            elif d < 0.0:
                l[i, spec[k]] -= w[k] * d
    return g, l


@njit(cache=True)
def _coverage(foff, fm, pk_off, pk_mz, pk_w, sp_del, sp_nd, lo, hi, hs, h, tol_da, tol_ppm):
    """Cleavage coverage per candidate and spectrum. Each candidate's depth-1 list (sorted, parent mass last) gives the
    cleavages as complementary pairs (A, M - A), A <= M - A. A pair is observable when either side's [frag + H]+ /
    [frag - H]- ion lies in [lo_s, hi_s]; covered when an observable side matches a peak (any ion delta, |s| <= hs).
    -> cov (covered / observable), covw (sum of the best matching peak weight / observable)."""
    n = len(foff) - 1
    S = len(pk_off) - 1
    cov = np.zeros((n, S))
    covw = np.zeros((n, S))
    for i in range(n):
        a0 = foff[i]; b0 = foff[i + 1]
        if b0 - a0 < 2:
            continue
        M = fm[b0 - 1]
        for s in range(S):
            p0 = pk_off[s]; p1 = pk_off[s + 1]
            if p1 <= p0:
                continue
            pk = pk_mz[p0:p1]
            w = pk_w[p0:p1]
            m = p1 - p0
            no = 0; nm = 0; wm = 0.0
            for j in range(a0, b0 - 1):
                A = fm[j]; B = M - A
                if A > B + 1e-6:
                    break
                ionA = A + sp_del[s, 0]; ionB = B + sp_del[s, 0]
                okA = ionA >= lo[s] and ionA <= hi[s]
                okB = ionB >= lo[s] and ionB <= hi[s]
                if not (okA or okB):
                    continue
                no += 1
                hit = False; bw = 0.0
                for side in range(2):
                    if side == 0:
                        if not okA:
                            continue
                        x0 = A
                    else:
                        if not okB:
                            continue
                        x0 = B
                    for d in range(sp_nd[s]):
                        for sh in range(-hs, hs + 1):
                            x = x0 + sp_del[s, d] + sh * h
                            t = x * tol_ppm * 1e-6
                            if t < tol_da:
                                t = tol_da
                            kk = np.searchsorted(pk, x - t)
                            while kk < m and pk[kk] <= x + t:
                                hit = True
                                if w[kk] > bw:
                                    bw = w[kk]
                                kk += 1
                if hit:
                    nm += 1; wm += bw
            if no > 0:
                cov[i, s] = nm / no
                covw[i, s] = wm / no
    return cov, covw


# ------------------------------------------------------------------------------------------------ helpers
def _peaks(v):
    def f():
        out = []
        for s in v.spectra:
            m, p = prep_query(np.asarray(s['mz'], np.float64), np.asarray(s['it'], np.float64), 0.002, 128, 1.0, False)
            if len(m):
                out.append((m, p / p.max(), s['adduct'], float(s['prec'])))
        return out
    return v.memo('frag2_peaks', f)


def _gen_frags(s):
    """(depth-2, depth-1) fragment masses of a generated candidate == frag.fragments_for_smiles(s, 2) / (s, 1), rounded to
    float32 like the pool tables (one RDKit parse for both)."""
    fg = _GF.get(s)
    if fg is None:
        if len(_GF) > 50000:
            _GF.clear()
        g = fragmod.mol_graph(s)
        if g is None:
            f2 = f1 = np.zeros(0, np.float64)
        else:
            w, bonds = g
            if len(bonds) == 0:
                f2 = f1 = np.array([w.sum()])
            else:
                f2 = frag_masses_fast(w, bonds, 2, 200000)
                f1 = frag_masses_fast(w, bonds, 1, 200000)
        fg = (f2.astype(np.float32).astype(np.float64), f1.astype(np.float32).astype(np.float64))
        _GF[s] = fg
    return fg


def _frags(v, i):
    """(depth-2, depth-1) sorted fragment masses of candidate i (float64 of float32 values)."""
    p = int(v.pid[i])
    if p >= 0:
        o = _R['f1_off']
        return _R['P'].frags(p), np.asarray(_R['f1_mass'][o[p]:o[p + 1]], np.float64)
    return _gen_frags(v.smiles[i])


def _caps(v):
    C = np.zeros((v.n, len(fragcaps.CAPS)), np.uint8)
    pi = np.flatnonzero(v.pid >= 0)
    if len(pi):
        C[pi] = _R['cap'][v.pid[pi]]
    for i in np.flatnonzero(v.pid < 0):
        s = v.smiles[i]
        c = _GC.get(s)
        if c is None:
            if len(_GC) > 200000:
                _GC.clear()
            c = fragcaps.caps_of_smiles(s); _GC[s] = c
        C[i] = c
    return C


def _group_ids(formulas):
    u = sorted(set(formulas))
    ix = {f: j for j, f in enumerate(u)}
    return np.array([ix[f] for f in formulas], np.int64), len(u)


def _group_rank(x, gid, G):
    """rank_norm of x within each formula group (0 = best, average ties; singletons 0)."""
    out = np.zeros(len(x))
    for g in range(G):
        j = np.flatnonzero(gid == g)
        out[j] = _rank_norm(x[j])
    return out


def _group_max(x, gid, G):
    mx = np.full(G, -np.inf)
    np.maximum.at(mx, gid, x)
    return mx[gid]


def _flat(fr):
    off = np.zeros(len(fr) + 1, np.int64); off[1:] = np.cumsum([len(f) for f in fr])
    return off, (np.concatenate(fr).astype(np.float64) if off[-1] else np.zeros(0))


def _mean_s(x):
    """(N, S) -> (N,) mean over spectra, fixed order (row-independent)."""
    acc = x[:, 0].copy()
    for s in range(1, x.shape[1]):
        acc = acc + x[:, s]
    return acc / x.shape[1]


def _max_s(x):
    acc = x[:, 0].copy()
    for s in range(1, x.shape[1]):
        acc = np.maximum(acc, x[:, s])
    return acc


# ------------------------------------------------------------------------------------------------ family
def compute(v):
    if not _R:
        prepare(v.res)
    n = v.n
    out = {k: np.zeros(n) for k in NAMES}
    peaks = _peaks(v) if n else []
    if not peaks:
        return {k: x.astype(np.float32) for k, x in out.items()}
    S = len(peaks)
    pk_mz = np.concatenate([p[0] for p in peaks]).astype(np.float64)
    pk_w = np.sqrt(np.concatenate([p[1] for p in peaks]).astype(np.float64))
    pk_spec = np.concatenate([np.full(len(p[0]), s, np.int64) for s, p in enumerate(peaks)])
    K = len(pk_mz)
    sp_del = np.zeros((S, MAXD)); sp_nd = np.zeros(S, np.int64)
    top = np.zeros(K)                                  # top-N peaks of each spectrum by weight (stable on peak order)
    wtot = np.zeros(S); ntop = np.zeros(S); wtop = np.zeros(S)
    lo = np.zeros(S); hi = np.zeros(S)
    pk_off = np.zeros(S + 1, np.int64)
    srt_mz = np.zeros(K); srt_w = np.zeros(K)          # per-spectrum m/z-sorted copies (coverage search)
    off = 0
    for s, (m, it, ad, prec) in enumerate(peaks):
        d = fragmod.ion_deltas(ad)[:MAXD]
        sp_del[s, :len(d)] = d; sp_nd[s] = len(d)
        w = pk_w[off:off + len(m)]
        o = np.argsort(-w, kind='mergesort')[:TOPN]
        top[off + o] = 1.0
        wtot[s] = w.sum(); ntop[s] = len(o); wtop[s] = w[o].sum()
        so = np.argsort(m, kind='mergesort')
        srt_mz[off:off + len(m)] = m[so]; srt_w[off:off + len(m)] = w[so]
        lo[s] = max(50.0, float(m.min()) - 0.5); hi[s] = prec - 2.0
        off += len(m); pk_off[s + 1] = off

    # ---------------- explanation on the engine's frag set -----------------------------------------------
    fz = v.X[:, COL['fz']]
    if n > FRAG_MAX_CAND:
        thr = np.sort(fz)[::-1][FRAG_MAX_CAND - 1]
        sel = np.flatnonzero(fz >= thr)                 # tie-inclusive (== engine frag_max_cand guard)
    else:
        sel = np.arange(n)
    N = len(sel)
    fr = [_frags(v, int(i)) for i in sel]
    foff2, fm2 = _flat([f[0] for f in fr])
    foff1, fm1 = _flat([f[1] for f in fr])
    M2 = _explain(foff2, fm2, pk_mz, pk_spec, sp_del, sp_nd, HSHIFT, TOL_DA, TOL_PPM, H_MASS)
    M1 = _explain(foff1, fm1, pk_mz, pk_spec, sp_del, sp_nd, HSHIFT, TOL_DA, TOL_PPM, H_MASS)
    E2 = (M2 <= HSHIFT).astype(np.float64)
    E2s = (M2 <= 1).astype(np.float64)
    E1 = (M1 <= HSHIFT).astype(np.float64)
    Em = np.maximum(E1, 0.5 * E2)                      # depth-1 = 1, depth-2 only = 0.5
    cnt2 = E2.sum(0); cnt2s = E2s.sum(0); cnt1 = E1.sum(0)
    df2 = cnt2 / N; df2s = cnt2s / N; dfm = Em.sum(0) / N
    idf = np.log((N + 1.0) / (cnt2 + 1.0)) / math.log(N + 1.0)
    uniq = ((cnt2 > 0) & (cnt2 <= max(1, int(math.floor(UNIQ_DF * N))))).astype(np.float64)
    p1 = (cnt1 + 0.5) / (N + 1.0)
    a_pos = np.log(LLR_Q / p1) - np.log((1.0 - LLR_Q) / (1.0 - p1))
    llr0 = np.bincount(pk_spec, weights=pk_w * np.log((1.0 - LLR_Q) / (1.0 - p1)), minlength=S)
    W2 = np.stack([pk_w, pk_w * (1.0 - df2), pk_w * idf, pk_w * uniq, pk_w * top, top], 1)
    R2 = _acc(E2, W2, pk_spec, S)                                     # (N, S, 6)
    R2s = _acc(E2s, (pk_w * (1.0 - df2s))[:, None].copy(), pk_spec, S)
    R1 = _acc(E1, np.stack([pk_w, pk_w * a_pos], 1), pk_spec, S)
    Rm = _acc(Em, (pk_w * (1.0 - dfm))[:, None].copy(), pk_spec, S)
    gid_s, G_s = _group_ids([v.formula[i] for i in sel])
    cG = np.zeros((G_s, K))
    np.add.at(cG, gid_s, E2)
    nG = np.bincount(gid_s, minlength=G_s).astype(np.float64)
    gexp, gmiss = _grp(E2, pk_w, pk_spec, S, gid_s, cG, nG)
    widf = np.bincount(pk_spec, weights=pk_w * idf, minlength=S)
    cov, covw = _coverage(foff1, fm1, pk_off, srt_mz, srt_w, sp_del, sp_nd, lo, hi, COV_HSHIFT, H_MASS, TOL_DA, TOL_PPM)

    wt = wtot[None, :]
    full = np.zeros

    def put(name, x):
        a = full(n); a[sel] = x; out[name] = a

    put('disc_max', _max_s(R2[:, :, 1] / wt))
    put('disc_mean', _mean_s(R2[:, :, 1] / wt))
    put('disc1_mean', _mean_s(R2s[:, :, 0] / wt))
    idf_ok = (widf > 1e-12)[None, :]
    put('idf_frac', _mean_s(np.where(idf_ok, R2[:, :, 2] / np.maximum(widf, 1e-12)[None, :], 0.0)))
    put('uniq_int', _max_s(R2[:, :, 3] / wt))
    put('top5_expl', _mean_s(R2[:, :, 5] / ntop[None, :]))
    put('top5_unexpl', _max_s((wtop[None, :] - R2[:, :, 4]) / wt))
    put('grp_disc', _mean_s(gexp / wt))
    put('grp_miss', _mean_s(gmiss / wt))
    put('d1_int', _mean_s(R1[:, :, 0] / wt))
    put('mix_disc', _mean_s(Rm[:, :, 0] / wt))
    put('llr1', _mean_s((llr0[None, :] + R1[:, :, 1]) / wt))
    put('cov1', _mean_s(cov))
    put('covw1', _mean_s(covw))
    # lead-relative: the lead = frag-set candidates with the maximal A+B f.z (ties: mean profile / mean score)
    fzs = fz[sel]
    lead = np.flatnonzero(fzs == fzs.max())
    EL = Em[lead].mean(0) if len(lead) > 1 else Em[lead[0]].copy()
    xg, xl = _excl(Em, EL, pk_w, pk_spec, S)
    put('excl_gain', _mean_s(xg / wt))
    put('excl_loss', _mean_s(xl / wt))
    for nm, src in (('mix_vs_lead', 'mix_disc'), ('cov_vs_lead', 'covw1'), ('llr_vs_lead', 'llr1')):
        xs = out[src][sel]
        put(nm, xs - float(np.sort(xs[lead]).mean()))
    if N < n:                                           # outside the frag set: no explanation evidence
        rest = np.setdiff1d(np.arange(n), sel)
        out['top5_unexpl'][rest] = _max_s(np.broadcast_to(wtop[None, :] / wt, (1, S)))[0]
        out['llr1'][rest] = float(_mean_s((llr0 / wtot)[None, :])[0])
    gid, G = _group_ids(v.formula)
    out['idf_rank'] = _rank_norm(out['idf_frac'])
    out['grp_rank'] = _group_rank(out['idf_frac'], gid, G)
    out['grp_gap'] = out['idf_frac'] - _group_max(out['idf_frac'], gid, G)
    out['mix_rank'] = _rank_norm(out['mix_disc'])
    out['mix_grp_gap'] = out['mix_disc'] - _group_max(out['mix_disc'], gid, G)
    out['cov_rank'] = _rank_norm(out['covw1'])
    out['cov_grp_gap'] = out['covw1'] - _group_max(out['covw1'], gid, G)

    # ---------------- neutral-loss diagnostics (all candidates) ------------------------------------------
    lm = _R['lmass']; lspec = _R['lspec']
    prec = np.array([p[3] for p in peaks])[pk_spec]
    loss = prec - pk_mz
    ltol = np.maximum(LOSS_TOL_DA, prec * LOSS_TOL_PPM * 1e-6)
    Mm = (np.abs(loss[:, None] - lm[None, :]) <= ltol[:, None]) & (loss[:, None] > 10.0)       # (K, D)
    acc = {k: np.zeros((n, S)) for k in ('ss', 'us', 'sg', 'ug')}
    hit = np.flatnonzero(Mm.any(1))
    if len(hit):
        Sup = fragcaps.support_matrix(_caps(v), _R['req'])                                    # (n, D)
        for k in hit:                                                                         # fixed peak order
            s = pk_spec[k]; w = pk_w[k]
            for spec_flag, a, b in ((True, 'ss', 'us'), (False, 'sg', 'ug')):
                ds = np.flatnonzero(Mm[k] & (lspec == spec_flag))
                if len(ds):
                    sup = Sup[:, ds].any(1)
                    acc[a][:, s] += w * sup
                    acc[b][:, s] += w * (~sup)
    for a, nm in (('ss', 'dl_sup_spec'), ('us', 'dl_unsup_spec'), ('sg', 'dl_sup_gen'), ('ug', 'dl_unsup_gen')):
        out[nm] = _max_s(acc[a] / wt)
    tot = out['dl_sup_spec'] + out['dl_sup_gen']
    out['dl_sup_gap'] = tot - tot.max()
    return {k: np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for k, x in out.items()}
