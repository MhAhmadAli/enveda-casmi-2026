"""Spectrum kernels (numba): cleaning, entropy similarity, batched library search."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

LOG4 = np.log(4.0)


@njit(cache=True)
def clean_peaks(mz, it, prec_mz, floor, topk, max_over_prec):
    """Filter peaks: > prec+max_over_prec removed, relative floor, top-k by intensity; sorted by m/z.
    Returns (mz float32, it float32 normalised to max=1)."""
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if mz[i] <= prec_mz + max_over_prec and it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if mz[i] <= prec_mz + max_over_prec and it[i] >= thr and mz[i] > 0:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if mz[i] <= prec_mz + max_over_prec and it[i] >= thr and mz[i] > 0:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float64)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        idx = k2
        c = topk
    om = np.empty(c, np.float64)
    oi = np.empty(c, np.float32)
    for i in range(c):
        om[i] = mz[idx[i]]
        oi[i] = it[idx[i]] / mx
    o = np.argsort(om)
    om2 = np.empty(c, np.float32)
    oi2 = np.empty(c, np.float32)
    for i in range(c):
        om2[i] = om[o[i]]
        oi2[i] = oi[o[i]]
    return om2, oi2


@njit(cache=True, fastmath=True)
def _weights(it, power, ent_weight):
    """Intensity transform + normalisation to sum 1, with optional entropy weighting (Li 2021)."""
    n = len(it)
    p = np.empty(n, np.float64)
    s = 0.0
    for i in range(n):
        v = it[i] ** power
        p[i] = v; s += v
    if s <= 0:
        return p
    for i in range(n):
        p[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(n):
            if p[i] > 0:
                S -= p[i] * np.log(p[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(n):
                p[i] = p[i] ** w; s2 += p[i]
            if s2 > 0:
                for i in range(n):
                    p[i] /= s2
    return p


@njit(cache=True, fastmath=True)
def prep_query(mz, it, floor, topk, power, ent_weight):
    """Clean (no precursor cut) + weights -> (mz float64, p float64)."""
    n = len(mz)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float64), np.empty(0, np.float64)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float64)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float64)
    oi = np.empty(c, np.float64)
    for i in range(c):
        om[i] = mz[idx[i]]
        oi[i] = it[idx[i]]
    return om, _weights(oi, power, ent_weight)


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol, ppm_tol):
    """Spectral entropy similarity between two weight-normalised, m/z-sorted peak lists.
    Match tolerance = max(tol Da, ppm_tol * mz * 1e-6)."""
    n = len(qmz); m = len(cmz)
    if n == 0 or m == 0:
        return 0.0
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    i = 0; j = 0
    while i < n and j < m:
        t = tol
        pt = qmz[i] * ppm_tol * 1e-6
        if pt > t:
            t = pt
        d = qmz[i] - cmz[j]
        if d < -t:
            v = qp[i] * 0.5; i += 1
        elif d > t:
            v = cp[j] * 0.5; j += 1
        else:
            v = (qp[i] + cp[j]) * 0.5; i += 1; j += 1
        if v > 0:
            SAB -= v * np.log(v)
    while i < n:
        v = qp[i] * 0.5
        if v > 0:
            SAB -= v * np.log(v)
        i += 1
    while j < m:
        v = cp[j] * 0.5
        if v > 0:
            SAB -= v * np.log(v)
        j += 1
    s = 1.0 - (2.0 * SAB - SA - SB) / LOG4
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return s


@njit(cache=True, fastmath=True)
def cosine_sim(qmz, qp, cmz, cp, tol, ppm_tol):
    """Weighted cosine on sqrt-transformed weights (peaks already weight-normalised)."""
    n = len(qmz); m = len(cmz)
    if n == 0 or m == 0:
        return 0.0
    dot = 0.0; na = 0.0; nb = 0.0
    for x in range(n):
        na += qp[x] * qp[x]
    for x in range(m):
        nb += cp[x] * cp[x]
    i = 0; j = 0
    while i < n and j < m:
        t = tol
        pt = qmz[i] * ppm_tol * 1e-6
        if pt > t:
            t = pt
        d = qmz[i] - cmz[j]
        if d < -t:
            i += 1
        elif d > t:
            j += 1
        else:
            dot += qp[i] * cp[j]; i += 1; j += 1
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / np.sqrt(na * nb)


@njit(cache=True, fastmath=True)
def hybrid_sim(qmz, qp, cmz, cp, tol, ppm_tol, shift):
    """Hybrid (direct + mass-shifted) entropy similarity: each candidate peak may match either
    directly or shifted by `shift`; each query peak is used at most once (greedy by intensity).
    Implemented as: merge candidate peaks with both positions, keeping the better assignment."""
    n = len(qmz); m = len(cmz)
    if n == 0 or m == 0:
        return 0.0
    if shift > -1e-4 and shift < 1e-4:
        return entropy_sim(qmz, qp, cmz, cp, tol, ppm_tol)
    # matched intensity for each candidate peak: direct or shifted, whichever exists
    used = np.zeros(n, np.bool_)
    matched_c = np.zeros(m, np.float64)   # query weight assigned to candidate peak j
    # pass 1: direct matches
    i = 0; j = 0
    while i < n and j < m:
        t = tol
        pt = qmz[i] * ppm_tol * 1e-6
        if pt > t:
            t = pt
        d = qmz[i] - cmz[j]
        if d < -t:
            i += 1
        elif d > t:
            j += 1
        else:
            matched_c[j] = qp[i]; used[i] = True; i += 1; j += 1
    # pass 2: shifted matches for still-unmatched candidate peaks
    i = 0; j = 0
    while i < n and j < m:
        cm = cmz[j] + shift
        t = tol
        pt = qmz[i] * ppm_tol * 1e-6
        if pt > t:
            t = pt
        d = qmz[i] - cm
        if d < -t:
            i += 1
        elif d > t:
            j += 1
        else:
            if (not used[i]) and matched_c[j] == 0.0:
                matched_c[j] = qp[i]; used[i] = True
            i += 1; j += 1
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    for j in range(m):
        v = (cp[j] + matched_c[j]) * 0.5
        if v > 0:
            SAB -= v * np.log(v)
    for i in range(n):
        if not used[i]:
            v = qp[i] * 0.5
            if v > 0:
                SAB -= v * np.log(v)
    s = 1.0 - (2.0 * SAB - SA - SB) / LOG4
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return s


@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol, ppm_tol, floor, topk, power, ent_weight, kind):
    """Score query against library spectra `cand` (indices into off). kind 0=entropy 1=cosine."""
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]; b = off[c + 1]
        if b <= a:
            continue
        cm, cp = prep_query(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0:
            continue
        if kind == 0:
            out[k] = entropy_sim(qmz, qp, cm, cp, tol, ppm_tol)
        else:
            out[k] = cosine_sim(qmz, qp, cm, cp, tol, ppm_tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, cand, shift, off, allmz, allin, tol, ppm_tol, floor, topk, power, ent_weight):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]; b = off[c + 1]
        if b <= a:
            continue
        cm, cp = prep_query(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0:
            continue
        out[k] = hybrid_sim(qmz, qp, cm, cp, tol, ppm_tol, shift[k])
    return out


def merge_spectra(pairs, mz_tol=0.005, max_peaks=None):
    """Merge several (mz, intensity) lists of one molecule into one peak list (max intensity
    per near-duplicate m/z; each spectrum first normalised to base peak 1)."""
    mzs, its = [], []
    for mz, it in pairs:
        mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
        if len(mz) == 0 or it.max() <= 0:
            continue
        mzs.append(mz); its.append(it / it.max())
    if not mzs:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mz = np.concatenate(mzs); it = np.concatenate(its)
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    last = 0
    for j in range(1, len(mz)):
        if mz[j] - mz[last] < mz_tol:
            if it[j] >= it[last]:
                keep[last] = False; last = j
            else:
                keep[j] = False
        else:
            last = j
    mz, it = mz[keep], it[keep]
    if max_peaks is not None and len(mz) > max_peaks:
        top = np.argsort(-it)[:max_peaks]; top.sort()
        mz, it = mz[top], it[top]
    return mz.astype(np.float32), it.astype(np.float32)
