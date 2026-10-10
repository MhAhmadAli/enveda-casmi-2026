"""Spectral entropy similarity (Li et al., Nat. Methods 2021) on raw peak lists.

Binned cosine was a compromise forced by scanning the whole library: it makes the
search a sparse matmul, at the cost of quantising m/z. But a *direct* search only
ever needs library spectra whose neutral mass already matches the query, which is a
few hundred to a few thousand spectra. At that scale we can afford exact
tolerance-based peak alignment and a better similarity measure.

Entropy similarity differs from cosine in two ways that matter here:

  - intensities are treated as a probability distribution and compared through the
    entropy of their mixture, which punishes a match that relies on one dominant
    peak -- precisely the failure mode of cosine on sparse high-energy spectra;
  - low-entropy spectra (few informative peaks) get their intensities flattened by
    a weight w = 0.25 + 0.25*S, so a 3-peak spectrum cannot score a confident match
    on intensity agreement alone.

`shift` supports analog search: a fragment that retains a modified moiety appears
displaced by the precursor mass difference, so the shifted alignment is scored too
and the better of the two kept.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

LN4 = float(np.log(4.0))


@njit(cache=True, fastmath=True)
def prepare(mz, inten, min_rel, max_peaks, entropy_weight):
    """Clean one spectrum and return (m/z, probability) with intensities summing to 1.

    Peaks are floored at min_rel of the base peak, capped to the max_peaks most
    intense, and renormalised. With entropy_weight, a low-entropy spectrum has its
    intensities raised to w = 0.25 + 0.25*S (S < 3), flattening it so that a sparse
    spectrum cannot claim a confident intensity match.
    """
    n = mz.shape[0]
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if inten[i] > mx:
            mx = inten[i]
    if mx <= 0.0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    thr = min_rel * mx
    cnt = 0
    for i in range(n):
        if inten[i] >= thr and mz[i] > 0.0:
            cnt += 1
    if cnt == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    idx = np.empty(cnt, np.int64)
    j = 0
    for i in range(n):
        if inten[i] >= thr and mz[i] > 0.0:
            idx[j] = i
            j += 1

    if cnt > max_peaks:
        vals = np.empty(cnt, np.float32)
        for i in range(cnt):
            vals[i] = inten[idx[i]]
        order = np.argsort(vals)
        keep = np.empty(max_peaks, np.int64)
        for i in range(max_peaks):
            keep[i] = idx[order[cnt - max_peaks + i]]
        keep.sort()
        idx = keep
        cnt = max_peaks

    out_mz = np.empty(cnt, np.float32)
    out_p = np.empty(cnt, np.float32)
    tot = 0.0
    for i in range(cnt):
        out_mz[i] = mz[idx[i]]
        v = inten[idx[i]]
        out_p[i] = v
        tot += v
    if tot <= 0.0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    for i in range(cnt):
        out_p[i] /= tot

    if entropy_weight:
        s = 0.0
        for i in range(cnt):
            if out_p[i] > 0.0:
                s -= out_p[i] * np.log(out_p[i])
        if s < 3.0:
            w = 0.25 + 0.25 * s
            tot2 = 0.0
            for i in range(cnt):
                out_p[i] = out_p[i] ** w
                tot2 += out_p[i]
            if tot2 > 0.0:
                for i in range(cnt):
                    out_p[i] /= tot2
    return out_mz, out_p


@njit(cache=True, fastmath=True)
def _entropy(p):
    s = 0.0
    for i in range(p.shape[0]):
        if p[i] > 0.0:
            s -= p[i] * np.log(p[i])
    return s


@njit(cache=True, fastmath=True)
def entropy_similarity(amz, ap, bmz, bp, tol):
    """1 - (2*S_AB - S_A - S_B)/ln4 over a merged two-pointer alignment.

    Both inputs must be m/z-sorted probability vectors. Returns 0 for no overlap and
    1 for identical spectra.
    """
    n = amz.shape[0]
    m = bmz.shape[0]
    if n == 0 or m == 0:
        return 0.0
    sa = _entropy(ap)
    sb = _entropy(bp)

    # merged distribution: matched peaks add, unmatched carry through
    merged = np.empty(n + m, np.float64)
    k = 0
    i = 0
    j = 0
    while i < n and j < m:
        d = amz[i] - bmz[j]
        if d < -tol:
            merged[k] = ap[i]
            i += 1
        elif d > tol:
            merged[k] = bp[j]
            j += 1
        else:
            merged[k] = ap[i] + bp[j]
            i += 1
            j += 1
        k += 1
    while i < n:
        merged[k] = ap[i]
        i += 1
        k += 1
    while j < m:
        merged[k] = bp[j]
        j += 1
        k += 1

    tot = 0.0
    for x in range(k):
        tot += merged[x]
    if tot <= 0.0:
        return 0.0
    sab = 0.0
    for x in range(k):
        v = merged[x] / tot
        if v > 0.0:
            sab -= v * np.log(v)
    sim = 1.0 - (2.0 * sab - sa - sb) / LN4
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


@njit(cache=True, fastmath=True)
def entropy_similarity_shift(amz, ap, bmz, bp, tol, shift):
    """Best of the direct alignment and one displaced by `shift` (analog search)."""
    direct = entropy_similarity(amz, ap, bmz, bp, tol)
    if shift > -tol and shift < tol:
        return direct
    shifted_mz = np.empty(bmz.shape[0], np.float32)
    for i in range(bmz.shape[0]):
        shifted_mz[i] = bmz[i] + shift
    alt = entropy_similarity(amz, ap, shifted_mz, bp, tol)
    return direct if direct > alt else alt


@njit(cache=True, parallel=True, fastmath=True)
def score_against(qmz, qp, cand, off, npk, all_mz, all_in, tol,
                  min_rel, max_peaks, entropy_weight, shifts):
    """Entropy similarity of one prepared query against a set of library spectra.

    cand indexes into (off, npk); shifts is per-candidate (0.0 for direct search,
    precursor difference for analog search).
    """
    out = np.zeros(cand.shape[0], np.float32)
    for k in prange(cand.shape[0]):
        c = cand[k]
        a = off[c]
        b = a + npk[c]
        if b <= a:
            continue
        cmz, cp = prepare(all_mz[a:b], all_in[a:b], min_rel, max_peaks,
                          entropy_weight)
        if cmz.shape[0] == 0:
            continue
        out[k] = entropy_similarity_shift(qmz, qp, cmz, cp, tol, shifts[k])
    return out
