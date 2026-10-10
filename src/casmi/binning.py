"""Vectorised conversion of many spectra into one sparse binned matrix.

Everything operates on the flat peak arrays of the cache, so binning 2.5M spectra
is a handful of numpy passes instead of a Python loop.

Two representations are produced from the same peaks:

  direct       : peaks at their observed m/z. Matches spectra of the same structure.
  neutral-loss : peaks re-expressed as precursor_mz - m/z. A library analog whose
                 structure differs by some modification still aligns here, because a
                 shared loss appears at the same position regardless of the shift.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def gather_rows(store, rows: np.ndarray):
    """Concatenate the peaks of `rows` plus the row id of every peak."""
    rows = np.asarray(rows, dtype=np.int64)
    n = store.n[rows].astype(np.int64)
    off = store.off[rows].astype(np.int64)
    total = int(n.sum())
    if total == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, np.zeros(0, dtype=np.int64), n
    # flat gather indices: for each row, off[i] .. off[i]+n[i]
    starts = np.repeat(off, n)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(n)[:-1])), n
    )
    idx = starts + within
    row_id = np.repeat(np.arange(len(rows), dtype=np.int64), n)
    return (
        np.asarray(store.mz)[idx],
        np.asarray(store.inten)[idx],
        row_id,
        n,
    )


def bin_matrix(
    store,
    rows: np.ndarray,
    precursor_mz: np.ndarray,
    *,
    bin_width: float = 0.01,
    max_mz: float = 1000.0,
    power: float = 0.5,
    neutral_loss: bool = False,
    precursor_guard: float = 1.5,
    min_rel: float = 0.0,
    soft: bool = True,
) -> sparse.csr_matrix:
    """Sparse CSR (len(rows) x n_bins), rows L2-normalised, values intensity**power.

    precursor_guard drops peaks within that distance of the precursor: the surviving
    precursor ion is present in nearly every spectrum and would otherwise dominate
    cosine while carrying no fragmentation information.

    soft=True splits each peak linearly across its two neighbouring bin centres.
    With hard binning, two peaks 0.0002 Da apart can fall either side of a boundary
    and contribute nothing to the cosine; soft binning makes the representation
    continuous in m/z, so near-coincident peaks always overlap. It also removes the
    float32/float64 truncation ambiguity at exact boundaries.
    """
    n_bins = int(round(max_mz / bin_width))
    rows = np.asarray(rows, dtype=np.int64)
    mz, it, row_id, _ = gather_rows(store, rows)
    if mz.size == 0:
        return sparse.csr_matrix((len(rows), n_bins), dtype=np.float32)

    pmz = np.asarray(precursor_mz, dtype=np.float64)[row_id]
    mz = mz.astype(np.float64)
    it = it.astype(np.float64)

    if neutral_loss:
        pos = pmz - mz
        keep = pos > precursor_guard
    else:
        pos = mz
        keep = mz < pmz - precursor_guard
    keep &= (pos >= 0) & (pos < max_mz) & (it > min_rel) & np.isfinite(pos)

    pos, val, row_id = pos[keep], it[keep] ** power, row_id[keep]
    if pos.size == 0:
        return sparse.csr_matrix((len(rows), n_bins), dtype=np.float32)

    if soft:
        f = pos / bin_width - 0.5          # bin centre k sits at (k + 0.5) * bin_width
        lo = np.floor(f).astype(np.int64)
        w = (f - lo).astype(np.float64)    # weight going to lo + 1
        b = np.concatenate([lo, lo + 1])
        val = np.concatenate([val * (1.0 - w), val * w])
        row_id = np.concatenate([row_id, row_id])
        ok = (b >= 0) & (b < n_bins) & (val > 0)
        b, val, row_id = b[ok], val[ok], row_id[ok]
    else:
        b = np.minimum((pos / bin_width).astype(np.int64), n_bins - 1)

    flat = row_id * n_bins + b
    order = np.argsort(flat, kind="stable")
    flat, val, row_id = flat[order], val[order], row_id[order]
    uniq, start = np.unique(flat, return_index=True)
    val = np.add.reduceat(val, start)
    row_id = row_id[start]

    # L2-normalise each row
    sq = np.bincount(row_id, weights=val * val, minlength=len(rows))
    nrm = np.sqrt(sq)
    nrm[nrm == 0] = 1.0
    val = val / nrm[row_id]

    cols = (uniq % n_bins).astype(np.int32)
    counts = np.bincount(row_id, minlength=len(rows))
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return sparse.csr_matrix(
        (val.astype(np.float32), cols, indptr), shape=(len(rows), n_bins)
    )
