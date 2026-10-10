"""Shared pieces for candidate generation and spectral-analog retrieval.

Candidate generation is a mass lookup: the precursor and adduct give a neutral mass,
and every pool structure within tolerance is a candidate. Analog retrieval is the
spectral search, returning library *structures* with a similarity, which is the raw
material for structure-based ranking of those candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- candidate generation --------------------------------------------------

class CandidatePool:
    """Pool structures indexed by exact mass for ppm-window lookup."""

    def __init__(self, index: pd.DataFrame, packed_fp: np.ndarray | None = None):
        self.df = index.reset_index(drop=True)
        self.mass = self.df.mass.to_numpy(dtype=np.float64)
        self.order = np.argsort(self.mass, kind="stable")
        self.mass_sorted = self.mass[self.order]
        self.key14 = self.df.key14.to_numpy()
        self.smiles = self.df.smiles.to_numpy()
        self.packed_fp = packed_fp

    @classmethod
    def load(cls, cache: str = "data/cache", with_fp: bool = True) -> "CandidatePool":
        idx = pd.read_parquet(f"{cache}/pool_index.parquet")
        fp = np.load(f"{cache}/pool_fp_packed.npy") if with_fp else None
        if "fp_ok" in idx.columns and fp is not None:
            keep = idx.fp_ok.to_numpy()
            idx, fp = idx[keep].reset_index(drop=True), fp[keep]
        return cls(idx, fp)

    def candidates(self, neutral_mass: float, ppm: float = 10.0) -> np.ndarray:
        """Pool row indices whose exact mass is within ppm of neutral_mass."""
        if not np.isfinite(neutral_mass):
            return np.zeros(0, dtype=np.int64)
        tol = neutral_mass * ppm * 1e-6
        lo = np.searchsorted(self.mass_sorted, neutral_mass - tol)
        hi = np.searchsorted(self.mass_sorted, neutral_mass + tol)
        return self.order[lo:hi]

    def unpack(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows) x n_bits) uint8 fingerprint matrix."""
        if self.packed_fp is None:
            raise ValueError("pool loaded without fingerprints")
        return np.unpackbits(self.packed_fp[rows], axis=1)


# --- fingerprint similarity ------------------------------------------------

def tanimoto_matrix(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """(n_a x n_b) Tanimoto between two binary fingerprint matrices."""
    a = a_bits.astype(np.float32)
    b = b_bits.astype(np.float32)
    inter = a @ b.T
    sa = a.sum(axis=1)[:, None]
    sb = b.sum(axis=1)[None, :]
    union = sa + sb - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# --- analog retrieval ------------------------------------------------------

def reference_table(cache: str = "data/cache", instrument: str = "tims",
                    min_peaks: int = 3) -> pd.DataFrame:
    """Library spectra joined to their tautomer-canonical structure key."""
    meta = pd.read_parquet(
        f"{cache}/train_meta.parquet",
        columns=["normalized_smiles", "instrument_type", "adduct", "precursor_mz",
                 "peak_n"])
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    structs = pd.read_parquet(f"{cache}/structures.parquet",
                              columns=["normalized_smiles", "key14"])
    meta = meta.merge(structs, on="normalized_smiles", how="left")
    mask = meta.key14.notna() & (meta.peak_n >= min_peaks)
    if instrument == "tims":
        mask &= meta.instrument_type == "timsTOF"
    return meta[mask].reset_index(drop=True)


def analogs_per_query(scores: np.ndarray, idx: np.ndarray, ref_key14: np.ndarray,
                      top_structures: int = 100) -> list[dict[str, float]]:
    """Per query spectrum: {structure key14 -> best spectral similarity}."""
    out = []
    for q in range(scores.shape[0]):
        rows, vals = idx[q], scores[q]
        ok = rows >= 0
        best: dict[str, float] = {}
        for r, v in zip(rows[ok], vals[ok]):
            k = ref_key14[r]
            if v > best.get(k, -1.0):
                best[k] = float(v)
        if len(best) > top_structures:
            best = dict(sorted(best.items(), key=lambda kv: -kv[1])[:top_structures])
        out.append(best)
    return out
