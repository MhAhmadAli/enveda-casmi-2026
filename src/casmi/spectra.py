"""Access to the memmapped peak cache, plus exact peak-level spectral comparison.

Bulk vectorisation for binned representations lives in casmi.binning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/cache")


@dataclass
class PeakStore:
    """Memmapped (mz, intensity) arrays addressed by per-spectrum (offset, n)."""
    mz: np.memmap
    inten: np.memmap
    off: np.ndarray
    n: np.ndarray

    @classmethod
    def load(cls, split: str = "train", cache: Path = CACHE) -> "PeakStore":
        info = json.load(open(cache / f"{split}_cache.json"))
        npk = info["n_peaks"]
        mz = np.memmap(cache / f"{split}_mz.f32", dtype=np.float32, mode="r", shape=(npk,))
        it = np.memmap(cache / f"{split}_int.f32", dtype=np.float32, mode="r", shape=(npk,))
        meta = pd.read_parquet(cache / f"{split}_meta.parquet",
                               columns=["peak_off", "peak_n"])
        return cls(mz, it, meta.peak_off.to_numpy(), meta.peak_n.to_numpy())

    def __len__(self) -> int:
        return len(self.off)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        o, k = self.off[i], self.n[i]
        return np.asarray(self.mz[o:o + k]), np.asarray(self.inten[o:o + k])


def modified_cosine(mz_a, it_a, pmz_a, mz_b, it_b, pmz_b,
                    tol: float = 0.01, power: float = 0.5,
                    allow_shift: bool = True) -> tuple[float, int]:
    """Greedy one-to-one peak matching allowing a precursor-difference shift.

    Returns (score, n_matched). Peaks may match directly (|da - db| <= tol) or
    shifted by the precursor mass difference, which is how a fragment that retains
    a modified moiety lines up between an unknown and a library analog.
    """
    a = it_a.astype(np.float64) ** power
    b = it_b.astype(np.float64) ** power
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0
    shift = (pmz_a - pmz_b) if allow_shift else 0.0
    offsets = {0.0} | ({shift} if allow_shift and abs(shift) > tol else set())
    pairs = []
    for off in offsets:
        target = mz_a - off                      # mz_b value that would match each mz_a
        j_lo = np.searchsorted(mz_b, target - tol)
        j_hi = np.searchsorted(mz_b, target + tol)
        for i in np.nonzero(j_hi > j_lo)[0]:
            for j in range(j_lo[i], j_hi[i]):
                pairs.append((a[i] * b[j], int(i), int(j)))
    pairs.sort(key=lambda p: -p[0])
    used_a, used_b, score, nm = set(), set(), 0.0, 0
    for w, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); score += w; nm += 1
    return float(score / (na * nb)), nm
