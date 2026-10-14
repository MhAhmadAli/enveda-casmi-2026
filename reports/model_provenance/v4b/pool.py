"""Candidate pool (train structures U COCONUT) with mass windows, fingerprints, fragments."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from . import frag as fragmod


class Pool:
    def __init__(self, workdir='work', load_frag=True, verbose=True):
        t0 = time.time()
        meta = pd.read_parquet(f'{workdir}/pool_meta.parquet')
        self.mass = meta.mass.values.astype(np.float64)          # sorted ascending
        self.key = meta.key.values.astype(object)
        self.smiles = meta.smiles.values.astype(object)
        self.formula = meta.formula.values.astype(object)
        self.n_heavy = meta.n_heavy.values.astype(np.int32)
        self.src = meta.src.values.astype(np.int8)
        self.train_sid = meta.train_sid.values.astype(np.int64)
        self.fp_packed = np.load(f'{workdir}/pool_fp.npy')
        self.bits = np.load(f'{workdir}/fp_bits.npy')
        self.nbits = int(len(self.bits))
        self.key2pid = dict(zip(self.key, range(len(self.key))))
        self.frag_off = self.frag_mass = None
        if load_frag and os.path.exists(f'{workdir}/pool_frag_off.npy'):
            self.frag_off = np.load(f'{workdir}/pool_frag_off.npy')
            self.frag_mass = np.load(f'{workdir}/pool_frag_mass.npy')
        self._fvec_cache = {}
        if verbose:
            print(f'[Pool] {len(self.mass):,} structures, nbits {self.nbits}, frag {"yes" if self.frag_off is not None else "no"} '
                  f'({time.time()-t0:.1f}s)', flush=True)

    def window(self, target: float, ppm: float) -> np.ndarray:
        tol = target * ppm * 1e-6
        a = np.searchsorted(self.mass, target - tol, 'left')
        b = np.searchsorted(self.mass, target + tol, 'right')
        return np.arange(a, b)

    def fps(self, pids) -> np.ndarray:
        return np.unpackbits(self.fp_packed[np.asarray(pids)], axis=1)[:, :self.nbits]

    def frags(self, pid: int) -> np.ndarray:
        if self.frag_off is None:
            return fragmod.fragments_for_smiles(self.smiles[pid])
        return self.frag_mass[self.frag_off[pid]:self.frag_off[pid + 1]].astype(np.float64)

    def fvec(self, pid: int):
        f = self.formula[pid]
        if f not in self._fvec_cache:
            self._fvec_cache[f] = fragmod.formula_vec(f)
        return self._fvec_cache[f]
