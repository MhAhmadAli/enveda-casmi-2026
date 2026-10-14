"""Spectral library index: direct (class-1) search and mass-shifted analog search."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import spectra as sp


@dataclass
class SearchCfg:
    floor: float = 0.002      # relative intensity floor for similarity
    topk: int = 256           # peaks kept for similarity
    power: float = 1.0        # intensity power before normalisation
    ent_weight: bool = True   # entropy weighting
    tol: float = 0.01         # Da tolerance
    ppm_tol: float = 20.0     # ppm tolerance (max of the two)
    ppm_win: float = 10.0     # precursor window for direct search
    analog_win: float = 200.0 # Da window for analog search
    n_analog: int = 100


class Library:
    """Holds the cleaned spectra of train.parquet plus per-spectrum metadata."""

    def __init__(self, workdir: str = 'work', verbose: bool = True):
        t0 = time.time()
        self.off = np.load(f'{workdir}/spec_off.npy')
        self.mz = np.load(f'{workdir}/spec_mz.npy')
        self.it = np.load(f'{workdir}/spec_it.npy')
        self.meta = pd.read_parquet(f'{workdir}/spec_meta.parquet')
        st = pd.read_parquet(f'{workdir}/train_structs.parquet', columns=['sid', 'mass', 'key', 'smiles'])
        self.struct_mass = st.mass.values.astype(np.float64)
        self.struct_key = st.key.values.astype(object)
        self.struct_smiles = st.smiles.values.astype(object)
        self.sid = self.meta.sid.values.astype(np.int64)
        self.lib = self.meta.lib.values.astype(np.int8)
        self.mode = self.meta["mode"].values.astype(np.int8)
        self.instr = self.meta.instr.values.astype(np.int8)
        self.adduct_ix = self.meta.adduct_ix.values.astype(np.int16)
        self.prec = self.meta.precursor_mz.values.astype(np.float64)
        self.n_clean = self.meta.n_clean.values.astype(np.int32)
        # neutral mass of the *labelled structure* (robust to precursor errors)
        self.smass = self.struct_mass[self.sid]
        self.order = np.argsort(self.smass, kind='mergesort')
        self.smass_sorted = self.smass[self.order]
        self._rep = {}
        if verbose:
            print(f'[Library] {len(self.sid):,} spectra, {len(self.struct_mass):,} structures, '
                  f'{len(self.mz):,} peaks ({time.time()-t0:.1f}s)', flush=True)

    # ------------------------------------------------------------------ windows
    def window(self, target: float, ppm: float, exclude: Optional[np.ndarray] = None,
               mode: Optional[int] = None) -> np.ndarray:
        tol = target * ppm * 1e-6
        lo = np.searchsorted(self.smass_sorted, target - tol, 'left')
        hi = np.searchsorted(self.smass_sorted, target + tol, 'right')
        c = self.order[lo:hi]
        if mode is not None:
            c = c[self.mode[c] == mode]
        if exclude is not None and len(c):
            c = c[~exclude[c]]
        return c

    # ------------------------------------------------------------------ representatives
    def representatives(self, exclude: Optional[np.ndarray] = None, key: str = 'default'):
        """One representative spectrum per (structure, polarity): the one with most cleaned peaks
        (ties -> larger n). Cached per key. Returns dict(idx, sid, mass, mode) sorted by mass."""
        if key in self._rep:
            return self._rep[key]
        ok = np.ones(len(self.sid), bool) if exclude is None else ~exclude
        idx = np.where(ok)[0]
        # score = n_clean; pick argmax per (sid, mode)
        grp = self.sid[idx].astype(np.int64) * 2 + (self.mode[idx] > 0)
        order = np.lexsort((-self.n_clean[idx], grp))
        g_sorted = grp[order]
        first = np.ones(len(order), bool)
        first[1:] = g_sorted[1:] != g_sorted[:-1]
        rep = idx[order[first]]
        m = self.smass[rep]
        o = np.argsort(m, kind='mergesort')
        rep = rep[o]
        out = dict(idx=rep, sid=self.sid[rep], mass=m[o], mode=self.mode[rep])
        self._rep[key] = out
        return out

    # ------------------------------------------------------------------ scoring
    def prep(self, mz, it, cfg: SearchCfg):
        return sp.prep_query(np.asarray(mz, np.float64), np.asarray(it, np.float64),
                             cfg.floor, cfg.topk, cfg.power, cfg.ent_weight)

    def direct_search(self, queries: Sequence[Tuple[np.ndarray, np.ndarray, int]], target: float,
                      cfg: SearchCfg, exclude: Optional[np.ndarray] = None,
                      same_mode: bool = True, kind: int = 0):
        """queries: list of (mz, it, mode). Returns dict: sid -> dict(max, mean_top, n_hits, best_spec)
        plus per-query-spectrum best similarity for diagnostics."""
        agg: Dict[int, List[float]] = {}
        per_q = []
        for mz, it, mode in queries:
            qm, qp = self.prep(mz, it, cfg)
            if len(qm) == 0:
                per_q.append(0.0); continue
            cand = self.window(target, cfg.ppm_win, exclude, mode if same_mode else None)
            if len(cand) == 0:
                per_q.append(0.0); continue
            sc = sp.search(qm, qp, cand, self.off, self.mz, self.it, cfg.tol, cfg.ppm_tol,
                           cfg.floor, cfg.topk, cfg.power, cfg.ent_weight, kind)
            per_q.append(float(sc.max()))
            for c, s in zip(cand, sc):
                agg.setdefault(int(self.sid[c]), []).append(float(s))
        out = {}
        for s, v in agg.items():
            v = np.asarray(v)
            out[s] = dict(max=float(v.max()), mean=float(v.mean()), n=len(v),
                          top3=float(np.sort(v)[-3:].mean()))
        return out, per_q

    def analog_search(self, queries: Sequence[Tuple[np.ndarray, np.ndarray, int]], target: float,
                      cfg: SearchCfg, rep: dict, lib_filter: Optional[np.ndarray] = None):
        """Mass-shifted hybrid similarity against representative spectra within +-analog_win Da.
        rep: output of representatives(). lib_filter: optional boolean mask over spectra to allow.
        Returns list of (sid, sim, shift) sorted by sim desc (top n_analog), sim = max over query spectra."""
        lo = np.searchsorted(rep['mass'], target - cfg.analog_win, 'left')
        hi = np.searchsorted(rep['mass'], target + cfg.analog_win, 'right')
        cand_all = rep['idx'][lo:hi]
        mass_all = rep['mass'][lo:hi]
        mode_all = rep['mode'][lo:hi]
        if lib_filter is not None:
            keep = lib_filter[cand_all]
            cand_all, mass_all, mode_all = cand_all[keep], mass_all[keep], mode_all[keep]
        best: Dict[int, Tuple[float, float]] = {}
        for mz, it, mode in queries:
            qm, qp = self.prep(mz, it, cfg)
            if len(qm) == 0:
                continue
            sel = mode_all == mode
            cand = cand_all[sel]
            if len(cand) == 0:
                continue
            shift = (target - mass_all[sel]).astype(np.float64)
            sc = sp.search_shift(qm, qp, cand, shift, self.off, self.mz, self.it, cfg.tol, cfg.ppm_tol,
                                 cfg.floor, cfg.topk, cfg.power, cfg.ent_weight)
            for c, s, sh in zip(cand, sc, shift):
                sid = int(self.sid[c])
                if s > best.get(sid, (-1.0, 0.0))[0]:
                    best[sid] = (float(s), float(sh))
        items = sorted(best.items(), key=lambda x: -x[1][0])[:cfg.n_analog]
        return [(s, v[0], v[1]) for s, v in items]
