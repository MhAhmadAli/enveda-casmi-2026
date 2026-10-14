"""CASMI26 v4 on the v3.6 backbone: the v1 engine (kaggle/ds_models/code = engine of the 0.358 model, its 94 features
unchanged) + recording hooks + fe_v4 feature families. Model views for the families come from FP models run through
casmi.fpnet6 (engine v6.3 fpnet incl. DreamsFP) on engine-v6.3-style items (cleaned spectra, deterministic merged adduct),
identically offline (simulate_v1ctx.py) and online (the v4 notebook).

    V = V1FE(E1, [A, B, D], families)
    C, X, F, names, info = V.run(spectra, target)            # X: v1 94 features; F: family features 'fam__feat'
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np

import ctxlib
import hooks_v1
from casmi import chem
from casmi.spectra import merge_spectra

# spectrum counts per adduct in train.parquet (engine v6.3 merged-view adduct tie-break)
ADDUCT_TRAIN_COUNT = {
    "[M+H]+": 1324585, "[M-H]-": 465688, "[2M+Na]+": 190033, "[M+Na]+": 178092, "[2M+H]+": 92041,
    "[M+CH2O2-H]-": 83188, "[2M-H]-": 56172, "[M+NH4]+": 53452, "[M+Cl]-": 27511, "[M-H2O+H]+": 21682,
    "[M+K]+": 18955, "[M]+": 8818, "[M+C2H4O2-H]-": 7436, "[M-2H2O+H]+": 3439, "[M+2H]2+": 1717,
    "[2M+Na-2H]-": 1431, "[2M+CH2O2-H]-": 665, "[M-H2O]+": 624, "[M-CH3]-": 619, "[M-2H]-": 454, "[M]-": 446,
    "[2M+NH4]+": 378, "[M-H2O-H]-": 285, "[M-H5O3]+": 230, "[2M+C2H4O2-H]-": 187, "[M+C2H4N]+": 158,
    "[M+3H]3+": 138, "[M+2Na-H]+": 110, "[2M+K]+": 109, "[M-H2]2-": 103, "[M+Ca]2+": 85, "[M+CH5O]+": 67,
    "[M-H2N]+": 61, "[M+HNa]2+": 44, "[M+Li]+": 6,
}


def merged_adduct(adducts: Sequence[str]) -> str:
    cnt: Dict[str, int] = {}
    for a in adducts:
        cnt[a] = cnt.get(a, 0) + 1
    best = max(cnt.values())
    return min((a for a in cnt if cnt[a] == best), key=lambda a: (-ADDUCT_TRAIN_COUNT.get(a, 0), a))


def v6_items(spectra):
    """== engine v6.3 Engine.model_logits item construction (spectra already prepared/cleaned)."""
    from casmi import fpnet6
    items_s = []
    for s in spectra:
        pm, pi = fpnet6.prep_peaks(s['mz'], s['it'], s['prec'])
        ce_ok = s['ce'] is not None and not np.isnan(s['ce'])
        items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']),
                            ce=float(s['ce']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(max(1, int(s.get('ce_n', 1))), 8), mode=float(s['mode'])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
        prec = float(np.median([s['prec'] for s in grp]))
        pm, pi = fpnet6.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
        ad = merged_adduct([s['adduct'] for s in grp])
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    return items_s, items_m


class V1FE:
    def __init__(self, E1, model_paths: Sequence[str], families: List[str], device: str = 'cuda'):
        from casmi import fpnet6
        self.E = E1
        hooks_v1.install(E1)
        self.banks = [fpnet6.ModelBank([p], device=device) for p in model_paths]     # slots 0, 1, 2 = A, B, D
        self.res = ctxlib.resources_from_engine(E1)
        self.families = list(families)
        self.mods = {f: importlib.import_module(f'fams.{f}') for f in self.families}
        for m in self.mods.values():
            if hasattr(m, 'prepare'):
                m.prepare(self.res)
        self.names: Optional[List[str]] = None

    def model_views(self, prep):
        items_s, items_m = v6_items(prep)
        zs, es, zm, em = [], [], [], []
        for b in self.banks:
            a, c = b.logits_el_each(items_s); d, e = b.logits_el_each(items_m)
            zs.append(a); es.append(c); zm.append(d); em.append(e)
        cat = lambda L: np.concatenate(L).astype(np.float32)
        return cat(zs), cat(es), cat(zm), cat(em), np.array([it['mode'] for it in items_m], np.float32)

    def context(self, spectra, target, C, X):
        """Everything the families need (the same record replay_ctx.py writes, plus the prepared spectra)."""
        E = self.E
        prep = self.res.prepare(spectra)
        zs, es, zm, em, modes = self.model_views(prep)
        gen = {g['key']: g for g in (E._last_gen or [])}
        pid = np.asarray(C.pid, np.int64)
        gi = np.flatnonzero(pid < 0)
        an = E._last_an or []
        return dict(
            n_cand=len(pid), pid=pid.astype(np.int32), chan=np.where(pid < 0, 2, 0).astype(np.int8),
            key_gen=[C.key[i] for i in gi], smiles_gen=[C.smiles[i] for i in gi], formula_gen=[C.formula[i] for i in gi],
            mass_gen=np.asarray(C.mass)[gi], n_heavy_gen=np.asarray(C.n_heavy)[gi],
            fp_gen=np.packbits(np.asarray(C.fp)[gi].astype(np.uint8), axis=1),
            parent_gen=np.array([gen[C.key[i]]['parent'] if C.key[i] in gen else -1 for i in gi], np.int64),
            gen_sim=np.asarray(C.gen_sim)[gi], gen_steps=np.asarray(C.gen_steps)[gi], gen_tan=np.asarray(C.gen_tan)[gi],
            combo_gen=[gen[C.key[i]].get('combo') if C.key[i] in gen else None for i in gi],
            raw_gen=[gen[C.key[i]].get('raw') if C.key[i] in gen else None for i in gi],
            an_sid=np.array([x[0] for x in an], np.int64), an_sim=np.array([x[1] for x in an], np.float32),
            an_shift=np.array([x[2] for x in an], np.float64), an_np=np.array([x[3] for x in an], bool),
            z_s=zs, el_s=es, z_m=zm, el_m=em, m_modes=modes, derlog=list(E._derlog or []),
            gen_order=[g['key'] for g in (E._last_gen or [])], n_gen=int(len(gi))), prep

    def view(self, rec, prep, target, C, X, features_v1):
        P = self.res.P
        pid = rec['pid'].astype(np.int64); n = len(pid); gi = np.flatnonzero(pid < 0)
        parent = np.full(n, -1, np.int64); parent[gi] = rec['parent_gen']
        combo = [None] * n
        for t, j in enumerate(gi):
            combo[j] = rec['combo_gen'][t]
        return ctxlib.QueryView(
            target=float(target), spectra=prep, pid=pid, key=list(C.key), smiles=list(C.smiles), formula=list(C.formula),
            mass=np.asarray(C.mass, np.float64), n_heavy=np.asarray(C.n_heavy, np.int32), fp=np.asarray(C.fp, np.float32),
            chan=rec['chan'], is_gen=(pid < 0).astype(np.float32), parent_sid=parent,
            gen_sim=np.asarray(C.gen_sim, np.float32), gen_steps=np.asarray(C.gen_steps, np.float32),
            gen_tan=np.asarray(C.gen_tan, np.float32), gen_combo=combo, an_sid=rec['an_sid'], an_sim=rec['an_sim'],
            an_shift=rec['an_shift'], an_np=rec['an_np'], z_s=rec['z_s'], el_s=rec['el_s'], z_m=rec['z_m'],
            el_m=rec['el_m'], m_modes=rec['m_modes'], X=ctxlib.to_v6_layout(X, features_v1), res=self.res,
            derlog=rec['derlog'])

    def features(self, v):
        cols, names = [], []
        for f in self.families:
            d = self.mods[f].compute(v)
            for k in sorted(d):
                cols.append(np.asarray(d[k], np.float32).reshape(-1)); names.append(f'{f}__{k}')
        return (np.stack(cols, 1) if cols else np.zeros((v.n, 0), np.float32)), names

    def run(self, spectra, target, want_record=False, **sim):
        from casmi.engine import FEATURES as F1
        E = self.E
        hooks_v1.reset(E)
        C, X, info = E.run(spectra, target, **sim)
        if len(C.pid) == 0:
            return (C, X, np.zeros((0, 0), np.float32), [], info) + ((None,) if want_record else ())
        rec, prep = self.context(spectra, target, C, X)
        F, names = np.zeros((len(C.pid), 0), np.float32), []
        if self.families:
            F, names = self.features(self.view(rec, prep, target, C, X, list(F1)))
            if self.names is None:
                self.names = names
            assert names == self.names
        return (C, X, F, names, info) + ((rec,) if want_record else ())
