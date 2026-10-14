"""Evidence engine: turns one molecule's spectra into a candidate list with ranking features.

Channels
  L  library search      (entropy similarity to visible reference spectra of the same structure)
  A  analog propagation  (mass-shifted similarity to relatives, propagated through Tanimoto)
  M  spectrum->FP model  (f.z Bayes score; per-spectrum and merged views; element head)
  F  fragmentation       (MetFrag-lite peak explanation)
  S  subformula          (formula-level peak explanation)
  G  generated derivatives of strong analogs (class-3 candidates), flagged for the ranker
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import chem, frag as fragmod, fpnet
from .library import Library, SearchCfg
from .pool import Pool
from . import spectra as spmod
from .spectra import merge_spectra, prep_query

NP_LIBS = np.array([2, 3, 4, 5, 6, 7, 10])   # riken gnps massbank mona spectraverse msdial masaryk
ELEMS = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']


@dataclass
class EngineCfg:
    ppm_win: float = 10.0
    ppm_fallback: float = 30.0
    search: SearchCfg = field(default_factory=SearchCfg)
    sim_power: float = 4.0
    n_analog: int = 100
    analog_query: str = 'merged'     # merged | each
    max_query_spectra: int = 16
    use_frag: bool = True
    use_subformula: bool = True
    frag_max_cand: int = 600         # runtime guard (ranked by model score if exceeded)
    generate: bool = False           # class-3 derivative generation
    gen_n_analog: int = 6            # analogs used as parents
    gen_min_sim: float = 0.3
    gen_max_per_parent: int = 60
    gen_max_total: int = 150
    gen_tol_da: float = 0.004


def _rank_norm(x):
    o = np.argsort(-x, kind='stable'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _tanimoto(cf, af):
    cf = cf.astype(np.float32); af = af.astype(np.float32)
    inter = cf @ af.T
    return inter / (cf.sum(1)[:, None] + af.sum(1)[None, :] - inter + 1e-9)


FEATURES: List[str] = [
    'lib_max', 'lib_mean', 'lib_top3', 'lib_nspec', 'lib_cos', 'lib_max_rank', 'lib_gap', 'lib_grp_max', 'lib_hit', 'lib_same_adduct',
    'ap', 'a1', 'best_tan', 'top_tan', 'mean_tan', 'ap_rank', 'ap_gap', 'ap_grp_max', 'top_sim',
    'ap_np', 'best_tan_np', 'top_sim_np', 'ap_np_gap', 'ap_sum', 'n_analog_hi', 'tan_zero_shift',
    'fz', 'fz_z', 'fz_rank', 'fz_gap', 'fz_norm', 'fz_top', 'fz_softmax', 'fz_grp_ent', 'fz_grp_gap',
    'fz_single', 'fz_single_gap', 'fz_merged', 'fz_merged_gap', 'fp_cos', 'fp_tan', 'fp_ll', 'fp_ll_gap',
    'el_c', 'el_n', 'el_o', 'el_sum', 'el_rank',
    'fr_int', 'fr_int_mean', 'fr_cnt', 'fr_top10', 'fr_strict', 'fr_rank', 'fr_gap', 'fr_z', 'fr_n',
    'sf_int', 'sf_cnt', 'sf_top10', 'sf_rank', 'sf_gap', 'formula_share', 'n_formula', 'formula_is_top',
    'mass_ppm', 'mass_ppm_rank', 'n_heavy', 'fp_bits', 'n_cand', 'n_query', 'n_pos', 'n_neg', 'target_mass',
    'q_npeaks', 'lib_x_fz', 'ap_x_fz', 'agree_lib', 'agree_ap', 'fr_x_fz',
    'is_gen', 'gen_parent_sim', 'gen_steps', 'gen_parent_tan', 'n_gen', 'gen_rank_fz',
    'xs_max', 'xs_mean', 'xs_merged', 'xs_gap', 'xs_rank', 'xs_z', 'xs_top', 'xs_grp_gap',
]
N_FEAT = len(FEATURES)
COL = {n: i for i, n in enumerate(FEATURES)}


@dataclass
class Cands:
    """Candidate set = pool entries (pid >= 0) plus generated structures (pid = -1)."""
    pid: np.ndarray
    key: List[str]
    smiles: List[str]
    formula: List[str]
    mass: np.ndarray
    n_heavy: np.ndarray
    train_sid: np.ndarray
    fp: np.ndarray                  # (n, nbits) float32
    frags: List[np.ndarray]
    is_gen: np.ndarray
    gen_sim: np.ndarray
    gen_steps: np.ndarray
    gen_tan: np.ndarray


class Engine:
    def __init__(self, L: Library, pool: Pool, train_fp: np.ndarray, cfg: EngineCfg,
                 bank: Optional[fpnet.ModelBank] = None, verbose=True, xscorer=None, gstore=None):
        """train_fp: packed (n_train_structs, nbytes) uint8 fingerprints of the selected bits (for analogs).
        xscorer: optional xenc.XScorer; gstore: graphs.GraphStore aligned with the pool."""
        self.L = L; self.pool = pool; self.cfg = cfg; self.bank = bank
        self.train_fp = train_fp; self.xscorer = xscorer; self.gstore = gstore
        t0 = time.time()
        self.rep = self._reps()
        if verbose:
            print(f'[Engine] reps {len(self.rep["idx"]):,} ({time.time()-t0:.1f}s)', flush=True)

    def _reps(self):
        """One representative spectrum per (structure, polarity, library): most cleaned peaks."""
        L = self.L
        grp = (L.sid.astype(np.int64) * 2 + (L.mode > 0)) * 16 + L.lib.astype(np.int64)
        order = np.lexsort((-L.n_clean, grp))
        g = grp[order]
        first = np.ones(len(order), bool); first[1:] = g[1:] != g[:-1]
        rep = order[first]
        m = L.smass[rep]; o = np.argsort(m, kind='mergesort'); rep = rep[o]
        return dict(idx=rep, sid=L.sid[rep], mass=m[o], mode=L.mode[rep], lib=L.lib[rep],
                    is_np=np.isin(L.lib[rep], NP_LIBS))

    # ------------------------------------------------------------------------------------
    def analogs(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1):
        """Top analogs: list of (sid, sim, shift, is_np) sorted by sim desc."""
        cfg = self.cfg; L = self.L
        if cfg.analog_query == 'merged':
            aq = []
            for md in (1, -1):
                grp = [(s['mz'], s['it']) for s in spectra if s['mode'] == md]
                if grp:
                    mm, ii = merge_spectra(grp); aq.append((mm, ii, md))
        else:
            aq = [(s['mz'], s['it'], s['mode']) for s in spectra]
        rep = self.rep
        lo = np.searchsorted(rep['mass'], target - cfg.search.analog_win, 'left')
        hi = np.searchsorted(rep['mass'], target + cfg.search.analog_win, 'right')
        ridx = rep['idx'][lo:hi]; rmass = rep['mass'][lo:hi]; rmode = rep['mode'][lo:hi]
        rsid = rep['sid'][lo:hi]; rlib = rep['lib'][lo:hi]; rnp = rep['is_np'][lo:hi]
        keep = np.ones(len(ridx), bool)
        if exclude_sid >= 0:
            if exclude_lib >= 0:
                keep &= ~((rsid == exclude_sid) & (rlib == exclude_lib))
            else:
                keep &= rsid != exclude_sid
        if exclude is not None:
            keep &= ~exclude[ridx]
        ridx, rmass, rmode, rsid, rnp = ridx[keep], rmass[keep], rmode[keep], rsid[keep], rnp[keep]
        best: Dict[int, Tuple[float, float, bool]] = {}
        for mz, it, md in aq:
            qm, qp = L.prep(mz, it, cfg.search)
            if len(qm) == 0:
                continue
            sel = rmode == md
            if not sel.any():
                continue
            cidx = ridx[sel]; shift = (target - rmass[sel]).astype(np.float64)
            sc = spmod.search_shift(qm, qp, cidx, shift, L.off, L.mz, L.it, cfg.search.tol, cfg.search.ppm_tol,
                                    cfg.search.floor, cfg.search.topk, cfg.search.power, cfg.search.ent_weight)
            csid = rsid[sel]; cnp = rnp[sel]
            ntop = min(4 * cfg.n_analog, len(sc))
            topj = np.argpartition(-sc, ntop - 1)[:ntop] if ntop < len(sc) else np.arange(len(sc))
            for j in topj:
                s = int(csid[j]); v = float(sc[j])
                if v > best.get(s, (-1.0, 0.0, False))[0]:
                    best[s] = (v, float(shift[j]), bool(cnp[j]))
        items = sorted(best.items(), key=lambda x: -x[1][0])[:cfg.n_analog]
        return [(s, v[0], v[1], v[2]) for s, v in items]

    def generate(self, analogs, target, exclude_keys: set) -> List[dict]:
        """Derivatives of the strongest analogs whose mass matches the target."""
        from . import derive
        cfg = self.cfg; L = self.L
        out = []; seen = set(exclude_keys)
        parents = [a for a in analogs if a[1] >= cfg.gen_min_sim][:cfg.gen_n_analog]
        tol = max(cfg.gen_tol_da, target * cfg.ppm_win * 1e-6)
        for rank, (sid, sim, shift, _) in enumerate(parents):
            psmi = L.struct_smiles[sid]
            pfp = np.unpackbits(self.train_fp[sid])[:self.pool.nbits].astype(np.float32)
            delta = target - float(L.struct_mass[sid])
            try:
                prods = derive.derive(psmi, delta, tol=tol, max_steps=2, max_per_analog=cfg.gen_max_per_parent)
            except Exception:
                prods = []
            for s in prods:
                std = chem.standardize_smiles(s)
                if std is None:
                    continue
                k = chem.score_key(std)
                if k is None or k in seen:
                    continue
                p = chem.mol_props(std)
                if p is None or abs(p[1] - target) > tol:
                    continue
                fp = chem.raw_fingerprint(std)
                if fp is None:
                    continue
                fp = fp[self.pool.bits].astype(np.float32)
                inter = float((fp * pfp).sum()); tan = inter / (fp.sum() + pfp.sum() - inter + 1e-9)
                seen.add(k)
                out.append(dict(smiles=std, key=k, formula=p[0], mass=p[1], n_heavy=p[2], fp=fp,
                                frags=fragmod.fragments_for_smiles(std), sim=sim, steps=1, tan=tan, parent=sid))
                if len(out) >= cfg.gen_max_total:
                    return out
        return out

    def candidates(self, target, drop_pid=-1, generated: Optional[List[dict]] = None) -> Cands:
        P = self.pool
        cand = P.window(target, self.cfg.ppm_win)
        if len(cand) == 0:
            cand = P.window(target, self.cfg.ppm_fallback)
        if drop_pid >= 0:
            cand = cand[cand != drop_pid]
        n = len(cand); g = generated or []
        fp = np.concatenate([P.fps(cand).astype(np.float32).reshape(n, P.nbits)] + [x['fp'][None] for x in g]) if (n + len(g)) else np.zeros((0, P.nbits), np.float32)
        return Cands(
            pid=np.concatenate([cand, np.full(len(g), -1)]).astype(np.int64),
            key=[P.key[c] for c in cand] + [x['key'] for x in g],
            smiles=[P.smiles[c] for c in cand] + [x['smiles'] for x in g],
            formula=[P.formula[c] for c in cand] + [x['formula'] for x in g],
            mass=np.concatenate([P.mass[cand], [x['mass'] for x in g]]).astype(np.float64),
            n_heavy=np.concatenate([P.n_heavy[cand], [x['n_heavy'] for x in g]]).astype(np.int32),
            train_sid=np.concatenate([P.train_sid[cand], np.full(len(g), -1)]).astype(np.int64),
            fp=fp,
            frags=[None] * n + [x['frags'] for x in g],
            is_gen=np.concatenate([np.zeros(n), np.ones(len(g))]).astype(np.float32),
            gen_sim=np.concatenate([np.zeros(n), [x['sim'] for x in g]]).astype(np.float32),
            gen_steps=np.concatenate([np.zeros(n), [x['steps'] for x in g]]).astype(np.float32),
            gen_tan=np.concatenate([np.zeros(n), [x['tan'] for x in g]]).astype(np.float32),
        )

    # ------------------------------------------------------------------------------------
    def run(self, spectra: List[dict], target: float, exclude: Optional[np.ndarray] = None,
            exclude_sid: int = -1, exclude_lib: int = -1, drop_pid: int = -1):
        """spectra: list of dict(mz, it, mode(+1/-1), adduct(str), prec, ce, ce_n).
        exclude: boolean mask over library spectra to hide (simulation); exclude_sid/exclude_lib:
        representatives to hide for the analog channel; drop_pid: remove this pool entry (class-3 sim).
        Returns (Cands, X, info)."""
        cfg = self.cfg; L = self.L; P = self.pool
        spectra = spectra[:cfg.max_query_spectra]
        queries = [(s['mz'], s['it'], s['mode']) for s in spectra]
        analogs = self.analogs(spectra, target, exclude, exclude_sid, exclude_lib)
        gen = None
        if cfg.generate and analogs:
            win = P.window(target, cfg.ppm_win)
            if drop_pid >= 0:
                win = win[win != drop_pid]          # class-3 simulation: the hidden truth may be generated
            gen = self.generate(analogs, target, set(P.key[c] for c in win))
        C = self.candidates(target, drop_pid, gen)
        nc = len(C.pid)
        if nc == 0:
            return C, np.zeros((0, N_FEAT), np.float32), dict(n_cand=0, lib_max=0.0, top_sim=0.0, fzmax=0.0)
        cfp = C.fp; cbits = cfp.sum(1)
        n_pos = sum(1 for s in spectra if s['mode'] > 0); n_neg = len(spectra) - n_pos
        X = np.zeros((nc, N_FEAT), np.float32)
        col = COL

        # ---------------- L: library search (per query spectrum, same polarity) --------------
        lib_max = np.zeros(nc); lib_mean = np.zeros(nc); lib_top3 = np.zeros(nc); lib_n = np.zeros(nc)
        lib_cos = np.zeros(nc); lib_same_ad = np.zeros(nc)
        sid2c: Dict[int, int] = {int(s): i for i, s in enumerate(C.train_sid) if s >= 0}
        per_spec_max: Dict[int, List[float]] = {}
        for qi, (mz, it, mode) in enumerate(queries):
            qm, qp = L.prep(mz, it, cfg.search)
            if len(qm) == 0:
                continue
            lc = L.window(target, cfg.search.ppm_win, exclude, mode)
            if len(lc) == 0:
                continue
            sc = spmod.search(qm, qp, lc, L.off, L.mz, L.it, cfg.search.tol, cfg.search.ppm_tol,
                              cfg.search.floor, cfg.search.topk, cfg.search.power, cfg.search.ent_weight, 0)
            scc = spmod.search(qm, qp, lc, L.off, L.mz, L.it, cfg.search.tol, cfg.search.ppm_tol,
                               cfg.search.floor, cfg.search.topk, cfg.search.power, cfg.search.ent_weight, 1)
            q_ad = chem.adduct_index(spectra[qi]['adduct'])
            best_this: Dict[int, float] = {}
            for j, s in enumerate(lc):
                ci = sid2c.get(int(L.sid[s]))
                if ci is None:
                    continue
                v = float(sc[j])
                if v > best_this.get(ci, 0.0):
                    best_this[ci] = v
                if v > lib_max[ci]:
                    lib_max[ci] = v
                if float(scc[j]) > lib_cos[ci]:
                    lib_cos[ci] = float(scc[j])
                if L.adduct_ix[s] == q_ad and v > lib_same_ad[ci]:
                    lib_same_ad[ci] = v
                lib_n[ci] += 1
            for ci, v in best_this.items():
                per_spec_max.setdefault(ci, []).append(v)
        nq = max(1, len(queries))
        for ci, vs in per_spec_max.items():
            lib_mean[ci] = sum(vs) / nq
            lib_top3[ci] = float(np.mean(sorted(vs)[-3:]))
        lib_n = lib_n / nq
        lmax = float(lib_max.max())
        X[:, col['lib_max']] = lib_max; X[:, col['lib_mean']] = lib_mean; X[:, col['lib_top3']] = lib_top3
        X[:, col['lib_nspec']] = np.log1p(lib_n); X[:, col['lib_cos']] = lib_cos
        X[:, col['lib_max_rank']] = _rank_norm(lib_max); X[:, col['lib_gap']] = lib_max - lmax
        X[:, col['lib_grp_max']] = lmax; X[:, col['lib_hit']] = (lib_max > 0).astype(np.float32)
        X[:, col['lib_same_adduct']] = lib_same_ad

        # ---------------- A: analog propagation ---------------------------------------------
        ap = a1 = best_tan = top_tan = mean_tan = ap_sum = tan0 = np.zeros(nc, np.float32)
        ap_np = best_tan_np = np.zeros(nc, np.float32)
        top_sim = top_sim_np = 0.0; n_hi = 0
        if analogs:
            aid = np.array([a[0] for a in analogs]); asim = np.array([a[1] for a in analogs], np.float32)
            ashift = np.array([a[2] for a in analogs]); anp = np.array([a[3] for a in analogs])
            afp = np.unpackbits(self.train_fp[aid], axis=1)[:, :P.nbits]
            tan = _tanimoto(cfp, afp)
            w4 = asim ** cfg.sim_power
            ap = (tan * w4[None, :]).max(1); a1 = (tan * asim[None, :]).max(1)
            best_tan = tan.max(1); top_tan = tan[:, 0]; top_sim = float(asim[0])
            mean_tan = (tan * w4[None, :]).sum(1) / (w4.sum() + 1e-9)
            ap_sum = np.log1p((tan * w4[None, :]).sum(1))
            n_hi = int((asim > 0.5).sum())
            z0 = np.abs(ashift) < 0.01
            tan0 = (tan[:, z0] * asim[None, z0]).max(1) if z0.any() else np.zeros(nc, np.float32)
            if anp.any():
                ap_np = (tan[:, anp] * w4[None, anp]).max(1); best_tan_np = tan[:, anp].max(1)
                top_sim_np = float(asim[anp][0])
        apmax = float(ap.max())
        X[:, col['ap']] = ap; X[:, col['a1']] = a1; X[:, col['best_tan']] = best_tan; X[:, col['top_tan']] = top_tan
        X[:, col['mean_tan']] = mean_tan; X[:, col['ap_rank']] = _rank_norm(ap); X[:, col['ap_gap']] = ap - apmax
        X[:, col['ap_grp_max']] = apmax; X[:, col['top_sim']] = top_sim
        X[:, col['ap_np']] = ap_np; X[:, col['best_tan_np']] = best_tan_np; X[:, col['top_sim_np']] = top_sim_np
        X[:, col['ap_np_gap']] = ap_np - float(ap_np.max())
        X[:, col['ap_sum']] = ap_sum; X[:, col['n_analog_hi']] = n_hi; X[:, col['tan_zero_shift']] = tan0

        # ---------------- M: spectrum -> fingerprint model ----------------------------------
        fz_avg = np.zeros(nc, np.float32); fz_s = np.zeros(nc, np.float32); fz_m = np.zeros(nc, np.float32)
        el_pred = None
        if self.bank is not None:
            items_s = []
            for s in spectra:
                pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec'])
                ce_ok = s['ce'] is not None and not np.isnan(s['ce'])
                items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']),
                                    ce=float(s['ce']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                                    n_merged=min(int(s.get('ce_n', 1)), 8), mode=float(s['mode'])))
            items_m = []
            for md in (1, -1):
                grp = [s for s in spectra if s['mode'] == md]
                if not grp:
                    continue
                mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
                prec = float(np.median([s['prec'] for s in grp]))
                pm, pi = fpnet.prep_peaks(mm, ii, prec)
                ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
                adducts = [s['adduct'] for s in grp]
                ad = max(set(adducts), key=adducts.count)
                items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                                    ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                                    n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
            zs, els = self.bank.logits_el(items_s)
            zm, elm = self.bank.logits_el(items_m)
            z_single = zs.mean(0); z_merged = zm.mean(0)
            el_pred = 0.5 * (els.mean(0) + elm.mean(0))
            fz_s = cfp @ z_single; fz_m = cfp @ z_merged
            z_avg = 0.5 * (z_single + z_merged)
            fz_avg = cfp @ z_avg
            pz = 1.0 / (1.0 + np.exp(-np.clip(z_avg, -30, 30))); pz = np.clip(pz, 1e-6, 1 - 1e-6)
            ll = cfp @ np.log(pz) + (1 - cfp) @ np.log(1 - pz)
            X[:, col['fp_ll']] = ll / max(1.0, P.nbits) * 100
            X[:, col['fp_ll_gap']] = (ll - ll.max()) / 100.0
            X[:, col['fp_cos']] = (cfp @ pz) / (np.sqrt(cbits) * np.linalg.norm(pz) + 1e-9)
            pb = (pz > 0.5).astype(np.float32)
            inter = cfp @ pb
            X[:, col['fp_tan']] = inter / (cbits + pb.sum() - inter + 1e-9)
        fzmax = float(fz_avg.max())
        sm = np.exp((fz_avg - fzmax) / math.sqrt(P.nbits) * 4.0); sm = sm / sm.sum()
        X[:, col['fz']] = fz_avg / 100.0; X[:, col['fz_z']] = _z(fz_avg); X[:, col['fz_rank']] = _rank_norm(fz_avg)
        X[:, col['fz_gap']] = (fz_avg - fzmax) / 100.0; X[:, col['fz_norm']] = fz_avg / np.sqrt(np.maximum(cbits, 1)) / 10.0
        X[:, col['fz_top']] = (fz_avg == fzmax).astype(np.float32); X[:, col['fz_softmax']] = sm
        X[:, col['fz_grp_ent']] = -(sm * np.log(sm + 1e-12)).sum()
        srt = np.sort(fz_avg)[::-1]
        X[:, col['fz_grp_gap']] = (srt[0] - srt[1]) / 100.0 if nc > 1 else 0.0
        X[:, col['fz_single']] = fz_s / 100.0; X[:, col['fz_single_gap']] = (fz_s - fz_s.max()) / 100.0
        X[:, col['fz_merged']] = fz_m / 100.0; X[:, col['fz_merged_gap']] = (fz_m - fz_m.max()) / 100.0
        if el_pred is not None:
            E = np.zeros((nc, len(ELEMS)), np.float32)
            for i in range(nc):
                fv = fragmod.formula_vec(C.formula[i])
                if fv is not None:
                    E[i] = np.log1p(fv)
            d = np.abs(E - el_pred[None, :])
            X[:, col['el_c']] = d[:, 0]; X[:, col['el_n']] = d[:, 2]; X[:, col['el_o']] = d[:, 3]
            X[:, col['el_sum']] = d.sum(1); X[:, col['el_rank']] = _rank_norm(-d.sum(1))

        # ---------------- X: cross-encoder ---------------------------------------------------
        if self.xscorer is not None and self.bank is not None:
            from . import graphs as gmod
            gitems = []
            for i in range(nc):
                g = self.gstore.get(int(C.pid[i])) if (self.gstore is not None and C.pid[i] >= 0) else gmod.mol_tokens(C.smiles[i])
                if g is None:
                    g = (np.zeros((0, gmod.N_ATOM_FEAT), np.uint8), np.zeros((0, 0), np.uint8), np.zeros((0, 0), np.uint8))
                gitems.append(g)
            xs_s = self.xscorer.score(items_s, gitems, cfp)           # (n_spec, nc)
            xs_m = self.xscorer.score(items_m, gitems, cfp)           # (n_views, nc)
            xmax = xs_s.max(0); xmean = xs_s.mean(0); xmer = xs_m.mean(0)
            xall = 0.5 * (xmax + xmer)
            X[:, col['xs_max']] = xmax; X[:, col['xs_mean']] = xmean; X[:, col['xs_merged']] = xmer
            X[:, col['xs_gap']] = xall - xall.max(); X[:, col['xs_rank']] = _rank_norm(xall); X[:, col['xs_z']] = _z(xall)
            X[:, col['xs_top']] = (xall >= xall.max() - 1e-6).astype(np.float32)
            xsrt = np.sort(xall)[::-1]
            X[:, col['xs_grp_gap']] = (xsrt[0] - xsrt[1]) if nc > 1 else 0.0
        # ---------------- F: fragmentation ---------------------------------------------------
        peaks = []
        for s in spectra:
            m, v = prep_query(np.asarray(s['mz'], np.float64), np.asarray(s['it'], np.float64), 0.002, 128, 1.0, False)
            if len(m):
                peaks.append((m, v / v.max(), s['adduct']))
        if cfg.use_frag and peaks:
            sel = np.arange(nc)
            if nc > cfg.frag_max_cand:
                sel = np.argsort(-fz_avg)[:cfg.frag_max_cand]
            fr = np.zeros((nc, 5), np.float32); frn = np.zeros(nc, np.float32)
            for i in sel:
                fg = C.frags[i] if C.frags[i] is not None else P.frags(int(C.pid[i]))
                fr[i] = fragmod.frag_features(fg, peaks); frn[i] = len(fg)
            X[:, col['fr_int']] = fr[:, 0]; X[:, col['fr_int_mean']] = fr[:, 1]; X[:, col['fr_cnt']] = fr[:, 2]
            X[:, col['fr_top10']] = fr[:, 3]; X[:, col['fr_strict']] = fr[:, 4]
            X[:, col['fr_rank']] = _rank_norm(fr[:, 0]); X[:, col['fr_gap']] = fr[:, 0] - fr[:, 0].max()
            X[:, col['fr_z']] = _z(fr[:, 0]); X[:, col['fr_n']] = np.log1p(frn)
        # ---------------- S: subformula ------------------------------------------------------
        uniq: Dict[str, int] = {}
        for f in C.formula:
            uniq[f] = uniq.get(f, 0) + 1
        X[:, col['n_formula']] = len(uniq)
        X[:, col['formula_share']] = np.array([uniq[f] / nc for f in C.formula], np.float32)
        if cfg.use_subformula and peaks:
            sf_cache = {f: fragmod.subformula_features(fragmod.formula_vec(f), peaks) for f in uniq}
            sf = np.stack([sf_cache[f] for f in C.formula])
            X[:, col['sf_int']] = sf[:, 0]; X[:, col['sf_cnt']] = sf[:, 2]; X[:, col['sf_top10']] = sf[:, 3]
            X[:, col['sf_rank']] = _rank_norm(sf[:, 0]); X[:, col['sf_gap']] = sf[:, 0] - sf[:, 0].max()
            X[:, col['formula_is_top']] = (sf[:, 0] >= sf[:, 0].max() - 1e-6).astype(np.float32)
        # ---------------- priors / group ------------------------------------------------------
        ppm = (C.mass - target) / target * 1e6
        X[:, col['mass_ppm']] = ppm; X[:, col['mass_ppm_rank']] = _rank_norm(-np.abs(ppm))
        X[:, col['n_heavy']] = C.n_heavy; X[:, col['fp_bits']] = cbits / 100.0
        X[:, col['n_cand']] = math.log(nc); X[:, col['n_query']] = len(spectra); X[:, col['n_pos']] = n_pos; X[:, col['n_neg']] = n_neg
        X[:, col['target_mass']] = target / 1000.0
        X[:, col['q_npeaks']] = math.log1p(float(np.mean([len(s['mz']) for s in spectra]))) if spectra else 0.0
        mr = _rank_norm(fz_avg)
        X[:, col['lib_x_fz']] = lib_max * (1.0 - mr); X[:, col['ap_x_fz']] = ap * (1.0 - mr)
        lbest = int(np.argmax(lib_max)) if lmax > 0 else -1
        X[:, col['agree_lib']] = (1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        X[:, col['agree_ap']] = (1.0 - mr[abest]) if abest >= 0 else 0.0
        X[:, col['fr_x_fz']] = X[:, col['fr_int']] * (1.0 - mr)
        # ---------------- G: generated candidates --------------------------------------------
        X[:, col['is_gen']] = C.is_gen; X[:, col['gen_parent_sim']] = C.gen_sim; X[:, col['gen_steps']] = C.gen_steps
        X[:, col['gen_parent_tan']] = C.gen_tan; X[:, col['n_gen']] = float(C.is_gen.sum())
        if C.is_gen.any():
            gi = np.where(C.is_gen > 0)[0]
            X[gi, col['gen_rank_fz']] = _rank_norm(fz_avg[gi])
        info = dict(n_cand=nc, lib_max=lmax, top_sim=top_sim, fzmax=fzmax, n_gen=int(C.is_gen.sum()))
        return C, X, info
