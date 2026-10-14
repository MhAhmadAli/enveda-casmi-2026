"""F-derivation: derivation evidence of the engine's class-3 generation step + holdout-clean class-3 priors.

DECISION NOTES (kept here so a restart can resume; newest last)
  2026-09-24 attempt 1 (killed): port of class3_v2/priors.py, supports() from v.derlog, 31 features; derivation_v1 on
      1,820 groups: ALL +0.0035 [-0.003, +0.010], C2FAM +0.002, C2FAM_wt +0.012 [+0.000, +0.026], C3FAM +0.048
      [+0.019, +0.082], c1 -0.006; provenance: gain vs train decoys (+0.001), none vs COCONUT decoys (-0.002) -> clean.
  2026-09-24 attempt 2 (this file):
    * pool smiles are already chem.standardize_smiles output (11,123 / 11,123 checked) -> smiles lookup in the prior
      groups is exact; but ~19% of pool hits are matched by SCORE KEY in the derlog (tautomer / stereo variants of the
      window smiles) and missed the prior (prior_miss).  Fix: fallback lookup by chem.score_key of the group products
      (cached per smiles; tested: resolves 41/41 remaining misses, ~1 ms/query).  Plain InChIKey / non-isomeric
      smiles do NOT resolve them (tautomers), so score_key is required.
    * zero-delta supports (combo mass ~ 0: remove+add isomerisations of a SAME-MASS analog) behave oppositely to real
      derivations: in c2* regimes truths are derived via zero-delta in 10-16% of their supports vs 35-50% for decoy hits
      (c1: ~equal). New columns split best_sim by zero-delta / non-zero-delta support and flag zero-delta-only.
    * nothing reads pool src / train_sid / is_train; pid never read (generated = v.is_gen); parents = analogs (library structures).
    * derlog 'gen'/'smi' idents that are not candidates by smiles are resolved by score key (earlier product rejected as a
      key duplicate); ~15% stay unresolved = off-mass products (not candidates) -> ignored.
    * the score-key fallback must be prefiltered by a heavy-atom skeleton (_skel): score_key (tautomer canonicalisation)
      of large polyphenol products took 26 s on one query; with the prefilter cold cost = 50 ms/q mean, p99 0.54 s.
    * parity of the prior port vs the original priors.py (subprocess, src/casmi): 1,500 products / 26,369 sibling
      entries, 0 mismatches (res_derivation.log).  gate_fam 120 q + extra permutation check on 44 hit-rich q: 0 failures.
    * provenance diagnostic (offline, src read for the diagnostic only): decoy derivability train vs COCONUT decoys
      c2 0.0106 vs 0.0084, c2L 0.0067 vs 0.0052, non-zero-delta hits equal (0.0060 / 0.0065) -> no provenance proxy.
    * evalfe (79 ctx files): derivation_v1 (36 feats, 3,541 groups) ALL +0.0082 [+0.0029, +0.0138], C2FAM +0.0011,
      C3FAM +0.103, np ALL +0.033, nplib C2FAM -0.005; v2_gen (dg_* only, 15) ALL +0.0054, C3FAM +0.089, nplib C2FAM
      -0.005 (the nplib c2 dip is not caused by the pool features); v3_core (19 feats, 3,774 groups) ALL +0.0081
      [+0.0032, +0.0130], C2FAM +0.0046 [-0.0010, +0.0101], C2FAM_wt +0.0105 [+0.0019, +0.0198], C3FAM +0.069.
      Gain importance: dg_rank_site, dg_n_sib, dg_min_steps, dg_rank_sup lead; dp_* individually small (1% of rows).
    * added ds_* sibling-group features (derived candidates sharing a (parent, combo) support = positional isomers of
      one derivation): group size, f.z rank / gap and site-prior rank inside the group, site-prior rank among derived
      candidates of the same formula -> tested as v4.
    * v4_core_ds (v3 core + 5 ds_*, 5,340 groups): ALL +0.0062 [+0.0014, +0.0109], C2FAM -0.0003, C3FAM +0.096.
      Paired v4 vs v3 on the same 5,340 groups (own OOF, seeds 0 and 1): ALL -0.001 / -0.002, C2FAM -0.0015 / -0.003,
      C3FAM +0.006 / +0.002 (all CIs span 0) -> ds_* add nothing; v3 core vs base: ALL +0.0082 / +0.0081 (lo90 > 0),
      C2FAM +0.0025 / +0.0035 (lo90 < 0), C3FAM +0.089 / +0.074, np ALL +0.021 / +0.028.  RECOMMENDED = v3 core (19).
    * c2 anatomy (v4 OOF): truth-derivable c2 groups (13%) +0.008, others 0.000; generated candidates above the truth
      0.197 -> 0.129 per group, pool decoys above the truth 2.17 -> 2.31 (reshuffling; better/worse groups 322 / 323).
    * pid switched to v.is_gen (identical outputs, rechecked on 60 built queries).
    * deployment: res_derivation.py now also writes res/derivation/parity_fixture.pkl (200 cases, rdkit 2026.03.3);
      self_test() re-runs it (0 mismatches here, 0.6 s) -> call once on Kaggle: the site-prior tables are keyed by
      unfolded Morgan ids and RDKit reaction enumeration order, both RDKit-version dependent.

Inputs (all inference-time, QueryView only):
  v.derlog   every derived product of Engine.generate that was NOT accepted as a new candidate:
             (parent_sid, parent_sim, combo, 'pool'|'gen', 'smi'|'key', ident)
             'pool' -> equals a pool-window candidate (ident = standardized smiles or score key),
             'gen'  -> equals an earlier product (multi-parent support of a generated candidate).
  generated candidates: parent_sid / gen_sim / gen_combo (their own, first = best-sim parent).
  v.an_sid / v.an_sim   analogs (parents are the top analogs with sim >= 0.3).

Support of a candidate = set of (parent sid, parent sim, transform combo) that derive it. Aggregates per candidate:
  n_par       distinct supporting parents            best_sim    best supporting parent spectral sim
  sum_sim4    sum over distinct parents of sim^4      min_steps   min len(combo) over supports (1 or 2)
  par_rank    best 0-based rank of a supporting parent among the analogs (10 = none)
  n_combo     distinct transform combos               tf_logfr    max over supports of sum_i log p(tf_i) (found-rate prior)
  site_logfr  max over supports of sum_i log p(site_i) (priors.py semantics; back-off to tf_logfr when the prior
              enumeration does not reproduce the product -> prior_miss = 1)
  sib_rank / n_sib / sib_frac   sibling rank of the best-site support among all products of that parent+combo
              (1 = most likely), number of siblings, (sib_rank-1)/max(n_sib-1,1)
  best_sim_nz / best_sim_zd     best parent sim over non-zero-delta / zero-delta (isomerisation) supports
  zd_only     1 if every support is a zero-delta isomerisation
  sentinels for "not derived": 0, par_rank 10, logfr -25, sib_frac -1.
Columns (a candidate outside the column's scope carries the "not derived" sentinel; is_gen is in the base):
  dp_*  pool candidates     dp_hit (derivable from >= 1 parent), dp_{n_par, best_sim, sum_sim4, min_steps, par_rank,
                            tf_logfr, site_logfr, sib_frac, prior_miss, best_sim_nz, best_sim_zd, zd_only};
                            dp_rank_sim (rank of best_sim among pool candidates), dp_rank_fz_hit (engine f.z rank among
                            pool hits), dp_iso_n_hit (log1p pool hits with the same formula), dp_iso_sim_gap / dp_iso_nz_gap
                            (best_sim / best_sim_nz minus the max of the same-formula pool candidates), dp_q_n_hit (log1p
                            pool hits in the query)
  dg_*  generated           dg_{n_par (own parent + 'gen' log entries), sum_sim4, min_steps, par_rank, n_combo, tf_logfr,
                            site_logfr, sib_rank, n_sib, sib_frac, prior_miss, zd_only} (== class3_v2 gen_tf_logfr,
                            gen_site_logfr, gen_sib_rank, gen_n_sib), dg_rank_site (site prior rank among generated),
                            dg_rank_sup (sum_sim4 rank among generated). best_sim of a generated candidate == base
                            gen_parent_sim (parents are visited in sim-descending order), so it is not repeated.
  dj_*  joint               dj_rank_site_der (site prior rank among pool hits + generated), dj_rank_sim (best_sim rank
                            among all candidates), dj_rank_fz_der (f.z rank among pool hits + generated)
  ds_*  sibling groups      derived candidates sharing a support (parent, combo); per candidate the best over its groups:
                            ds_n (log1p group size), ds_fz_rank / ds_fz_gap (engine f.z rank / (f.z - group max)/100),
                            ds_site_rank (site-prior rank), ds_fm_site_rank (site-prior rank among derived candidates with
                            the same formula); -1 (gap -10) for non-derived candidates
All ranks: ctxlib._rank_norm (average ties, 0 = best), -1 = outside the ranked subset.

The class-3 priors (transform found-rate, site-environment back-off, sibling rank) are a verbatim port of
work/analysis/class3_v2/priors.py onto the engine_v6 casmi package (identical derive.py and chem.standardize_smiles);
tables res/derivation/prior_tables.pkl (built from random COCONUT parents with every non-train-fold key removed from
parents and from 'found' -> holdout-clean; the target is pool existence of the product, not provenance).
Verified against the original by res_derivation.py.

No provenance: nothing reads pool src / train_sid; pid is never read (generated = v.is_gen). Per-structure values are
functions of (parent smiles, combo, product smiles); parents are analogs (library structures, deployable).
"""
from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from ctxlib import _rank_norm
from casmi import chem          # chem.py is byte-identical in the v1 and v6.3 engines
import derive6 as derive        # engine v6.3 transform grammar (71 transforms) for the priors, on either engine path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(HERE, 'res', 'derivation', 'prior_tables.pkl')

# ============================================================================================ priors.py port (verbatim)
MIN_N = 20
A = 2.0
MAX_PROD_TABLE = 200

_RX = {}
_GEN = {}


def _rdk():
    if 'Chem' not in _GEN:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, rdFingerprintGenerator
        RDLogger.DisableLog('rdApp.*')
        _GEN['Chem'] = Chem; _GEN['AllChem'] = AllChem
        _GEN['morgan'] = rdFingerprintGenerator.GetMorganGenerator(radius=3)
        _GEN['rfg'] = rdFingerprintGenerator
    return _GEN


def _tf(name):
    """(smarts, element delta) of a transform: engine v6.3 grammar, else the running engine's own derive table (the v1
    engine has one transform, 'dehexose2', that v6.3 lacks)."""
    t = derive._TRANSFORMS.get(name)
    if t is None:
        from casmi import derive as _dv
        t = _dv._TRANSFORMS[name]
    return t


def rxn(name):
    if name not in _RX:
        _RX[name] = _rdk()['AllChem'].ReactionFromSmarts(_tf(name)[0])
    return _RX[name]


def atom_envs(mol) -> Dict[int, Tuple[int, int, int]]:
    """atom idx -> (env1, env2, env3) unfolded Morgan ids of the atom environment (fallback to smaller radius)."""
    g = _rdk()
    ao = g['rfg'].AdditionalOutput(); ao.AllocateBitInfoMap()
    g['morgan'].GetSparseCountFingerprint(mol, additionalOutput=ao)
    env = {}
    for bit, lst in ao.GetBitInfoMap().items():
        for a, r in lst:
            env[(a, r)] = int(bit)
    out = {}
    for a in range(mol.GetNumAtoms()):
        e0 = env.get((a, 0), -1)
        e1 = env.get((a, 1), e0)
        e2 = env.get((a, 2), e1)
        e3 = env.get((a, 3), e2)
        out[a] = (e1, e2, e3)
    return out


def _dheavy(name):
    return sum(v for e, v in _tf(name)[1].items() if e != 'H')


def enumerate_products(mol, name, max_products=MAX_PROD_TABLE) -> List[Tuple[object, str, int]]:
    """Distinct products of one transform: (product mol, canonical smiles, site atom idx in `mol`); products whose
    heavy-atom count is not parent + transform delta (ring/chain cuts returning a fragment) are dropped."""
    Chem = _rdk()['Chem']
    try:
        prods = rxn(name).RunReactants((mol,))
    except Exception:
        return []
    want = mol.GetNumHeavyAtoms() + _dheavy(name)
    seen = {}
    out = []
    for p in prods:
        m = p[0]
        if m.GetNumAtoms() != want:
            continue
        site = -1
        for at in m.GetAtoms():
            if at.HasProp('old_mapno') and at.GetIntProp('old_mapno') == 1:
                site = at.GetIntProp('react_atom_idx'); break
        try:
            Chem.SanitizeMol(m); s = Chem.MolToSmiles(m)
        except Exception:
            continue
        if s in seen:
            continue
        seen[s] = 1
        out.append((m, s, site))
        if len(out) >= max_products:
            break
    return out


def _combo(transform) -> Tuple[str, ...]:
    if isinstance(transform, str):
        return tuple(transform.split('+'))
    return tuple(transform)


class Priors:
    def __init__(self, path: str = TABLES):
        with open(path, 'rb') as f:
            T = pickle.load(f)
        self.glob = T['glob']
        self.tf = T['tf']
        self.env = T['env']
        self.meta = T.get('meta', {})
        self.p_tf = {k: (s + A * self.glob) / (n + A) for k, (s, n) in self.tf.items()}
        self._cache = {}

    def p_transform(self, name: str) -> float:
        return self.p_tf.get(name, self.glob)

    def p_site(self, name: str, envs: Tuple[int, int, int]) -> float:
        p = self.p_transform(name)
        for r in (1, 2, 3):
            s_n = self.env[r].get((name, envs[r - 1]))
            if s_n is None:
                break
            s, n = s_n
            if n < MIN_N:
                break
            p = (s + A * p) / (n + A)
        return p

    def group(self, parent_smiles: str, transform, max_first=32, max_second=16, max_mols=24) -> Dict[str, dict]:
        """All products of `transform` (1 or 2 steps, derive.derive's enumeration limits) on the parent, keyed by
        standardized SMILES, with prior features."""
        combo = _combo(transform)
        ck = (parent_smiles, combo)
        if ck in self._cache:
            return self._cache[ck]
        Chem = _rdk()['Chem']
        mol = Chem.MolFromSmiles(parent_smiles)
        res: Dict[str, dict] = {}
        if mol is None:
            self._cache[ck] = res
            return res
        front = [(mol, 0.0, 0.0)]
        for depth, name in enumerate(combo):
            nxt = []
            lt = math.log(self.p_transform(name))
            for m, ls, ltf in front[:max_mols]:
                envs = atom_envs(m)
                for pm, s, site in enumerate_products(m, name, max_first if depth == 0 else max_second):
                    e = envs.get(site, (-1, -1, -1))
                    nxt.append((pm, ls + math.log(self.p_site(name, e)), ltf + lt))
            front = nxt
            if not front:
                break
        for pm, ls, ltf in front:
            try:
                s = Chem.MolToSmiles(pm)
            except Exception:
                continue
            std = chem.standardize_smiles(s)
            if std is None:
                continue
            if std not in res or ls > res[std]['gen_site_logfr']:
                res[std] = dict(gen_tf_logfr=ltf, gen_site_logfr=ls)
        if res:
            vals = np.array([v['gen_site_logfr'] for v in res.values()])
            for v in res.values():
                v['gen_sib_rank'] = int(1 + (vals > v['gen_site_logfr'] + 1e-12).sum())
                v['gen_n_sib'] = len(vals)
        if len(self._cache) > 5000:
            self._cache.clear()
        self._cache[ck] = res
        return res

    def features(self, parent_smiles: str, product_smiles: str, transform) -> Optional[dict]:
        std = chem.standardize_smiles(product_smiles) or product_smiles
        return self.group(parent_smiles, transform).get(std)


FEATURE_NAMES = ['gen_tf_logfr', 'gen_site_logfr', 'gen_sib_rank', 'gen_n_sib']

# ============================================================================================ family
_S = {}
NONE_LOGFR = -25.0
NONE_RANK = 10.0
ZD_TOL = 0.01                                        # |combo mass| below this = zero-delta isomerisation
SENT_GAP = 10.0                                      # f.z gap sentinel / clip (f.z / 100 units, as in the isomer family)
AGG = ['n_par', 'best_sim', 'sum_sim4', 'min_steps', 'par_rank', 'n_combo', 'tf_logfr', 'site_logfr', 'sib_rank',
       'n_sib', 'sib_frac', 'prior_miss', 'best_sim_nz', 'best_sim_zd', 'zd_only']
AGG_NONE = dict(n_par=0.0, best_sim=0.0, sum_sim4=0.0, min_steps=0.0, par_rank=NONE_RANK, n_combo=0.0,
                tf_logfr=NONE_LOGFR, site_logfr=NONE_LOGFR, sib_rank=0.0, n_sib=0.0, sib_frac=-1.0, prior_miss=0.0,
                best_sim_nz=0.0, best_sim_zd=0.0, zd_only=0.0)
POOL_AGG = ['n_par', 'best_sim', 'sum_sim4', 'min_steps', 'par_rank', 'tf_logfr', 'site_logfr', 'sib_frac', 'prior_miss',
            'best_sim_nz', 'best_sim_zd', 'zd_only']
GEN_AGG = ['n_par', 'sum_sim4', 'min_steps', 'par_rank', 'n_combo', 'tf_logfr', 'site_logfr', 'sib_rank', 'n_sib',
           'sib_frac', 'prior_miss', 'zd_only']      # best_sim of a generated candidate == base gen_parent_sim
NAMES = (['dp_hit'] + [f'dp_{k}' for k in POOL_AGG] + ['dp_rank_sim', 'dp_rank_fz_hit', 'dp_iso_n_hit', 'dp_iso_sim_gap',
                                                          'dp_iso_nz_gap', 'dp_q_n_hit']
         + [f'dg_{k}' for k in GEN_AGG] + ['dg_rank_site', 'dg_rank_sup']
         + ['dj_rank_site_der', 'dj_rank_sim', 'dj_rank_fz_der']
         + ['ds_n', 'ds_fz_rank', 'ds_fz_gap', 'ds_site_rank', 'ds_fm_site_rank'])
GEN_ONLY = [k for k in NAMES if k.startswith('dg_')]


def prepare(res):
    _S['pr'] = Priors(TABLES)
    _S['L'] = res.L


def _priors() -> Priors:
    if 'pr' not in _S:
        _S['pr'] = Priors(TABLES)
    return _S['pr']


def _combo_mass(combo) -> float:
    cm = _S.get('cm')
    if cm is None:
        derive.matching_transforms(0.0)               # initialises derive._COMBOS (all 1- and 2-step combos)
        cm = _S['cm'] = {tuple(c): float(m) for c, m in derive._COMBOS}
    m = cm.get(tuple(combo))
    if m is None:                                     # not a derive combo (never expected): mass from the formula deltas
        d = {}
        for nm in combo:
            tf = derive._TRANSFORMS.get(nm)
            if tf is None:                            # v1-engine-only transform (e.g. 'dehexose2'): its own table
                from casmi import derive as _dv
                tf = _dv._TRANSFORMS[nm]
            for e, k in tf[1].items():
                d[e] = d.get(e, 0) + k
        m = cm[tuple(combo)] = float(derive._delta_mass(d))
    return m


def _tf_logfr(pr: Priors, combo) -> float:
    return float(sum(math.log(pr.p_transform(nm)) for nm in combo))


def _score_key(smi: str):
    c = _S.setdefault('skey', {})
    if smi not in c:
        if len(c) > 200000:
            c.clear()
        c[smi] = chem.score_key(smi)
    return c[smi]


def _skel(smi: str) -> Optional[str]:
    """Heavy-atom skeleton (elements + connectivity; bond orders, aromaticity, charges, H and stereo erased). Tautomers
    and stereoisomers share it -> cheap prefilter before the (expensive, tautomer-canonical) score key."""
    c = _S.setdefault('skel', {})
    if smi not in c:
        if len(c) > 200000:
            c.clear()
        out = None
        try:
            Chem = _rdk()['Chem']
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                rw = Chem.RWMol(m)
                for b in rw.GetBonds():
                    b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
                for at in rw.GetAtoms():
                    at.SetIsAromatic(False); at.SetFormalCharge(0); at.SetNumExplicitHs(0); at.SetNoImplicit(True)
                    at.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                out = Chem.MolToSmiles(rw, isomericSmiles=False)
        except Exception:
            out = None
        c[smi] = out
    return c[smi]


def _prior_by_key(pr: Priors, psmi: str, combo, key: str, skel: Optional[str]) -> Optional[tuple]:
    """Best prior tuple over the group's products whose score key == key (products prefiltered by skeleton; the score
    key is computed only for skeleton matches). Order-free max -> deterministic."""
    if skel is None:
        return None
    best = None
    for s, g in pr.group(psmi, combo).items():
        if _skel(s) != skel or _score_key(s) != key:
            continue
        t = (float(g['gen_site_logfr']), float(g['gen_tf_logfr']), -float(g['gen_sib_rank']), -float(g['gen_n_sib']))
        if best is None or t > best:
            best = t
    return best


def supports(v) -> List[List[tuple]]:
    """Per candidate: list of (parent_sid, parent_sim, combo) deriving it (aggregated order-free: sets / max / sorted sums)."""
    n = v.n
    gen = v.is_gen > 0                               # generated candidates (== pid -1); pid never read
    maps = {(False, 'smi'): {}, (False, 'key'): {}, (True, 'smi'): {}, (True, 'key'): {}}
    for i in range(n):
        g = bool(gen[i])
        maps[(g, 'smi')].setdefault(v.smiles[i], []).append(i)
        maps[(g, 'key')].setdefault(v.key[i], []).append(i)
    sup: List[List[tuple]] = [[] for _ in range(n)]
    for i in np.flatnonzero(gen):
        if v.parent_sid[i] >= 0 and v.gen_combo[i] is not None:
            sup[i].append((int(v.parent_sid[i]), float(v.gen_sim[i]), tuple(v.gen_combo[i])))
    for psid, psim, combo, kind, how, ident in v.derlog:
        if combo is None:
            continue
        hit = maps[(kind == 'gen', how)].get(ident)
        if hit is None and how == 'smi' and kind == 'gen':
            # an earlier product with this smiles was not accepted (score-key duplicate of a pool / generated candidate,
            # or off-mass): resolve by its score key (intrinsic, deterministic)
            k = _score_key(ident)
            hit = (maps[(False, 'key')].get(k, []) + maps[(True, 'key')].get(k, [])) if k is not None else None
        for i in hit or ():
            sup[i].append((int(psid), float(psim), tuple(combo)))
    return sup


def aggregate(v, sup, pr: Priors, L) -> Dict[str, np.ndarray]:
    """Per-candidate support aggregates (AGG); non-derived candidates carry AGG_NONE."""
    n = v.n
    an_rank = {}
    for j, s in enumerate(v.an_sid):
        an_rank.setdefault(int(s), j)
    a = {k: np.full(n, AGG_NONE[k], np.float64) for k in AGG}
    for i in range(n):
        S = sup[i]
        if not S:
            continue
        uniq = sorted(set((p, c) for p, _, c in S))
        psim = {}; sim_nz = 0.0; sim_zd = 0.0; all_zd = True
        for p, s, c in S:
            psim[p] = max(psim.get(p, 0.0), s)
            if abs(_combo_mass(c)) < ZD_TOL:
                sim_zd = max(sim_zd, s)
            else:
                sim_nz = max(sim_nz, s); all_zd = False
        a['n_par'][i] = len(psim)
        a['best_sim'][i] = max(psim.values())
        a['sum_sim4'][i] = math.fsum(s ** 4 for s in sorted(psim.values()))
        a['min_steps'][i] = min(len(c) for _, c in uniq)
        a['par_rank'][i] = min(an_rank.get(p, int(NONE_RANK)) for p in psim)
        a['n_combo'][i] = len(set(c for _, c in uniq))
        a['best_sim_nz'][i] = sim_nz; a['best_sim_zd'][i] = sim_zd; a['zd_only'][i] = float(all_zd)
        best = None                                  # (site, tf, -sib_rank, -n_sib) over supports the prior reproduces
        tf_best = NONE_LOGFR
        for p, c in uniq:
            tf_best = max(tf_best, _tf_logfr(pr, c))
            g = pr.group(L.struct_smiles[p], c).get(v.smiles[i])
            if g is not None:
                t = (float(g['gen_site_logfr']), float(g['gen_tf_logfr']), -float(g['gen_sib_rank']),
                     -float(g['gen_n_sib']))
                if best is None or t > best:
                    best = t
        if best is None and v.key[i] is not None:    # matched by score key (tautomer / stereo variant): key lookup
            sk = _skel(v.smiles[i])
            for p, c in uniq:
                t = _prior_by_key(pr, L.struct_smiles[p], c, v.key[i], sk)
                if t is not None and (best is None or t > best):
                    best = t
        a['tf_logfr'][i] = tf_best
        if best is None:                             # not reproduced by the prior enumeration: transform back-off
            a['site_logfr'][i] = tf_best
            a['prior_miss'][i] = 1.0
        else:
            sr, ns = -best[2], -best[3]
            a['site_logfr'][i] = best[0]
            a['sib_rank'][i] = sr; a['n_sib'][i] = ns
            a['sib_frac'][i] = (sr - 1.0) / max(ns - 1.0, 1.0)
    return a


def _sub_rank(x, sel):
    r = np.full(len(x), -1.0)
    idx = np.flatnonzero(sel)
    if len(idx):
        r[idx] = _rank_norm(np.asarray(x, np.float64)[idx])
    return r


def compute(v):
    n = v.n
    pr = _priors()
    L = _S['L'] if 'L' in _S else v.res.L
    gen = v.is_gen > 0                               # generated candidates (== pid -1); pid never read
    pool = ~gen
    sup = supports(v)
    a = aggregate(v, sup, pr, L)
    der = a['n_par'] > 0
    hit = der & pool
    gd = der & gen
    f: Dict[str, np.ndarray] = {}
    f['dp_hit'] = hit.astype(np.float64)
    for k in POOL_AGG:
        f[f'dp_{k}'] = np.where(pool, a[k], AGG_NONE[k])
    for k in GEN_AGG:
        f[f'dg_{k}'] = np.where(gen, a[k], AGG_NONE[k])
    fz = v.fz_ab.astype(np.float64)
    # ---- pool candidates: rank of the derivation strength among pool candidates, f.z among pool hits
    f['dp_rank_sim'] = _sub_rank(a['best_sim'], pool)
    f['dp_rank_fz_hit'] = _sub_rank(fz, hit)
    # ---- isomer context: derivable pool candidates in the same-formula group
    iso_n = np.zeros(n); iso_gap = np.zeros(n); iso_nz = np.zeros(n)
    grp: Dict[str, List[int]] = {}
    for i, fm in enumerate(v.formula):
        grp.setdefault(fm, []).append(i)
    for fm, ii in grp.items():
        ii = np.asarray(ii)
        iso_n[ii] = math.log1p(int(hit[ii].sum()))
        pi = ii[pool[ii]]
        if len(pi):
            iso_gap[pi] = f['dp_best_sim'][pi] - f['dp_best_sim'][pi].max()
            iso_nz[pi] = f['dp_best_sim_nz'][pi] - f['dp_best_sim_nz'][pi].max()
    f['dp_iso_n_hit'] = iso_n; f['dp_iso_sim_gap'] = iso_gap; f['dp_iso_nz_gap'] = iso_nz
    f['dp_q_n_hit'] = np.full(n, math.log1p(int(hit.sum())))
    # ---- generated candidates: site prior rank and support strength rank among generated
    f['dg_rank_site'] = _sub_rank(a['site_logfr'], gd)
    f['dg_rank_sup'] = _sub_rank(a['sum_sim4'], gd)
    # ---- joint (pool hits + generated)
    f['dj_rank_site_der'] = _sub_rank(a['site_logfr'], der)
    f['dj_rank_sim'] = _rank_norm(a['best_sim'])
    f['dj_rank_fz_der'] = _sub_rank(fz, der)
    # ---- sibling groups: derived candidates (pool hits + generated) sharing a support (parent, combo) = positional /
    #      site isomers of one derivation; f.z and site-prior rank inside the group (best over the candidate's groups)
    sg: Dict[tuple, List[int]] = {}
    for i in np.flatnonzero(der):
        for p, c in set((p, c) for p, _, c in sup[i]):
            sg.setdefault((p, c), []).append(i)
    ds_n = np.zeros(n); ds_fr = np.full(n, 2.0); ds_gap = np.full(n, -SENT_GAP); ds_sr = np.full(n, 2.0)
    for ii in sg.values():
        ii = np.asarray(sorted(set(ii)))
        ds_n[ii] = np.maximum(ds_n[ii], math.log1p(len(ii)))
        ds_fr[ii] = np.minimum(ds_fr[ii], _rank_norm(fz[ii]))
        ds_gap[ii] = np.maximum(ds_gap[ii], np.clip((fz[ii] - fz[ii].max()) / 100.0, -SENT_GAP, 0.0))
        ds_sr[ii] = np.minimum(ds_sr[ii], _rank_norm(a['site_logfr'][ii]))
    ds_fr[~der] = -1.0; ds_sr[~der] = -1.0
    f['ds_n'] = ds_n; f['ds_fz_rank'] = ds_fr; f['ds_fz_gap'] = ds_gap; f['ds_site_rank'] = ds_sr
    fm_sr = np.full(n, -1.0)
    for fm, ii in grp.items():
        jj = np.asarray(ii)[der[np.asarray(ii)]]
        if len(jj):
            fm_sr[jj] = _rank_norm(a['site_logfr'][jj])
    f['ds_fm_site_rank'] = fm_sr
    assert sorted(f) == sorted(NAMES), set(f) ^ set(NAMES)
    return {k: np.nan_to_num(f[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for k in NAMES}


def self_test(path: str = os.path.join(HERE, 'res', 'derivation', 'parity_fixture.pkl'), n: int = 200) -> int:
    """Deployment check (run once in the Kaggle notebook): re-evaluates the parity fixture written by res_derivation.py
    (product features + full sibling groups of the prior port) and returns the number of mismatching cases. A non-zero
    count means the RDKit build changes reaction enumeration / Morgan environment ids -> site priors are not the ones
    the ranker was trained with."""
    with open(path, 'rb') as f:
        fx = pickle.load(f)
    pr = Priors(TABLES)
    bad = 0
    for (psmi, combo, prod), fm, gm in fx['cases'][:n]:
        g = pr.group(psmi, combo)
        if pr.features(psmi, prod, combo) != fm or {k: dict(v) for k, v in g.items()} != gm:
            bad += 1
    return bad
