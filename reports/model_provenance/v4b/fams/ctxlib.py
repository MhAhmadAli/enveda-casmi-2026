"""fe_v4 context library: one QueryView per simulated (or real) query = everything the v4 engine knows at inference time.

A feature family is a function  fam(v: QueryView) -> Dict[str, np.ndarray (n_cand,) float32]  that uses ONLY QueryView
fields (and the static resources in v.res). The same function runs offline on replayed contexts (fe_v4/ctx) and online
inside the v4 engine, so the ranker's training features and deployment features are the same computation.

Rules for families (enforced by review + gates):
  * order invariance: no dependence on candidate row order (use eng._rank_norm / average ties, sorted statistics);
  * no provenance: never read pool src / train_sid / pid-range; `chan` (visibility) is allowed; train-structure identity
    of ANALOGS (an_sid) is allowed (the analog channel is deployable: analogs are library structures);
  * no labels, no sid of the query, no regime;
  * deterministic, float32 output, finite values (use 0 / sentinel for missing).

QueryView fields
  target, n_query, spectra (engine-prepared: mz, it, mode, adduct, prec, ce, ce_n), n
  pid (int64, -1 = generated), key, smiles, formula, mass (float64), n_heavy, fp (n, nbits) float32 0/1, chan (0 visible
  pool, 1 invisible pool, 2 generated), is_gen, parent_sid, gen_sim, gen_steps, gen_tan, gen_combo (tuple | None)
  an_sid, an_sim, an_shift, an_np  (top-100 analogs, sim desc), an_smiles (lazy), an_fp (lazy, (k, nbits))
  z_s (M, n_s, nbits), el_s (M, n_s, 10), z_m (M, n_m, nbits), el_m (M, n_m, 10), m_modes (n_m,)  M: 0=A 1=B 2=D
  X (n, 89) base engine features, FEAT (name -> column), fz_ab (n,) = the engine's A+B f.z (float32)
  derlog: generation log of Engine.generate (every derived product that was NOT accepted as a new candidate):
          (parent_sid, parent_sim, combo, 'pool' = equals a pool-window candidate | 'gen' = equals an earlier product,
           'smi' | 'key' = matched by standardized smiles | by score key, the smiles or key)
  res: resources (L library, P pool, train_fp, E engine-like helpers)
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
V6CODE = os.path.join(ROOT, 'work', 'analysis', 'engine_v6', 'code')
if 'casmi' not in sys.modules and V6CODE not in sys.path:   # offline default; a process that already imported a
    sys.path.insert(0, V6CODE)                              # casmi package (e.g. the v1 engine) keeps it
import numpy as np
import pandas as pd

# package-agnostic copies of engine v6.3 definitions (families must behave identically on the v1 and v6 paths)
CLEAN_FLOOR, CLEAN_TOPK, CLEAN_OVER = 0.001, 512, 2.0
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
    'chan', 'n_chan_vis', 'n_chan_novis',
]
COL = {n: i for i, n in enumerate(FEATURES)}


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """1-based ranks of x ascending; ties share the average rank (== engine v6.3 _avg_rank)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(x, kind='mergesort')
    xs = x[o]
    b = np.r_[True, xs[1:] != xs[:-1]]
    bid = np.cumsum(b) - 1
    starts = np.flatnonzero(b); ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) / 2.0 + 1.0
    r = np.empty(n); r[o] = avg[bid]
    return r


def _rank_norm(x):
    """0 = best (largest x) .. 1 = worst, average ranks for ties (== engine v6.3 _rank_norm)."""
    n = len(x)
    return (_avg_rank(-np.asarray(x, np.float64)) - 1.0) / max(1, n - 1)


def _z(x):
    """z-score with order-invariant statistics (== engine v6.3 _z)."""
    xs = np.sort(np.asarray(x, np.float64))
    s = xs.std()
    return ((np.asarray(x, np.float64) - xs.mean()) / s).astype(np.float32) if s > 1e-9 else np.zeros(len(xs), np.float32)


def to_v6_layout(X: np.ndarray, names: List[str]) -> np.ndarray:
    """Base rows of another engine (e.g. the v1 engine's 94 features) in the FEATURES layout by name (missing -> 0)."""
    if list(names) == FEATURES:
        return X
    out = np.zeros((X.shape[0], len(FEATURES)), np.float32)
    pos = {n: i for i, n in enumerate(names)}
    for j, n in enumerate(FEATURES):
        if n in pos:
            out[:, j] = X[:, pos[n]]
    return out

MODEL_NAMES = ['A', 'B', 'D']


class Resources:
    """Static inference-time resources (library, pool, train fingerprints). Memory-mapped; one per process."""

    def __init__(self, workdir: Optional[str] = None, verbose=False):
        from casmi.library import Library
        from casmi.pool import Pool
        wd = workdir or os.path.join(ROOT, 'work')
        self.L = Library(wd, verbose=verbose, mmap=True, lean=True)
        self.P = Pool(wd, verbose=verbose, mmap=True)
        self.train_fp = np.load(f'{wd}/train_fp_sel.npy', mmap_mode='r')
        self.nbits = self.P.nbits
        self.cfg = type('Cfg', (), dict(max_query_spectra=16))()
        self.cache: Dict[str, object] = {}          # per-process caches for families (e.g. descriptor tables)

    def prepare(self, spectra: List[dict]) -> List[dict]:
        """== Engine.prepare (first max_query_spectra, cleaned like the library cache)."""
        from casmi.spectra import clean_peaks
        spectra = spectra[:self.cfg.max_query_spectra]
        out = []
        for s in spectra:
            m, v = clean_peaks(np.asarray(s['mz'], np.float64), np.asarray(s['it'], np.float64), float(s['prec']),
                               CLEAN_FLOOR, CLEAN_TOPK, CLEAN_OVER)
            d = dict(s); d['mz'] = m; d['it'] = v
            out.append(d)
        return out


@dataclass
class QueryView:
    target: float
    spectra: List[dict]
    pid: np.ndarray
    key: List[str]
    smiles: List[str]
    formula: List[str]
    mass: np.ndarray
    n_heavy: np.ndarray
    fp: np.ndarray
    chan: np.ndarray
    is_gen: np.ndarray
    parent_sid: np.ndarray
    gen_sim: np.ndarray
    gen_steps: np.ndarray
    gen_tan: np.ndarray
    gen_combo: List[Optional[tuple]]
    an_sid: np.ndarray
    an_sim: np.ndarray
    an_shift: np.ndarray
    an_np: np.ndarray
    z_s: np.ndarray
    el_s: np.ndarray
    z_m: np.ndarray
    el_m: np.ndarray
    m_modes: np.ndarray
    X: np.ndarray
    res: Resources
    derlog: List[tuple] = field(default_factory=list)   # generation log: (parent_sid, parent_sim, combo, 'pool'|'gen', 'smi'|'key', ident)
    ug: int = -1                                   # offline bookkeeping only (never a feature)
    _an_fp: Optional[np.ndarray] = None
    _an_smiles: Optional[List[str]] = None
    _memo: Dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.pid)

    @property
    def n_query(self) -> int:
        return len(self.spectra)

    @property
    def FEAT(self):
        return COL

    @property
    def fz_ab(self) -> np.ndarray:
        return self.X[:, COL['fz']] * 100.0

    @property
    def an_fp(self) -> np.ndarray:
        if self._an_fp is None:
            nb = self.res.nbits
            self._an_fp = (np.unpackbits(np.asarray(self.res.train_fp[self.an_sid]), axis=1)[:, :nb].astype(np.float32)
                           if len(self.an_sid) else np.zeros((0, nb), np.float32))
        return self._an_fp

    @property
    def an_smiles(self) -> List[str]:
        if self._an_smiles is None:
            self._an_smiles = [self.res.L.struct_smiles[int(s)] for s in self.an_sid]
        return self._an_smiles

    def memo(self, name: str, fn: Callable[[], object]):
        """Per-query memo shared by families (e.g. candidate Tanimoto matrix)."""
        if name not in self._memo:
            self._memo[name] = fn()
        return self._memo[name]

    def cand_tanimoto(self) -> np.ndarray:
        """(n, n) Tanimoto among candidates (float32)."""
        def f():
            A = self.fp.astype(np.float32)
            inter = A @ A.T
            s = A.sum(1)
            return (inter / np.maximum(s[:, None] + s[None, :] - inter, 1e-9)).astype(np.float32)
        return self.memo('cand_tan', f)

    def an_tanimoto(self) -> np.ndarray:
        """(n, k) Tanimoto candidate x analog (float32)."""
        def f():
            A = self.fp.astype(np.float32); B = self.an_fp
            if len(B) == 0:
                return np.zeros((self.n, 0), np.float32)
            inter = A @ B.T
            return (inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1e-9)).astype(np.float32)
        return self.memo('an_tan', f)

    def fz_model(self, m: int, view: str = 'avg') -> np.ndarray:
        """f.z of one model: view 'single' (mean logits over spectra), 'merged' (mean over polarity items), 'avg' (both,
        == the engine's definition for the A+B mean)."""
        def f():
            c64 = self.fp.astype(np.float64)
            zs = self.z_s[m].astype(np.float64).mean(0) if self.z_s.shape[1] else np.zeros(self.fp.shape[1])
            zm = self.z_m[m].astype(np.float64).mean(0) if self.z_m.shape[1] else np.zeros(self.fp.shape[1])
            return dict(single=(c64 @ zs).astype(np.float32), merged=(c64 @ zm).astype(np.float32),
                        avg=(c64 @ (0.5 * (zs + zm))).astype(np.float32))
        return self.memo(f'fz_{m}', f)[view]


def view_from_record(rec: dict, res: Resources, X: np.ndarray) -> QueryView:
    """QueryView of one replayed context record (fe_v4/replay_ctx.py) + its stored base rows X."""
    from casmi.pipeline import spectra_from_library
    P = res.P
    pid = rec['pid'].astype(np.int64)
    n = len(pid)
    pool_i = np.flatnonzero(pid >= 0); gen_i = np.flatnonzero(pid < 0)
    pp = pid[pool_i]
    key = [None] * n; smiles = [None] * n; formula = [None] * n
    for j, p in zip(pool_i, pp):
        key[j] = P.key[p]; smiles[j] = P.smiles[p]; formula[j] = P.formula[p]
    for t, j in enumerate(gen_i):
        key[j] = rec['key_gen'][t]; smiles[j] = rec['smiles_gen'][t]; formula[j] = rec['formula_gen'][t]
    mass = np.zeros(n); mass[pool_i] = P.mass[pp]; mass[gen_i] = rec['mass_gen']
    nh = np.zeros(n, np.int32); nh[pool_i] = P.n_heavy[pp]; nh[gen_i] = rec['n_heavy_gen']
    fp = np.zeros((n, P.nbits), np.float32)
    if len(pool_i):
        fp[pool_i] = P.fps(pp)
    if len(gen_i):
        fp[gen_i] = np.unpackbits(rec['fp_gen'], axis=1)[:, :P.nbits]
    z = lambda k, d: np.zeros(n, d)
    parent = np.full(n, -1, np.int64); parent[gen_i] = rec['parent_gen']
    gs = z(0, np.float32); gs[gen_i] = rec['gen_sim']
    gst = z(0, np.float32); gst[gen_i] = rec['gen_steps']
    gt = z(0, np.float32); gt[gen_i] = rec['gen_tan']
    combo = [None] * n
    for t, j in enumerate(gen_i):
        combo[j] = rec['combo_gen'][t]
    q = rec['_q']                                   # qidx / target from the meta row (attached by the loader)
    spectra = res.prepare(spectra_from_library(res.L, q['qidx']))
    return QueryView(target=float(q['target']), spectra=spectra, pid=pid, key=key, smiles=smiles, formula=formula,
                     mass=mass, n_heavy=nh, fp=fp, chan=rec['chan'].astype(np.int8), is_gen=(pid < 0).astype(np.float32),
                     parent_sid=parent, gen_sim=gs, gen_steps=gst, gen_tan=gt, gen_combo=combo,
                     an_sid=rec['an_sid'], an_sim=rec['an_sim'], an_shift=rec['an_shift'], an_np=rec['an_np'],
                     z_s=rec['z_s'], el_s=rec['el_s'], z_m=rec['z_m'], el_m=rec['el_m'], m_modes=rec['m_modes'],
                     X=X, res=res, derlog=list(rec.get('derlog', [])), ug=int(rec['shard']) * 10_000_000 + int(rec['gid']))


# ------------------------------------------------------------------------------------------------ offline data access
SIM = os.environ.get('FE_SIM') or os.path.join(ROOT, 'work', 'analysis', 'engine_v6', 'sim10_s2', 'rows')
SNAP = os.environ.get('FE_SNAP') or (SIM if os.environ.get('FE_SIM') else os.path.join(HERE, 'rows_snap', 'rows'))
CTX = os.environ.get('FE_CTX') or os.path.join(HERE, 'ctx')


def _shards() -> List[int]:
    """Shard ids of the row set (merged files rows_<s>.npz; the sim10_s2 default has 4)."""
    import re as _re
    got = {int(m.group(1)) for f in glob.glob(f'{SIM}_*.npz') for m in [_re.search(r'_(\d+)\.npz$', f)] if m}
    got |= {int(m.group(1)) for f in glob.glob(f'{SNAP}_*_p[0-9][0-9][0-9].npz') for m in [_re.search(r'_(\d+)_p\d{3}\.npz$', f)] if m}
    return sorted(got) or [0, 1, 2, 3]


def load_meta_all() -> pd.DataFrame:
    """Meta of every sim10_s2 query (merged shard files, else the snapshot parts) with ug = shard*1e7 + gid."""
    ms = []
    for s in _shards():
        f = f'{SIM}_{s}_meta.parquet'
        if os.path.exists(f):
            m = pd.read_parquet(f)
        else:
            ps = sorted(glob.glob(f'{SNAP}_{s}_p[0-9][0-9][0-9]_meta.parquet'))
            if not ps:
                continue
            m = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
        m['shard'] = s; ms.append(m)
    m = pd.concat(ms, ignore_index=True)
    m['ug'] = m.shard.values.astype(np.int64) * 10_000_000 + m.gid.values
    return m.sort_values('ug').reset_index(drop=True)


class BaseRows:
    """Stored base rows (X, y) by ug; merged shard files, else the snapshot parts."""

    def __init__(self):
        self.span: Dict[int, tuple] = {}
        Xs, ys = [], []; off = 0
        for s in _shards():
            f = f'{SIM}_{s}.npz'
            files = [f] if os.path.exists(f) else sorted(glob.glob(f'{SNAP}_{s}_p[0-9][0-9][0-9].npz'))
            for fn in files:
                with np.load(fn, allow_pickle=True) as z:
                    X = z['X']; y = z['y']; g = z['g']; self.features = [str(x) for x in z['features']]
                u, st, cnt = np.unique(g, return_index=True, return_counts=True)
                for a, b, c in zip(u, st, cnt):
                    self.span[s * 10_000_000 + int(a)] = (off + int(b), int(c))
                Xs.append(X); ys.append(y); off += len(y)
        self.X = np.concatenate(Xs); self.y = np.concatenate(ys)

    def get(self, ug):
        a, n = self.span[int(ug)]
        return self.X[a:a + n], self.y[a:a + n]


def ctx_files(ctx_dir: str = CTX) -> List[str]:
    import time as _t
    return sorted(f for f in glob.glob(f'{ctx_dir}/ctx_*.pkl') if _t.time() - os.path.getmtime(f) > 60)


EXTRA_DIR = os.environ.get('FE_EXTRA_DIR') or os.path.join(HERE, 'ctx_extra')


def extra_models() -> List[str]:
    """Extra spectrum->FP models appended after A, B, D (model slots 3, 4, ...): env FE_EXTRA='D2,...' (ctx_extra/<tag>/)."""
    return [t for t in os.environ.get('FE_EXTRA', '').split(',') if t]


def _load_extra(tag: str, ctx_file: str) -> Dict[int, tuple]:
    z = np.load(os.path.join(EXTRA_DIR, tag, os.path.basename(ctx_file).replace('.pkl', '.npz')))
    os_ = np.r_[0, np.cumsum(z['n_s'])]; om = np.r_[0, np.cumsum(z['n_m'])]
    return {int(u): (z['z_s'][os_[i]:os_[i + 1]], z['el_s'][os_[i]:os_[i + 1]], z['z_m'][om[i]:om[i + 1]], z['el_m'][om[i]:om[i + 1]])
            for i, u in enumerate(z['ug'])}


def iter_views(files: List[str], res: Resources, base: BaseRows, meta: pd.DataFrame, extra: Optional[List[str]] = None) -> Iterator[QueryView]:
    qmap = meta.set_index('ug')
    extra = extra_models() if extra is None else extra
    for f in files:
        with open(f, 'rb') as fh:
            recs = pickle.load(fh)
        ex = [_load_extra(t, f) for t in extra]
        for rec in recs:
            ug = int(rec['shard']) * 10_000_000 + int(rec['gid'])
            for d in ex:
                zs, es, zm, em = d[ug]
                rec['z_s'] = np.concatenate([rec['z_s'], zs[None]]); rec['el_s'] = np.concatenate([rec['el_s'], es[None]])
                rec['z_m'] = np.concatenate([rec['z_m'], zm[None]]); rec['el_m'] = np.concatenate([rec['el_m'], em[None]])
            row = qmap.loc[ug]
            rec['_q'] = dict(qidx=np.asarray(list(row.qidx), np.int64), target=float(row.target))
            X, _ = base.get(ug)
            assert len(X) == len(rec['pid']), (ug, len(X), len(rec['pid']))
            yield view_from_record(rec, res, to_v6_layout(X, base.features))


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid is running (Windows: OpenProcess + GetExitCodeProcess; POSIX: kill 0)."""
    if os.name == 'nt':
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code)); k32.CloseHandle(h)
        return bool(ok) and code.value == 259            # STILL_ACTIVE
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


class FileLock:
    """Cross-process exclusive lock (heavy evaluations run one at a time to protect RAM)."""

    def __init__(self, path=os.path.join(HERE, 'eval', '.lock'), stale_s=3 * 3600, poll_s=15):
        self.path, self.stale_s, self.poll_s = path, stale_s, poll_s

    def __enter__(self):
        import time as _t
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        waited = 0
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f'{os.getpid()} {_t.strftime("%H:%M:%S")}'.encode()); os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = int(open(self.path).read().split()[0])
                except Exception:
                    holder = -1
                try:
                    if (_t.time() - os.path.getmtime(self.path) > self.stale_s) or (holder > 0 and not _pid_alive(holder)):
                        print(f'  [lock] removing stale lock of pid {holder}', flush=True)
                        os.remove(self.path); continue
                except FileNotFoundError:
                    continue
                if waited % 300 == 0:
                    print(f'  [lock] waiting for {self.path} ({waited}s)', flush=True)
                _t.sleep(self.poll_s); waited += self.poll_s

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


PER_CAND = ('pid', 'mass', 'n_heavy', 'fp', 'chan', 'is_gen', 'parent_sid', 'gen_sim', 'gen_steps', 'gen_tan', 'X')
PER_CAND_LIST = ('key', 'smiles', 'formula', 'gen_combo')


def permute_view(v: QueryView, perm: np.ndarray) -> QueryView:
    """Same query with candidates reordered (gate: features must permute with the candidates)."""
    kw = {f: getattr(v, f) for f in v.__dataclass_fields__ if not f.startswith('_')}
    for f in PER_CAND:
        kw[f] = getattr(v, f)[perm]
    for f in PER_CAND_LIST:
        lst = getattr(v, f); kw[f] = [lst[i] for i in perm]
    return QueryView(**kw)


def resources_from_engine(E) -> Resources:
    """Resources sharing the engine's library / pool / train fingerprints (inference: no second copy)."""
    r = Resources.__new__(Resources)
    r.L = E.L; r.P = E.pool; r.train_fp = E.train_fp; r.nbits = E.pool.nbits; r.cfg = E.cfg; r.cache = {}
    return r
