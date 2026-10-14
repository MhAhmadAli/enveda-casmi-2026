"""End-to-end inference (Kaggle notebook body). Mirrors e02_channels.py + e05_rank.py feature blocks exactly,
but reads train.parquet / test.parquet directly. Runs locally too (set CASMI_LOCAL=1).

Stages: library -> candidate pool -> index -> per-molecule channels -> MetFrag-lite -> features -> ranker -> submission
"""
import os, sys, glob, time, pickle, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import pv
from pv import CFG
import re as _re
_MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100, P=30.97376163,
             Cl=34.96885268, Br=78.9183371, F=18.99840322, I=126.904473, Si=27.9769265325, Se=79.9165213,
             Na=22.9897692809, K=38.96370668, B=11.0093054, As=74.9215965, Fe=55.9349375, D=2.0141017778)
_TOK = _re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(f):
    """ADDED: neutral monoisotopic mass by arithmetic on molecular_formula (prvsiyan fix; immune to
    riken's 0.005 Da precursor precision and gnps adduct mislabels). NaN if an element is untabulated."""
    if not isinstance(f, str) or not f: return np.nan
    m = 0.0
    for el, n in _TOK.findall(f):
        if el not in _MONO: return np.nan
        m += _MONO[el] * (int(n) if n else 1)
    return m

POOL = None


class RANK:
    W_A = (0.35, 0.55)      # weight on class-1 simulation rows; two priors averaged
    SEEDS = (0, 1)
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)

T0 = time.time()
LOCAL = os.environ.get("CASMI_LOCAL") == "1"
ROOTS = [r"C:\Users\HW-LEE\Desktop\CASMI"] if LOCAL else ["/kaggle/input"]


def find(name):
    for root in ROOTS:
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits: return sorted(hits, key=len)[0]
    raise FileNotFoundError(name)


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------------------------
def load_library(path):
    """Row group by row group, peaks straight to float32: peak RAM stays near the final ~3.5 GB."""
    f = pq.ParquetFile(path)
    n_rows = f.metadata.num_rows
    npk = 0
    for i in range(f.num_row_groups):
        c = f.read_row_group(i, columns=["ms2_mzs"]).column(0).combine_chunks()
        npk += len(c.values)
    off = np.zeros(n_rows + 1, np.int64)
    allmz = np.empty(npk, np.float32); allin = np.empty(npk, np.float32)
    prec, add, pol, ik, smi = [], [], [], [], []
    fo, tims = [], []
    r0 = p0 = 0
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ionization_mode",
                                         "ms2_mzs", "ms2_normalized_intensities", "molecular_formula", "instrument_type"])
        mzc = t.column("ms2_mzs").combine_chunks(); itc = t.column("ms2_normalized_intensities").combine_chunks()
        o = mzc.offsets.to_numpy().astype(np.int64)
        n = len(o) - 1; k = int(o[-1] - o[0])
        allmz[p0:p0 + k] = mzc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]
        allin[p0:p0 + k] = itc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]
        off[r0 + 1:r0 + n + 1] = p0 + (o[1:] - o[0])
        prec.append(t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64))
        add += t.column("adduct").cast(pa.string()).to_pylist()
        pol.append(np.array(t.column("ionization_mode").cast(pa.string()).to_pylist(), dtype=object) == "positive")
        ik += t.column("inchikey14").cast(pa.string()).to_pylist()
        smi += t.column("normalized_smiles").cast(pa.string()).to_pylist()
        fo += t.column("molecular_formula").cast(pa.string()).to_pylist()
        tims.append(np.array([isinstance(x, str) and x.lower().replace("-", "").replace(" ", "") == "timstof"
                              for x in t.column("instrument_type").cast(pa.string()).to_pylist()]))
        r0 += n; p0 += k
        del t, mzc, itc
    prec = np.concatenate(prec); add = np.asarray(add, dtype=object)
    pol = np.where(np.concatenate(pol), 1, -1).astype(np.int8)
    nmp = pv.neutral_mass(prec, add)
    _fm = {x: formula_mass(x) for x in set(v for v in fo if isinstance(v, str))}
    nmf = np.array([_fm.get(x, np.nan) if isinstance(x, str) else np.nan for x in fo])
    nm = np.where(np.isfinite(nmf), nmf, nmp)
    tims = np.concatenate(tims)
    log(f"library mass index: formula {np.isfinite(nmf).mean():.1%}, timsTOF rows {tims.mean():.1%}")
    del fo
    return dict(off=off, mz=allmz, it=allin, prec=prec, nm=nm, pol=pol, tims=tims,
                ik=np.asarray(ik, dtype=object), smi=np.asarray(smi, dtype=object))


# ---------------------------------------------------------------------------------------------
# candidate pool = COCONUT (prvsiyan fingerprints) U training structures (fingerprinted here)
# ---------------------------------------------------------------------------------------------
_g = {}


def _fp_init():
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _g["bits"] = np.load(find("fp_bits.npy"))
    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)


def fp_and_mass(smi):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]
        return np.packbits(fp), float(ExactMolWt(m))
    except Exception:
        return None


def canon_key(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _g: _g["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def build_pool(L, workers):
    from multiprocessing import Pool as MPool
    d = os.path.dirname(find("coco_fp.npy"))
    cm = pickle.load(open(os.path.join(d, "coco_meta.pkl"), "rb"))
    co_fp = np.load(os.path.join(d, "coco_fp.npy")); co_mass = np.load(os.path.join(d, "coco_mass.npy"))
    co_keys = np.asarray(cm["keys"], dtype=object); co_smi = np.asarray(cm["smiles"], dtype=object)
    # ADDED: ChEBI + LIPID MAPS (prvsiyan/chebi-lipidmaps-casmi26), same 6,930-bit packed fingerprints
    bd = os.path.dirname(find("bio_fp.npy")); bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))
    bkeys = np.asarray(bm["keys"], dtype=object); bnew = ~pd.Series(bkeys).isin(set(co_keys)).values
    co_fp = np.vstack([co_fp, np.load(os.path.join(bd, "bio_fp.npy"))[bnew]])
    co_mass = np.concatenate([co_mass, np.load(os.path.join(bd, "bio_mass.npy"))[bnew]])
    co_keys = np.concatenate([co_keys, bkeys[bnew]])
    co_smi = np.concatenate([co_smi, np.asarray(bm["smiles"], dtype=object)[bnew]])
    log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")
    tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")
    tr = tr[~tr.ik.isin(set(co_keys))]
    log(f"COCONUT {len(co_keys):,}; training structures to fingerprint {len(tr):,}")
    with MPool(workers) as mp:
        res = mp.map(fp_and_mass, list(tr.smi), chunksize=500)
    ok = np.array([r is not None for r in res])
    tr_fp = np.stack([r[0] for r in res if r is not None]); tr_mass = np.array([r[1] for r in res if r is not None])
    fp = np.vstack([co_fp, tr_fp]); mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_keys, tr.ik.values[ok]]); smis = np.concatenate([co_smi, tr.smi.values[ok]])
    good = np.isfinite(mass)
    o = np.argsort(np.where(good, mass, 1e18), kind="mergesort")[: good.sum()]
    P = dict(fp=fp[o], mass=mass[o], keys=keys[o], smiles=smis[o])
    P["k2i"] = pd.Series(np.arange(len(o)), index=P["keys"]); P["k2i"] = P["k2i"][~P["k2i"].index.duplicated()]
    log(f"pool {len(o):,} structures")
    return P


def pool_fps(idx):
    return np.unpackbits(np.asarray(POOL["fp"][np.asarray(idx, np.int64)]), axis=1)[:, :6930]


def pool_window(t, ppm):
    a = np.searchsorted(POOL["mass"], t * (1 - ppm / 1e6), "left")
    b = np.searchsorted(POOL["mass"], t * (1 + ppm / 1e6), "right")
    return np.arange(a, b)


# ---------------------------------------------------------------------------------------------
# index: neutral-mass sorted library rows + cleaned representatives (all / positive / negative)
# ---------------------------------------------------------------------------------------------
def _reps(L, rows):
    npk = np.diff(L["off"])[rows]
    df = pd.DataFrame({"row": rows, "p": L["row_p"][rows], "npk": npk, "tims": L["tims"][rows]})
    # ADDED: prefer the test instrument (timsTOF) representative, then the richest spectrum
    df = df.sort_values(["p", "tims", "npk", "row"], ascending=[True, False, False, True]).drop_duplicates("p")
    rep = df.row.values; rep_nm = L["nm"][rep]
    o = np.argsort(rep_nm, kind="mergesort"); rep = rep[o]
    roff, rmz, rit = pv.clean_store(rep, L["off"], L["mz"], L["it"], CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
    return dict(rep=rep, rep_p=L["row_p"][rep], rep_nm=L["nm"][rep], roff=roff, rmz=rmz, rit=rit)


def build_index(L):
    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)
    rows = np.where(ok)[0]
    o = np.argsort(L["nm"][rows], kind="mergesort")
    I = dict(lib_rows=rows[o], lib_nm=L["nm"][rows[o]])
    I["all"] = _reps(L, rows)
    I["pos"] = _reps(L, rows[L["pol"][rows] == 1])
    I["neg"] = _reps(L, rows[L["pol"][rows] == -1])
    log(f"index: {len(rows):,} library rows; representatives {len(I['all']['rep']):,}")
    return I


# ---------------------------------------------------------------------------------------------
# per-molecule channels (identical to e02_channels.molecule_channels)
# ---------------------------------------------------------------------------------------------
def clean(mz, it):
    return pv._clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                     CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)


def analog_search(R, Q, target):
    lo = np.searchsorted(R["rep_nm"], target - CFG.ANALOG_WIN, "left")
    hi = np.searchsorted(R["rep_nm"], target + CFG.ANALOG_WIN, "right")
    asim = np.zeros(hi - lo, np.float32)
    if hi > lo and Q:
        shift = (target - R["rep_nm"][lo:hi]).astype(np.float32)
        for qm, qp in Q:
            asim = np.maximum(asim, pv.search_shift_pre(qm, qp, lo, hi, R["roff"], R["rmz"], R["rit"], CFG.MZ_TOL, shift))
    return R["rep_p"][lo:hi], asim


def top_analogs(p, s):
    if len(p) == 0: return []
    df = pd.DataFrame({"p": p, "s": s}).groupby("p", sort=False).s.max().sort_values(ascending=False, kind="mergesort")
    return list(zip(df.index[:CFG.N_ANALOG].astype(int), df.values[:CFG.N_ANALOG].astype(float)))


def molecule_channels(L, I, specs, precs, pols, target):
    Q = [clean(a, b) for a, b in specs]
    keep = [i for i, (a, _) in enumerate(Q) if len(a)]
    Q = [Q[i] for i in keep]; precs = np.asarray(precs)[keep]; pols = np.asarray(pols)[keep]
    tol = target * CFG.PPM_WIN / 1e6
    lo = np.searchsorted(I["lib_nm"], target - tol, "left"); hi = np.searchsorted(I["lib_nm"], target + tol, "right")
    crow = I["lib_rows"][lo:hi]
    u, inv = np.unique(L["row_p"][crow], return_inverse=True)
    M = np.zeros((max(len(Q), 1), len(u)), np.float32); MS = np.zeros_like(M)
    for j, (qm, qp) in enumerate(Q):
        if len(crow) == 0: break
        s = pv.search(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                      CFG.INT_POWER, CFG.ENT_WEIGHT)
        np.maximum.at(M[j], inv, s)
        sh = (precs[j] - L["prec"][crow]).astype(np.float32)
        s2 = pv.search_shift_rows(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                                  CFG.INT_POWER, CFG.ENT_WEIGHT, sh)
        np.maximum.at(MS[j], inv, s2)
    cnt = np.bincount(inv, minlength=len(u)).astype(np.float32)
    z = np.zeros(0, np.float32)
    lib = (u, M.max(0) if len(u) else z, M.mean(0) if len(u) else z, MS.max(0) if len(u) else z, cnt)
    p, s = analog_search(I["all"], Q, target)
    ana = top_analogs(p, s)
    ps, ss = [], []
    for sign, name in ((1, "pos"), (-1, "neg")):
        QQ = [q for q, pl in zip(Q, pols) if pl == sign]
        if QQ:
            p, s = analog_search(I[name], QQ, target); ps.append(p); ss.append(s)
    anapol = top_analogs(np.concatenate(ps), np.concatenate(ss)) if ps else []
    return lib, ana, anapol


def parse_ce(v):
    try:
        a = np.abs(np.atleast_1d(np.asarray(v, float)))
        return float(a.mean()) if len(a) else 25.0
    except Exception:
        return 25.0




# ---------------------------------------------------------------------------------------------
# ranking features: prvsiyan's 31 + our blocks (identical to e05_rank.py BLOCKS)
# ---------------------------------------------------------------------------------------------
def _rel(v):
    v = np.asarray(v, np.float32)
    return [v, pv._rank_norm(v).astype(np.float32), v - (v.max() if len(v) else 0.0)]


def _analog_feats(cfp, ana):
    nc = len(cfp)
    if not ana: return np.zeros((nc, 5), np.float32)
    afp = pool_fps([p for p, _ in ana]).astype(np.float32); cf = cfp.astype(np.float32)
    inter = cf @ afp.T
    tan = inter / (cf.sum(1)[:, None] + afp.sum(1)[None, :] - inter + 1e-9)
    w = np.clip(np.array([s for _, s in ana], np.float32), 0, None)
    ap = (tan * (w ** 3)[None, :]).max(1)
    ap6 = (tan * (w ** 6)[None, :]).max(1)
    return np.column_stack(_rel(ap) + [ap6, np.full(nc, w[0])]).astype(np.float32)


def blk_base(r, cfp):
    ana = r["ana"]
    afp = pool_fps([p for p, _ in ana]) if ana else None
    sims = np.array([s for _, s in ana], np.float32)
    return pv.rank_features(cfp, r["lib"], afp, sims, r["zlog"], r["frag"])


def blk_mass(r, cfp):
    ppm = (POOL["mass"][r["cand"]] - r["target"]) / r["target"] * 1e6
    return np.column_stack([ppm, np.abs(ppm), np.abs(ppm + 1.5)]).astype(np.float32)


def blk_libx(r, cfp):
    cnt = np.log1p(r["lib_cnt"])
    return np.column_stack(_rel(r["lib_mean"]) + _rel(r["libsh"]) + [cnt, cnt - cnt.max()]).astype(np.float32)


def blk_anapol(r, cfp):
    return _analog_feats(cfp, r["anapol"])


def blk_fragx(r, cfp):
    return np.column_stack(_rel(r["frag_ad"]) + _rel(r["frag_mean"]) + [r["frag_pol"]]).astype(np.float32)


BLOCKS = {"base": blk_base, "mass": blk_mass, "libx": blk_libx, "anapol": blk_anapol, "fragx": blk_fragx}


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------
def compute_channels(L, I, models, te):
    mols = list(te.groupby("molecule_id", sort=True))
    recs = []
    for gi, (mid, sub) in enumerate(mols):
        specs = []
        for r in sub.itertuples():
            a_ = np.asarray(r.ms2_mzs, np.float32); b_ = np.asarray(r.ms2_normalized_intensities, np.float32)
            k_ = a_ <= float(r.precursor_mz) + 2.0
            a_, b_ = a_[k_], b_[k_]
            specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))
        nms = pv.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.astype(str).values)
        nms = nms[np.isfinite(nms)]
        rec = dict(mid=mid, cand=np.zeros(0, np.int64))
        if len(nms):
            target = float(np.median(nms))
            modes = np.where(sub.ionization_mode.astype(str).values == "positive", 1.0, -1.0)
            lib, ana, anapol = molecule_channels(L, I, specs, sub.precursor_mz.values.astype(np.float64), modes, target)
            cand = pool_window(target, CFG.PPM_WIN)
            if len(cand) == 0: cand = pool_window(target, CFG.PPM_FALLBACK)
            ces = np.array([parse_ce(v) for v in sub.collision_energy_ev.values], np.float32)
            import pv_fp
            zlog = pv_fp.molecule_logits(models, specs, sub.precursor_mz.values.astype(np.float32),
                                         list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),
                                         ces, modes)
            u, lmax, lmean, lsh, lcnt = lib
            pos = {int(x): i for i, x in enumerate(u)}
            ci = np.array([pos.get(int(c), -1) for c in cand], np.int64); has = ci >= 0
            rec.update(target=target, cand=cand, zlog=None if zlog is None else zlog.astype(np.float32), specs=specs,
                       mode=float(np.mean(modes)), modes=modes, adducts=list(sub.adduct.astype(str).values),
                       ana=ana, anapol=anapol)
            for nm_, arr in (("lib", lmax), ("lib_mean", lmean), ("libsh", lsh), ("lib_cnt", lcnt)):
                v = np.zeros(len(cand), np.float32); v[has] = arr[ci[has]]; rec[nm_] = v
        recs.append(rec)
        if gi % 50 == 0: log(f"  channels {gi}/{len(mols)}")
    return mols, recs


def compute_frag(recs, workers):
    from multiprocessing import Pool as MPool
    uc = np.unique(np.concatenate([r["cand"] for r in recs]))
    log(f"MetFrag-lite on {len(uc):,} candidates")
    with MPool(workers) as mp:
        fr = mp.map(pv.frag_masses_safe, list(POOL["smiles"][uc]), chunksize=16)
    fmap = dict(zip(uc.tolist(), fr))
    for r in recs:
        if not len(r["cand"]): continue
        peaks = []
        for a_, b_ in r.pop("specs"):
            m2, i2 = pv._clean(a_, b_, CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)
            peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
        F1 = np.zeros((len(r["cand"]), max(len(peaks), 1)), np.float32); F2 = np.zeros_like(F1); F3 = np.zeros_like(F1)
        for j, c in enumerate(r["cand"]):
            fm = fmap[int(c)]
            for k, (x, y) in enumerate(peaks):
                F1[j, k] = pv.explain_score(fm, x, y, mode=r["mode"], tol=CFG.MZ_TOL)
                F2[j, k] = pv.explain_score_adduct(fm, x, y, r["adducts"][k], tol=CFG.MZ_TOL)
                F3[j, k] = pv.explain_score(fm, x, y, mode=r["modes"][k], tol=CFG.MZ_TOL)
        r["frag"] = F1.max(1); r["frag_ad"] = F2.max(1); r["frag_mean"] = F3.mean(1); r["frag_pol"] = F3.max(1)


def _hgb(X, Y, W, seed, gbm):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=seed, **gbm); m.fit(X, Y, sample_weight=W)
    return m


def fit_rankers(rank_train_path):
    """Our ranker: simulation rows from 7 query sets (S = 0 class-1 sim, 1 = class-2 sim)."""
    z = np.load(rank_train_path, allow_pickle=True)
    X, Y, S = z["X"], z["Y"], z["S"]
    rankers = [_hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, RANK.GBM) for wA in RANK.W_A for sd in RANK.SEEDS]
    log(f"our ranker: {len(rankers)} GBMs on {X.shape[0]:,} rows x {X.shape[1]} features")
    return rankers, [str(b) for b in z["blocks"]]


def fit_pv_rankers(pv_rank_path, priors=(0.30, 0.60), seeds=(0, 1, 2, 3)):
    """prvsiyan's ranker, exactly as in his notebook: shipped rank_train.npz (M = 0 class-1 sim)."""
    z = np.load(pv_rank_path)
    rankers = [_hgb(z["X"], z["Y"], np.where(z["M"] == 0, w1, 1.0 - w1), sd, CFG.GBM) for w1 in priors for sd in seeds]
    log(f"prvsiyan ranker: {len(rankers)} GBMs on {z['X'].shape[0]:,} rows x {z['X'].shape[1]} features")
    return rankers, z["X"].shape[1]


def score_molecules(recs, rankers, blocks, use_fp):
    out = {}
    for r in recs:
        if not len(r["cand"]): continue
        rr = r if use_fp else dict(r, zlog=None)
        cfp = pool_fps(r["cand"])
        Xm = np.hstack([BLOCKS[b](rr, cfp) for b in blocks]).astype(np.float32)
        out[r["mid"]] = np.mean([m.predict_proba(Xm)[:, 1] for m in rankers], axis=0)
    return out


def blend_scores(a, b, wa):
    """Within-molecule rank blend (both inputs are per-candidate probabilities)."""
    out = {}
    for mid in a:
        ra = 1.0 - pv._rank_norm(a[mid]); rb = 1.0 - pv._rank_norm(b[mid])
        out[mid] = wa * ra + (1.0 - wa) * rb + 1e-6 * a[mid]
    return out


def make_submissions(mols, recs, score_sets, sample_path, workers, topn=25):
    """score_sets: {name: {mid: scores}} -> {name: submission DataFrame}; one canonicalisation pass for all."""
    from multiprocessing import Pool as MPool
    cand = {r["mid"]: r["cand"] for r in recs if len(r["cand"])}
    ordered = {n: {mid: cand[mid][np.argsort(-sc, kind="mergesort")][:topn + 15] for mid, sc in S.items()}
               for n, S in score_sets.items()}
    allc = [v for o in ordered.values() for v in o.values()]
    short = np.unique(np.concatenate(allc)) if allc else np.zeros(0, np.int64)
    with MPool(workers) as mp:
        ck = mp.map(canon_key, list(POOL["smiles"][short]), chunksize=64)
    ckey = dict(zip(short.tolist(), ck))
    samp = pd.read_csv(sample_path)
    subs = {}
    for n, o in ordered.items():
        rows = []
        for mid, _ in mols:
            out, seen = [], set()
            for c in o.get(mid, []):
                k = ckey.get(int(c)) or POOL["keys"][c]
                if k in seen: continue
                seen.add(k); out.append(POOL["smiles"][c])
                if len(out) == topn: break
            while len(out) < topn:
                out.append("CCO")
            rows.append((mid, ";".join(out[:topn])))
        sub = samp[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id", how="left")
        sub["smiles"] = sub["smiles"].fillna("CCO")
        assert len(sub) == len(samp) and sub.molecule_id.duplicated().sum() == 0 and sub.smiles.isnull().sum() == 0
        assert (sub.smiles.str.split(";").map(len) <= topn).all()
        subs[n] = sub
    return subs


def main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88):
    global POOL
    import torch
    import pv_fp
    L = load_library(train_path); log(f"library {len(L['prec']):,} spectra")
    POOL = build_pool(L, workers)
    L["row_p"] = POOL["k2i"].reindex(L["ik"]).fillna(-1).values.astype(np.int64)
    del L["ik"], L["smi"]
    I = build_index(L)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    single, merged, _ = pv_fp.load_fp_models(fp_model_paths, dev)
    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}")
    te = pq.read_table(test_path).to_pandas()
    if limit:
        keep = sorted(te.molecule_id.unique())[:limit]; te = te[te.molecule_id.isin(keep)]
    log(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")
    mols, recs = compute_channels(L, I, (single, merged, dev), te)
    del L, I
    compute_frag(recs, workers)
    ours, blocks = fit_rankers(our_rank_path)
    s_ours = score_molecules(recs, ours, blocks, use_fp=False)
    del ours
    pvr, nfeat = fit_pv_rankers(pv_rank_path)
    s_pv = score_molecules(recs, pvr, ["base"], use_fp=True)
    del pvr
    subs = make_submissions(mols, recs, {"blend": blend_scores(s_pv, s_ours, w_pv), "pv": s_pv, "ours": s_ours},
                            sample_path, workers)
    log("submissions ready")
    return subs, recs, {"pv": s_pv, "ours": s_ours}
