# %% [markdown]
# # CASMI 2026 — candidate ranking by spectral-analog propagation
#
# The visible `test.parquet` is a placeholder made of training rows; at scoring time
# Kaggle re-runs this notebook against a hidden test set of natural products from
# biological extracts. So nothing here may key off the visible file's molecule ids,
# mass range, adducts or instrument — everything is derived at runtime.
#
# ### Method
#
# For each test molecule:
#
# 1. **Candidates** — the precursor m/z and adduct give a neutral monoisotopic mass;
#    every structure in a 729,596-structure pool (COCONUT 2.0 + all training
#    structures) within a ppm window is a candidate. Typically ~90 of them.
# 2. **Analogs** — search the molecule's spectra against all 2.5M library spectra by
#    soft-binned cosine, in both the direct and the neutral-loss domain, and keep the
#    best-matching library *structures*.
# 3. **Rank** — `score(c) = max over analogs a of tanimoto(c, a) * spectral_sim(a)**4`.
#    A library compound that fragments like the unknown is evidence for candidates
#    that resemble that compound.
#
# This covers both regimes with no special casing: if the true structure is itself in
# the library, it is its own best analog (tanimoto 1.0) and sorts to rank 1; if it is
# genuinely novel, it is ranked by resemblance to the compounds that fragment like it.
#
# Local measurements, structure-disjoint (every spectrum of the answer removed):
#
# | dev set | molecules | MRR@25 | top-1 | pool recall |
# |---|---|---|---|---|
# | `enveda-np-examples`, timsTOF natural products | 518 | **0.475** | 0.336 | 0.975 |
# | public-library natural products | 3,496 | **0.429** | 0.295 | 0.905 |
# | random-order floor | 518 | 0.104 | 0.035 | 0.975 |

# %%
import gc
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# Kaggle's mount layout is not guaranteed: datasets appear under
# /kaggle/input/datasets/<owner>/<slug> and competition data under a similarly
# nested path, neither of which matches the slug-only path the docs imply. So both
# locations are discovered by looking for a file we know they contain.
def find_dir(marker, extra_roots=()):
    roots = ["/kaggle/input", *extra_roots, "data", "."]
    seen = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - root.count(os.sep) > 4:
                dirnames[:] = []
                continue
            seen.append(dirpath)
            if marker in filenames:
                return dirpath, seen
    return None, seen


def require_dir(marker, what):
    found, seen = find_dir(marker)
    if found is None:
        print(f"could not locate {what} (looking for {marker}); searched:")
        for d in seen[:300]:
            try:
                print("   ", d, sorted(os.listdir(d))[:8])
            except OSError:
                pass
        raise FileNotFoundError(what)
    return found


IN = require_dir("test.parquet", "competition data")
POOL = require_dir("pool_index.parquet", "candidate pool dataset")
log(f"competition data: {IN}")
log(f"candidate pool:   {POOL}")

# peak cleaning
MIN_REL = 1e-3
MAX_PEAKS = 512
HEADROOM = 5.0
# binning / similarity
BIN_WIDTH = 0.01
POWER = 0.5
GUARD = 1.5
MAX_MZ = 1000.0
# retrieval
TOPK_SPECTRA = 500
TOP_ANALOGS = 100
SIM_POWER = 4.0
# candidates
PPM_STEPS = (10.0, 20.0, 50.0)   # widen only if a molecule has no candidates
N_OUT = 25
MIN_PEAKS_REF = 3

# %% [markdown]
# ## Peak cleaning and a flat peak store
#
# `train.parquet` is 3 GB of nested lists, so it is streamed row group by row group
# and reduced to flat float32 arrays addressed by (offset, count).

# %%
def clean(mz, it, pmz):
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(it, dtype=np.float64)
    empty = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    if mz.size == 0 or it.size != mz.size or not np.isfinite(it).any():
        return empty
    peak = np.nanmax(it)
    if not (peak > 0):
        return empty
    keep = (it >= MIN_REL * peak) & np.isfinite(mz) & (mz > 0)
    if np.isfinite(pmz):
        keep &= mz <= pmz + HEADROOM
    mz, it = mz[keep], it[keep]
    if mz.size > MAX_PEAKS:
        sel = np.argpartition(it, -MAX_PEAKS)[-MAX_PEAKS:]
        mz, it = mz[sel], it[sel]
    order = np.argsort(mz, kind="stable")
    return mz[order].astype(np.float32), (it[order] / peak).astype(np.float32)


def load_split(path, meta_cols):
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    cols = list(dict.fromkeys(
        [c for c in meta_cols if c in have]
        + ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]))
    metas, mzs_out, its_out, offs, ns = [], [], [], [], []
    off = 0
    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        mzs = df.pop("ms2_mzs").to_numpy()
        its = df.pop("ms2_normalized_intensities").to_numpy()
        pmz = df.precursor_mz.to_numpy()
        for i in range(len(df)):
            m, v = clean(mzs[i], its[i], pmz[i])
            mzs_out.append(m)
            its_out.append(v)
            offs.append(off)
            ns.append(m.size)
            off += m.size
        metas.append(df)
        del mzs, its, df
        gc.collect()
        if (rg + 1) % 5 == 0 or rg + 1 == pf.metadata.num_row_groups:
            log(f"  row group {rg+1}/{pf.metadata.num_row_groups}, {off:,} peaks")
    meta = pd.concat(metas, ignore_index=True)
    meta["peak_off"] = np.asarray(offs, dtype=np.int64)
    meta["peak_n"] = np.asarray(ns, dtype=np.int32)
    return meta, np.concatenate(mzs_out), np.concatenate(its_out)


# %% [markdown]
# ## Soft binning
#
# Each peak is split linearly between its two neighbouring bin centres. With hard
# bins, two peaks a fraction of a milli-dalton apart can straddle an edge and
# contribute nothing; locally soft binning held 0.81 self-cosine under a 0.004 Da
# calibration shift versus 0.64.

# %%
N_BINS = int(round(MAX_MZ / BIN_WIDTH))


def bin_matrix(mz_all, it_all, off, n, pmz, neutral_loss=False):
    n = np.asarray(n, dtype=np.int64)
    off = np.asarray(off, dtype=np.int64)
    total = int(n.sum())
    if total == 0:
        return sparse.csr_matrix((len(n), N_BINS), dtype=np.float32)
    starts = np.repeat(off, n)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(n)[:-1])), n)
    idx = starts + within
    row_id = np.repeat(np.arange(len(n), dtype=np.int64), n)
    mz = mz_all[idx].astype(np.float64)
    val = it_all[idx].astype(np.float64)
    p = np.asarray(pmz, dtype=np.float64)[row_id]

    if neutral_loss:
        pos = p - mz
        keep = pos > GUARD
    else:
        pos = mz
        keep = mz < p - GUARD
    keep &= (pos >= 0) & (pos < MAX_MZ) & (val > 0) & np.isfinite(pos)
    pos, val, row_id = pos[keep], val[keep] ** POWER, row_id[keep]
    if pos.size == 0:
        return sparse.csr_matrix((len(n), N_BINS), dtype=np.float32)

    f = pos / BIN_WIDTH - 0.5
    lo = np.floor(f).astype(np.int64)
    w = f - lo
    b = np.concatenate([lo, lo + 1])
    val = np.concatenate([val * (1.0 - w), val * w])
    row_id = np.concatenate([row_id, row_id])
    ok = (b >= 0) & (b < N_BINS) & (val > 0)
    b, val, row_id = b[ok], val[ok], row_id[ok]

    flat = row_id * N_BINS + b
    order = np.argsort(flat, kind="stable")
    flat, val, row_id = flat[order], val[order], row_id[order]
    uniq, start = np.unique(flat, return_index=True)
    val = np.add.reduceat(val, start)
    row_id = row_id[start]
    nrm = np.sqrt(np.bincount(row_id, weights=val * val, minlength=len(n)))
    nrm[nrm == 0] = 1.0
    val = val / nrm[row_id]
    counts = np.bincount(row_id, minlength=len(n))
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return sparse.csr_matrix(
        (val.astype(np.float32), (uniq % N_BINS).astype(np.int32), indptr),
        shape=(len(n), N_BINS))


# %% [markdown]
# ## Adduct arithmetic
#
# The neutral mass implied by the precursor is the strongest single constraint, so
# adducts are parsed rather than table-matched: the hidden test may use adducts the
# visible file does not contain.

# %%
ELECTRON = 0.00054857990946
MONO = {"H": 1.0078250319, "D": 2.0141017780, "C": 12.0, "N": 14.0030740052,
        "O": 15.9949146221, "F": 18.99840320, "Na": 22.98976928, "Mg": 23.98504170,
        "Si": 27.9769265327, "P": 30.97376151, "S": 31.97207069, "Cl": 34.96885271,
        "K": 38.96370649, "Ca": 39.96259098, "Fe": 55.93493633, "Br": 78.91833710,
        "I": 126.904473, "Li": 7.01600344, "Se": 79.91652132, "B": 11.0093054}
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADD = re.compile(r"^\[(\d*)M(?P<mods>(?:[+-][A-Za-z0-9]+)*)\](?P<chg>\d*[+-])$")


def tok_mass(tok):
    m, pos = 0.0, 0
    for mt in _TOK.finditer(tok):
        if mt.start() != pos:
            raise KeyError(tok)
        pos = mt.end()
        m += MONO[mt.group(1)] * (int(mt.group(2)) if mt.group(2) else 1)
    if pos != len(tok):
        raise KeyError(tok)
    return m


def neutral_mass(pmz, adduct):
    """Neutral monoisotopic mass; falls back to [M+H]+ if the adduct is unparsable."""
    try:
        m = _ADD.match(str(adduct).strip().replace(" ", ""))
        if not m:
            raise KeyError(adduct)
        n_mers = int(m.group(1)) if m.group(1) else 1
        chg_s = m.group("chg")
        mag = int(chg_s[:-1]) if len(chg_s) > 1 else 1
        charge = mag if chg_s[-1] == "+" else -mag
        delta = 0.0
        for sign, tk in re.findall(r"([+-])([A-Za-z0-9]+)", m.group("mods")):
            delta += tok_mass(tk) if sign == "+" else -tok_mass(tk)
        return (pmz * abs(charge) - delta + charge * ELECTRON) / n_mers
    except Exception:
        return pmz - MONO["H"] + ELECTRON


# %% [markdown]
# ## Load the candidate pool

# %%
pool = pd.read_parquet(f"{POOL}/pool_index.parquet")
pool_fp = np.load(f"{POOL}/pool_fp_packed.npy")
assert len(pool) == len(pool_fp)
pool_mass = pool.mass.to_numpy(dtype=np.float64)
pool_key = pool.key14.to_numpy()
pool_smiles = pool.smiles.to_numpy()
mass_order = np.argsort(pool_mass, kind="stable")
mass_sorted = pool_mass[mass_order]
# first pool row holding each structure, for looking up an analog's fingerprint
pool_row_of_key = {}
for r, k in enumerate(pool_key):
    if k not in pool_row_of_key:
        pool_row_of_key[k] = r
log(f"pool: {len(pool):,} structures, fp {pool_fp.shape}")

# %% [markdown]
# ## Load test and library

# %%
test_meta, test_mz, test_it = load_split(
    f"{IN}/test.parquet",
    ["molecule_id", "spectrum_id", "adduct", "instrument_type", "precursor_mz"])
log(f"test: {len(test_meta)} spectra, {test_meta.molecule_id.nunique()} molecules, "
    f"instruments {test_meta.instrument_type.value_counts().to_dict() if 'instrument_type' in test_meta else 'n/a'}")

# %%
ref_meta, ref_mz, ref_it = load_split(
    f"{IN}/train.parquet",
    ["normalized_smiles", "adduct", "instrument_type", "precursor_mz"])
log(f"library: {len(ref_meta):,} spectra")

# map library spectra to tautomer-canonical structure keys, then to compact ids
structs = pd.read_parquet(f"{POOL}/structures.parquet",
                          columns=["normalized_smiles", "key14"])
ref_meta = ref_meta.merge(structs, on="normalized_smiles", how="left")
ref_meta.drop(columns=["normalized_smiles"], inplace=True)
ok = ref_meta.key14.notna().to_numpy() & (ref_meta.peak_n.to_numpy() >= MIN_PEAKS_REF)
ref_meta = ref_meta[ok].reset_index(drop=True)
struct_codes, struct_uniques = pd.factorize(ref_meta.key14)
ref_struct = struct_codes.astype(np.int32)
struct_keys = np.asarray(struct_uniques)
ref_meta.drop(columns=["key14"], inplace=True)
gc.collect()
log(f"library usable: {len(ref_meta):,} spectra / {len(struct_keys):,} structures")

# %% [markdown]
# ## Search both domains, accumulating analog evidence
#
# Sparse x sparse, so cost tracks query non-zeros rather than library size. Domains
# are processed one at a time and the reference matrix freed in between, to keep peak
# memory to roughly one matrix.

# %%
q_off = test_meta.peak_off.to_numpy()
q_n = test_meta.peak_n.to_numpy()
q_pmz = test_meta.precursor_mz.to_numpy()
r_off = ref_meta.peak_off.to_numpy()
r_n = ref_meta.peak_n.to_numpy()
r_pmz = ref_meta.precursor_mz.to_numpy()
nq = len(test_meta)

# per query spectrum: best spectral similarity for each library structure
analog_best = [dict() for _ in range(nq)]

for nl in (False, True):
    R = bin_matrix(ref_mz, ref_it, r_off, r_n, r_pmz, neutral_loss=nl)
    Q = bin_matrix(test_mz, test_it, q_off, q_n, q_pmz, neutral_loss=nl)
    log(f"  binned {'neutral-loss' if nl else 'direct'}: ref nnz {R.nnz:,}")
    CH = 64
    for s in range(0, nq, CH):
        e = min(s + CH, nq)
        prod = (R @ Q[s:e].T).tocsc()
        for c in range(e - s):
            lo, hi = prod.indptr[c], prod.indptr[c + 1]
            rows, vals = prod.indices[lo:hi], prod.data[lo:hi]
            if rows.size == 0:
                continue
            k = min(TOPK_SPECTRA, rows.size)
            sel = np.argpartition(vals, -k)[-k:]
            d = analog_best[s + c]
            for sid, v in zip(ref_struct[rows[sel]], vals[sel]):
                v = float(v)
                if v > d.get(sid, -1.0):
                    d[sid] = v
        del prod
    del R, Q
    gc.collect()
    log(f"  searched {'neutral-loss' if nl else 'direct'}")

# keep only the strongest analogs per query spectrum
for q in range(nq):
    if len(analog_best[q]) > TOP_ANALOGS:
        analog_best[q] = dict(sorted(analog_best[q].items(),
                                     key=lambda kv: -kv[1])[:TOP_ANALOGS])
log("analogs collected")

# %% [markdown]
# ## Merge a molecule's spectra, then rank its candidates
#
# A molecule's several collision energies are merged by keeping each analog's best
# similarity — locally max beat averaging, because the informative energy differs
# per compound.

# %%
mol_of_q = test_meta.molecule_id.to_numpy()
by_mol = {}
for q in range(nq):
    tgt = by_mol.setdefault(mol_of_q[q], {})
    for sid, v in analog_best[q].items():
        if v > tgt.get(sid, -1.0):
            tgt[sid] = v

mol_info = test_meta.groupby("molecule_id").agg(
    pmz=("precursor_mz", "first"),
    adduct=("adduct", "first") if "adduct" in test_meta else ("precursor_mz", "first"))


def tanimoto(cand_rows, analog_rows):
    a = np.unpackbits(pool_fp[cand_rows], axis=1).astype(np.float32)
    b = np.unpackbits(pool_fp[analog_rows], axis=1).astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


rows_out = []
stats = {"no_candidates": 0, "widened": 0, "no_analogs": 0}
for mol in mol_info.index:
    nm = neutral_mass(float(mol_info.pmz[mol]), mol_info.adduct[mol])
    cand = np.zeros(0, dtype=np.int64)
    for ppm in PPM_STEPS:
        tol = nm * ppm * 1e-6
        lo = np.searchsorted(mass_sorted, nm - tol)
        hi = np.searchsorted(mass_sorted, nm + tol)
        cand = mass_order[lo:hi]
        if len(cand):
            if ppm != PPM_STEPS[0]:
                stats["widened"] += 1
            break

    analogs = by_mol.get(mol, {})
    a_items = [(struct_keys[sid], v) for sid, v in analogs.items()]
    a_items = [(k, v) for k, v in a_items if k in pool_row_of_key]

    ranked_keys, seen = [], set()
    if len(cand) and a_items:
        a_rows = np.array([pool_row_of_key[k] for k, _ in a_items], dtype=np.int64)
        w = np.array([v for _, v in a_items], dtype=np.float32) ** SIM_POWER
        sc = (tanimoto(cand, a_rows) * w[None, :]).max(axis=1)
        best = {}
        for k, v in zip(pool_key[cand], sc):
            if v > best.get(k, -1.0):
                best[k] = float(v)
        ranked_keys = [k for k, _ in sorted(best.items(), key=lambda kv: -kv[1])]
    elif len(cand):
        stats["no_analogs"] += 1
        ranked_keys = list(dict.fromkeys(pool_key[cand]))
    else:
        stats["no_candidates"] += 1

    smis = []
    for k in ranked_keys:
        if k in seen:
            continue
        seen.add(k)
        smis.append(pool_smiles[pool_row_of_key[k]])
        if len(smis) >= N_OUT:
            break
    # last resort: the best-matching library structures themselves, mass or not
    if len(smis) < N_OUT:
        for k, _ in sorted(a_items, key=lambda kv: -kv[1]):
            if k in seen:
                continue
            seen.add(k)
            smis.append(pool_smiles[pool_row_of_key[k]])
            if len(smis) >= N_OUT:
                break
    if not smis:
        smis = [pool_smiles[0]]
    rows_out.append((mol, ";".join(smis)))

log(f"ranked {len(rows_out)} molecules; {json.dumps(stats)}")

# %%
sub = pd.DataFrame(rows_out, columns=["molecule_id", "smiles"])
sample = pd.read_csv(f"{IN}/sample_submission.csv")
sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
missing = sub.smiles.isna().sum()
if missing:
    log(f"WARNING: {missing} molecule_ids absent from predictions; filling")
    sub["smiles"] = sub.smiles.fillna(pool_smiles[0])
sub.to_csv("submission.csv", index=False)
n = sub.smiles.str.count(";") + 1
log(f"wrote submission.csv: {len(sub)} rows, candidates/row "
    f"min {n.min()} median {n.median():.0f} max {n.max()}")
print(sub.head())
