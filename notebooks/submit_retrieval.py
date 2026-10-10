# %% [markdown]
# # CASMI 2026 - spectral library retrieval probe
#
# Self-contained baseline: for each test molecule, search its MS2 spectra against the
# timsTOF portion of the training library by soft-binned cosine, map hit spectra to
# their structures, and rank structures with the precursor-derived neutral mass as a
# constraint.
#
# The point of this run is diagnostic as much as competitive. Pure retrieval can only
# score if the test structures are themselves present in the training library. A score
# near the ~0.47 local in-library figure means the test molecules are largely in the
# library; a score near zero means they are genuine unknowns and the pipeline must be
# built around candidate-database ranking instead.

# %%
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

T0 = time.time()
IN = "/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"
if not os.path.isdir(IN):
    IN = "enveda-CASMI26-molecule-id-mass-spectra"

MIN_REL = 1e-3          # relative-intensity floor
MAX_PEAKS = 512
HEADROOM = 5.0          # drop fragments above precursor + this
BIN_WIDTH = 0.01
POWER = 0.5             # intensity**power before L2 normalisation
GUARD = 1.5             # drop peaks within this of the precursor
MAX_MZ = 1000.0
TOPK = 200
MASS_PPM = 20.0
N_OUT = 25


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# %% [markdown]
# ## Peak cleaning and flat peak store
#
# train.parquet is 3 GB with nested peak lists, so it is streamed row group by row
# group and only the timsTOF rows (the instrument the whole test set was measured on)
# are kept, as flat float32 arrays.

# %%
def clean(mz, it, pmz):
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(it, dtype=np.float64)
    if mz.size == 0 or not np.isfinite(it).any():
        return mz[:0].astype(np.float32), it[:0].astype(np.float32)
    peak = np.nanmax(it)
    if not (peak > 0):
        return mz[:0].astype(np.float32), it[:0].astype(np.float32)
    keep = (it >= MIN_REL * peak) & np.isfinite(mz) & (mz > 0)
    if np.isfinite(pmz):
        keep &= mz <= pmz + HEADROOM
    mz, it = mz[keep], it[keep]
    if mz.size > MAX_PEAKS:
        sel = np.argpartition(it, -MAX_PEAKS)[-MAX_PEAKS:]
        mz, it = mz[sel], it[sel]
    order = np.argsort(mz, kind="stable")
    return mz[order].astype(np.float32), (it[order] / peak).astype(np.float32)


def load_split(path, columns, instrument=None):
    pf = pq.ParquetFile(path)
    cols = [c for c in columns if c in set(pf.schema_arrow.names)]
    need = cols + ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]
    need = list(dict.fromkeys(need))
    metas, mz_parts, it_parts, offs, ns = [], [], [], [], []
    off = 0
    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg, columns=need).to_pandas()
        if instrument is not None:
            df = df[df.instrument_type == instrument]
            if df.empty:
                continue
            df = df.reset_index(drop=True)
        mzs = df.pop("ms2_mzs").to_numpy()
        its = df.pop("ms2_normalized_intensities").to_numpy()
        pmz = df.precursor_mz.to_numpy()
        for i in range(len(df)):
            m, v = clean(mzs[i], its[i], pmz[i])
            mz_parts.append(m)
            it_parts.append(v)
            offs.append(off)
            ns.append(m.size)
            off += m.size
        metas.append(df)
        log(f"  row group {rg+1}/{pf.metadata.num_row_groups}: {off:,} peaks")
        del mzs, its, df
        gc.collect()
    meta = pd.concat(metas, ignore_index=True)
    meta["peak_off"] = np.asarray(offs, dtype=np.int64)
    meta["peak_n"] = np.asarray(ns, dtype=np.int32)
    return meta, np.concatenate(mz_parts), np.concatenate(it_parts)


# %% [markdown]
# ## Soft binning
#
# Each peak is split linearly across its two neighbouring bin centres. Hard binning
# would drop the match between two peaks a fraction of a milli-dalton apart that
# happen to straddle a bin edge; locally this was worth ~0.17 cosine under a 0.004 Da
# calibration shift.

# %%
def bin_matrix(mz_all, it_all, off, n, pmz, neutral_loss=False):
    n_bins = int(round(MAX_MZ / BIN_WIDTH))
    n = np.asarray(n, dtype=np.int64)
    off = np.asarray(off, dtype=np.int64)
    total = int(n.sum())
    if total == 0:
        return sparse.csr_matrix((len(n), n_bins), dtype=np.float32)
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
        return sparse.csr_matrix((len(n), n_bins), dtype=np.float32)

    f = pos / BIN_WIDTH - 0.5
    lo = np.floor(f).astype(np.int64)
    w = f - lo
    b = np.concatenate([lo, lo + 1])
    val = np.concatenate([val * (1.0 - w), val * w])
    row_id = np.concatenate([row_id, row_id])
    ok = (b >= 0) & (b < n_bins) & (val > 0)
    b, val, row_id = b[ok], val[ok], row_id[ok]

    flat = row_id * n_bins + b
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
        (val.astype(np.float32), (uniq % n_bins).astype(np.int32), indptr),
        shape=(len(n), n_bins))


# %% [markdown]
# ## Adduct arithmetic
#
# The neutral monoisotopic mass implied by the observed precursor is the single
# strongest constraint available, so adducts are parsed properly rather than
# table-matched.

# %%
ELECTRON = 0.00054857990946
MONO = {"H": 1.0078250319, "C": 12.0, "N": 14.0030740052, "O": 15.9949146221,
        "F": 18.99840320, "Na": 22.98976928, "Si": 27.9769265327, "P": 30.97376151,
        "S": 31.97207069, "Cl": 34.96885271, "K": 38.96370649, "Br": 78.91833710,
        "I": 126.904473, "Li": 7.01600344, "Mg": 23.98504170, "Ca": 39.96259098}
import re  # noqa: E402

_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADD = re.compile(r"^\[(\d*)M(?P<mods>(?:[+-][A-Za-z0-9]+)*)\](?P<chg>\d*[+-])$")


def tok_mass(tok):
    m = 0.0
    for el, cnt in _TOK.findall(tok):
        m += MONO[el] * (int(cnt) if cnt else 1)
    return m


def neutral_mass(pmz, adduct):
    m = _ADD.match(str(adduct).strip().replace(" ", ""))
    if not m:
        return float("nan")
    n_mers = int(m.group(1)) if m.group(1) else 1
    chg_s = m.group("chg")
    mag = int(chg_s[:-1]) if len(chg_s) > 1 else 1
    charge = mag if chg_s[-1] == "+" else -mag
    delta = 0.0
    try:
        for sign, tk in re.findall(r"([+-])([A-Za-z0-9]+)", m.group("mods")):
            delta += tok_mass(tk) if sign == "+" else -tok_mass(tk)
    except KeyError:
        return float("nan")
    return (pmz * abs(charge) - delta + charge * ELECTRON) / n_mers


# %% [markdown]
# ## Load library and queries

# %%
log("loading test")
test_meta, test_mz, test_it = load_split(
    f"{IN}/test.parquet",
    ["molecule_id", "spectrum_id", "adduct", "instrument_type", "precursor_mz"])
log(f"test: {len(test_meta)} spectra, {test_meta.molecule_id.nunique()} molecules")

log("loading train (timsTOF only)")
ref_meta, ref_mz, ref_it = load_split(
    f"{IN}/train.parquet",
    ["normalized_smiles", "inchikey14", "adduct", "instrument_type", "precursor_mz"],
    instrument="timsTOF")
log(f"reference: {len(ref_meta):,} spectra, {ref_meta.inchikey14.nunique():,} structures")
gc.collect()

# %%
ref_meta = ref_meta[ref_meta.peak_n >= 3].reset_index(drop=True)
R = bin_matrix(ref_mz, ref_it, ref_meta.peak_off.to_numpy(), ref_meta.peak_n.to_numpy(),
               ref_meta.precursor_mz.to_numpy())
log(f"reference binned: nnz {R.nnz:,}")
Q = bin_matrix(test_mz, test_it, test_meta.peak_off.to_numpy(),
               test_meta.peak_n.to_numpy(), test_meta.precursor_mz.to_numpy())
log(f"query binned: nnz {Q.nnz:,}")

# %% [markdown]
# ## Search
#
# Sparse x sparse, so cost tracks query non-zeros rather than library size.

# %%
nq = Q.shape[0]
top_s = np.zeros((nq, TOPK), dtype=np.float32)
top_i = np.full((nq, TOPK), -1, dtype=np.int64)
CHUNK = 64
for s in range(0, nq, CHUNK):
    e = min(s + CHUNK, nq)
    prod = (R @ Q[s:e].T).tocsc()
    for c in range(e - s):
        lo, hi = prod.indptr[c], prod.indptr[c + 1]
        rows, vals = prod.indices[lo:hi], prod.data[lo:hi]
        if rows.size == 0:
            continue
        k = min(TOPK, rows.size)
        sel = np.argpartition(vals, -k)[-k:]
        order = sel[np.argsort(-vals[sel])]
        top_s[s + c, :k] = vals[order]
        top_i[s + c, :k] = rows[order]
    del prod
    if (s // CHUNK) % 5 == 0:
        log(f"  searched {e}/{nq}")
log("search done")

# %% [markdown]
# ## Structure masses
#
# RDKit exact masses for the library structures, used for the mass constraint.

# %%
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")
smi_by_key = ref_meta.groupby("inchikey14").normalized_smiles.first()
mass_by_key = {}
for key, smi in smi_by_key.items():
    mol = Chem.MolFromSmiles(smi)
    mass_by_key[key] = Descriptors.ExactMolWt(mol) if mol is not None else float("nan")
log(f"masses for {len(mass_by_key):,} structures")

# %% [markdown]
# ## Aggregate to a ranked structure list per molecule
#
# Per query spectrum, each structure keeps its best-matching spectrum (max), then the
# several spectra of one molecule are combined by max: locally that beat averaging
# (0.292 vs 0.243 MRR) because the informative collision energy varies by compound.

# %%
ref_key_arr = ref_meta.inchikey14.to_numpy()
q_mol = test_meta.molecule_id.to_numpy()

per_mol = {}
for q in range(nq):
    rows, vals = top_i[q], top_s[q]
    ok = rows >= 0
    best = {}
    for r, v in zip(rows[ok], vals[ok]):
        k = ref_key_arr[r]
        if v > best.get(k, -1.0):
            best[k] = float(v)
    tgt = per_mol.setdefault(q_mol[q], {})
    for k, v in best.items():
        if v > tgt.get(k, -1.0):
            tgt[k] = v

mol_info = test_meta.groupby("molecule_id").agg(
    pmz=("precursor_mz", "first"), adduct=("adduct", "first"))
mol_mass = {m: neutral_mass(p, a) for m, p, a in
            zip(mol_info.index, mol_info.pmz, mol_info.adduct)}

# a mass-matched popularity fallback for molecules retrieval says nothing about
key_counts = ref_meta.inchikey14.value_counts()
pop_keys = [(k, mass_by_key.get(k, float("nan"))) for k in key_counts.index[:40000]]

rows_out = []
for mol in mol_info.index:
    cands = per_mol.get(mol, {})
    mm = mol_mass[mol]
    if np.isfinite(mm):
        tol = mm * MASS_PPM * 1e-6
        hit_keys = {k for k in cands
                    if np.isfinite(mass_by_key.get(k, np.nan))
                    and abs(mass_by_key[k] - mm) <= tol}
        hit = [(k, v) for k, v in cands.items() if k in hit_keys]
        miss = [(k, v) for k, v in cands.items() if k not in hit_keys]
    else:
        hit, miss = list(cands.items()), []
    hit.sort(key=lambda kv: -kv[1])
    miss.sort(key=lambda kv: -kv[1])
    ranked = [k for k, _ in hit] + [k for k, _ in miss]
    if len(ranked) < N_OUT and np.isfinite(mm):
        tol = mm * MASS_PPM * 1e-6
        seen = set(ranked)
        for k, m in pop_keys:
            if len(ranked) >= N_OUT:
                break
            if k not in seen and np.isfinite(m) and abs(m - mm) <= tol:
                ranked.append(k)
                seen.add(k)
    smis = [smi_by_key[k] for k in ranked[:N_OUT] if k in smi_by_key.index]
    if not smis:
        smis = [smi_by_key.iloc[0]]
    rows_out.append((mol, ";".join(smis)))

sub = pd.DataFrame(rows_out, columns=["molecule_id", "smiles"])
log(f"submission rows {len(sub)}, mean candidates "
    f"{np.mean([len(s.split(';')) for s in sub.smiles]):.1f}")

# %%
sample = pd.read_csv(f"{IN}/sample_submission.csv")
sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
sub["smiles"] = sub.smiles.fillna(smi_by_key.iloc[0])
sub.to_csv("submission.csv", index=False)
log("wrote submission.csv")
print(sub.head())
print(json.dumps({"n_rows": int(len(sub)),
                  "n_with_candidates": int(sum(1 for m in mol_info.index
                                               if per_mol.get(m)))}, indent=1))
