# %% [markdown]
# # CASMI 2026 — two-channel evidence, combined by a learned ranker
#
# The visible `test.parquet` is a placeholder built from training rows; Kaggle
# re-runs this notebook against a hidden test set of natural products at scoring
# time. Nothing here keys off the visible file — molecule ids, mass range, adducts
# and instrument are all read at runtime.
#
# ### Method
#
# Two channels, each with the similarity measure it needs, combined per candidate by
# a gradient-boosted ranker:
#
# | channel | what it finds | similarity |
# |---|---|---|
# | 1 — direct | the molecule itself, if its structure has library spectra | spectral entropy on raw peaks, over library spectra whose **neutral mass matches** |
# | 2 — analog | structural relatives of a novel molecule | binned cosine, direct + neutral-loss domains, propagated by `tanimoto x cos^4` |
# | 3 — fingerprint | the structure itself, with no library involved | set transformer over peaks predicts 2,022 Morgan bits; candidates scored by naive-Bayes log-likelihood against the bit prior |
#
# Measured separately on held-out natural products (structure-disjoint):
#
# | scorer | Class 1 | Class 2 | Class 2, no close relative |
# |---|---|---|---|
# | channel 1 alone | 0.653 | 0.111 | 0.127 |
# | channel 2 alone | 0.721 | 0.477 | 0.361 |
# | **learned ranker** | **0.748** | **0.522** | **0.413** |
#
# Why a *learned* combination: every fixed rule tried was worse. Rescoring the
# analog channel with entropy similarity halved Class 2 (0.475 → 0.226); hard
# confidence gating came out below either channel alone, because Channel-1
# similarity separates the two classes only weakly (median 0.821 vs 0.468, with
# 10.4% of novel molecules above the Class-1 median). The ranker is trained on both
# regimes at several assumed Class-1 fractions and the predictions averaged, since
# the hidden test's proportion is unknown.
#
# The feature code is imported from the shipped `casmi` package — the same module
# that produced the training features — so an inference/training feature skew
# cannot arise.

# %%
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def find_dir(marker):
    """Kaggle mounts are nested (/kaggle/input/datasets/<owner>/<slug>, and a
    similar path for competition data), never the slug-only path the docs imply,
    so locate both by a file we know they contain."""
    seen = []
    for root in ("/kaggle/input", "data", "."):
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
# marker is the ranker, not pool_index.parquet: the local dev cache also
# holds a pool_index.parquet, and only the shipped payload has the model
POOL = require_dir("ranker.joblib", "candidate pool + ranker dataset")
def locate_package():
    """Return a sys.path entry from which `import casmi` works.

    Two layouts are possible: the dataset may expose casmi/ as a real directory,
    or the CLI may have stored it as an archive. For the archive case the files are
    extracted to a writable dir -- numba compiles kernels in casmi.entropy and
    wants a real path, so importing straight from the zip is not relied on.
    """
    found, seen = find_dir("channels.py")
    if found is not None:
        return os.path.dirname(found), f"directory {found}"

    import zipfile
    for root in ("/kaggle/input", "data", "."):
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".zip"):
                    continue
                zp = os.path.join(dirpath, fn)
                try:
                    with zipfile.ZipFile(zp) as z:
                        names = z.namelist()
                        if not any(n.endswith("channels.py") for n in names):
                            continue
                        out = "/tmp/casmi_pkg" if os.path.isdir("/tmp") else "casmi_pkg"
                        os.makedirs(out, exist_ok=True)
                        z.extractall(out)
                        inner, _ = find_dir_under(out, "channels.py")
                        if inner:
                            return os.path.dirname(inner), f"archive {zp}"
                except zipfile.BadZipFile:
                    continue
    print("could not locate the casmi package; searched:")
    for d in seen[:300]:
        try:
            print("   ", d, sorted(os.listdir(d))[:8])
        except OSError:
            pass
    raise FileNotFoundError("shipped casmi package")


def find_dir_under(root, marker):
    for dirpath, _, filenames in os.walk(root):
        if marker in filenames:
            return os.path.join(dirpath, marker), dirpath
    return None, None


PKG_PATH, PKG_HOW = locate_package()
sys.path.insert(0, PKG_PATH)
PKG = PKG_HOW
log(f"competition data: {IN}")
log(f"candidate pool:   {POOL}")
log(f"casmi package:    {PKG}")

from casmi.channels import ChannelConfig, SpectraSet, compute_features  # noqa: E402
from casmi.retrieval import CandidatePool  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CFG = ChannelConfig()           # ppm 8.5, tol 0.005, min_rel 0.005, sim_power 4
MIN_REL_CLEAN = 1e-3            # peak-store floor, looser than the channel floor
MAX_PEAKS_CLEAN = 512
HEADROOM = 5.0
MIN_PEAKS_REF = 3
N_OUT = 25

# %% [markdown]
# ## Stream the spectra into flat arrays
#
# `train.parquet` is 3 GB of nested lists, so it is read row group by row group and
# reduced to the flat float32 layout `PeakStore` expects.

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
    keep = (it >= MIN_REL_CLEAN * peak) & np.isfinite(mz) & (mz > 0)
    if np.isfinite(pmz):
        keep &= mz <= pmz + HEADROOM
    mz, it = mz[keep], it[keep]
    if mz.size > MAX_PEAKS_CLEAN:
        sel = np.argpartition(it, -MAX_PEAKS_CLEAN)[-MAX_PEAKS_CLEAN:]
        mz, it = mz[sel], it[sel]
    order = np.argsort(mz, kind="stable")
    return mz[order].astype(np.float32), (it[order] / peak).astype(np.float32)


def load_split(path, meta_cols):
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    cols = list(dict.fromkeys([c for c in meta_cols if c in have]
                              + ["ms2_mzs", "ms2_normalized_intensities",
                                 "precursor_mz"]))
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
    store = PeakStore(mz=np.concatenate(mzs_out), inten=np.concatenate(its_out),
                      off=np.asarray(offs, np.int64), n=np.asarray(ns, np.int32))
    return meta, store


# %%
pool = CandidatePool.load(POOL, with_fp=True)
pool_row_of_key = {}
for _r, _k in enumerate(pool.key14):
    if _k not in pool_row_of_key:
        pool_row_of_key[_k] = _r
log(f"pool: {len(pool.df):,} structures")

test_meta, test_store = load_split(
    f"{IN}/test.parquet",
    ["molecule_id", "spectrum_id", "adduct", "instrument_type", "precursor_mz",
     "collision_energy_ev"])
test_meta_raw = test_meta
log(f"test: {len(test_meta)} spectra, {test_meta.molecule_id.nunique()} molecules")

train_meta, train_store = load_split(
    f"{IN}/train.parquet",
    ["normalized_smiles", "adduct", "instrument_type", "precursor_mz"])
log(f"library: {len(train_meta):,} spectra")

# %% [markdown]
# ## Map library spectra to tautomer-canonical structure keys
#
# The shipped `structures.parquet` carries the mapping, computed once offline the way
# the official scorer does it. The `inchikey14` column in the competition data is
# *not* tautomer-canonical (274,195 canonical keys vs 275,810 shipped), so using it
# would mis-merge structures.

# %%
structs = pd.read_parquet(f"{POOL}/structures.parquet",
                          columns=["normalized_smiles", "key14"])
train_meta = train_meta.merge(structs, on="normalized_smiles", how="left")
ok = (train_meta.key14.notna().to_numpy()
      & (train_store.n[: len(train_meta)] >= MIN_PEAKS_REF))
lib_idx = np.nonzero(ok)[0]
log(f"library usable: {len(lib_idx):,} spectra / "
    f"{train_meta.key14.iloc[lib_idx].nunique():,} structures")

queries = SpectraSet(
    store=test_store, rows=np.arange(len(test_meta)),
    precursor_mz=test_meta.precursor_mz.to_numpy(),
    adduct=(test_meta.adduct.to_numpy() if "adduct" in test_meta
            else np.array(["[M+H]+"] * len(test_meta), dtype=object)),
    group=test_meta.molecule_id.to_numpy())
library = SpectraSet(
    store=train_store, rows=lib_idx,
    precursor_mz=train_meta.precursor_mz.to_numpy()[lib_idx],
    adduct=train_meta.adduct.to_numpy()[lib_idx],
    key14=train_meta.key14.to_numpy()[lib_idx])

# %% [markdown]
# ## Features, then the ranker

# %%
feat = compute_features(queries, library, pool, CFG, truth=None,
                        regime="test", log=log)
del train_store          # the library is finished with; queries.store is still needed
gc.collect()

# %% [markdown]
# ## Channel 3: predicted fingerprints
#
# Independent of the library entirely — the model reads the spectrum and predicts
# which substructures are present, so a molecule with no close library relative is
# no harder for it. A molecule's several collision energies are averaged in
# probability space, since different energies expose different substructures.

# %%
import torch  # noqa: E402

from casmi.fpmodel import (FPModelConfig, build_model,  # noqa: E402
                           candidate_scores, predict_fingerprints)

fp_dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{POOL}/fpmodel.pt", map_location=fp_dev, weights_only=False)
fp_cfg = FPModelConfig(**ck["cfg"])
fp_model = build_model(fp_cfg).to(fp_dev)
fp_model.load_state_dict(ck["state"])
keep_bits, bit_prior, add_map = ck["keep_bits"], ck["prior"], ck["adduct_id"]
log(f"fp model on {fp_dev}: epoch {ck.get('epoch')}, {fp_cfg.n_bits} bits")

# collision energy: the column is a list per spectrum, so take its maximum, and
# fall back to 0 when absent rather than failing on a hidden-test schema variant
if "collision_energy_ev" in test_meta_raw.columns:
    ce_q = np.array([
        (max(x) if x is not None and len(x) else 0.0)
        for x in test_meta_raw.collision_energy_ev], dtype=np.float32)
else:
    ce_q = np.zeros(len(test_meta), np.float32)
aid_q = np.array([add_map.get(str(a), 0) for a in queries.adduct], np.int64)

pred = predict_fingerprints(fp_model, queries.store, queries.rows,
                            queries.precursor_mz.astype(np.float32),
                            aid_q, ce_q, fp_cfg, fp_dev)
log(f"predicted fingerprints for {len(pred)} spectra")

pred_by_mol = {}
for m, g in pd.DataFrame({"m": queries.group,
                          "i": np.arange(len(pred))}).groupby("m"):
    pred_by_mol[m] = pred[g.i.to_numpy()].mean(axis=0)

# attach fp features to the candidate rows the other channels produced
fp_rows = []
for mol, g in feat.groupby("molecule_id", sort=False):
    rows_pool = np.array([pool_row_of_key[k] for k in g.key14], dtype=np.int64)
    bits = np.unpackbits(pool.packed_fp[rows_pool], axis=1)[:, keep_bits]
    sc = candidate_scores(pred_by_mol[mol], bits, bit_prior)
    order = np.argsort(-sc)
    rank = np.empty(len(sc), np.int64)
    rank[order] = np.arange(1, len(sc) + 1)
    fp_rows.append(pd.DataFrame({
        "molecule_id": mol, "key14": g.key14.to_numpy(),
        "fp_score": sc.astype(np.float32),
        "fp_margin": (sc - sc.max()).astype(np.float32),
        "fp_rank": rank}))
fp_df = pd.concat(fp_rows, ignore_index=True)
before = len(feat)
feat = feat.merge(fp_df, on=["molecule_id", "key14"], how="left")
assert len(feat) == before, "fp merge changed the row count"
log(f"fp channel merged: {feat.fp_score.notna().mean():.1%} of rows covered")
feat["fp_score"] = feat.fp_score.fillna(feat.fp_score.min())
feat["fp_margin"] = feat.fp_margin.fillna(feat.fp_margin.min())
feat["fp_rank"] = feat.fp_rank.fillna(feat.n_candidates)

# %%
import joblib  # noqa: E402

bundle = joblib.load(f"{POOL}/ranker.joblib")
models, cols = bundle["models"], bundle["features"]
log(f"ranker: {len(models)} models over {len(cols)} features, trained on "
    f"{bundle.get('n_train_molecules', '?')} molecules")

X = feat[cols].to_numpy(np.float32)
score = np.zeros(len(X))
for mdl in models:
    score += mdl.predict_proba(X)[:, 1]
score /= len(models)
feat["score"] = score
log("scored")

# %% [markdown]
# ## Ranked SMILES, with fallbacks that never leave a row empty

# %%
smi_of_key = {}
for k, s in zip(pool.key14, pool.smiles):
    if k not in smi_of_key:
        smi_of_key[k] = s

feat = feat.sort_values(["molecule_id", "score"], ascending=[True, False])
rows_out = []
for mol, g in feat.groupby("molecule_id", sort=False):
    seen, smis = set(), []
    for k in g.key14:
        if k in seen or k not in smi_of_key:
            continue
        seen.add(k)
        smis.append(smi_of_key[k])
        if len(smis) >= N_OUT:
            break
    rows_out.append((mol, ";".join(smis)))
sub = pd.DataFrame(rows_out, columns=["molecule_id", "smiles"])

sample = pd.read_csv(f"{IN}/sample_submission.csv")
sub = sample[["molecule_id"]].merge(sub, on="molecule_id", how="left")
missing = int(sub.smiles.isna().sum()) + int((sub.smiles == "").sum())
if missing:
    log(f"WARNING: {missing} molecules without predictions; filling with pool head")
    filler = ";".join(pd.unique(pool.smiles[:N_OUT]))
    sub["smiles"] = sub.smiles.replace("", np.nan).fillna(filler)
sub.to_csv("submission.csv", index=False)

n = sub.smiles.str.count(";") + 1
log(f"wrote submission.csv: {len(sub)} rows, candidates/row min {n.min()} "
    f"median {n.median():.0f} max {n.max()}")
print(json.dumps({"rows": int(len(sub)), "molecules_scored": int(feat.molecule_id.nunique()),
                  "rows_filled_by_fallback": missing}, indent=1))
print(sub.head())
