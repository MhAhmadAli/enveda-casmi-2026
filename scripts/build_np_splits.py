"""Validation splits for the real task: identify natural products from timsTOF spectra.

The hidden test is natural products measured on Enveda's instrument. Two dev sets,
in increasing order of fidelity to that:

  npx   - the 250 `enveda-np-examples` structures. timsTOF, eV collision energies,
          94% COCONUT natural products: the same distribution as the hidden test.
          Small, so noisy, but it is the only truly matched sample available.
  npbig - natural-product structures drawn from the public libraries (gnps, riken,
          mona, massbank, msdial, pluskal, spectraverse). Thousands of molecules, so
          the MRR estimate is tight, but the queries are Orbitrap/QTOF rather than
          timsTOF. Use for model selection, then confirm on `npx`.

Both are emitted structure-disjoint-ready: the reference set must exclude every
spectrum of a held-out structure, which the evaluator enforces by key14.

Queries are grouped into pseudo-molecules the way test is: several spectra of one
structure at different collision energies, same adduct, scored once.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE = "data/cache"
NP_LIBS = ["gnps", "riken", "mona", "massbank", "msdial", "pluskal_ms2",
           "spectraverse", "drug_plus", "masaryk"]


def load() -> pd.DataFrame:
    meta = pd.read_parquet(
        f"{CACHE}/train_meta.parquet",
        columns=["normalized_smiles", "ingest_lib", "instrument_type", "adduct",
                 "precursor_mz", "peak_n", "ce_hi"])
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    st = pd.read_parquet(f"{CACHE}/structures.parquet",
                         columns=["normalized_smiles", "key14", "exact_mass"])
    meta = meta.merge(st, on="normalized_smiles", how="left")
    co = set(pd.read_parquet("data/external/coconut.parquet",
                             columns=["key14"]).key14.dropna())
    meta["is_np"] = meta.key14.isin(co)
    return meta


def group_queries(df: pd.DataFrame, rng, max_per_mol: int, prefix: str) -> pd.DataFrame:
    """One pseudo-molecule per (structure, adduct), keeping up to max_per_mol spectra."""
    rows = []
    for i, ((key, add), g) in enumerate(df.groupby(["key14", "adduct"], sort=False)):
        take = g.row.to_numpy()
        if len(take) > max_per_mol:
            take = rng.permutation(take)[:max_per_mol]
        mid = f"{prefix}_{i:05d}"
        for r in take:
            rows.append((mid, key, add, int(r)))
    return pd.DataFrame(rows, columns=["molecule_id", "key14", "adduct", "row"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--max-per-mol", type=int, default=6)
    ap.add_argument("--npbig-mols", type=int, default=1500)
    ap.add_argument("--min-peaks", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = load()
    usable = meta[(meta.peak_n >= args.min_peaks) & meta.key14.notna()
                  & meta.exact_mass.notna()]

    # --- npx: the matched timsTOF natural-product sample -------------------
    npx_src = usable[usable.ingest_lib == "enveda-np-examples"]
    npx = group_queries(npx_src, rng, args.max_per_mol, "npx")
    npx.to_parquet(f"{CACHE}/val_npx.parquet", index=False)
    print(f"npx:   {npx.molecule_id.nunique()} molecules, {len(npx)} spectra, "
          f"{npx.key14.nunique()} structures "
          f"({len(npx)/max(npx.molecule_id.nunique(),1):.2f} spectra/molecule)")

    # --- npbig: natural products from the public libraries -----------------
    pub = usable[usable.ingest_lib.isin(NP_LIBS) & usable.is_np]
    # exclude anything already in npx so the two dev sets stay independent
    pub = pub[~pub.key14.isin(set(npx.key14))]
    keys = pub.key14.unique()
    keys = rng.permutation(keys)[: args.npbig_mols]
    big = group_queries(pub[pub.key14.isin(set(keys))], rng, args.max_per_mol, "npb")
    big.to_parquet(f"{CACHE}/val_npbig.parquet", index=False)
    print(f"npbig: {big.molecule_id.nunique()} molecules, {len(big)} spectra, "
          f"{big.key14.nunique()} structures")

    for name, df in [("npx", npx), ("npbig", big)]:
        mm = meta.set_index("row").exact_mass.reindex(df.row)
        inst = meta.set_index("row").instrument_type.reindex(df.row)
        print(f"\n{name}: mass median {mm.median():.0f} "
              f"({mm.quantile(.1):.0f}-{mm.quantile(.9):.0f}), "
              f"top instruments {inst.value_counts().head(3).to_dict()}")
        print(f"  adducts {df.groupby('molecule_id').adduct.first().value_counts(normalize=True).head(4).round(3).to_dict()}")

    json.dump({"seed": args.seed, "max_per_mol": args.max_per_mol,
               "npx_molecules": int(npx.molecule_id.nunique()),
               "npbig_molecules": int(big.molecule_id.nunique()),
               "np_libs": NP_LIBS},
              open(f"{CACHE}/val_np_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
