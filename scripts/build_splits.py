"""Build test-mimicking validation splits from the timsTOF (enveda-180) pool.

Test is 100% timsTOF: 400 molecules, 1213 spectra, 87% [M+H]+ / 9% [M-H]-, and
neutral masses confined to roughly 246-440 Da (mean 326). Train timsTOF spans
132-1154 Da, so an unstratified sample would not predict leaderboard behaviour.
We therefore mirror the test joint distribution over (adduct, 25 Da mass bin)
and the spectra-per-molecule distribution.

Two reference-set regimes are supported downstream from the same query files:

  disjoint : every spectrum of a held-out structure is banned from the reference.
             The honest "unknown molecule" setting the challenge is about.
  inlib    : only the query spectra themselves are withheld; sibling spectra of
             the same structure remain. Library-lookup ceiling.

The gap between the two says how much of a score is mere library lookup.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.chem import precursor_to_neutral_mass  # noqa: E402

CACHE = "data/cache"
MASS_BIN = 25.0


def load_pool() -> pd.DataFrame:
    meta = pd.read_parquet(
        f"{CACHE}/train_meta.parquet",
        columns=["normalized_smiles", "instrument_type", "adduct", "peak_n"],
    )
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    structs = pd.read_parquet(
        f"{CACHE}/structures.parquet", columns=["normalized_smiles", "key14", "exact_mass"]
    )
    meta = meta.merge(structs, on="normalized_smiles", how="left")
    pool = meta[
        (meta.instrument_type == "timsTOF")
        & meta.key14.notna()
        & meta.exact_mass.notna()
        & (meta.peak_n >= 5)
    ]
    return pool[["row", "key14", "adduct", "exact_mass"]].reset_index(drop=True)


def test_profile():
    test = pd.read_parquet(
        f"{CACHE}/test_meta.parquet", columns=["molecule_id", "adduct", "precursor_mz"]
    )
    n_per_mol = test.groupby("molecule_id").size().to_numpy()
    tm = test.groupby("molecule_id").agg(
        pmz=("precursor_mz", "first"), adduct=("adduct", "first")
    )
    tm["neutral"] = [precursor_to_neutral_mass(p, a) for p, a in zip(tm.pmz, tm.adduct)]
    tm["mbin"] = (tm.neutral // MASS_BIN).astype(int)
    return tm, n_per_mol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mols", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    tm, n_per_mol = test_profile()
    print("test adduct mix:", tm.adduct.value_counts(normalize=True).round(3).to_dict())
    print(
        f"test neutral mass {tm.neutral.min():.0f}-{tm.neutral.max():.0f}"
        f" mean {tm.neutral.mean():.0f}"
    )

    pool = load_pool()
    print(f"timsTOF pool: {len(pool):,} spectra, {pool.key14.nunique():,} structures")

    pool["mbin"] = (pool.exact_mass // MASS_BIN).astype(int)
    # rows available per (structure, adduct), and structures available per cell
    rows_by_mol_add = pool.groupby(["key14", "adduct"], sort=False).row.apply(
        lambda s: s.to_numpy()
    )
    mols_by_cell = pool.groupby(["adduct", "mbin"], sort=False).key14.unique()

    cells = tm.groupby(["adduct", "mbin"], sort=False).size()
    cells = cells[cells.index.isin(mols_by_cell.index)]
    cell_p = (cells / cells.sum()).sort_index()
    covered = cells.sum() / len(tm)
    print(f"{len(cell_p)} (adduct, mass-bin) cells cover {covered:.1%} of test molecules")

    # pre-shuffled per-cell structure queues so sampling without replacement is O(1)
    queues = {c: list(rng.permutation(mols_by_cell.loc[c])) for c in cell_p.index}

    taken_global: set[str] = set()
    summary = []
    for fold in range(args.folds):
        draws = rng.choice(len(cell_p), size=args.n_mols * 6, p=cell_p.to_numpy())
        rows, mol_i = [], 0
        for d in draws:
            if mol_i >= args.n_mols:
                break
            cell = cell_p.index[d]
            q = queues[cell]
            key = None
            while q:
                cand = q.pop()
                if cand not in taken_global:
                    key = cand
                    break
            if key is None:
                continue
            taken_global.add(key)
            avail = rows_by_mol_add.loc[(key, cell[0])]
            n = int(rng.choice(n_per_mol))
            take = rng.permutation(avail)[: min(n, len(avail))]
            mid = f"v{fold}_{mol_i:04d}"
            for r in take:
                rows.append((mid, key, cell[0], int(r)))
            mol_i += 1

        q = pd.DataFrame(rows, columns=["molecule_id", "key14", "adduct", "row"])
        q = q.merge(pool[["row", "exact_mass"]], on="row", how="left")
        q.to_parquet(f"{CACHE}/val_fold{fold}.parquet", index=False)
        per_mol = q.groupby("molecule_id").exact_mass.first()
        print(
            f"fold {fold}: {q.molecule_id.nunique()} molecules, {len(q)} queries, "
            f"{len(q) / max(q.molecule_id.nunique(), 1):.2f} spec/mol, "
            f"mass mean {per_mol.mean():.0f} ({per_mol.min():.0f}-{per_mol.max():.0f})"
        )
        summary.append({"fold": fold, "n_mols": int(q.molecule_id.nunique()),
                        "n_queries": int(len(q))})

    json.dump(
        {"seed": args.seed, "n_mols": args.n_mols, "folds": args.folds,
         "mass_bin": MASS_BIN, "cell_coverage": float(covered), "folds_summary": summary},
        open(f"{CACHE}/val_config.json", "w"), indent=1,
    )


if __name__ == "__main__":
    main()
