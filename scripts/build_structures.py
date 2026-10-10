"""Unique-structure table: normalized_smiles -> tautomer-canonical InChIKey14 + descriptors.

The competition scores on the connectivity block of a *tautomer-canonical* InChIKey,
which is not the same as the `inchikey14` column shipped with train.parquet. This
builds the mapping once so every downstream step scores on the real metric unit.
"""
import os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = "data/cache/structures.parquet"


def work(smiles_chunk):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    te = rdMolStandardize.TautomerEnumerator()
    rows = []
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append((smi, None, None, None, np.nan, np.nan, None))
            continue
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mass = Descriptors.ExactMolWt(mol)
        try:
            can = te.Canonicalize(mol)
        except Exception:
            can = mol
        try:
            key = Chem.MolToInchiKey(can)
        except Exception:
            key = None
        try:
            can_smi = Chem.MolToSmiles(can)
        except Exception:
            can_smi = smi
        rows.append((smi, key[:14] if key else None, key, can_smi, mass,
                     Descriptors.MolLogP(mol), formula))
    return rows


def main():
    smi = pd.read_parquet("data/cache/train_meta.parquet",
                          columns=["normalized_smiles"])["normalized_smiles"]
    uniq = pd.unique(smi.dropna()).tolist()
    print(f"{len(uniq):,} unique SMILES", flush=True)
    chunks = [uniq[i:i + 200] for i in range(0, len(uniq), 200)]
    rows, t0 = [], time.time()
    with Pool(processes=max(os.cpu_count() - 4, 2)) as pool:
        for i, part in enumerate(pool.imap_unordered(work, chunks, chunksize=1)):
            rows.extend(part)
            if (i + 1) % 100 == 0:
                done = len(rows)
                rate = done / (time.time() - t0)
                print(f"{done:,}/{len(uniq):,}  {rate:.0f}/s  "
                      f"eta {(len(uniq)-done)/rate/60:.1f} min", flush=True)
    df = pd.DataFrame(rows, columns=["normalized_smiles", "key14", "inchikey_full",
                                     "canonical_smiles", "exact_mass", "logp", "formula"])
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} rows, {df.key14.notna().sum():,} with keys, "
          f"{df.key14.nunique():,} unique key14")


if __name__ == "__main__":
    main()
