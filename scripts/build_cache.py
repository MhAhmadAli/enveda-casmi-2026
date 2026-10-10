"""Convert train.parquet into a compact memmap peak store + metadata parquet.

Peaks are cleaned (relative-intensity floor, top-K cap) and stored as float32
in two flat arrays addressed by (offset, n_peaks) in the metadata table.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = "enveda-CASMI26-molecule-id-mass-spectra"
OUT = "data/cache"

META_COLS = [
    "ingest_lib", "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
    "ionization_mode", "instrument_type", "adduct", "adduct_orig", "precursor_mz",
    "precursor_error_ppm", "num_peaks", "base_peak_intensity",
    "collision_energy_orig", "collision_energy_orig_units",
]
LIST_COLS = ["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]


def clean_peaks(mz, inten, min_rel, max_peaks, max_mz):
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)
    if mz.size == 0:
        return mz.astype(np.float32), inten.astype(np.float32)
    peak = inten.max()
    if peak <= 0:
        return mz[:0].astype(np.float32), inten[:0].astype(np.float32)
    keep = (inten >= min_rel * peak) & np.isfinite(mz) & (mz > 0)
    if max_mz is not None:
        keep &= mz <= max_mz
    mz, inten = mz[keep], inten[keep]
    if mz.size > max_peaks:
        idx = np.argpartition(inten, -max_peaks)[-max_peaks:]
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz, kind="stable")
    return mz[order].astype(np.float32), (inten[order] / peak).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--min-rel", type=float, default=1e-3)
    ap.add_argument("--max-peaks", type=int, default=512)
    ap.add_argument("--mz-headroom", type=float, default=5.0,
                    help="drop fragments above precursor_mz + headroom")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    src = f"{RAW}/{args.split}.parquet"
    pf = pq.ParquetFile(src)
    avail = set(pf.schema_arrow.names)
    meta_cols = [c for c in META_COLS if c in avail]
    if args.split == "test":
        meta_cols = ["molecule_id", "spectrum_id"] + meta_cols
    cols = meta_cols + [c for c in LIST_COLS if c in avail]

    mz_path = f"{OUT}/{args.split}_mz.f32"
    it_path = f"{OUT}/{args.split}_int.f32"
    fmz, fit = open(mz_path, "wb"), open(it_path, "wb")

    metas, offsets, npks, ce_lo, ce_hi, ce_n = [], [], [], [], [], []
    off = 0
    t0 = time.time()
    for rg in range(pf.metadata.num_row_groups):
        tb = pf.read_row_group(rg, columns=cols)
        df = tb.to_pandas()
        mzs = df.pop("ms2_mzs").to_numpy()
        ints = df.pop("ms2_normalized_intensities").to_numpy()
        ces = df.pop("collision_energy_ev").to_numpy() if "collision_energy_ev" in df else None
        pmz = df["precursor_mz"].to_numpy()
        for i in range(len(df)):
            lim = pmz[i] + args.mz_headroom if np.isfinite(pmz[i]) else None
            m, v = clean_peaks(mzs[i], ints[i], args.min_rel, args.max_peaks, lim)
            fmz.write(m.tobytes()); fit.write(v.tobytes())
            offsets.append(off); npks.append(m.size); off += m.size
            c = np.asarray(ces[i], dtype=np.float64) if ces is not None and ces[i] is not None else np.array([])
            c = c[np.isfinite(c)]
            ce_lo.append(c.min() if c.size else np.nan)
            ce_hi.append(c.max() if c.size else np.nan)
            ce_n.append(c.size)
        metas.append(df)
        print(f"rg {rg+1}/{pf.metadata.num_row_groups} rows={off:,} peaks  {time.time()-t0:.0f}s", flush=True)
    fmz.close(); fit.close()

    meta = pd.concat(metas, ignore_index=True)
    meta["peak_off"] = np.asarray(offsets, dtype=np.int64)
    meta["peak_n"] = np.asarray(npks, dtype=np.int32)
    meta["ce_lo"] = np.asarray(ce_lo, dtype=np.float32)
    meta["ce_hi"] = np.asarray(ce_hi, dtype=np.float32)
    meta["ce_count"] = np.asarray(ce_n, dtype=np.int16)
    meta.to_parquet(f"{OUT}/{args.split}_meta.parquet", index=False)
    json.dump({"split": args.split, "n_rows": int(len(meta)), "n_peaks": int(off),
               "min_rel": args.min_rel, "max_peaks": args.max_peaks,
               "mz_headroom": args.mz_headroom},
              open(f"{OUT}/{args.split}_cache.json", "w"), indent=1)
    print(f"done: {len(meta):,} spectra, {off:,} peaks, "
          f"{os.path.getsize(mz_path)/2**30:.2f} GB per array")


if __name__ == "__main__":
    main()
