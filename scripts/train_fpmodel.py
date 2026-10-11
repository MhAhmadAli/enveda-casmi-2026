"""Train the spectrum -> fingerprint model on the GPU.

The non-negotiable constraint: every structure appearing in a dev set (npx, npbig,
and therefore nphard) is held out of training entirely, by structure and not by
spectrum. A model that has seen the answer's fingerprint during training would score
its candidate perfectly and tell us nothing about a molecule we have never seen,
which is the only case that matters.

Bits are restricted to those with frequency in [min_freq, 1-min_freq] among the
training structures: a bit that is almost always on or almost always off cannot
discriminate between candidates and only adds loss that looks like progress.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.fpmodel import (FPModelConfig, build_model,  # noqa: E402
                           make_batch, predict_fingerprints)
from casmi.retrieval import CandidatePool, reference_table  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-peaks", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--min-freq", type=float, default=0.002)
    ap.add_argument("--val-structures", type=int, default=3000,
                    help="structures held out of training for loss monitoring")
    ap.add_argument("--holdout", default="npx,npbig",
                    help="dev sets whose structures must never be trained on")
    ap.add_argument("--out", default="data/cache/fpmodel.pt")
    ap.add_argument("--limit-spectra", type=int, default=0, help="debug")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- data: spectra joined to their structure's fingerprint ----
    ref = reference_table(CACHE, instrument="all", min_peaks=3)
    meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                           columns=["adduct", "ce_hi"])
    ref = ref.join(meta[["ce_hi"]], on="row")

    banned: set[str] = set()
    for name in args.holdout.split(","):
        p = f"{CACHE}/val_{name}.parquet"
        if os.path.exists(p):
            banned |= set(pd.read_parquet(p, columns=["key14"]).key14.dropna())
    print(f"holding out {len(banned):,} dev structures from training", flush=True)
    ref = ref[~ref.key14.isin(banned)].reset_index(drop=True)

    pool = CandidatePool.load(CACHE, with_fp=True)
    row_of_key = {}
    for r, k in enumerate(pool.key14):
        if k not in row_of_key:
            row_of_key[k] = r
    ref["pool_row"] = ref.key14.map(row_of_key)
    ref = ref[ref.pool_row.notna()].reset_index(drop=True)
    ref["pool_row"] = ref.pool_row.astype(np.int64)
    if args.limit_spectra:
        ref = ref.sample(n=min(args.limit_spectra, len(ref)),
                         random_state=0).reset_index(drop=True)
    print(f"{len(ref):,} training spectra over {ref.key14.nunique():,} structures "
          f"({time.time()-t0:.0f}s)", flush=True)

    # informative bits, measured over the training structures only
    uniq_rows = np.sort(ref.pool_row.unique())
    bits_uniq = np.unpackbits(pool.packed_fp[uniq_rows], axis=1)
    freq = bits_uniq.mean(axis=0)
    keep_bits = np.nonzero((freq >= args.min_freq) & (freq <= 1 - args.min_freq))[0]
    prior = freq[keep_bits]
    print(f"predicting {len(keep_bits)} of {bits_uniq.shape[1]} bits "
          f"(frequency within [{args.min_freq}, {1-args.min_freq}])", flush=True)
    del bits_uniq

    cfg = FPModelConfig(n_bits=len(keep_bits), max_peaks=args.max_peaks,
                        d_model=args.d_model, n_layers=args.layers)

    adducts = sorted(ref.adduct.astype(str).unique())
    add_id = {a: min(i, cfg.n_adducts - 1) for i, a in enumerate(adducts)}
    ref["adduct_id"] = ref.adduct.astype(str).map(add_id).astype(np.int64)
    ref["ce"] = ref.ce_hi.fillna(0.0).to_numpy(np.float32)

    # held-out structures for loss monitoring (still disjoint from the dev sets)
    rng = np.random.default_rng(0)
    structs = ref.key14.unique()
    val_structs = set(rng.permutation(structs)[: args.val_structures])
    is_val = ref.key14.isin(val_structs).to_numpy()
    tr_idx = np.nonzero(~is_val)[0]
    va_idx = np.nonzero(is_val)[0]
    print(f"train {len(tr_idx):,} spectra | monitor {len(va_idx):,} spectra",
          flush=True)

    store = PeakStore.load("train", CACHE)
    pool_fp = pool.packed_fp
    pool_rows = ref.pool_row.to_numpy()
    prec = ref.precursor_mz.to_numpy(np.float32)
    add_arr = ref.adduct_id.to_numpy()
    ce_arr = ref.ce.to_numpy(np.float32)
    rows_arr = ref.row.to_numpy()

    def targets(idx):
        return np.unpackbits(pool_fp[pool_rows[idx]], axis=1)[:, keep_bits]

    model = build_model(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par/1e6:.1f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * max(len(tr_idx) // args.batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()

    def run_eval():
        model.eval()
        tot, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, min(len(va_idx), 20000), args.batch):
                idx = va_idx[s:s + args.batch]
                mz, it, pad, pm, ad, c = make_batch(
                    store, rows_arr[idx], prec[idx], add_arr[idx], ce_arr[idx], cfg)
                y = torch.from_numpy(targets(idx).astype(np.float32)).to(dev)
                t = [torch.from_numpy(x).to(dev) for x in (mz, it, pad, pm, ad, c)]
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(dev == "cuda")):
                    out = model(t[0], t[1], t[2], t[3], t[4], t[5])
                tot += lossf(out.float(), y).item()
                nb += 1
        model.train()
        return tot / max(nb, 1)

    step = 0
    for ep in range(args.epochs):
        perm = rng.permutation(tr_idx)
        run_loss, seen = 0.0, 0
        for s in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[s:s + args.batch]
            mz, it, pad, pm, ad, c = make_batch(
                store, rows_arr[idx], prec[idx], add_arr[idx], ce_arr[idx], cfg)
            y = torch.from_numpy(targets(idx).astype(np.float32)).to(dev)
            t = [torch.from_numpy(x).to(dev) for x in (mz, it, pad, pm, ad, c)]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(dev == "cuda")):
                out = model(t[0], t[1], t[2], t[3], t[4], t[5])
                loss = lossf(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            seen += 1
            step += 1
            if seen % 500 == 0:
                print(f"  ep{ep} step {seen} loss {run_loss/seen:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        vl = run_eval()
        print(f"epoch {ep}: train {run_loss/max(seen,1):.4f}  "
              f"holdout {vl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"cfg": cfg.__dict__, "keep_bits": keep_bits, "prior": prior,
                    "adduct_id": add_id, "state": model.state_dict(),
                    "epoch": ep, "holdout_loss": vl}, args.out)
    print(f"saved {args.out}")
    json.dump({"n_params": int(n_par), "n_bits": int(len(keep_bits)),
               "train_spectra": int(len(tr_idx)), "epochs": args.epochs,
               "holdout_loss": vl, "held_out_dev_structures": len(banned)},
              open("reports/fpmodel_train.json", "w"), indent=1)


if __name__ == "__main__":
    main()
