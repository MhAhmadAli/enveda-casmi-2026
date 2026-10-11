"""Assemble the Kaggle dataset payload from source, so it cannot drift.

The notebook imports the real `casmi` package rather than carrying its own copy of
the feature code: a divergence between a training feature and its inference twin is
the classic way a ranker degrades silently, and shipping one implementation is the
only robust guard. That only holds if the shipped copy is rebuilt from source, which
is what this does.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

SRC = "src/casmi"
DEST = "data/kaggle_dataset"
PKG_DEST = os.path.join(DEST, "casmi")
REQUIRED = ["pool_index.parquet", "pool_fp_packed.npy", "structures.parquet",
            "ranker.joblib"]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    os.makedirs(PKG_DEST, exist_ok=True)
    for f in os.listdir(PKG_DEST):
        if f.endswith((".py", ".pyc")):
            os.remove(os.path.join(PKG_DEST, f))
    shipped = []
    for f in sorted(os.listdir(SRC)):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(SRC, f), os.path.join(PKG_DEST, f))
            shipped.append(f)
    print(f"shipped casmi modules: {', '.join(shipped)}")

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DEST, f))]
    if missing:
        raise SystemExit(f"missing payload files: {missing}")

    manifest = {}
    total = 0
    for root, _, files in os.walk(DEST):
        for f in sorted(files):
            if f.endswith((".pyc", ".json")) and f != "dataset-metadata.json":
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, DEST).replace("\\", "/")
            size = os.path.getsize(p)
            total += size
            manifest[rel] = {"bytes": size, "sha256_16": sha(p)}
    json.dump(manifest, open(os.path.join(DEST, "MANIFEST.json"), "w"), indent=1)
    print(f"{len(manifest)} files, {total/2**20:.1f} MB total")
    for k, v in manifest.items():
        print(f"  {k:40s} {v['bytes']/2**20:8.2f} MB  {v['sha256_16']}")


if __name__ == "__main__":
    main()
