"""Replay the fork's post-ranker stages offline from a validation dump and sweep their weights.

Reads `val/dump.pkl` written by the kernels from scripts/build_val_kernel.py, then
re-applies, exactly as the fork does,
  1. the pool popularity re-rank of the main engine's top-60 (cell 6, POOLPOP_MU), and
  2. the ICEBERG / GLACIER same-formula re-rank (cell 7, ICE_LAM / GL_LAM),
and scores the main list's top 25. The first rows replay the kernel's own config and
must reproduce its `pop_main` / `sim_main` scores; that checks the replay and the local
RDKit (the pinned 2026.03.3 has no Intel-Mac wheel).

Deltas are paired against the shipped config, with a bootstrap 95% interval over
molecules.

usage: python scripts/replay_val.py reports/kernel_val_disjoint/val [more dump dirs]
"""
from __future__ import annotations

import collections
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "reports/model_provenance/iceberg"), str(ROOT / "reports/model_provenance/glacier")]
import fuse as ice_fuse  # noqa: E402
import gl_fuse  # noqa: E402

RDLogger.DisableLog("rdApp.*")
_TAUT = rdMolStandardize.TautomerEnumerator()
_key_cache: dict[str, str | None] = {}
_comp_cache: dict[str, tuple | None] = {}

POP_GRID = (0.0, 0.15, 0.25, 0.4, 0.6, 1.0)
SIM_GRID = ((1.0, 1.0), (0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0))


def metric_key(smiles: str) -> str | None:
    """Official metric identity: strip stereo, tautomer-canonicalise, InChIKey14."""
    if smiles not in _key_cache:
        key = None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.RemoveStereochemistry(mol)
                key = Chem.MolToInchiKey(_TAUT.Canonicalize(mol))[:14]
        except (ValueError, RuntimeError, TypeError):
            key = None
        _key_cache[smiles] = key
    return _key_cache[smiles]


def composition(smiles: str) -> tuple | None:
    if smiles not in _comp_cache:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        _comp_cache[smiles] = None if mol is None else tuple(sorted(collections.Counter(
            a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "H").items()))
    return _comp_cache[smiles]


def rank_of(smiles: list[str], truth_smiles: str, truth_key: str, k: int = 25) -> int:
    comp = composition(truth_smiles)
    for r, s in enumerate(smiles[:k], 1):
        if composition(s) == comp and metric_key(s) == truth_key:
            return r
    return 0


def rr(ranks: np.ndarray) -> np.ndarray:
    return np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)


def pop_rerank(raw: dict, mu: float) -> tuple[list, list, list, list]:
    """Cell 6: z(score) + mu * pop over pool candidates; generated candidates keep their slots."""
    smis, keys, scs, forms, pids = raw["smiles"], raw["keys"], raw["score"], raw["formula"], raw["pid"]
    if not (mu > 0 and len(scs) > 1):
        return list(smis), list(keys), list(scs), list(forms)
    s = np.asarray(scs, np.float64)
    z = (s - s.mean()) / (s.std() + 1e-9)
    pid = np.asarray(pids)
    gen = pid < 0
    pop = np.nan_to_num(np.asarray(raw["pop"], np.float64))
    f = z + mu * np.where(gen, 0.0, pop)
    pool_i = np.where(~gen)[0]
    o = np.empty(len(pid), np.int64)
    o[gen] = np.where(gen)[0]
    o[~gen] = pool_i[np.argsort(-f[pool_i], kind="stable")]
    return ([smis[i] for i in o], [keys[i] for i in o], [float(f[i]) for i in o],
            [forms[i] + ("|gen" if gen[i] else "") for i in o])


def sim_rerank(smis, keys, scs, forms, ice: dict, gl: dict, ice_lam: float, gl_lam: float, top_n: int) -> list:
    """Cell 7: ICEBERG re-rank, then GLACIER (with ICEBERG) from the same base order when GLACIER scored."""
    if not (ice and any(v is not None for v in ice.values())):
        return list(smis)
    order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ice_lam, top_n=top_n)
    if gl:
        order = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ice_lam, gl_lam], top_n=top_n)
    return [smis[i] for i in order]


def bootstrap(delta: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def replay(dump_dir: Path) -> list[dict]:
    d = pickle.load(open(dump_dir / "dump.pkl", "rb"))
    truth = {r["molecule_id"]: r for r in json.load(open(dump_dir / "truth.json"))}
    kernel = json.load(open(dump_dir / "results.json"))
    cfg = d["cfg"]
    mids = sorted(truth)
    gl_all = {m: g for m, g in (d["gl"] or {}).items()
              if isinstance(g, dict) and any(v is not None for v in g.values())}

    def score(lists: dict) -> np.ndarray:
        return np.array([rank_of(lists.get(m, []), truth[m]["smiles"], truth[m]["key"]) for m in mids])

    rows, ref = [], None
    for mu in POP_GRID:
        pop_lists = {m: pop_rerank(d["raw"][m], mu) for m in mids if m in d["raw"]}
        for ice_lam, gl_lam in SIM_GRID:
            sim = {m: sim_rerank(*pop_lists[m], (d["ice"] or {}).get(m, {}), gl_all.get(m, {}),
                                 ice_lam, gl_lam, cfg["TOPN"]) for m in pop_lists}
            ranks = score(sim)
            row = dict(dump=dump_dir.parent.name, pop_mu=mu, ice_lam=ice_lam, gl_lam=gl_lam,
                       mrr25=float(rr(ranks).mean()), top1=float(np.mean(ranks == 1)),
                       top25=float(np.mean(ranks > 0)), _rr=rr(ranks))
            if math.isclose(mu, cfg["POOLPOP_MU"]) and (ice_lam, gl_lam) == (cfg["ICE_LAM"], cfg["GL_LAM"]):
                ref = row
            rows.append(row)
        if math.isclose(mu, cfg["POOLPOP_MU"]):
            pop_mrr = float(rr(score({m: v[0] for m, v in pop_lists.items()})).mean())
            print(f"replay check {dump_dir.parent.name}: pop_main {pop_mrr:.4f} vs kernel "
                  f"{kernel['pop_main']['mrr25']:.4f}")
    assert ref is not None, "shipped config missing from the grid"
    print(f"replay check {dump_dir.parent.name}: sim_main {ref['mrr25']:.4f} vs kernel "
          f"{kernel['sim_main']['mrr25']:.4f}")
    for row in rows:
        lo, hi = bootstrap(row["_rr"] - ref["_rr"])
        row.update(delta=row["mrr25"] - ref["mrr25"], ci95=(round(lo, 4), round(hi, 4)))
    return rows


def main() -> None:
    out = []
    for arg in sys.argv[1:]:
        rows = replay(Path(arg))
        print(f"\n{arg}  (n={len(rows[0]['_rr'])}; delta vs shipped pop_mu/ice/gl, bootstrap 95% CI)")
        print(f"{'pop_mu':>6} {'ice':>4} {'gl':>4} {'mrr25':>7} {'top1':>6} {'top25':>6} {'delta':>7}  ci95")
        for r in sorted(rows, key=lambda r: -r["mrr25"]):
            print(f"{r['pop_mu']:6.2f} {r['ice_lam']:4.1f} {r['gl_lam']:4.1f} {r['mrr25']:7.4f} "
                  f"{r['top1']:6.3f} {r['top25']:6.3f} {r['delta']:+7.4f}  {r['ci95']}")
        out += [{k: v for k, v in r.items() if k != "_rr"} for r in rows]
    with open(ROOT / "reports/val_replay.jsonl", "a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
