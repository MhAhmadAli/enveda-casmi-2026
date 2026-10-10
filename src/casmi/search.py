"""GPU spectral retrieval: sparse binned-cosine prefilter over millions of spectra."""
from __future__ import annotations

import numpy as np


class SpectralIndex:
    """Holds an L2-normalised sparse reference matrix on the GPU and answers top-k
    cosine queries. Scores are plain dot products because rows are pre-normalised.

    The reference is (n_spectra x n_bins); a query batch becomes a dense
    (n_bins x B) block so the whole search is one sparse-dense matmul.
    """

    def __init__(self, ref_csr, device: str = "cuda", dtype=None):
        import torch
        self.torch = torch
        dtype = dtype or torch.float32
        self.device = device
        self.n_ref, self.n_bins = ref_csr.shape
        ref_csr = ref_csr.astype(np.float32)
        self.mat = torch.sparse_csr_tensor(
            torch.from_numpy(ref_csr.indptr.astype(np.int64)),
            torch.from_numpy(ref_csr.indices.astype(np.int32)),
            torch.from_numpy(ref_csr.data),
            size=ref_csr.shape, device=device, dtype=dtype,
        )

    @property
    def nbytes(self) -> int:
        return self.mat.values().numel() * 4 + self.mat.col_indices().numel() * 4

    def search(self, q_csr, topk: int = 1000, batch: int = 32,
               mask_rows: np.ndarray | None = None, progress: bool = False):
        """Return (scores, idx) arrays of shape (n_queries, topk), descending.

        mask_rows: boolean array over reference rows; True rows are excluded. Used to
        enforce structure-disjoint validation (ban every spectrum of a held-out
        molecule) without rebuilding the index.
        """
        torch = self.torch
        nq = q_csr.shape[0]
        topk = min(topk, self.n_ref)
        out_s = np.zeros((nq, topk), dtype=np.float32)
        out_i = np.zeros((nq, topk), dtype=np.int64)
        ban = None
        if mask_rows is not None:
            ban = torch.from_numpy(np.ascontiguousarray(mask_rows)).to(self.device)
        rng = range(0, nq, batch)
        if progress:
            from tqdm import tqdm
            rng = tqdm(rng, total=(nq + batch - 1) // batch, desc="search")
        for s in rng:
            e = min(s + batch, nq)
            dense = torch.from_numpy(
                np.ascontiguousarray(q_csr[s:e].toarray().T)).to(self.device)
            sims = torch.mm(self.mat, dense)          # (n_ref, b)
            if ban is not None:
                sims[ban] = -1.0
            v, i = torch.topk(sims, k=topk, dim=0)
            out_s[s:e] = v.T.cpu().numpy()
            out_i[s:e] = i.T.cpu().numpy()
            del dense, sims, v, i
        return out_s, out_i


def search_cpu(ref_csr, q_csr, topk: int = 1000, chunk: int = 64,
               mask_rows: np.ndarray | None = None, progress: bool = False):
    """Top-k cosine search via sparse-sparse matmul.

    Cost scales with (query nnz) x (mean reference entries per bin), not with the
    full reference size, so this stays cheap even against millions of spectra -- far
    better than materialising a dense score matrix.
    """
    from scipy import sparse

    nq = q_csr.shape[0]
    topk = min(topk, ref_csr.shape[0])
    out_s = np.zeros((nq, topk), dtype=np.float32)
    out_i = np.full((nq, topk), -1, dtype=np.int64)
    rng = range(0, nq, chunk)
    if progress:
        from tqdm import tqdm
        rng = tqdm(rng, total=(nq + chunk - 1) // chunk, desc="search")
    for s in rng:
        e = min(s + chunk, nq)
        prod = (ref_csr @ q_csr[s:e].T).tocsc()      # (n_ref, b)
        for c in range(e - s):
            lo, hi = prod.indptr[c], prod.indptr[c + 1]
            rows, vals = prod.indices[lo:hi], prod.data[lo:hi]
            if mask_rows is not None and rows.size:
                ok = ~mask_rows[rows]
                rows, vals = rows[ok], vals[ok]
            if rows.size == 0:
                continue
            k = min(topk, rows.size)
            sel = np.argpartition(vals, -k)[-k:]
            order = sel[np.argsort(-vals[sel])]
            out_s[s + c, :k] = vals[order]
            out_i[s + c, :k] = rows[order]
    return out_s, out_i


def merge_topk(score_lists, idx_lists, topk: int):
    """Union several (nq, k) candidate sets, keeping each reference row's best score."""
    nq = score_lists[0].shape[0]
    out_i, out_s = [], []
    for q in range(nq):
        best: dict[int, float] = {}
        for sc, ix in zip(score_lists, idx_lists):
            for s, i in zip(sc[q], ix[q]):
                if i < 0:
                    continue
                if s > best.get(int(i), -2.0):
                    best[int(i)] = float(s)
        items = sorted(best.items(), key=lambda kv: -kv[1])[:topk]
        out_i.append(np.array([k for k, _ in items], dtype=np.int64))
        out_s.append(np.array([v for _, v in items], dtype=np.float32))
    return out_s, out_i
