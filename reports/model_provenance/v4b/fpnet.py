"""Spectrum -> molecular fingerprint transformer (FPNet) with mass-window contrastive ranking loss.

Score of a candidate with fingerprint f under predicted logits z is f.z (Bayes log-likelihood up to
a candidate-independent constant), so the ranking objective is trained directly with hard negatives
drawn from the candidate pool within +-10 ppm of the true mass.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import chem

MAX_PEAKS = 160
N_ADDUCT = len(chem.ADDUCT_LIST)


def prep_peaks(mz, it, prec_mz, max_peaks=MAX_PEAKS, floor=1e-3):
    """Cleaned, intensity-top-k, m/z-sorted peaks with sqrt intensities in (0,1]."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 2.0) & (mz > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0 or it.max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    it = it / it.max()
    keep = it >= floor
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        sel = np.argsort(-it)[:max_peaks]
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    return mz[o].astype(np.float32), np.sqrt(it[o]).astype(np.float32)


class SinEmb(nn.Module):
    """Multi-scale sinusoidal embedding of a scalar (m/z or mass)."""
    def __init__(self, dim, lo=-2.0, hi=3.3, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)

    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        B, N, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=8, heads=8, drop=0.1, n_elem=10):
        super().__init__()
        self.nbits = nbits; self.d = d; self.layers = layers; self.heads = heads
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.int_emb = nn.Linear(2, d)
        self.pk = nn.Linear(3 * d, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(N_ADDUCT, d)
        self.ce_emb = SinEmb(d // 4, lo=0.0, hi=2.2)
        self.gl = nn.Linear(d + d // 4 + 4, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))
        self.elem_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, n_elem))

    def encode(self, mz, it, pad, prec, ad, ce, ce_known, merged, mode):
        """Token states after the encoder: (x (B,N+1,d) incl. global token first, pad (B,N+1))."""
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        pf = torch.cat([self.mz_emb(mz), self.nl_emb(nl),
                        self.int_emb(torch.stack([it, it * it], -1))], -1)
        p = self.pk(pf)
        g = self.gl(torch.cat([self.prec_emb(prec), self.ce_emb(ce / 10.0 + 1.0),
                               ce_known.unsqueeze(-1), merged.unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks:
            x = b(x, pad)
        return self.norm(x), pad

    def forward(self, mz, it, pad, prec, ad, ce, ce_known, merged, mode, return_emb=False):
        """mz,it: (B,N) float; pad: (B,N) bool (True=pad); prec: (B,) float; ad: (B,) long;
        ce: (B,) float eV; ce_known: (B,) float 0/1; merged: (B,) float (n merged spectra / 4);
        mode: (B,) float +-1."""
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        pf = torch.cat([self.mz_emb(mz), self.nl_emb(nl),
                        self.int_emb(torch.stack([it, it * it], -1))], -1)
        p = self.pk(pf)
        g = self.gl(torch.cat([self.prec_emb(prec), self.ce_emb(ce / 10.0 + 1.0),
                               ce_known.unsqueeze(-1), merged.unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        emb = torch.cat([cls, mean], -1)
        z = self.head(emb)
        if return_emb:
            return z, self.elem_head(emb), emb
        return z, self.elem_head(emb)


# ----------------------------------------------------------------------------------------------
# batching helpers (shared by training and inference)
# ----------------------------------------------------------------------------------------------
def collate(items: Sequence[dict], device) -> Tuple[torch.Tensor, ...]:
    """items: dicts with mz, it (prepped arrays), prec, adduct_ix, ce, ce_known, n_merged, mode."""
    B = len(items)
    N = max(1, max(len(x['mz']) for x in items))
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
    for i, x in enumerate(items):
        n = len(x['mz'])
        mz[i, :n] = x['mz']; it[i, :n] = x['it']; pad[i, :n] = False
        if n == 0:
            pad[i, 0] = False
    T = lambda a, dt: torch.as_tensor(np.asarray(a, dt), device=device)
    return (T(mz, np.float32), T(it, np.float32), T(pad, bool),
            T([x['prec'] for x in items], np.float32),
            T([x['adduct_ix'] for x in items], np.int64),
            T([x['ce'] for x in items], np.float32),
            T([x['ce_known'] for x in items], np.float32),
            T([x['n_merged'] / 4.0 for x in items], np.float32),
            T([x['mode'] for x in items], np.float32))


def unpack_gpu(packed: torch.Tensor, nbits: int) -> torch.Tensor:
    """packed: (..., nbytes) uint8 on GPU -> (..., nbits) float32 in {0,1}."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], -1)[..., :nbits].float()


def autocast_dtype(device='cuda'):
    """bf16 on Ampere+ GPUs, fp16 on older ones (T4/P100)."""
    if device != 'cuda' or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def save_ckpt(path, model, step, extra=None):
    d = dict(model=model.state_dict(), step=step, nbits=model.nbits, d=model.d, layers=model.layers,
             heads=model.heads)
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'], heads=ck.get('heads', 8))
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


class ModelBank:
    """Runs one or more FPNets on molecule spectra; returns averaged logits per view."""

    def __init__(self, paths: Sequence[str], device='cuda'):
        self.device = device
        self.models = [load_model(p, device) for p in paths]
        self.nbits = self.models[0].nbits
        self.dtype = autocast_dtype(device)

    @torch.no_grad()
    def logits_el(self, items: Sequence[dict], bs=64):
        """Returns (logits (n,nbits), element predictions (n,10)) averaged over models."""
        if not items:
            return np.zeros((0, self.nbits), np.float32), np.zeros((0, 10), np.float32)
        zo, eo = [], []
        for s in range(0, len(items), bs):
            args = collate(items[s:s + bs], self.device)
            with torch.autocast('cuda', dtype=self.dtype, enabled=self.device == 'cuda'):
                outs = [m(*args) for m in self.models]
            zo.append(torch.stack([o[0].float() for o in outs]).mean(0).cpu().numpy())
            eo.append(torch.stack([o[1].float() for o in outs]).mean(0).cpu().numpy())
        return np.concatenate(zo), np.concatenate(eo)

    def logits(self, items: Sequence[dict], bs=64) -> np.ndarray:
        return self.logits_el(items, bs)[0]
