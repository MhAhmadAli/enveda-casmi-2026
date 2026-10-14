"""Spectrum -> molecular fingerprint transformer (FPNet) with mass-window contrastive ranking loss.

Score of a candidate with fingerprint f under predicted logits z is f.z (Bayes log-likelihood up to
a candidate-independent constant), so the ranking objective is trained directly with hard negatives
drawn from the candidate pool within +-10 ppm of the true mass.

engine v6: load_model dispatches natively on the checkpoint's 'arch' ('dreams_fp' -> dreams_model.DreamsFP, else the
unchanged FPNet path); ModelBank averages mixed FPNet / DreamsFP logits with a per-device precision recipe.
"""
from __future__ import annotations

import contextlib
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
    """bf16 on Ampere+ GPUs (sm >= 8), fp16 on older ones (T4 sm75 / P100 sm60), fp32 (= no autocast) on CPU."""
    if device != 'cuda' or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def save_ckpt(path, model, step, extra=None):
    d = dict(model=model.state_dict(), step=step, nbits=model.nbits, d=model.d, layers=model.layers,
             heads=model.heads)
    if extra:
        d.update(extra)
    torch.save(d, path)


# ----------------------------------------------------------------------------------------------
# checkpoints: FPNet and DreamsFP (engine v6: native dispatch on ck['arch'])
# ----------------------------------------------------------------------------------------------
NATIVE_DREAMS_FP = True          # dreams_model.patch_fpnet_load_model() is a no-op for this module


def model_from_ckpt(ck: dict, device='cuda'):
    """Checkpoint dict -> eval-mode model on `device`.
    arch absent or 'fpnet' -> FPNet (the unchanged pre-v6 path); 'dreams_fp' -> dreams_model.DreamsFP
    (a DreaMS backbone fine-tuned by work/analysis/dreams_ft/train_dreams_fp.py; same forward signature and outputs).
    Parameters stored in fp16 by export_model.py are copied into the fp32 modules (load_state_dict converts)."""
    arch = ck.get('arch', 'fpnet')
    if arch == 'dreams_fp':
        from . import dreams_model
        return dreams_model.dreams_fp_from_ckpt(ck, device)
    if arch != 'fpnet':
        raise ValueError(f'unknown checkpoint arch {arch!r}')
    m = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'], heads=ck.get('heads', 8))
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return model_from_ckpt(ck, device)


def model_arch(m) -> str:
    return 'dreams_fp' if hasattr(m, 'backbone') and hasattr(m, 'forward_spec') else 'fpnet'


# DreamsFP inference precision recipe (measured in work/analysis/engine_v6/precision_check.py, see its log):
#   * Fourier features are always computed in fp32 (FourierFeatures disables autocast) and BackboneCfg.fourier_fp32
#     keeps the 5-layer ff_fourier MLP on top of them in fp32 as well (the phases 2*pi*m*b reach 3.1e7 rad);
#   * attention logits + softmax: fp32 under fp16 autocast (T4/P100); under bf16 the checkpoint's training setting
#     is kept (pilot_d1 trained with --attn_bf16, i.e. attn_fp32 False).
DREAMS_ATTN_FP32 = {torch.float16: True, torch.bfloat16: None, torch.float32: True}   # None = as trained


def apply_dreams_recipe(m, dtype, attn_fp32: Optional[bool] = None) -> dict:
    """Set a DreamsFP's inference precision for autocast `dtype` in place; returns the recipe (for logging)."""
    bb = m.backbone
    trained = bool(bb.cfg.attn_fp32)
    if attn_fp32 is None:
        attn_fp32 = DREAMS_ATTN_FP32.get(dtype, True)
        if attn_fp32 is None:
            attn_fp32 = trained
    bb.cfg.fourier_fp32 = True
    bb.cfg.attn_fp32 = bool(attn_fp32)
    for att in bb.transformer_encoder.atts:
        att.fp32 = bool(attn_fp32)
    return dict(fourier_fp32=True, attn_fp32=bool(attn_fp32), attn_fp32_trained=trained, n_peaks=int(m.n_peaks))


class ModelBank:
    """Runs one or more spectrum->FP models on molecule spectra -- FPNet and DreamsFP checkpoints mixed freely -- and
    returns the per-item logits / element predictions averaged over the models (equal weights).
    Precision: one autocast dtype per device (autocast_dtype: bf16 on sm >= 8, fp16 on T4/P100, none on CPU); every
    DreamsFP gets apply_dreams_recipe(dtype) (Fourier path fp32; attention fp32 under fp16).
    dtype / dreams_attn_fp32 override the defaults (diagnostics only)."""

    def __init__(self, paths: Sequence[str], device='cuda', dtype=None, dreams_attn_fp32: Optional[bool] = None):
        self.device = device
        self.paths = [str(p) for p in paths]
        self.models = [load_model(p, device) for p in self.paths]
        if not self.models:
            raise ValueError('ModelBank needs at least one checkpoint')
        self.arch = [model_arch(m) for m in self.models]
        nb = sorted({int(m.nbits) for m in self.models})
        if len(nb) != 1:
            raise ValueError(f'models disagree on nbits: {nb}')
        self.nbits = self.models[0].nbits
        self.dtype = autocast_dtype(device) if dtype is None else dtype
        self.autocast = device == 'cuda' and self.dtype != torch.float32
        self.recipe = [apply_dreams_recipe(m, self.dtype, dreams_attn_fp32) if a == 'dreams_fp' else {}
                       for m, a in zip(self.models, self.arch)]

    def describe(self) -> List[dict]:
        """Per-model arch / precision recipe (for run manifests and logs)."""
        return [dict(path=p, arch=a, nbits=int(m.nbits), autocast=str(self.dtype) if self.autocast else 'none', **r)
                for p, a, m, r in zip(self.paths, self.arch, self.models, self.recipe)]

    def _ctx(self):
        return torch.autocast('cuda', dtype=self.dtype) if self.autocast else contextlib.nullcontext()

    @torch.no_grad()
    def logits_el_each(self, items: Sequence[dict], bs=64):
        """Per-model (logits (M,n,nbits), element predictions (M,n,10)) before averaging (parity checks)."""
        M = len(self.models)
        if not items:
            return np.zeros((M, 0, self.nbits), np.float32), np.zeros((M, 0, 10), np.float32)
        zo, eo = [], []
        for s in range(0, len(items), bs):
            args = collate(items[s:s + bs], self.device)
            with self._ctx():
                outs = [m(*args) for m in self.models]
            zo.append(torch.stack([o[0].float() for o in outs]).cpu().numpy())
            eo.append(torch.stack([o[1].float() for o in outs]).cpu().numpy())
        return np.concatenate(zo, 1), np.concatenate(eo, 1)

    @torch.no_grad()
    def logits_el(self, items: Sequence[dict], bs=64):
        """Returns (logits (n,nbits), element predictions (n,10)) averaged over models."""
        if not items:
            return np.zeros((0, self.nbits), np.float32), np.zeros((0, 10), np.float32)
        zo, eo = [], []
        for s in range(0, len(items), bs):
            args = collate(items[s:s + bs], self.device)
            with self._ctx():
                outs = [m(*args) for m in self.models]
            zo.append(torch.stack([o[0].float() for o in outs]).mean(0).cpu().numpy())
            eo.append(torch.stack([o[1].float() for o in outs]).mean(0).cpu().numpy())
        return np.concatenate(zo), np.concatenate(eo)

    def logits(self, items: Sequence[dict], bs=64) -> np.ndarray:
        return self.logits_el(items, bs)[0]
