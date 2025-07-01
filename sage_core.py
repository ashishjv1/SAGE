"""
SAGE Core Implementation
========================

This module contains the core SAGE algorithms including:
- Frequent Directions (FD) streaming
- Agreement-based subset selection  
- Gradient computation utilities
- Baseline methods (GradMatch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def gpu_fd_stream(
    A: np.ndarray,
    ell: int,
    device=None,
    batch_size: int = 2048,
    dtype: torch.dtype = torch.float16,
    svd_dtype: torch.dtype = torch.float32,
):
    """GPU-accelerated Frequent Directions streaming"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m, n = A.shape
    B = torch.zeros((ell, n), device=device, dtype=dtype)
    next_row = 0

    for start in range(0, m, batch_size):
        batch_gpu = torch.from_numpy(A[start:start + batch_size])\
                 .to(device=device, dtype=dtype, non_blocking=True)

        insert = 0
        while insert < batch_gpu.shape[0]:
            space = ell - next_row
            take = min(space, batch_gpu.shape[0] - insert)
            B[next_row:next_row + take] = batch_gpu[insert:insert + take]
            next_row += take
            insert += take

            if next_row == ell:  # overflow → compress
                U, s, Vt = torch.linalg.svd(
                    B.to(svd_dtype), full_matrices=False
                )
                delta = s[-1] ** 2
                s = torch.sqrt(torch.clamp(s ** 2 - delta, min=0.0))
                B = (torch.diag(s).to(dtype) @ Vt.to(dtype))
                next_row = (s > 1e-8).sum().item()
                if next_row < ell:
                    B[next_row:].zero_()
                torch.cuda.empty_cache()

    return B[:next_row].float().cpu().numpy()


class FDStreamer:
    """Frequent Directions Streamer for memory-efficient sketching"""
    
    def __init__(self, ell: int, batch_size: int = 2048, dtype: torch.dtype = torch.float16):
        self.ell = ell
        self.batch_size = batch_size
        self.dtype = dtype
        self._buf = []  # CPU mini-batches
        self._B = None

    def add(self, grad_batch: np.ndarray):
        """Add a batch of gradients to the sketch"""
        self._buf.append(grad_batch)
        if sum(b.shape[0] for b in self._buf) >= self.batch_size:
            self._flush()

    def _flush(self):
        """Flush accumulated gradients through FD compression"""
        if not self._buf:
            return
        A_cpu = np.vstack(self._buf)
        self._buf.clear()
        if self._B is None:
            self._B = gpu_fd_stream(A_cpu, self.ell,
                                  batch_size=self.batch_size,
                                  dtype=self.dtype)
        else:
            A_big = np.vstack([self._B, A_cpu])
            self._B = gpu_fd_stream(A_big, self.ell,
                                  batch_size=self.batch_size,
                                  dtype=self.dtype)

    def finalize(self) -> np.ndarray:
        """Return final sketch matrix"""
        self._flush()
        return self._B


def per_sample_grads_slow(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """
    Compute per-sample gradients using backward hooks.
    Returns a (B, D) NumPy array: one gradient vector per sample.
    """
    model.eval()
    B = x.size(0)
    grads = []

    for i in range(B):
        model.zero_grad(set_to_none=True)
        logits = model(x[i:i+1])  # keep batch dim
        loss = F.cross_entropy(logits, y[i:i+1])
        loss.backward()

        # flatten all parameter grads into one long vector
        g = torch.cat([
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ]).cpu().numpy()
        grads.append(g)
        torch.cuda.empty_cache()
    
    return np.stack(grads, axis=0)  # (B, D)


def _project_single_grad(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor,
    criterion: nn.Module,
    proj_matrix: torch.Tensor  # (ℓ, D) on same device as model
) -> torch.Tensor:  # (ℓ,)
    """Project the gradient of ONE sample to ℓ dims"""
    model.zero_grad(set_to_none=True)
    out = model(x.unsqueeze(0)).squeeze(0)
    loss = criterion(out, y)
    loss.backward()

    # flatten all parameter grads in registration order
    g_proj = torch.zeros(proj_matrix.size(0), device=proj_matrix.device)
    offset = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g_flat = p.grad.flatten().to(proj_matrix.dtype)
        P_slice = proj_matrix[:, offset: offset + g_flat.numel()]
        g_proj += P_slice @ g_flat
        offset += g_flat.numel()
    return g_proj


def class_balanced_agreeing_subset_fast(
    model: nn.Module,
    dataset,
    num_classes: int,
    samples_per_class: int,
    criterion: nn.Module,
    device: torch.device,
    proj_matrix: torch.Tensor,   # (ℓ, D) on same device as model
    batch_size_data: int = 64,   # images per forward pass
    chunk_size_grad: int = 4     # images whose grads we keep at once
) -> List[int]:
    """
    Pick `samples_per_class` images per class with the highest
    gradient-agreement score, **without ever running out of GPU memory**.
    Returns a list of dataset indices.
    """
    model.eval()
    proj_matrix = proj_matrix.to(device)

    loader = DataLoader(dataset,
                       batch_size=batch_size_data,
                       shuffle=False,
                       num_workers=4,
                       pin_memory=True)

    # buckets for low-dim projected grads (stored on CPU)
    grads_per_class = defaultdict(list)   # list[(Ni_c , ℓ)]
    indices_per_class = defaultdict(list)

    running_idx = 0
    for X, Y in tqdm(loader, desc="Computing projected gradients"):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        # split current mini-batch into micro-chunks
        B = Y.size(0)
        for s in range(0, B, chunk_size_grad):
            xc = X[s : s + chunk_size_grad]
            yc = Y[s : s + chunk_size_grad]

            # compute projected grad for **each** sample in micro-chunk
            proj_chunk = torch.stack([
                _project_single_grad(model, xc[i], yc[i],
                                   criterion, proj_matrix)
                for i in range(yc.size(0))
            ])  # (m, ℓ) on GPU
            proj_chunk_cpu = proj_chunk.cpu()  # immediately off-load

            # bucket by class
            for cls in range(num_classes):
                mask = (yc == cls)
                if mask.any():
                    grads_per_class[cls].append(proj_chunk_cpu[mask.cpu()])
                    base = running_idx + s
                    idxs = torch.arange(base, base + yc.size(0))[mask.cpu()]
                    indices_per_class[cls].append(idxs)

            del proj_chunk, proj_chunk_cpu
            torch.cuda.empty_cache()

        running_idx += B

    # agreement scoring
    selected = []
    for cls in range(num_classes):
        if cls not in grads_per_class:
            continue  # class absent
        G = torch.cat(grads_per_class[cls], dim=0)  # (Nc , ℓ)
        I = torch.cat(indices_per_class[cls], dim=0)  # (Nc ,)

        # cosine agreement with centroid
        G_norm = F.normalize(G, dim=1)
        centroid = G_norm.mean(0)
        scores = G_norm @ centroid
        top_k = torch.topk(scores, min(samples_per_class, len(G))).indices
        selected.extend(I[top_k].tolist())

    return selected


def agreeing_subset_fast(
    model: nn.Module,
    dataset,
    subset_size: int,           # total samples to keep
    criterion: nn.Module,
    device: torch.device,
    proj_matrix: torch.Tensor,  # (ℓ, D) on same device
    batch_size_data: int = 64,
    chunk_size_grad: int = 4
) -> List[int]:
    """
    Pick `subset_size` images whose projected gradients have the highest
    agreement with the global centroid. OOM-safe and single-pass.
    """
    model.eval()
    proj_matrix = proj_matrix.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size_data,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_grads = []    # list[(m, ℓ)] on CPU
    all_indices = []  # matching dataset indices (tensor CPU)

    running_idx = 0
    for X, Y in tqdm(loader, desc="Computing projected gradients"):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        B = Y.size(0)
        for s in range(0, B, chunk_size_grad):
            xc = X[s : s + chunk_size_grad]
            yc = Y[s : s + chunk_size_grad]

            proj_chunk = torch.stack([
                _project_single_grad(model, xc[i], yc[i],
                                   criterion, proj_matrix)
                for i in range(yc.size(0))
            ])  # (m, ℓ) GPU
            all_grads.append(proj_chunk.cpu())  # off-load
            base = running_idx + s
            all_indices.append(
                torch.arange(base, base + yc.size(0)).cpu()
            )

            del proj_chunk
            torch.cuda.empty_cache()

        running_idx += B

    # agreement scoring
    G = torch.cat(all_grads, dim=0)   # (N, ℓ) CPU
    I = torch.cat(all_indices, dim=0) # (N,) CPU

    G_norm = F.normalize(G, dim=1)
    centroid = G_norm.mean(0)
    scores = G_norm @ centroid  # (N,)

    top = torch.topk(scores, min(subset_size, len(G))).indices
    return I[top].tolist()





def select_agreeing_subset_with_diversity(
    grads: torch.Tensor,
    subset_size: int,
    lambda_div: float = 0.2
) -> np.ndarray:
    """
    SAGE subset selection with diversity penalty
    """
    grads = F.normalize(grads, dim=1)
    centroid = grads.mean(dim=0)
    scores = grads @ centroid
    
    chosen = []
    for _ in range(subset_size):
        if not chosen:
            # First pick: pure agreement
            idx = scores.argmax()
        else:
            # Later picks: agreement - diversity penalty
            idxs_t = torch.tensor(chosen, device=grads.device)
            S = grads[idxs_t]
            if S.dim() == 1:
                S = S.unsqueeze(0)
            
            sim_to_S = (grads @ S.T).max(dim=1).values
            adj_score = scores - lambda_div * sim_to_S
            idx = adj_score.argmax()
        
        chosen.append(idx.item())
        # Set score to -inf to avoid re-selection
        scores[idx] = float('-inf')
    
    return np.array(chosen)


# CutMix augmentation utilities
def cutmix_data(x, y, alpha=1.0, device=None):
    """Apply CutMix augmentation to a batch of data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device
    
    # Shuffle indices
    index = torch.randperm(batch_size).to(device)

    # Generate random box
    W = x.size(3)
    H = x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index], lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMixDataLoader:
    """Wrapper for DataLoader that applies CutMix augmentation"""
    
    def __init__(self, dataloader, alpha=1.0, prob=0.5):
        self.dataloader = dataloader
        self.alpha = alpha
        self.prob = prob
    
    def __iter__(self):
        for x, y in self.dataloader:
            if np.random.rand() < self.prob:
                x, y_a, y_b, lam = cutmix_data(x, y, self.alpha)
                yield x, y_a, y_b, lam, True  # True indicates CutMix was applied
            else:
                yield x, y, y, 1.0, False  # False indicates no CutMix
    
    def __len__(self):
        return len(self.dataloader)


# Utility functions for experiments
def compute_agreement_scores(grads: torch.Tensor) -> torch.Tensor:
    """Compute agreement scores for analysis"""
    grads_norm = F.normalize(grads, dim=1)
    centroid = grads_norm.mean(dim=0)
    scores = grads_norm @ centroid
    return scores


def compute_feature_representations(model: nn.Module, dataset, device: torch.device) -> tuple:
    """Extract feature representations for visualization"""
    model.eval()
    features = []
    labels = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting features"):
            x = x.to(device)
            # Get features from second-to-last layer
            if hasattr(model, 'fc'):
                # ResNet-style models
                feat = model.features(x) if hasattr(model, 'features') else model(x, last=True)[1]
            else:
                # Other models - use penultimate layer
                feat = model(x)
            
            features.append(feat.cpu())
            labels.append(y)
    
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)
