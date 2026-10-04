"""
Graph-Aware Vessel Adapter (GAVA)

A lightweight, VGN-inspired (https://github.com/syshin1014/VGN) graph
convolution adapter branch, meant to run in parallel with the AdaptFormer
adapter (models/sansa/adapter.py) inside a Hiera MultiScaleBlock.

Design notes:
  - Operates directly on the block's native [B, H, W, C] layout.
  - Builds a *local* dynamic graph inside real 2-D windows (reusing Hiera's
    own `window_partition` / `window_unpartition`), so both horizontally-
    and vertically-running vessel segments stay connected within a window
    -- unlike naive 1-D raster chunking (a window of N consecutive raster
    tokens is just one or two image rows, and can't see across rows).
  - All windows (across the batch and across the image) are processed as a
    single batched op instead of a Python loop, with no change to the
    memory footprint of a windowed loop: both are
    O(B * H * W * window_size**2), linear in image size. A full, unwindowed
    [B, N, N] graph (what an earlier version of this module built) is what
    actually caused an OOM; local windowing avoids that regardless of
    whether the windows are visited in a loop or in one batched call --
    batching just removes the Python/kernel-launch overhead.
  - Neighbor selection uses cosine similarity plus a learnable salience bias
    that varies over the *candidate* (key) axis, so it can actually change
    which neighbors get picked. A bias that only varies per query row is a
    no-op under top-k selection, since scaling a whole row by one positive
    scalar can't change that row's ranking.
  - Aggregation is a softmax-weighted sum over the selected top-k
    similarities (lightweight graph attention), not a plain mean, reusing
    scores that top-k already computed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam2.modeling.backbones.utils import window_partition, window_unpartition


class GraphVesselConv(nn.Module):
    """One round of local, windowed graph convolution.

    Args:
        dim: feature dimension (operates in the adapter's bottleneck space).
        k: number of neighbors to aggregate per node.
        window_size: side length of the square local window (tokens per
            window = window_size ** 2). Kept small so a window covers a
            spatially coherent neighborhood in both H and W.
    """

    def __init__(self, dim: int, k: int = 8, window_size: int = 8):
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.node_proj = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # Learnable strength of the key-side salience bias; starts small so
        # the module leans on plain cosine similarity until training shows
        # the bias helps.
        self.bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        ws = self.window_size

        windows, (Hp, Wp) = window_partition(x, ws)          # [B*nW, ws, ws, C]
        M = windows.shape[0]
        Nw = ws * ws
        win_flat = windows.reshape(M, Nw, C)

        # Validity mask for padding tokens introduced by window_partition,
        # built with the exact same padding logic (so it always matches).
        valid = None
        if Hp != H or Wp != W:
            ones = torch.ones(1, H, W, 1, device=x.device, dtype=x.dtype)
            valid_windows, _ = window_partition(ones, ws)      # [nW, ws, ws, 1]
            valid = valid_windows.reshape(-1, Nw) > 0.5         # [nW, Nw]
            valid = valid.repeat(B, 1)                          # [M, Nw]

        kw = min(self.k, Nw - 1) if Nw > 1 else 0

        x_norm = F.normalize(win_flat, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))         # [M, Nw, Nw]

        # Salience bias: varies over the KEY axis (last dim), so it can
        # actually influence which neighbors get selected by topk below
        # (a bias that only varies per query row cannot, since scaling a
        # whole row by a positive scalar never changes its own ranking).
        mag = F.normalize(win_flat.norm(dim=-1), dim=-1)        # [M, Nw]
        sim = sim + self.bias_scale * mag.unsqueeze(1)          # broadcast over query axis

        # mask self
        eye = torch.eye(Nw, device=x.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, float('-inf'))

        # mask padded candidates
        if valid is not None:
            sim = sim.masked_fill(~valid.unsqueeze(1), float('-inf'))
            # Degenerate windows with no valid neighbor at all (can only
            # happen for a near-empty boundary window): fall back to a
            # finite row so softmax doesn't NaN. Harmless -- such queries
            # are themselves padding-adjacent corner cases and, if they are
            # padding tokens, their output is cropped away by
            # window_unpartition below regardless.
            dead = torch.isneginf(sim).all(dim=-1, keepdim=True)
            sim = torch.where(dead, torch.zeros_like(sim), sim)

        if kw == 0:
            agg = torch.zeros_like(win_flat)
        else:
            topk_val, topk_idx = sim.topk(kw, dim=-1)            # [M, Nw, kw]
            weights = F.softmax(topk_val, dim=-1)                # [M, Nw, kw]

            neighbors = torch.gather(
                win_flat.unsqueeze(1).expand(-1, Nw, -1, -1), 2,
                topk_idx.unsqueeze(-1).expand(-1, -1, -1, C),
            )                                                     # [M, Nw, kw, C]
            agg = (neighbors * weights.unsqueeze(-1)).sum(dim=2)  # [M, Nw, C]

        out = self.node_proj(win_flat) + self.edge_proj(agg)
        out = self.norm(out).reshape(M, ws, ws, C)

        return window_unpartition(out, ws, (Hp, Wp), (H, W))      # [B, H, W, C]


class VesselAdapter(nn.Module):
    """Bottleneck adapter with a stack of local graph-conv rounds.

    Mirrors AdaptFormer's Adapter (models/sansa/adapter.py) convention: the
    up-projection is zero-initialized so the branch starts as a no-op when
    summed into the block's residual stream.
    """

    def __init__(self, dim: int, bottleneck: int, k: int = 8,
                 scale: float = 0.1, num_rounds: int = 2,
                 window_size: int = 8):
        super().__init__()
        self.scale = scale
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.graph_convs = nn.ModuleList([
            GraphVesselConv(bottleneck, k=k, window_size=window_size)
            for _ in range(num_rounds)
        ])
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        h = self.act(self.down(x))
        for graph_conv in self.graph_convs:
            h = h + graph_conv(h)   # residual across rounds
        return self.up(h) * self.scale
