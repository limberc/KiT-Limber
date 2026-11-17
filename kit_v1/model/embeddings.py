"""Condition embeddings: global identity and per-bar calendar (timestep: diffusers, see layers.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondEmbeddings(nn.Module):
    """Global identity condition: market + sector + proj(instrument) + scale -> [B, d].

    Slot 0 is the learnable unknown/null entry (no padding_idx).
    """
    def __init__(self, d: int, market_slots: int, sector_slots: int,
                 scale_slots: int, inst_vocab: int, inst_dim: int) -> None:
        super().__init__()
        self.market = nn.Embedding(market_slots, d)
        self.sector = nn.Embedding(sector_slots, d)
        self.scale = nn.Embedding(scale_slots, d)
        self.inst = nn.Embedding(inst_vocab, inst_dim)  # low-rank instrument table
        self.inst_proj = nn.Linear(inst_dim, d, bias=False)  # project to d

    def forward(self, market: torch.Tensor, sector: torch.Tensor,
                inst: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Four [B] long tensors -> c_base [B, d]."""
        return (self.market(market) + self.sector(sector)
                + self.inst_proj(self.inst(inst)) + self.scale(scale))


class CalendarEmbed(nn.Module):
    """Per-bar calendar embedding [B, L, d], added to token embeddings (never noised)."""

    def __init__(self, d: int, clock_dim: int, sess_hidden: int,
                 yday_dim: int = 8) -> None:
        super().__init__()
        # time of day (Fourier features)
        self.clock_proj = nn.Linear(clock_dim, d)
        # relative position within the session
        self.sess_fc1 = nn.Linear(2, sess_hidden)
        self.sess_fc2 = nn.Linear(sess_hidden, d)
        # day of week
        self.dow = nn.Embedding(7, d)
        # special event (0 = normal)
        self.event = nn.Embedding(3, d)
        # month (0 = January)
        self.month = nn.Embedding(12, d)
        # day of year (low-order Fourier features)
        self.yday_proj = nn.Linear(yday_dim, d)

    def forward(self, clock: torch.Tensor, sess: torch.Tensor,
                dow: torch.Tensor, event: torch.Tensor,
                month: torch.Tensor, yday: torch.Tensor) -> torch.Tensor:
        """clock [B,L,16], sess [B,L,2], yday [B,L,8] float; dow/event/month [B,L] long -> [B,L,d]."""
        return (self.clock_proj(clock)
                + self.sess_fc2(F.silu(self.sess_fc1(sess)))
                + self.dow(dow) + self.event(event)
                + self.month(month) + self.yday_proj(yday))


def make_role_embedding(d: int) -> nn.Embedding:
    """Role embedding: 0 = ctx, 1 = tgt."""
    return nn.Embedding(2, d)


def make_register_tokens(n_registers: int, d: int) -> nn.Parameter:
    """Learnable register tokens [n_registers, d]."""
    p = nn.Parameter(torch.empty(n_registers, d))
    nn.init.normal_(p, mean=0.0, std=0.02)
    return p
