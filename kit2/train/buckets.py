"""Training buckets: one (market, target scale) per step, fixed shapes per bucket (torch.compile friendly).

History = target-scale patches + the two next-coarser scales. Horizons are multiples of the 8-bar patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

LADDER: Tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
SCALE_ID: Dict[str, int] = {s: i + 1 for i, s in enumerate(LADDER)}       # 0 = null
SCALE_SECONDS: Dict[str, int] = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
                                 "1d": 86400}
MARKET_ID: Dict[str, int] = {"us": 1, "crypto": 2}
CHAIN_ID: Dict[str, int] = {"": 0, "eth": 1, "base": 2, "arbitrum": 3}

#                 target patches, coarse patches (each), horizon patches (x 8 bars)
WINDOWS: Dict[str, Tuple[int, int, int]] = {
    "1m": (128, 32, 15),   # 1024 bars history-window, 120-bar horizon
    "5m": (96, 32, 12),
    "15m": (64, 32, 8),
    "30m": (64, 32, 5),
    "1h": (48, 32, 3),
    "2h": (32, 32, 3),
    "1d": (32, 32, 3),
}


@dataclass(frozen=True)
class Bucket:
    market: str
    scale: str
    n_coarse_scales: int = 2

    @property
    def scales(self) -> List[str]:
        i = LADDER.index(self.scale)
        return list(LADDER[i: i + 1 + self.n_coarse_scales])

    @property
    def n_patches(self) -> List[int]:
        nt, nc, _ = WINDOWS[self.scale]
        return [nt] + [nc] * (len(self.scales) - 1)

    @property
    def horizon(self) -> int:
        return WINDOWS[self.scale][2]

    @property
    def anchor_range(self) -> Tuple[int, int]:
        """Target patch indices allowed as anchors: enough history before, the whole horizon after."""
        nt = WINDOWS[self.scale][0]
        return nt // 2 - 1, nt - self.horizon - 1

    @property
    def name(self) -> str:
        return f"{self.market}_{self.scale}"

    def tokens(self, G: int) -> int:
        return G * sum(self.n_patches)


def parse_buckets(markets: str, scales: str) -> List[Bucket]:
    return [Bucket(m, s) for m in markets.split(",") for s in scales.split(",")]
