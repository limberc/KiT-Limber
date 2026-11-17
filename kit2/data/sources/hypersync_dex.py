"""Envio HyperSync access to Uniswap-v3 pools: pool registry, activity scan, decoded swaps.

All timestamps are block timestamps (UTC seconds) - the moment the swap became public on chain.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import hypersync
from hypersync import (BlockField, FieldSelection, HexOutput, LogField, LogSelection, Query,
                       StreamConfig, TransactionField)

from kit2.env import secret

SWAP_SIG = ("Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, "
            "uint160 sqrtPriceX96, uint128 liquidity, int24 tick)")
POOL_CREATED_SIG = ("PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, "
                    "int24 tickSpacing, address pool)")
SWAP_T0 = hypersync.signature_to_topic0("Swap(address,address,int256,int256,uint160,uint128,int24)")
POOL_CREATED_T0 = hypersync.signature_to_topic0("PoolCreated(address,address,uint24,int24,address)")


@dataclass(frozen=True)
class Token:
    symbol: str
    decimals: int
    stable: bool = False


@dataclass(frozen=True)
class Chain:
    name: str
    url: str
    factory: str
    factory_block: int
    quotes: Dict[str, Token] = field(default_factory=dict)   # lower-case address -> quote token

    def quote_rank(self, addr: str) -> int:
        """Lower = preferred quote: stables, then ETH, then BTC; non-quotes rank last."""
        t = self.quotes.get(addr)
        if t is None:
            return 99
        return 0 if t.stable else (1 if "ETH" in t.symbol else 2)


CHAINS: Dict[str, Chain] = {
    "eth": Chain("eth", "https://eth.hypersync.xyz", "0x1f98431c8ad98523631ae4a59f267346ea31f984", 12369621, {
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": Token("USDC", 6, True),
        "0xdac17f958d2ee523a2206206994597c13d831ec7": Token("USDT", 6, True),
        "0x6b175474e89094c44da98b954eedeac495271d0f": Token("DAI", 18, True),
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": Token("WETH", 18),
        "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": Token("WBTC", 8)}),
    "base": Chain("base", "https://base.hypersync.xyz", "0x33128a8fc17869897dce68ed026d694621f6fdfd", 1371680, {
        "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": Token("USDC", 6, True),
        "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca": Token("USDbC", 6, True),
        "0x50c5725949a6f0c72e6c4a641f24049a917db0cb": Token("DAI", 18, True),
        "0x4200000000000000000000000000000000000006": Token("WETH", 18),
        "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": Token("cbBTC", 8)}),
    "arbitrum": Chain("arbitrum", "https://arbitrum.hypersync.xyz", "0x1f98431c8ad98523631ae4a59f267346ea31f984", 165, {
        "0xaf88d065e77c8cc2239327c5edb3a432268e5831": Token("USDC", 6, True),
        "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8": Token("USDC.e", 6, True),
        "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9": Token("USDT", 6, True),
        "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1": Token("DAI", 18, True),
        "0x82af49447d8a07e3bd95bd0d56f35241523fbab1": Token("WETH", 18),
        "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f": Token("WBTC", 8)}),
}


def hex_to_float(col, signed: bool = False) -> np.ndarray:
    """0x-hex strings (ABI words) -> float64; signed values are two's complement 256-bit."""
    out = np.empty(len(col), dtype=np.float64)
    for i, h in enumerate(col):
        if h is None:
            out[i] = np.nan
            continue
        v = int(h, 16)
        if signed and v >= 1 << 255:
            v -= 1 << 256
        out[i] = float(v)
    return out


def hex_to_int(col) -> np.ndarray:
    return np.array([int(h, 16) if h is not None else -1 for h in col], dtype=np.int64)


def _addr_from_word(col) -> List[str]:
    """32-byte ABI word or 20-byte address hex -> lower-case 0x address."""
    return [("0x" + h[-40:]).lower() if h is not None else None for h in col]


class Dex:
    def __init__(self, chain: str) -> None:
        self.chain = CHAINS[chain]
        self.client = hypersync.HypersyncClient(hypersync.ClientConfig(
            url=self.chain.url, bearer_token=secret("ENVIO_API_TOKEN")))

    async def height(self) -> int:
        return await self.client.get_height()

    async def _scan(self, query: Query, on_batch: Callable[[object], None], to_block: int,
                    signature: Optional[str] = None, progress: Optional[Callable[[int], None]] = None) -> None:
        """Page through [query.from_block, to_block) with get_arrow; on_batch(response.data) per page."""
        cfg = StreamConfig(event_signature=signature, hex_output=HexOutput.PREFIXED)
        query.to_block = to_block
        for attempt in range(8):                   # a failed stream restarts from the last finished page
            try:
                stream = await self.client.stream_arrow(query, cfg)
                while True:
                    r = await stream.recv()
                    if r is None:
                        return
                    on_batch(r.data)
                    query.from_block = r.next_block
                    if progress:
                        progress(query.from_block)
                    if r.next_block >= to_block:
                        return
            except Exception:
                if attempt == 7:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def pools(self) -> pd.DataFrame:
        """Every pool created by the Uniswap v3 factory: pool, token0, token1, fee, tick_spacing, block."""
        parts: List[pd.DataFrame] = []

        def on_batch(d):
            if d.logs.num_rows == 0:
                return
            logs = d.logs.to_pydict()
            parts.append(pd.DataFrame({
                "block": logs["block_number"],
                "token0": _addr_from_word(logs["topic1"]), "token1": _addr_from_word(logs["topic2"]),
                "fee": hex_to_int(logs["topic3"]),
                "tick_spacing": [int(h[-64:], 16) for h in (x[2:66] for x in logs["data"])],
                "pool": _addr_from_word([x[66:130] for x in logs["data"]]),
            }))

        q = Query(from_block=self.chain.factory_block,
                  logs=[LogSelection(address=[self.chain.factory], topics=[[POOL_CREATED_T0]])],
                  field_selection=FieldSelection(log=[LogField.BLOCK_NUMBER, LogField.TOPIC1, LogField.TOPIC2,
                                                      LogField.TOPIC3, LogField.DATA]))
        await self._scan(q, on_batch, await self.height())
        return pd.concat(parts, ignore_index=True)

    async def activity(self, from_block: int, to_block: int,
                       progress: Optional[Callable[[int], None]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Swap counts per (pool, UTC day) for every Swap(v3 ABI) log in the range; also first block per day."""
        counts: List[pd.DataFrame] = []
        day_blocks: List[pd.DataFrame] = []

        def on_batch(d):
            if d.logs.num_rows == 0:
                return
            logs = d.logs.select(["address", "block_number"]).to_pandas()
            blocks = d.blocks.select(["number", "timestamp"]).to_pandas()
            blocks["day"] = pd.to_datetime(hex_to_int(blocks["timestamp"]), unit="s").normalize()
            logs = logs.merge(blocks[["number", "day"]], left_on="block_number", right_on="number", how="left")
            counts.append(logs.groupby(["address", "day"]).size().rename("n_swaps").reset_index())
            day_blocks.append(blocks.groupby("day")["number"].min().rename("first_block").reset_index())

        q = Query(from_block=from_block, logs=[LogSelection(topics=[[SWAP_T0]])],
                  field_selection=FieldSelection(block=[BlockField.NUMBER, BlockField.TIMESTAMP],
                                                 log=[LogField.ADDRESS, LogField.BLOCK_NUMBER]))
        await self._scan(q, on_batch, to_block, progress=progress)
        act = pd.concat(counts).groupby(["address", "day"], as_index=False)["n_swaps"].sum()
        days = pd.concat(day_blocks).groupby("day", as_index=False)["first_block"].min()
        return act.rename(columns={"address": "pool"}), days

    async def swaps(self, pools: List[str], from_block: int, to_block: int) -> pd.DataFrame:
        """Decoded swaps of `pools` in [from_block, to_block), one row per swap, ordered by (block, log_index).

        Columns: block, log_index, ts (UTC s), pool, amount0, amount1 (raw token units, + = into the pool),
        log_price (ln token1-per-token0 in raw units, post-swap), liquidity, tick, tx_from, base_fee_gwei.
        """
        parts: List[pd.DataFrame] = []

        def on_batch(d):
            if d.logs.num_rows == 0:
                return
            logs = d.logs.select(["block_number", "log_index", "address", "transaction_hash"]).to_pandas()
            dec = d.decoded_logs.to_pydict()
            sqrt = hex_to_float(dec["sqrtPriceX96"])
            df = pd.DataFrame({
                "block": logs["block_number"].to_numpy(np.int64), "log_index": logs["log_index"].to_numpy(np.int32),
                "pool": logs["address"].str.lower(), "tx": logs["transaction_hash"],
                "amount0": hex_to_float(dec["amount0"], signed=True),
                "amount1": hex_to_float(dec["amount1"], signed=True),
                "log_price": 2.0 * (np.log(sqrt) - 96.0 * np.log(2.0)),
                "liquidity": hex_to_float(dec["liquidity"]),
                "tick": hex_to_float(dec["tick"], signed=True).astype(np.int32),
            })
            blocks = d.blocks.select(["number", "timestamp", "base_fee_per_gas"]).to_pandas()
            blocks["ts"] = hex_to_int(blocks["timestamp"])
            blocks["base_fee_gwei"] = hex_to_float(blocks["base_fee_per_gas"]) / 1e9
            txs = d.transactions.select(["hash", "from"]).to_pandas().rename(columns={"hash": "tx", "from": "tx_from"})
            df = df.merge(blocks[["number", "ts", "base_fee_gwei"]], left_on="block", right_on="number", how="left")
            df = df.merge(txs.drop_duplicates("tx"), on="tx", how="left").drop(columns=["number", "tx"])
            parts.append(df)

        q = Query(from_block=from_block,
                  logs=[LogSelection(address=pools, topics=[[SWAP_T0]])],
                  field_selection=FieldSelection(
                      block=[BlockField.NUMBER, BlockField.TIMESTAMP, BlockField.BASE_FEE_PER_GAS],
                      log=[LogField.BLOCK_NUMBER, LogField.LOG_INDEX, LogField.ADDRESS, LogField.TRANSACTION_HASH,
                           LogField.TOPIC0, LogField.TOPIC1, LogField.TOPIC2, LogField.DATA],
                      transaction=[TransactionField.HASH, TransactionField.FROM]))
        await self._scan(q, on_batch, to_block, signature=SWAP_SIG)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        return out.sort_values(["block", "log_index"]).reset_index(drop=True)
