"""Crypto split (kit2.data.crypto.prepare.split_chain): a split restricted to some tokens writes exactly the files a
full split writes for those tokens, plus the quote tokens the USD references need, and nothing else."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from kit2.data.crypto import prepare
from kit2.tests.test_no_lookahead import synth_swaps


def test_targeted_split_equals_full_split(tmp_path, monkeypatch):
    sw, _ = synth_swaps()
    y = sw[sw["pool"] == "usdc_x"].assign(pool="usdc_y")              # a second token, Y
    sw = pd.concat([sw, y]).sort_values(["block", "log_index"], kind="stable").reset_index(drop=True)
    months = list(pd.period_range("2025-01", "2025-02", freq="M"))
    raw = tmp_path / "raw"
    (raw / "swaps").mkdir(parents=True)
    month = pd.to_datetime(sw["ts"], unit="s").dt.to_period("M")
    for m in months:
        sw[month == m].to_parquet(raw / "swaps" / f"{m}.parquet", index=False)
    pools = pd.DataFrame({"base": ["X", "X", "WETH", "Y"]}, index=pd.Index(["usdc_x", "weth_x", "ref", "usdc_y"],
                                                                           name="pool"))
    stub = SimpleNamespace(pools=pools, swap_months=months, nonstable={"WETH": None})
    monkeypatch.setattr(prepare, "ChainTables", lambda raw, chain: stub)

    n_full = prepare.split_chain(raw, tmp_path / "full", "eth")
    n_some = prepare.split_chain(raw, tmp_path / "some", "eth", ["X"])
    assert n_full == len(sw) and n_some == int(sw["pool"].isin(["usdc_x", "weth_x", "ref"]).sum())
    assert sorted(p.name for p in (tmp_path / "some" / "eth").iterdir()) == ["WETH", "X"]
    for base in ("X", "WETH"):
        for m in months:
            pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "some" / "eth" / base / f"{m}.parquet"),
                                          pd.read_parquet(tmp_path / "full" / "eth" / base / f"{m}.parquet"))
