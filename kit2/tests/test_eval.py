"""Evaluation forecasts (kit2.eval.kit2pred): every group sent to the model is full, panel names first."""
from __future__ import annotations

import numpy as np

from kit2.eval.kit2pred import fill_groups, panel_chunks
from kit2.tests.test_train_data import root  # noqa: F401  (synthetic on-disk store fixture)
from kit2.train import Bucket, GroupSampler


def test_groups_are_full_and_score_each_panel_name_once(root):  # noqa: F811
    bk, G = Bucket("us", "1h"), 4
    gs = GroupSampler(root, [bk], G=G, A=1, batch_tokens=1, split="train")
    panel = np.array([1, 3, 5, 7, 9, 11])
    active = np.arange(12)
    anchors = np.zeros(0, dtype=np.int64)                        # the synthetic store has no SPY/QQQ
    chunks = [panel[:G], panel[G:]]                              # 4 + 2: the second group would be half empty
    groups = fill_groups(chunks, anchors, panel, active, G, np.random.default_rng(0))
    e = int(gs._ends[0][-1])
    raw = gs.build(0, np.full(len(groups), e), groups, np.full((len(groups), 1), bk.anchor_range[1]))
    assert (raw["valid"].sum(1) == G).all()
    inst = raw["inst_idx"]
    scored = inst[np.isin(inst, panel)]
    assert sorted(scored.tolist()) == sorted(panel.tolist())
    assert set(inst[1, 2:].tolist()).isdisjoint(panel)           # the fill is context only


def test_panel_name_that_is_a_market_anchor_is_not_duplicated(root):  # noqa: F811
    bk, G = Bucket("us", "1h"), 4
    gs = GroupSampler(root, [bk], G=G, A=1, batch_tokens=1, split="train")
    anchors, panel = np.array([0]), np.array([0, 2, 4, 6, 8])  # instrument 0 is both an anchor and a panel name
    rng = np.random.default_rng(0)
    chunks = panel_chunks(panel, anchors, G, rng)
    assert sorted(np.concatenate(chunks).tolist()) == [2, 4, 6, 8]
    groups = fill_groups(chunks, anchors, panel, np.arange(12), G, rng)
    e = int(gs._ends[0][-1])
    raw = gs.build(0, np.full(len(groups), e), groups, np.full((len(groups), 1), bk.anchor_range[1]))
    for row, ok in zip(raw["inst_idx"], raw["valid"]):
        assert ok.all() and len(set(row.tolist())) == G           # full, no instrument twice in a group
