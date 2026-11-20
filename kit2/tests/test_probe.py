"""Vol probe scoring (kit2.eval.probe): a signal planted in one feature block is recovered out of sample and
scored with the vol RankIC definition; noise features score ~0."""
from __future__ import annotations

import numpy as np

from kit2.eval.probe import evaluate


def test_probe_recovers_a_planted_signal():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=8)                     # the signal lives along one fixed direction of the states
    recs = []
    for w in range(40):
        for a in range(2):
            sig = rng.normal(size=32)
            enc = np.c_[np.outer(sig, direction) + 0.1 * rng.normal(size=(32, 8)), rng.normal(size=(32, 24))]
            recs.append({"win": w, "t": 10 * w + a, "y": sig + 0.3 * rng.normal(size=32), "model": -sig,
                         "trail": rng.normal(size=(32, 5)), "enc": enc})
    out = {o["method"]: o["rankic"] for o in evaluate(recs)}
    assert out["probe_enc"] > 0.8 and out["probe_enc+trail"] > 0.8
    assert abs(out["probe_trail"]) < 0.15 and abs(out["trail_lh"]) < 0.15
    assert out["model"] < -0.8
