
<h1 align="center">KiT-2: A Fast Multi-Resolution Candlestick Foundation Model</h1>

<div align="center">

<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-AGPL_v3-green.svg" alt="License: AGPL-3.0">
</a>
</div>

> KiT-2 is a candlestick (K-line) foundation model trained on U.S. equities and on-chain crypto (Uniswap v3 tokens on Ethereum, Base and Arbitrum). It generates joint future paths for a group of instruments by flow matching, and is the smaller, faster successor of [KiT](https://github.com/Luciferbobo/KiT) ([arXiv:2609.34507](https://arxiv.org/abs/2609.34507)).

> ⚠️ For research purposes only. Not financial advice.

## 📖 Model

KiT-2 keeps KiT's five-feature candle state, $x_t=(r_{\mathrm{gap}}, r_{\mathrm{body}}, r_{\mathrm{up}}, r_{\mathrm{dn}}, v_t)$, and its flow-matching path generation. It changes how the model reads the history:

- **Patches:** one token per 8 bars instead of one per bar. History tokens come from the target timescale and the next two coarser ones.
- **Encode once:** the history encoder runs once per forecast. A small denoiser reads its cached states at every sampling step.
- **Group attention** across 32 instruments of one market at the same moment.
- **8 Heun steps** instead of 32 Euler steps.

| Model | Params | Architecture | Training data |
|---|---|---|---|
| KiT-L (`step_market_284M`) | 285.8M | d = 1024, 21 layers | A-shares, U.S., crypto (exchange USDT pairs), up to 2025-12-31 |
| **KiT-2** | **63.2M** | d = 512, 8 encoder + 4 denoiser layers | U.S. equities, on-chain crypto, up to 2026-06-30 |

KiT-2 was trained for 48k steps (about 4 h) on one RTX PRO 4000 (24 GB) in bf16-mixed precision. An adaptive mixture sets how often each (market, timescale) bucket is sampled: an even share with a cap on how often a bucket's bars are reused, tilted towards buckets whose validation loss is still falling. The config is `configs/kit2/dev_us_crypto_mix.yaml`. The checkpoint is not released yet.

## 📈 Results on held-out data

All models forecast the same instruments at the same moments:

- **Test period:** 2026-09-01 to 2026-11, unseen by every model.
- **Panel:** 64 instruments per market, with 6 forecast moments at each of 5m, 15m, 1h and 1d. That is 24 moments and 1,536 forecasts per market.
- **Horizon:** KiT's horizon `Lh` at each timescale: 96 bars at 5m, 64 at 15m, 20 at 1h and 20 at 1d.

Metrics:

- **Return / volatility RankIC:** Spearman correlation across the panel at each moment, averaged over moments.
- **CRPS skill:** 1 − CRPS(model) / CRPS(historical bootstrap) of the horizon return. Higher is better; 0 is the bootstrap.
- **cov80:** share of realised horizon returns inside the forecast's 10–90% range. The target is 0.80.

These numbers are **not comparable with the tables in the KiT paper**, which use different instruments, periods and protocol.

Baselines:

- **KiT-L:** KiT's published checkpoint `step_market_284M` with its published sampler (16 paths, 32 Euler steps).
- **LightGBM:** two gradient-boosted tree models per market and timescale, one for the horizon return and one for log realised volatility. They use 29 hand-made features of the last 4·`Lh` bars: trailing volatility and return at five windows, wick, volume and time-of-day features. Each is trained on 150k samples from KiT-2's training period. Its point forecasts become 16 Gaussian paths for CRPS and coverage (`kit2/eval/lgbm.py`).
- **Historical bootstrap:** 16 random blocks of the instrument's own last 2,000 bars.

**U.S. equities, test period**

| Model | Return RankIC ↑ | Vol RankIC ↑ | CRPS skill ↑ | cov80 (→ 0.80) |
|---|---|---|---|---|
| KiT-L | **+0.012** | **0.840** | −0.005 | 0.597 |
| **KiT-2** | −0.102 | 0.827 | −0.044 | 0.622 |
| LightGBM | −0.091 | 0.836 | **+0.071** | **0.722** |
| Historical bootstrap | −0.020 | 0.784 | 0 | 0.699 |

**On-chain crypto, test period**

| Model | Return RankIC ↑ | Vol RankIC ↑ | CRPS skill ↑ | cov80 (→ 0.80) |
|---|---|---|---|---|
| KiT-L ¹ | +0.005 | 0.669 | −0.088 | **0.752** |
| **KiT-2** | **+0.081** | 0.734 | +0.067 | 0.720 |
| LightGBM | +0.049 | **0.742** | **+0.087** | 0.639 |
| Historical bootstrap | −0.047 | 0.699 | 0 | 0.702 |

¹ KiT-L knows only 3 of the 64 on-chain tokens; the rest go through its unknown-instrument slot. In the U.S. panel, 40 of the 64 names are in its vocabulary.

**Volatility RankIC by timescale, test period**

| Model | U.S. 5m | U.S. 15m | U.S. 1h | U.S. 1d | crypto 5m | crypto 15m | crypto 1h | crypto 1d |
|---|---|---|---|---|---|---|---|---|
| KiT-L | .877 | **.840** | **.788** | **.855** | .562 | .673 | .729 | **.713** |
| **KiT-2** | **.880** | .818 | .772 | .838 | .651 | .787 | **.794** | .704 |
| LightGBM | .875 | .834 | .787 | .849 | .691 | **.810** | .770 | .699 |
| Historical bootstrap | .852 | .837 | .724 | .721 | **.739** | .759 | .683 | .616 |

**Validation period** (2026-07-01 to 2026-08-31, same protocol; KiT-L was not run)

| Market | Model | Return RankIC ↑ | Vol RankIC ↑ | CRPS skill ↑ | cov80 (→ 0.80) |
|---|---|---|---|---|---|
| U.S. | **KiT-2** | −0.201 | **0.841** | −0.074 | 0.561 |
| U.S. | LightGBM | −0.109 | 0.824 | **+0.070** | **0.673** |
| U.S. | Historical bootstrap | **−0.065** | 0.773 | 0 | 0.642 |
| crypto | **KiT-2** | **+0.065** | **0.741** | +0.042 | **0.747** |
| crypto | LightGBM | +0.012 | 0.690 | **+0.069** | 0.624 |
| crypto | Historical bootstrap | +0.040 | 0.705 | 0 | 0.732 |

Standard errors of these means, over 24 moments, are about 0.04 for return RankIC, 0.01–0.025 for volatility RankIC and 0.01–0.05 for CRPS skill. Return RankIC differences between models are within noise.

What the tables show:

- **KiT-2 vs KiT-L:** KiT-2 is 4.5× smaller and about 300× faster. It is close on U.S. volatility ranking (0.827 vs 0.840) and ahead on on-chain crypto, where KiT-L is out of domain.
- **LightGBM is a strong baseline.** On the test period it has the best CRPS in both markets, and its U.S. volatility RankIC is level with KiT-L. On the validation period, KiT-2 ranks volatility better: 0.841 vs 0.824 on U.S. and 0.741 vs 0.690 on crypto.

### Inference speed

Each time covers one market's test plan: 1,536 forecasts with 16 sampled paths each. GPU models ran on one NVIDIA RTX PRO 4000 Blackwell (24 GB). Times are end-to-end wall clock, including data assembly.

| Model | Params | Sampler | Hardware | Time | Per forecast | vs KiT-L |
|---|---|---|---|---|---|---|
| KiT-L | 285.8M | 32 Euler steps, bf16, `torch.compile` | GPU | 74–75 min | 2.9 s | 1× |
| **KiT-2** | **63.2M** | 8 Heun steps, bf16 | GPU | 13–15 s | ~9 ms | **~300×** |
| LightGBM | 8 tree models per market | — | CPU, 8 threads | 1.7–1.9 s (+ 33–41 s training) | ~1.2 ms | ~2,400× |
| Historical bootstrap | — | — | CPU | negligible | — | — |

## 🗂️ Layout

| Path | What it is |
|---|---|
| `kit2/data/` | Data pipeline: Alpha Vantage (U.S.) and HyperSync (on-chain) downloaders, point-in-time bars, the training store |
| `kit2/model/` | The KiT-2 model |
| `kit2/train/` | Lightning module, data module, bucket mixture, CLI |
| `kit2/eval/` | Evaluation plan, forecasts (KiT-2, KiT-L, LightGBM), scoring, joint-scenario, conditional and surprise scores |
| `kit2/bench/` | Training throughput benchmark |
| `kit2/tests/` | Tests, including no-look-ahead checks of the data pipeline |
| `configs/kit2/` | LightningCLI training configs |
| `ckpt/` | Local checkpoints (git-ignored), see `ckpt/README.md` |

## 🚀 Get started

### Setup

Python >= 3.10 and a CUDA GPU for training and sampling.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # adjust to your CUDA version
pip install -r requirements.txt
```

Secrets are read from the environment, else from the first `.env` file found: `$KIT_ENV`, `~/.config/kit/.env`, then `<repo>/.env`.

| Key | Needed for |
|---|---|
| `ALPHAVANTAGE_API_KEY` | U.S. download (premium key) |
| `ENVIO_API_TOKEN` | on-chain crypto download (HyperSync) |

The data tree lives under `$KIT2_DATA` (default `/workspace/kit2/data`). Prefer a local disk to a network volume: the pipeline writes many small files.

### Data

```bash
python -m kit2.data.us.download                    # resumable; see kit2/data/us/README.md
python -m kit2.data.crypto.download --chain eth    # one process per chain: eth, base, arbitrum
python -m kit2.data.us.prepare                     # build the training store under $KIT2_DATA/store/
python -m kit2.data.crypto.prepare
```

### Train

```bash
python -m kit2.train.cli fit --config configs/kit2/dev_us_crypto_mix.yaml
```

### Evaluate

Outputs go to `$KIT2_EVAL`. For the validation period, point `$KIT2_EVAL` at a separate tree and run `plan --split val`.

```bash
python -m kit2.eval.plan --market us
python -m kit2.eval.bars --market us
python -m kit2.eval.kit2pred --market us --ckpt <checkpoint> --tag kit2_mix
python -m kit2.eval.lgbm --market us                                   # LightGBM baseline
KIT_ROOT=<KiT checkout> python -m kit2.eval.kit --market us            # KiT-L baseline, optional
python -m kit2.eval.score --models kit2_mix,kit,lgbm
```

The KiT-L baseline needs a KiT v1 checkout with the `kit_v1` package and `ckpt/step_market_284M/`. This repo contained it up to commit `1113d69`.

### Test

```bash
python -m pytest kit2/tests
```

## Citation

KiT-2 builds on KiT. If you use it in your research, please cite:

```bibtex
@article{kit2026,
  title={KiT: A Foundation Model for Financial Time-Series Forecasting using Diffusion Transformers},
  author={Boyu Zhang, Haorui Li},
  journal={arXiv preprint arXiv:2609.34507},
  year={2026}
}
```
