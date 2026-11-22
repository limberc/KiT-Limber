# ckpt/

Put the model checkpoint here (contents are git-ignored). `huggingface-cli download Lucifer744/KiT_model --local-dir ckpt/`
([KiT_model](https://huggingface.co/Lucifer744/KiT_model)) puts the default checkpoint at:

```
ckpt/step_market_284M/step_market_284M.pt     KiT-L (283.7M params), EMA weights included
ckpt/step_market_284M/instruments.json
ckpt/step_market_284M/stats.json
```

`eval/infer_for_eval.py` and `eval/infer_for_pred.py` default to this path; pass `--ckpt` for another checkpoint.
