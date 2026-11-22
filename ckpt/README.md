# ckpt/

Local KiT-2 checkpoints (contents are git-ignored). KiT-2 checkpoints are not released yet.

```
ckpt/kit2/<run>/<step>-<val_loss>.ckpt   Lightning checkpoint
ckpt/kit2/<run>/config.yaml              LightningCLI config of the run
ckpt/kit2/<run>/hparams.yaml, metrics.csv
ckpt/kit2/MANIFEST.md                    runs, eval tags, md5 sums, where the other checkpoints are archived
```

Pass a checkpoint with `--ckpt` to `kit2.eval.kit2pred`, `kit2.eval.surprise` or `kit2.eval.probe`.
