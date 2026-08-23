# Pure-bulk baseline

The comparison arm for every cell of the main table is
`survival_prediction/bulk_mlp_survival_cv.py` — a bulk-only Cox / DeepSurv / DeepHit MLP
trained on exactly the same outer folds, the same per-fold DEG gene sets, and the same
inner fit/validation split as the fusion model.

## Command

All eight cancers were run with one identical hyperparameter set (recovered from the
`args` block of each run's `cv_summary.json`):

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

for CANCER in BRCA COAD HNSC KIRC LGG LIHC LUAD STAD; do
  python survival_prediction/bulk_mlp_survival_cv.py \
    --cancer "$CANCER" \
    --config config/path_seetacloud_survival.json \
    --models cox deepsurv deephit \
    --folds 1 2 3 4 5 \
    --epochs 200 \
    --early_stop_fraction 0.2 \
    --selection_smoothing_window 1 \
    --early_stop_patience 20 \
    --fixed_config \
    --seed 3407 \
    --device cuda \
    --results_dir output/bulk_mlp_survival_cv/8cancer_fixed_w1/"$CANCER"
done
```

Two of these differ from the script's own argparse defaults and must be passed explicitly:
`--selection_smoothing_window 1` (default 5) and `--early_stop_patience 20` (default 30).
`--fixed_config` takes the first pre-specified candidate per model family instead of
running an inner hyperparameter search.

`resolve_config` reads `bulk`, `surv_label`, `deg_dir`, `gene_list`, and `gene_list_path`
from the chosen config under the requested cancer, resolving relative paths against the
repository root.

## Head matching

Compare like with like — a fusion head against the baseline head that shares its loss:

| baseline `--models` | fusion flag |
|---|---|
| `cox` | `--direct_cox_from_fusion` |
| `deepsurv` | default head (no head flag) |
| `deephit` | `--loss_fn deephit` |

## Provenance caveat

The published baseline numbers were produced with a config whose `bulk` entries pointed at
a session-local staging directory of symlinks into machine-specific paths, not at
`data/{CANCER}/bulk` in this repository. The hyperparameters above are exact; the data
paths in the command are the repository-relative equivalents. Point the config at the same
bulk matrices and survival labels the fusion runs used, and the two arms stay comparable.
