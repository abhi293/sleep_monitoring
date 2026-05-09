# CNN + Optional BiLSTM + Transformer Sleep Model

This directory contains a standalone model family for `CNN_BiLSTM_Transformer/test_new.csv`.

There is no `config.yaml`, no MOPSO, and no external optimizer-selected architecture. Defaults are defined directly in code.

## Variants

- `CNN+BiLSTM+Transformer`: default. Uses CNN feature extraction, BiLSTM temporal modeling, then Transformer attention.
- `CNN+Transformer`: disables the BiLSTM branch and sends CNN features directly into the Transformer.

Outputs are separated by variant so runs do not overwrite each other:

- `logs/cnn_bilstm_transformer/`
- `logs/cnn_transformer/`
- `checkpoints/cnn_bilstm_transformer/`
- `checkpoints/cnn_transformer/`

## Train

From inside this directory:

```powershell
python -u train.py --epochs 50 --with_bilstm
python -u train.py --epochs 50 --without_bilstm
```

From the repository root:

```powershell
venv\Scripts\python.exe CNN_BiLSTM_Transformer\train.py --epochs 50 --with_bilstm
venv\Scripts\python.exe CNN_BiLSTM_Transformer\train.py --epochs 50 --without_bilstm
```

Useful quick checks:

```powershell
python -u train.py --smoke_test --with_bilstm
python -u train.py --smoke_test --without_bilstm
```

Cleaner anti-overfit baseline:

```powershell
python -u train.py --epochs 20 --without_bilstm --loss ce --stride 10 --batch_size 128 --learning_rate 0.0003
```

Class weights and focal loss are opt-in because this dataset is only moderately imbalanced and the combination can push the model to over-predict minority stages:

```powershell
python -u train.py --epochs 20 --with_bilstm --loss focal --class_weights
```

## Evaluate

Default held-out test split:

```powershell
python evaluate.py --with_bilstm
python evaluate.py --without_bilstm
```

One user:

```powershell
python evaluate.py --with_bilstm --user_id 2 --out_dir evaluation_results\user2_bilstm
python evaluate.py --without_bilstm --user_id 2 --out_dir evaluation_results\user2_no_bilstm
```

## Generated Artifacts

Each training/evaluation run creates the same plot suite as the earlier models:

- `confusion_matrix.png`
- `confusion_matrix_counts.png`
- `dashboard.png`
- `feature_correlation.png`
- `hypnogram_comparison.png`
- `metrics_radar.png`
- `metrics_summary_table.png`
- `roc_curves.png`
- `stage_probabilities.png`
- transition matrices
- class distributions
- predictions CSV
- probabilities CSV
- metrics JSON
