# MAResNet-BiGRU Sleep Model

This folder contains a standalone MAResNet + BiGRU sleep-stage classifier for `MAResNet_BiGRU/test_new.csv`.

There is no optimizer-search path and no `config.yaml` dependency. The architecture and training defaults are defined directly in `train.py`, with only simple CLI overrides for runtime values such as epochs, batch size, and learning rate.

## Architecture

- Sliding windows over each `User_ID` and `Day` session.
- `StandardScaler` fitted only on the training split.
- Features include physiological, movement, environment, `GSR_uS`, `Audio_dB`, delta, and rolling-statistic signals.
- Multi-scale residual CNN blocks with kernels `3, 5, 7`.
- Sensor attention gate and channel attention for feature refinement.
- Temporal self-attention before a bidirectional GRU layer.
- Focal loss with class weights, label smoothing, dropout, and L2 regularization.

## Train

From the repository root:

```powershell
venv\Scripts\python.exe MAResNet_BiGRU\train.py --epochs 50
```

From inside this folder:

```powershell
python train.py --epochs 50
```

Useful overrides:

```powershell
python train.py --epochs 30 --batch_size 128 --learning_rate 0.0005
python train.py --smoke_test
```

The training script writes outputs to:

- `MAResNet_BiGRU/checkpoints/best_model.keras`
- `MAResNet_BiGRU/checkpoints/scaler.pkl`
- `MAResNet_BiGRU/logs/test_metrics.json`
- `MAResNet_BiGRU/logs/test_predictions.csv`
- `MAResNet_BiGRU/logs/test_probabilities.csv`
- all diagnostic plots in `MAResNet_BiGRU/logs/`

## Evaluate

Evaluate the default held-out test split:

```powershell
python evaluate.py --out_dir evaluation_results\test_split
```

Evaluate one user:

```powershell
python evaluate.py --user_id 2 --out_dir evaluation_results\user2
```

Evaluate a custom CSV:

```powershell
python evaluate.py --input test_new.csv --out_dir evaluation_results\custom
```

## Plots

Training and evaluation generate the same style of artifacts as the earlier pipeline:

- `confusion_matrix.png`
- `confusion_matrix_counts.png`
- `dashboard.png`
- `feature_correlation.png`
- `hypnogram_comparison.png`
- `metrics_radar.png`
- `metrics_summary_table.png`
- `roc_curves.png`
- `stage_probabilities.png`
- transition matrices, class distributions, predictions CSVs, probabilities CSVs, and metrics JSON.

## Note On Validation

The dataset has only five users. A user-held-out validation split can be noisy because one validation user may not represent the others well. The updated default model is intentionally smaller and more regularized than the previous version, and it no longer duplicates minority windows through oversampling, which should reduce the train-validation gap.
