# CLI Control Guide for Training & Ablation Studies

## Overview
This guide explains how to use the new CLI controls to run clean ablation studies and MOPSO optimizations without data leakage.



## Three Operating Modes

### Mode 1: Ablation Studies (Recommended for Clean Results)
**Use case**: Testing model architecture components (CNN, GRU, LSTM)

```bash
# Default ablation (all components enabled)
python train.py

# Disable specific components
python train.py --without_gru              # Only CNN + LSTM
python train.py --without_lstm             # Only CNN + GRU
python train.py --without_cnn              # Only GRU + LSTM
python train.py --with_cnn --without_gru   # Only CNN + LSTM (explicit)

# With custom data windowing (be careful about data leakage!)
python train.py --without_gru --window_size 25 --stride 5
```

**Characteristics**:
- ✓ Uses **ablation-friendly defaults**: window_size=30, stride=10
- ✓ Uses **built-in model/training defaults** (not YAML)
- ✓ Minimizes data leakage
- ✓ All results directly comparable

**Behavior**:
```
Config mode: Ablation defaults
Data windowing: window_size=30 stride=10
Using built-in model/training defaults
```

---

### Mode 2: YAML Configuration (For Fine-Tuned Hyperparameters)
**Use case**: Running training with custom hyperparameters from `model_config.yaml`

```bash
# Use all hyperparameters from YAML
python train.py --use_config_model

# Use YAML config + specific architecture ablation
python train.py --use_config_model --without_gru
python train.py --use_config_model --window_size 25  # CLI override window_size
```

**Characteristics**:
- ✓ Loads model AND data hyperparameters from `model_config.yaml`
- ✓ Allows fine-tuned settings
- ✓ Data windowing defaults to YAML values (window_size=20, stride=3)
- ⚠️ May create more overlapping windows → more data leakage

**Behavior**:
```
Config mode: YAML config
Data windowing: window_size=20 stride=3 (from YAML)
Using model/training hyperparameters from configs/model_config.yaml
```

---

### Mode 3: MOPSO Optimization (Hyperparameter Search)
**Use case**: Automatically find best hyperparameters via multi-objective optimization

```bash
# Run MOPSO optimization
python train.py --mopso

# With custom MOPSO parameters
python train.py --mopso --mopso_iter 20 --mopso_particles 15

# Load previous MOPSO results instead of re-running
python train.py --mopso --load_mopso

# MOPSO + specific architecture
python train.py --mopso --without_lstm
```

**Characteristics**:
- ✓ Ignores `--use_config_model` (MOPSO generates its own config)
- ✓ Optimizes across: CNN filters, GRU/LSTM units, dropout, learning_rate, window_size
- ✓ Multi-objective: accuracy vs. false_alarm_rate vs. model_efficiency
- ✓ Outputs Pareto archive of non-dominated solutions

**Behavior**:
```
Config mode: MOPSO optimization
Data windowing: window_size=20 stride=3 (MOPSO default during optimization)
Will run optimization and use MOPSO-selected hyperparameters
```

---

## Common Workflows

### Workflow A: Clean Ablation Study
```bash
# Test full model (reference)
python train.py
# Results: logs/ablation_cnn_gru_lstm/test_metrics.json

# Test without GRU
python train.py --without_gru
# Results: logs/ablation_cnn_lstm/test_metrics.json

# Test without LSTM
python train.py --without_lstm
# Results: logs/ablation_cnn_gru/test_metrics.json

# Test without CNN
python train.py --without_cnn
# Results: logs/ablation_gru_lstm/test_metrics.json
```
✓ All runs use same data windowing (30, 10) → comparable results

### Workflow B: MOPSO Optimization → Ablation with MOPSO Config
```bash
# Step 1: Run MOPSO to find best hyperparameters
python train.py --mopso --mopso_iter 15
# Results: logs/mopso_cnn_gru_lstm/, checkpoints/mopso_cnn_gru_lstm/

# Step 2: Use those hyperparameters for subsequent runs
python train.py --use_config_model --without_gru
python train.py --use_config_model --without_lstm
```
✓ All ablations now use MOPSO-optimized hyperparameters

### Workflow C: Hybrid Ablation (Ablation defaults + Custom Data Windowing)
```bash
# Test different window sizes while keeping model architecture
python train.py --window_size 20 --stride 5 --without_gru
python train.py --window_size 25 --stride 8 --without_gru
python train.py --window_size 30 --stride 10 --without_gru
```
⚠️ Different windowing = different number of samples, not directly comparable

---

## Parameter Priority & Precedence

When multiple sources specify the same parameter:

```
CLI args > Ablation/MOPSO logic > YAML config > Built-in defaults
```

Example:
```bash
# Window size resolution:
python train.py --window_size 25  # CLI arg wins (25)
# Resolves to: 25

python train.py --use_config_model  # YAML config applies
# Resolves to: 20 (from model_config.yaml)

python train.py  # Ablation default applies
# Resolves to: 30 (ablation-friendly default)
```

---

## Troubleshooting

### Issue: Ablation results are still suspiciously high
**Cause**: Using `--use_config_model` without realizing it changes data windowing
**Fix**: Check the logs for "Config mode:" line. Should say "Ablation defaults" for clean ablations.

### Issue: MOPSO and --use_config_model seem contradictory
**Expected behavior**: `--mopso` ignores `--use_config_model` because MOPSO generates its own config

### Issue: Different runs have different data shapes
**Likely cause**: Different windowing parameters (window_size / stride)
**Fix**: Make sure ablations use same window_size and stride

---

## Verification

To verify your setup is correct, check the log output:

```
Config mode: Ablation defaults  ← This should appear for clean ablations
Data windowing: window_size=30 stride=10
Using built-in model/training defaults
```

If you see different values, check your CLI arguments and compare metrics carefully.
