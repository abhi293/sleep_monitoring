# Sleep Monitoring Project Documentation

## Project Overview
This project implements a sleep monitoring system using machine learning to analyze sleep patterns and detect anomalies. The system processes sleep data through preprocessing, model training, and evaluation pipelines.

## Workspace Structure
```
DOCUMENTATION.md        # Detailed technical documentation
README.md               # Project overview (this file)
evaluate.py             # Model evaluation script
requirements.txt        # Python dependencies
train.py                # Model training script
__pycache__/            # Python cache files
checkpoints/            # Model checkpoints
  best_model.keras      # Best trained model
configs/                # Configuration files
  model_config.yaml     # Model hyperparameters
mopso_results/          # Multi-objective optimization results
old_dataset/            # Raw dataset files
  realistic_sleep_dataset_v3.csv
src/                    # Source code
  model.py              # Model architecture
  mopso.py              # Multi-objective optimization
  preprocessing.py      # Data preprocessing
  utils.py              # Utility functions
```

## Setup Instructions
1. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
2. **Dataset preparation**: 
   - Place CSV files in `old_dataset/`
   - Preprocessed data will be saved in `processed_data/`
3. **Model configuration**: 
   - Adjust hyperparameters in `configs/model_config.yaml`

## Training & Evaluation
- **Training**: Run `python train.py`
- **Evaluation**: Run `python evaluate.py`
- **Results**: 
  - Evaluation metrics: `evaluation_results/mixed_dataset/user2/`
  - Training logs: `logs/training_log.csv`

## Technical Details
- **Model architecture**: LSTM-based neural network (see `src/model.py`)
- **Optimization**: Multi-objective PSO algorithm (see `src/mopso.py`)
- **Data pipeline**: 
  1. Raw data → `preprocessing.py` → Cleaned data
  2. Cleaned data → `train.py` → Model training

## License
MIT License - see LICENSE file