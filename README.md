# Flipkart GriLock Pipeline

This repository contains the production pipeline for the Flipkart GriLock demand modeling workflow.

## Main Files

- `config.py` — pipeline configuration and paths.
- `pipeline.py` — the full refactored training and prediction pipeline.
- `run_pipeline.py` — entry point for running the pipeline.

## Usage

Run the pipeline from the repository root:

```bash
python run_pipeline.py
```

This will:

1. Load training data from `dataset/train.csv`
2. Load test data from `dataset/test.csv`
3. Execute the refactored pipeline logic
4. Save predictions under `submit/sample_submission_v9.csv`

## Notes

- Do not use `train_model_v2.py`, `train_model_v3.py`, or `train_model_v9.py` for the current pipeline workflow.
- The current pipeline is implemented in `pipeline.py` and executed via `run_pipeline.py`.
- `config.py` contains the file paths and pipeline settings.

## Requirements

Make sure the required Python packages are installed, including:

- `pandas`
- `numpy`
- `lightgbm`
- `xgboost`
- `catboost`
- `optuna`
- `scikit-learn`

Install via pip if needed:

```bash
pip install pandas numpy lightgbm xgboost catboost optuna scikit-learn
```
