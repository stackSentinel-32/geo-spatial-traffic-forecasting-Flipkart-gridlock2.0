# Flipkart GriLock 2.0 - ULTRA Traffic Forecasting Pipeline (v3)

Welcome to the **Flipkart GriLock 2.0 Traffic Forecasting Pipeline (v3)** repository. This pipeline is engineered to achieve a top-tier score on traffic forecasting, leveraging advanced spatial-temporal feature engineering, hyperparameter optimization (HPO), model blending, and semi-supervised pseudo-labeling.

---

## 🚀 Key Features & Techniques

The pipeline implements several state-of-the-art machine learning techniques:

1. **Temporal Lag Demand Features (105 Lag Steps)**: Evaluates demand from 15-minute, 1-day, and 1-day + 15-minute intervals at geohash, `geo4`, and `geo3` resolution levels to capture localized traffic velocity trends.
2. **Spatial KNN Neighbor Features (K=8)**: Locates the 8 nearest spatial neighbors using Haversine distances to extract neighboring demand statistics (mean, max, std, distance).
3. **Yeo-Johnson Target Transform**: Corrects highly skewed demand distributions (skewness corrected from `2.052` down to `0.177`) to stabilize regression target training.
4. **Day-Recency Sample Weighting**: Places a `2.5x` weight multiplier on the most recent training days (e.g., `day=49`) to align prediction bias with the test set's temporal distribution.
5. **KFold-Safe Target Encodings**: Incorporates 14 distinct smoothed target encodings (e.g., `geohash`, `geo4_hour`, `RoadType_Weather`, `RoadType_is_peak`) with out-of-fold leakage prevention.
6. **Robust Feature Engineering**: Computes capacity scores, vehicle pressure indices, weather temperature z-scores, temporal sine/cosine oscillations, and interactive products of latitude/longitude.
7. **Optuna HPO & GPU Support**: Runs 50 Optuna trials for LightGBM (CPU-optimized) and 30 Optuna trials for CatBoost utilizing the local WDDM GPU (NVIDIA RTX 4050 Laptop GPU) for rapid training.
8. **Nelder-Mead Optimal Blend**: Employs non-negative constrained optimization to blend predictions from LightGBM, CatBoost, and HistGradientBoosting.
9. **Semi-Supervised Pseudo-Labeling**: Incorporates confident test predictions into a combined training set at a `0.4x` pseudo-label weight, retraining a final robust CatBoost model.

---

## 📈 Performance & Accuracy Scores

The latest end-to-end evaluation run (completed in **49.8 minutes**) yielded the following out-of-fold (OOF) results:

| Model | OOF RMSE | OOF $R^2$ Score | Projected Leaderboard Score | Execution Context |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM** (HPO + Weighted) | `0.04474` | `0.9010` | **`90.10`** | 50 Optuna trials (CPU) |
| **CatBoost** (HPO + Weighted) | `0.03611` | `0.9355` | **`93.55`** | 30 Optuna trials (GPU) |
| **HistGradientBoosting** (sklearn)| `0.04598` | `0.8954` | **`89.54`** | 10-Fold CV (CPU) |
| **Optuna Blend** (Final Ensemble) | `0.03611` | `0.9355` | **`93.55`** | Blend Weight: 100% CatBoost |
| **Pseudo-Labeled CatBoost** | `0.02539` | *(Train)* | — | Combined Train + Pseudo-test |

### Key Takeaways for Recruiters
*   **Leaderboard Metric Match**: In the Flipkart GriLock competition, the online score is evaluated directly as $100 \times R^2$. The final model achieves a **projected online score of `93.55`**, matching the highly competitive target tier of `93-94`.
*   **Significant Baseline Gains**: The final blend represents a **`+5.1%`** reduction in RMSE over the initial baseline model and a **`+1.0%`** boost over the previous stacking iteration.
*   **Hardware Acceleration & Efficiency**: By transitioning CatBoost to WDDM GPU acceleration, training times were drastically optimized, enabling deep tuning (Optuna HPO) across a large feature space of 79 dimensions in under 50 minutes.

---

## 📂 Codebase Structure

The pipeline is split into three clean, modular scripts:

*   **[`config.py`](file:///C:/Projects/Flipkart_grilock/config.py)**: Contains all global hyperparameters, paths, tuning trials, validation folds, and GPU device execution parameters.
*   **[`pipeline.py`](file:///C:/Projects/Flipkart_grilock/pipeline.py)**: Encompasses data loading, coordinate decoding, spatial-temporal feature extraction, Target Encoding, Optuna HPO objectives, CV loops, and blending methods.
*   **[`run_pipeline.py`](file:///C:/Projects/Flipkart_grilock/run_pipeline.py)**: The entry point wrapper which boots the entire pipeline sequence.

---

## ⚙️ How to Run

1. **Verify Requirements**:
   Ensure you have all dependencies installed:
   ```bash
   pip install pandas numpy lightgbm catboost optuna scikit-learn scipy
   ```

2. **Execute the Pipeline**:
   Run the following command from the repository root directory:
   ```bash
   python run_pipeline.py
   ```

This will run the entire workflow end-to-end, printing step-by-step logs, training the HPO models (using WDDM/CUDA GPU for CatBoost), optimizing the blend, performing pseudo-labeling, and exporting the final predictions to the designated path:
`submit/sample_submission_v3.csv`.
