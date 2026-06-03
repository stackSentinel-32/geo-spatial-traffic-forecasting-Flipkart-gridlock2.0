# run_pipeline.py

import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import modular layers from sibling scripts
import config
from pipeline import base_preprocessing, fit_historical_lookups, apply_historical_lookups

def main():
    print("====================================================")
    print("STARTING TRAFFIC DEMAND FORECASTING PIPELINE")
    print("====================================================\n")
    
    # --------------------------------------------------
    # PHASE 1: LOAD AND INITIALIZE TRAINING SET
    # --------------------------------------------------
    if not os.path.exists(config.TRAIN_PATH):
        raise FileNotFoundError(f"Training dataset missing at path: '{config.TRAIN_PATH}'")
        
    print(f"[1/5] Loading primary training log: {config.TRAIN_PATH} ...")
    raw_train = pd.read_csv(config.TRAIN_PATH)
    
    # Run pipeline base parsing transformations
    preprocessed_train = base_preprocessing(raw_train)
    
    # Compute lookup maps exclusively on training constraints
    print("[2/5] Compiling historical baseline anchors on Day 48 constraints...")
    lookups = fit_historical_lookups(preprocessed_train)
    
    # Map lookups back to complete train frame
    final_train_df = apply_historical_lookups(preprocessed_train, lookups)
    
    # --------------------------------------------------
    # PHASE 2: TIME-BASED VALIDATION SPLIT
    # --------------------------------------------------
    # Train on complete Day 48 history -> Validate against current Day 49 slices
    X_train = final_train_df[final_train_df['day'] == 48][config.FEATURES]
    y_train = final_train_df[final_train_df['day'] == 48][config.TARGET]
    
    X_val = final_train_df[final_train_df['day'] == 49][config.FEATURES]
    y_val = final_train_df[final_train_df['day'] == 49][config.TARGET]
    
    print(f" -> Training footprint dimension   : {X_train.shape[0]} samples (Day 48)")
    print(f" -> Validation footprint dimension : {X_val.shape[0]} samples (Day 49 training subset)")
    
    train_pool = Pool(X_train, y_train, cat_features=config.CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=config.CAT_FEATURES)
    
    # --------------------------------------------------
    # PHASE 3: MODEL CONFIGURATION AND TRAINING
    # --------------------------------------------------
    print("\n[3/5] Initializing Regularized CatBoost Core and starting optimization iterations...")
    model = CatBoostRegressor(**config.CATBOOST_PARAMS)
    
    model.fit(
        train_pool, 
        eval_set=val_pool, 
        early_stopping_rounds=100, 
        use_best_model=True
    )
    
    # --------------------------------------------------
    # PHASE 4: ACCURACY RATE VERIFICATION & DIAGNOSTICS
    # --------------------------------------------------
    print("\n[4/5] Running validation engine calculations...")
    val_predictions = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    mae = mean_absolute_error(y_val, val_predictions)
    r2 = r2_score(y_val, val_predictions)
    
    print("\n" + "="*45)
    print("       LOCAL VALIDATION ACCURACY RATE       ")
    print("="*45)
    print(f" RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f" MAE  (Mean Absolute Error):     {mae:.6f}")
    print(f" R2 Score (Variance Explained):  {r2:.3%}")
    print("="*45 + "\n")
    
    # --------------------------------------------------
    # PHASE 5: RUN PRODUCTION CONSTRAINTS ON SEPARATE TEST FILE
    # --------------------------------------------------
    if not os.path.exists(config.TEST_PATH):
        print(f"[⚠️ WARNING] Separate Test file path not found at '{config.TEST_PATH}'. Skipping predictions export step.")
        print("Please ensure your test target matches the location file to generate your submission.")
        return
        
    print(f"[5/5] Ingesting production separate test target: {config.TEST_PATH} ...")
    raw_test = pd.read_csv(config.TEST_PATH)
    
    # Format structural design matches
    preprocessed_test = base_preprocessing(raw_test)
    final_test_df = apply_historical_lookups(preprocessed_test, lookups)
    
    # Isolate test input features matrix
    X_test = final_test_df[config.FEATURES]
    
    print(f" -> Testing footprint dimension   : {X_test.shape[0]} rows of Day 49 test data")
    print(" -> Generating out-of-sample predictions...")
    
    # Generate final out-of-sample predictions
    raw_test[config.TARGET] = model.predict(X_test)
    
    # Retain the exact required columns for clean submission output mapping
    if 'Index' in raw_test.columns:
        submission_output = raw_test[['Index', config.TARGET]]
    else:
        submission_output = raw_test[['geohash', 'timestamp', config.TARGET]]
        
    submission_output.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"\n[SUCCESS] Production submission file saved to: '{config.SUBMISSION_PATH}'")
    print("====================================================")

if __name__ == "__main__":
    main()