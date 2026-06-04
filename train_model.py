import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
"""
=============================================================================
  Flipkart GriLock 2.0 - Traffic Demand Prediction
  Full Pipeline: Feature Engineering + LightGBM + XGBoost + Stacking
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import time

print("=" * 65)
print("  FLIPKART GRILOCK 2.0 — TRAFFIC DEMAND PREDICTION PIPELINE")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading data...")
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

print(f"      Train shape : {train.shape}")
print(f"      Test shape  : {test.shape}")
print(f"      Target range: [{train['demand'].min():.4f}, {train['demand'].max():.4f}]")
print(f"      Target mean : {train['demand'].mean():.4f}")
print(f"      Target std  : {train['demand'].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUE AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Missing value audit...")
for col in ['RoadType', 'Temperature', 'Weather']:
    tr_miss = train[col].isna().sum()
    te_miss = test[col].isna().sum()
    print(f"      {col:<18}: train={tr_miss} ({tr_miss/len(train)*100:.1f}%)  test={te_miss} ({te_miss/len(test)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Engineering features...")

def parse_timestamp(ts):
    """Convert 'H:MM' → minutes since midnight"""
    try:
        parts = str(ts).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return np.nan

def engineer_features(df):
    df = df.copy()

    # ── Timestamp features ──────────────────────────────────────────────────
    df['minutes']      = df['timestamp'].apply(parse_timestamp)
    df['hour']         = (df['minutes'] / 60).astype(float)
    df['time_bucket']  = pd.cut(df['hour'],
                                bins=[-1, 5, 9, 12, 16, 20, 25],
                                labels=[0, 1, 2, 3, 4, 5]).astype(float)
    df['is_peak']      = df['hour'].apply(lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 20) else 0)
    df['is_night']     = df['hour'].apply(lambda h: 1 if (h < 5 or h > 23) else 0)

    # ── Geohash features ────────────────────────────────────────────────────
    df['geo3']  = df['geohash'].str[:3]
    df['geo4']  = df['geohash'].str[:4]
    df['geo5']  = df['geohash'].str[:5]

    # ── Binary encoding ─────────────────────────────────────────────────────
    df['LargeVehicles_num'] = (df['LargeVehicles'] == 'Allowed').astype(float)
    df['Landmarks_num']     = (df['Landmarks'] == 'Yes').astype(float)

    return df

train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. IMPUTATION (must use train stats only, then apply to test)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Imputing missing values...")

# ── RoadType: mode per geo4 from train ─────────────────────────────────────
geo4_roadtype_mode = (train.dropna(subset=['RoadType'])
                          .groupby('geo4')['RoadType']
                          .agg(lambda x: x.mode()[0]))
global_rt_mode = train['RoadType'].mode()[0]

def impute_roadtype(df, mode_map, fallback):
    df = df.copy()
    mask = df['RoadType'].isna()
    df.loc[mask, 'RoadType'] = df.loc[mask, 'geo4'].map(mode_map)
    df['RoadType'].fillna(fallback, inplace=True)
    return df

train = impute_roadtype(train, geo4_roadtype_mode, global_rt_mode)
test  = impute_roadtype(test,  geo4_roadtype_mode, global_rt_mode)

# ── Temperature: median per (Weather, day) ──────────────────────────────────
temp_map = (train.dropna(subset=['Temperature'])
                 .groupby(['Weather', 'day'])['Temperature']
                 .median())
global_temp_median = train['Temperature'].median()

def impute_temperature(df, t_map, fallback):
    df = df.copy()
    mask = df['Temperature'].isna()
    df.loc[mask, 'Temperature'] = df.loc[mask].apply(
        lambda r: t_map.get((r['Weather'], r['day']), np.nan), axis=1)
    df['Temperature'].fillna(fallback, inplace=True)
    return df

train = impute_temperature(train, temp_map, global_temp_median)
test  = impute_temperature(test,  temp_map, global_temp_median)

# ── Weather: mode per geo4 ──────────────────────────────────────────────────
geo4_weather_mode = (train.dropna(subset=['Weather'])
                          .groupby('geo4')['Weather']
                          .agg(lambda x: x.mode()[0]))
global_wx_mode = train['Weather'].mode()[0]

def impute_weather(df, mode_map, fallback):
    df = df.copy()
    mask = df['Weather'].isna()
    df.loc[mask, 'Weather'] = df.loc[mask, 'geo4'].map(mode_map)
    df['Weather'].fillna(fallback, inplace=True)
    return df

train = impute_weather(train, geo4_weather_mode, global_wx_mode)
test  = impute_weather(test,  geo4_weather_mode, global_wx_mode)

print(f"      Remaining NaN in train after imputation: {train[['RoadType','Temperature','Weather']].isna().sum().sum()}")
print(f"      Remaining NaN in test  after imputation: {test[['RoadType','Temperature','Weather']].isna().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TARGET ENCODING (from TRAIN only — prevents data leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Building target encodings...")

# Mean demand per geohash (high-resolution location signal)
geo_demand_mean = train.groupby('geohash')['demand'].mean()
geo3_demand_mean = train.groupby('geo3')['demand'].mean()
geo4_demand_mean = train.groupby('geo4')['demand'].mean()
geo5_demand_mean = train.groupby('geo5')['demand'].mean()

roadtype_demand_mean    = train.groupby('RoadType')['demand'].mean()
weather_demand_mean     = train.groupby('Weather')['demand'].mean()
lanes_demand_mean       = train.groupby('NumberofLanes')['demand'].mean()
geo4_hour_demand_mean   = train.groupby(['geo4', 'hour'])['demand'].mean()
roadtype_lanes_mean     = train.groupby(['RoadType', 'NumberofLanes'])['demand'].mean()

global_mean = train['demand'].mean()

def apply_target_encodings(df):
    df = df.copy()
    df['te_geohash']     = df['geohash'].map(geo_demand_mean).fillna(global_mean)
    df['te_geo3']        = df['geo3'].map(geo3_demand_mean).fillna(global_mean)
    df['te_geo4']        = df['geo4'].map(geo4_demand_mean).fillna(global_mean)
    df['te_geo5']        = df['geo5'].map(geo5_demand_mean).fillna(global_mean)
    df['te_roadtype']    = df['RoadType'].map(roadtype_demand_mean).fillna(global_mean)
    df['te_weather']     = df['Weather'].map(weather_demand_mean).fillna(global_mean)
    df['te_lanes']       = df['NumberofLanes'].map(lanes_demand_mean).fillna(global_mean)
    df['te_geo4_hour']   = df.set_index(['geo4', 'hour']).index.map(geo4_hour_demand_mean).values
    df['te_geo4_hour']   = pd.to_numeric(df['te_geo4_hour'], errors='coerce').fillna(global_mean)
    df['te_rt_lanes']    = df.set_index(['RoadType', 'NumberofLanes']).index.map(roadtype_lanes_mean).values
    df['te_rt_lanes']    = pd.to_numeric(df['te_rt_lanes'], errors='coerce').fillna(global_mean)
    return df

train = apply_target_encodings(train)
test  = apply_target_encodings(test)

# ── Label encode categoricals ─────────────────────────────────────────────
le_road    = LabelEncoder().fit(pd.concat([train['RoadType'], test['RoadType']]))
le_weather = LabelEncoder().fit(pd.concat([train['Weather'],  test['Weather']]))
le_geo3    = LabelEncoder().fit(pd.concat([train['geo3'],     test['geo3']]))
le_geo4    = LabelEncoder().fit(pd.concat([train['geo4'],     test['geo4']]))

for df in [train, test]:
    df['RoadType_enc'] = le_road.transform(df['RoadType'])
    df['Weather_enc']  = le_weather.transform(df['Weather'])
    df['geo3_enc']     = le_geo3.transform(df['geo3'])
    df['geo4_enc']     = le_geo4.transform(df['geo4'])

# ── Interaction features ──────────────────────────────────────────────────
for df in [train, test]:
    df['lanes_x_roadtype']    = df['NumberofLanes'] * df['RoadType_enc']
    df['temp_x_weather']      = df['Temperature'] * df['Weather_enc']
    df['large_x_lanes']       = df['LargeVehicles_num'] * df['NumberofLanes']
    df['landmark_x_roadtype'] = df['Landmarks_num'] * df['RoadType_enc']
    df['hour_x_roadtype']     = df['hour'] * df['RoadType_enc']

print("      Feature engineering complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL TRAINING — 5-Fold Cross-Validation + Out-of-Fold Stacking
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'day', 'minutes', 'hour', 'time_bucket', 'is_peak', 'is_night',
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_num', 'Landmarks_num',
    'Temperature', 'Weather_enc',
    'geo3_enc', 'geo4_enc',
    'te_geohash', 'te_geo3', 'te_geo4', 'te_geo5',
    'te_roadtype', 'te_weather', 'te_lanes',
    'te_geo4_hour', 'te_rt_lanes',
    'lanes_x_roadtype', 'temp_x_weather', 'large_x_lanes',
    'landmark_x_roadtype', 'hour_x_roadtype'
]

X = train[FEATURES].values
y = train['demand'].values
X_test = test[FEATURES].values

print(f"\n[6/7] Training models on {len(FEATURES)} features with 5-Fold CV...")
print(f"      Training samples : {len(X)}")
print(f"      Test samples     : {len(X_test)}")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage for OOF and test predictions
lgb_oof   = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))
xgb_oof   = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

lgb_scores = []
xgb_scores = []

# ── LightGBM parameters ──────────────────────────────────────────────────
lgb_params = {
    'objective':        'regression',
    'metric':           'rmse',
    'n_estimators':     3000,
    'learning_rate':    0.03,
    'num_leaves':       127,
    'max_depth':        -1,
    'min_child_samples': 20,
    'subsample':         0.8,
    'subsample_freq':    1,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,
    'reg_lambda':        0.2,
    'random_state':      42,
    'n_jobs':           -1,
    'verbose':          -1,
}

# ── XGBoost parameters ────────────────────────────────────────────────────
xgb_params = {
    'objective':            'reg:squarederror',
    'eval_metric':          'rmse',
    'n_estimators':         3000,
    'learning_rate':        0.03,
    'max_depth':            7,
    'min_child_weight':     5,
    'subsample':            0.8,
    'colsample_bytree':     0.8,
    'gamma':                0.1,
    'reg_alpha':            0.1,
    'reg_lambda':           0.2,
    'early_stopping_rounds': 200,   # passed in constructor for XGBoost >= 1.6
    'random_state':         42,
    'n_jobs':              -1,
    'verbosity':            0,
}

print("\n  +-------------------------------------------------------------+")
print("  |                  LightGBM 5-Fold CV                        |")
print("  +-------------------------------------------------------------+")

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )

    val_pred = model_lgb.predict(X_val)
    lgb_oof[val_idx] = val_pred
    lgb_preds += model_lgb.predict(X_test) / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    fold_r2   = r2_score(y_val, val_pred)
    lgb_scores.append(fold_rmse)
    best_iter = model_lgb.best_iteration_
    print(f"  Fold {fold}/5 → RMSE: {fold_rmse:.5f}  R²: {fold_r2:.4f}  Best iter: {best_iter}")

lgb_time = time.time() - t0
lgb_oof_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
lgb_oof_mae  = mean_absolute_error(y, lgb_oof)
lgb_oof_r2   = r2_score(y, lgb_oof)

print(f"\n  ► LightGBM OOF RMSE : {lgb_oof_rmse:.5f}")
print(f"  ► LightGBM OOF MAE  : {lgb_oof_mae:.5f}")
print(f"  ► LightGBM OOF R²   : {lgb_oof_r2:.4f}")
print(f"  ► Time taken        : {lgb_time:.1f}s")

print("\n  +-------------------------------------------------------------+")
print("  |                   XGBoost 5-Fold CV                        |")
print("  +-------------------------------------------------------------+")

t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred = model_xgb.predict(X_val)
    xgb_oof[val_idx] = val_pred
    xgb_preds += model_xgb.predict(X_test) / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    fold_r2   = r2_score(y_val, val_pred)
    xgb_scores.append(fold_rmse)
    best_iter = model_xgb.best_iteration
    print(f"  Fold {fold}/5 → RMSE: {fold_rmse:.5f}  R²: {fold_r2:.4f}  Best iter: {best_iter}")

xgb_time = time.time() - t0
xgb_oof_rmse = np.sqrt(mean_squared_error(y, xgb_oof))
xgb_oof_mae  = mean_absolute_error(y, xgb_oof)
xgb_oof_r2   = r2_score(y, xgb_oof)

print(f"\n  ► XGBoost OOF RMSE  : {xgb_oof_rmse:.5f}")
print(f"  ► XGBoost OOF MAE   : {xgb_oof_mae:.5f}")
print(f"  ► XGBoost OOF R²    : {xgb_oof_r2:.4f}")
print(f"  ► Time taken        : {xgb_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Blended Ensemble (weighted average based on OOF RMSE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  +-------------------------------------------------------------+")
print("  |               Ensemble (Weighted Blend)                     |")
print("  +-------------------------------------------------------------+")

# Lower RMSE → higher weight
lgb_weight = 1.0 / lgb_oof_rmse
xgb_weight = 1.0 / xgb_oof_rmse
total_w    = lgb_weight + xgb_weight
lgb_w = lgb_weight / total_w
xgb_w = xgb_weight / total_w

print(f"  LightGBM weight  : {lgb_w:.3f}")
print(f"  XGBoost  weight  : {xgb_w:.3f}")

blend_oof   = lgb_w * lgb_oof   + xgb_w * xgb_oof
blend_preds = lgb_w * lgb_preds + xgb_w * xgb_preds

blend_oof_rmse = np.sqrt(mean_squared_error(y, blend_oof))
blend_oof_mae  = mean_absolute_error(y, blend_oof)
blend_oof_r2   = r2_score(y, blend_oof)

print(f"\n  ► Ensemble OOF RMSE : {blend_oof_rmse:.5f}")
print(f"  ► Ensemble OOF MAE  : {blend_oof_mae:.5f}")
print(f"  ► Ensemble OOF R²   : {blend_oof_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. GENERATE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Generating submission file...")

# Choose best model
if blend_oof_rmse <= min(lgb_oof_rmse, xgb_oof_rmse):
    final_preds = blend_preds
    chosen = "Ensemble (LightGBM + XGBoost)"
    best_rmse = blend_oof_rmse
    best_r2   = blend_oof_r2
elif lgb_oof_rmse <= xgb_oof_rmse:
    final_preds = lgb_preds
    chosen = "LightGBM"
    best_rmse = lgb_oof_rmse
    best_r2   = lgb_oof_r2
else:
    final_preds = xgb_preds
    chosen = "XGBoost"
    best_rmse = xgb_oof_rmse
    best_r2   = xgb_oof_r2

# Clip to valid range
final_preds = np.clip(final_preds, 0.0, 1.0)

submission = pd.DataFrame({
    'Index':  test['Index'].values,
    'demand': final_preds
})
submission.to_csv('submit/sample_submission.csv', index=False)

print(f"      Saved → submit/sample_submission.csv")
print(f"      Rows  : {len(submission)}")
print(f"      Preview:\n{submission.head(5).to_string(index=False)}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  [OK] FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"  Best model used  : {chosen}")
print(f"  Number of features: {len(FEATURES)}")
print()
print(f"  {'Model':<28}  {'OOF RMSE':>10}  {'OOF MAE':>10}  {'OOF R²':>8}")
print(f"  {'-'*62}")
print(f"  {'LightGBM':<28}  {lgb_oof_rmse:>10.5f}  {lgb_oof_mae:>10.5f}  {lgb_oof_r2:>8.4f}")
print(f"  {'XGBoost':<28}  {xgb_oof_rmse:>10.5f}  {xgb_oof_mae:>10.5f}  {xgb_oof_r2:>8.4f}")
print(f"  {'Ensemble (Weighted Blend)':<28}  {blend_oof_rmse:>10.5f}  {blend_oof_mae:>10.5f}  {blend_oof_r2:>8.4f}")
print(f"  {'-'*62}")
print(f"  {'★ CHOSEN: ' + chosen:<28}  {best_rmse:>10.5f}                {best_r2:>8.4f}")
print()
print(f"  Prediction range : [{final_preds.min():.4f}, {final_preds.max():.4f}]")
print(f"  Prediction mean  : {final_preds.mean():.4f}")
print(f"  Submission file  : dataset/sample_submission.csv")
print("=" * 65)
print("  Pipeline complete!")
print("=" * 65)
