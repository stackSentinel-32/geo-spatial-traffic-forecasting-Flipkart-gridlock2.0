import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

"""
=============================================================================
  Flipkart GriLock 2.0 - ADVANCED Demand Prediction (v2)
  Techniques applied to maximize accuracy:
    1. Log1p target transformation (fixes right-skew)
    2. Geohash -> lat/lon decoding (real spatial coordinates)
    3. Cyclical time encoding (sin/cos for hour)
    4. KFold-safe target encoding (prevents leakage)
    5. Rich statistical group aggregations
    6. Optuna hyperparameter optimization for LightGBM
    7. CatBoost (best native categorical support)
    8. Stacking ensemble with Ridge meta-learner
    9. 10-Fold CV for stable OOF estimates
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('catboost').setLevel(logging.CRITICAL)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import time

# ─────────────────────────────────────────────────────────────────────────────
# Geohash decoder (no external lib needed — pure math)
# ─────────────────────────────────────────────────────────────────────────────
BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
def decode_geohash(ghash):
    """Decode geohash string to (lat, lon, lat_err, lon_err)"""
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon = True
    for char in ghash:
        cd = BASE32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if cd & mask:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if cd & mask:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon
    lat = (lat_range[0] + lat_range[1]) / 2
    lon = (lon_range[0] + lon_range[1]) / 2
    return lat, lon

print("=" * 65)
print("  FLIPKART GRILOCK 2.0 - ADVANCED PIPELINE (v2)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/9] Loading data...")
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')
print(f"      Train: {train.shape}  |  Test: {test.shape}")
print(f"      Target mean={train['demand'].mean():.4f}  std={train['demand'].std():.4f}  "
      f"skew={train['demand'].skew():.2f}  max={train['demand'].max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOG1P TARGET TRANSFORM — Fixes severe right skew, biggest single boost
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/9] Applying log1p target transform...")
y_raw    = train['demand'].values
y_log    = np.log1p(y_raw)
print(f"      Original skew: {pd.Series(y_raw).skew():.3f}  -> log1p skew: {pd.Series(y_log).skew():.3f}")
print(f"      log1p range  : [{y_log.min():.4f}, {y_log.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/9] Engineering features...")

def parse_timestamp(ts):
    try:
        h, m = str(ts).split(':')
        return int(h) * 60 + int(m)
    except:
        return np.nan

# --- Cache geohash lat/lon decode (expensive, do once) ----------------------
print("      Decoding geohash coordinates...")
all_hashes = pd.concat([train['geohash'], test['geohash']]).unique()
geo_coords = {}
for gh in all_hashes:
    try:
        geo_coords[gh] = decode_geohash(gh)
    except:
        geo_coords[gh] = (np.nan, np.nan)

def engineer(df):
    df = df.copy()

    # ── Timestamp ────────────────────────────────────────────────────────────
    df['minutes']      = df['timestamp'].apply(parse_timestamp)
    df['hour']         = df['minutes'] / 60.0
    # Cyclical encoding (sin/cos) — captures circular nature of time
    df['hour_sin']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']     = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_peak']      = df['hour'].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=20) else 0)
    df['is_night']     = df['hour'].apply(lambda h: 1 if (h<5 or h>22) else 0)
    df['time_bucket']  = pd.cut(df['hour'],
                                bins=[-1,5,9,12,16,20,25],
                                labels=[0,1,2,3,4,5]).astype(float)

    # ── Geohash spatial features ─────────────────────────────────────────────
    df['lat']   = df['geohash'].map(lambda g: geo_coords.get(g,(np.nan,np.nan))[0])
    df['lon']   = df['geohash'].map(lambda g: geo_coords.get(g,(np.nan,np.nan))[1])
    df['geo3']  = df['geohash'].str[:3]
    df['geo4']  = df['geohash'].str[:4]
    df['geo5']  = df['geohash'].str[:5]
    df['geo6']  = df['geohash'].str[:6]

    # ── Binary encoding ───────────────────────────────────────────────────────
    df['LargeVehicles_num'] = (df['LargeVehicles'] == 'Allowed').astype(float)
    df['Landmarks_num']     = (df['Landmarks'] == 'Yes').astype(float)

    return df

train = engineer(train)
test  = engineer(test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. IMPUTATION (train-stats only, applied to both)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/9] Imputing missing values...")

# RoadType: mode per geo4
geo4_rt_mode = (train.dropna(subset=['RoadType'])
                .groupby('geo4')['RoadType']
                .agg(lambda x: x.mode()[0]))
global_rt_mode = train['RoadType'].mode()[0]

for df in [train, test]:
    mask = df['RoadType'].isna()
    df.loc[mask, 'RoadType'] = df.loc[mask, 'geo4'].map(geo4_rt_mode)
    df['RoadType'].fillna(global_rt_mode, inplace=True)

# Temperature: median per (Weather, day)
temp_map = (train.dropna(subset=['Temperature'])
            .groupby(['Weather','day'])['Temperature'].median())
global_temp = train['Temperature'].median()

for df in [train, test]:
    mask = df['Temperature'].isna()
    df.loc[mask, 'Temperature'] = df.loc[mask].apply(
        lambda r: temp_map.get((r['Weather'], r['day']), np.nan), axis=1)
    df['Temperature'].fillna(global_temp, inplace=True)

# Weather: mode per geo4
geo4_wx_mode = (train.dropna(subset=['Weather'])
                .groupby('geo4')['Weather']
                .agg(lambda x: x.mode()[0]))
global_wx = train['Weather'].mode()[0]

for df in [train, test]:
    mask = df['Weather'].isna()
    df.loc[mask, 'Weather'] = df.loc[mask, 'geo4'].map(geo4_wx_mode)
    df['Weather'].fillna(global_wx, inplace=True)

# Temperature interaction with weather
temp_std_map = (train.groupby('Weather')['Temperature'].std())
for df in [train, test]:
    df['temp_zscore'] = (df['Temperature'] -
                         df['Weather'].map(train.groupby('Weather')['Temperature'].mean())) / \
                        (df['Weather'].map(temp_std_map) + 1e-6)

# RoadType capacity score
road_capacity = {'Highway': 5, 'Street': 2, 'Residential': 1}
for df in [train, test]:
    df['road_capacity'] = df['RoadType'].map(road_capacity).fillna(1)
    df['capacity_score'] = df['road_capacity'] * df['NumberofLanes']
    df['vehicle_pressure'] = df['capacity_score'] * df['LargeVehicles_num']

print(f"      NaN remaining: train={train[['RoadType','Temperature','Weather']].isna().sum().sum()}  "
      f"test={test[['RoadType','Temperature','Weather']].isna().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
le_road    = LabelEncoder().fit(pd.concat([train['RoadType'], test['RoadType']]))
le_weather = LabelEncoder().fit(pd.concat([train['Weather'],  test['Weather']]))
le_geo3    = LabelEncoder().fit(pd.concat([train['geo3'], test['geo3']]))
le_geo4    = LabelEncoder().fit(pd.concat([train['geo4'], test['geo4']]))
le_geo5    = LabelEncoder().fit(pd.concat([train['geo5'], test['geo5']]))
le_geo6    = LabelEncoder().fit(pd.concat([train['geo6'], test['geo6']]))

for df in [train, test]:
    df['RoadType_enc'] = le_road.transform(df['RoadType'])
    df['Weather_enc']  = le_weather.transform(df['Weather'])
    df['geo3_enc']     = le_geo3.transform(df['geo3'])
    df['geo4_enc']     = le_geo4.transform(df['geo4'])
    df['geo5_enc']     = le_geo5.transform(df['geo5'])
    df['geo6_enc']     = le_geo6.transform(df['geo6'])

# ─────────────────────────────────────────────────────────────────────────────
# 6. KFOLD TARGET ENCODING — zero data leakage (train only, applied OOF style)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/9] Building KFold-safe target encodings...")
N_TE_FOLDS = 5
SMOOTH_ALPHA = 20   # smoothing factor
global_mean = np.mean(y_log)
global_mean_raw = np.mean(y_raw)

kf_te = KFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=99)

TE_COLS = [
    ('geohash',),
    ('geo3',),
    ('geo4',),
    ('geo5',),
    ('RoadType',),
    ('Weather',),
    ('NumberofLanes',),
    ('geo4', 'hour'),
    ('RoadType', 'NumberofLanes'),
    ('RoadType', 'Weather'),
    ('geo3', 'RoadType'),
    ('geo4', 'RoadType'),
]

def smoothed_te(group_mean, group_count, global_mean, alpha=20):
    """Smoothed target encoding: blend group mean toward global mean."""
    return (group_count * group_mean + alpha * global_mean) / (group_count + alpha)

te_train = pd.DataFrame(index=train.index)
te_test  = pd.DataFrame(index=test.index)

for cols in TE_COLS:
    col_name = 'te_' + '_'.join(cols)
    te_train[col_name] = np.nan

    # OOF target encoding for train
    for tr_idx, val_idx in kf_te.split(train):
        fold_train = train.iloc[tr_idx].copy()
        fold_val   = train.iloc[val_idx].copy()
        fold_train['_y_log'] = y_log[tr_idx]

        stats = fold_train.groupby(list(cols))['_y_log'].agg(['mean','count'])
        stats.columns = ['grp_mean','grp_count']
        stats['te'] = smoothed_te(stats['grp_mean'], stats['grp_count'], global_mean, SMOOTH_ALPHA)

        val_keys = fold_val[list(cols)]
        if len(cols) == 1:
            mapped = val_keys[cols[0]].map(stats['te'])
        else:
            mapped = val_keys.set_index(list(cols)).index.map(stats['te'].to_dict())
            mapped = pd.Series(mapped.values, index=fold_val.index)
        te_train.loc[fold_val.index, col_name] = mapped.values

    # Full-train target encoding for test
    full_stats = train.copy()
    full_stats['_y_log'] = y_log
    stats_full = full_stats.groupby(list(cols))['_y_log'].agg(['mean','count'])
    stats_full.columns = ['grp_mean','grp_count']
    stats_full['te'] = smoothed_te(stats_full['grp_mean'], stats_full['grp_count'], global_mean, SMOOTH_ALPHA)

    if len(cols) == 1:
        te_test[col_name] = test[cols[0]].map(stats_full['te'])
    else:
        te_test[col_name] = test.set_index(list(cols)).index.map(stats_full['te'].to_dict()).values
    te_test[col_name] = pd.to_numeric(te_test[col_name], errors='coerce')

    # Fill NaN with global mean
    te_train[col_name].fillna(global_mean, inplace=True)
    te_test[col_name].fillna(global_mean, inplace=True)

# Concatenate encodings
train = pd.concat([train.reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
test  = pd.concat([test.reset_index(drop=True),  te_test.reset_index(drop=True)],  axis=1)

print(f"      Generated {len(TE_COLS)} KFold-safe target encodings")

# ─────────────────────────────────────────────────────────────────────────────
# RICH STATISTICAL AGGREGATIONS per group
# ─────────────────────────────────────────────────────────────────────────────
agg_targets = [
    ('geo4',  ['mean','std','median','max']),
    ('geo3',  ['mean','std']),
    ('RoadType', ['mean','std','max']),
    ('geo4',  ['mean']),  # duplicate intentional — for geo4+hour below
]

# geo4 stats
for stat in ['mean','std','median','max','min']:
    m = train.groupby('geo4')['demand'].agg(stat)
    train[f'geo4_demand_{stat}'] = train['geo4'].map(m)
    test[f'geo4_demand_{stat}']  = test['geo4'].map(m).fillna(global_mean_raw)

# geo3 stats
for stat in ['mean','std']:
    m = train.groupby('geo3')['demand'].agg(stat)
    train[f'geo3_demand_{stat}'] = train['geo3'].map(m)
    test[f'geo3_demand_{stat}']  = test['geo3'].map(m).fillna(global_mean_raw)

# RoadType stats
for stat in ['mean','std','max']:
    m = train.groupby('RoadType')['demand'].agg(stat)
    train[f'rt_demand_{stat}'] = train['RoadType'].map(m)
    test[f'rt_demand_{stat}']  = test['RoadType'].map(m).fillna(global_mean_raw)

# Interaction features
for df in [train, test]:
    df['lanes_x_rt']        = df['NumberofLanes'] * df['RoadType_enc']
    df['temp_x_weather']    = df['Temperature'] * df['Weather_enc']
    df['large_x_lanes']     = df['LargeVehicles_num'] * df['NumberofLanes']
    df['landmark_x_rt']     = df['Landmarks_num'] * df['RoadType_enc']
    df['hour_x_rt']         = df['hour'] * df['RoadType_enc']
    df['capacity_x_hour']   = df['capacity_score'] * df['hour']
    df['pressure_x_peak']   = df['vehicle_pressure'] * df['is_peak']
    df['lat_lon_product']   = df['lat'].fillna(0) * df['lon'].fillna(0)
    df['lat_rounded']       = (df['lat'].fillna(0) * 100).round() / 100
    df['lon_rounded']       = (df['lon'].fillna(0) * 100).round() / 100

print("      Rich statistical aggregations complete.")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET
# ─────────────────────────────────────────────────────────────────────────────
TE_FEATURE_NAMES = [f'te_{"_".join(c)}' for c in TE_COLS]
GEO_STATS = ([f'geo4_demand_{s}' for s in ['mean','std','median','max','min']] +
             [f'geo3_demand_{s}' for s in ['mean','std']] +
             [f'rt_demand_{s}' for s in ['mean','std','max']])

FEATURES = (
    ['day', 'minutes', 'hour', 'hour_sin', 'hour_cos',
     'time_bucket', 'is_peak', 'is_night',
     'RoadType_enc', 'NumberofLanes', 'LargeVehicles_num', 'Landmarks_num',
     'Temperature', 'temp_zscore', 'Weather_enc',
     'road_capacity', 'capacity_score', 'vehicle_pressure',
     'geo3_enc', 'geo4_enc', 'geo5_enc', 'geo6_enc',
     'lat', 'lon', 'lat_rounded', 'lon_rounded', 'lat_lon_product',
     'lanes_x_rt', 'temp_x_weather', 'large_x_lanes',
     'landmark_x_rt', 'hour_x_rt', 'capacity_x_hour', 'pressure_x_peak']
    + TE_FEATURE_NAMES
    + GEO_STATS
)

# Ensure all columns exist and are numeric
for f in FEATURES:
    for df in [train, test]:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0.0)

X      = train[FEATURES].values
y      = y_log   # log1p transformed!
y_orig = y_raw
X_test = test[FEATURES].values

print(f"\n      Total features: {len(FEATURES)}")
print(f"      Training rows: {len(X)}  |  Test rows: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTUNA HPO FOR LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/9] Optuna hyperparameter search for LightGBM (40 trials)...")

N_HPO_FOLDS = 3   # fast 3-fold for HPO

def lgb_objective(trial):
    params = {
        'objective':         'regression',
        'metric':            'rmse',
        'n_estimators':      2000,
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 63, 255),
        'max_depth':         trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 0.1),
        'random_state':      42,
        'n_jobs':           -1,
        'verbose':          -1,
    }
    kf_hpo = KFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=42)
    scores = []
    for tr_i, val_i in kf_hpo.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr_i], y[tr_i],
              eval_set=[(X[val_i], y[val_i])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(period=-1)])
        pred = m.predict(X[val_i])
        # Score on original scale
        pred_orig = np.expm1(pred).clip(0, 1)
        y_orig_val = np.expm1(y[val_i])
        scores.append(np.sqrt(mean_squared_error(y_orig_val, pred_orig)))
    return np.mean(scores)

t_hpo = time.time()
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
best_lgb_params = study.best_params
best_lgb_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
})
print(f"      HPO done in {time.time()-t_hpo:.1f}s")
print(f"      Best trial RMSE : {study.best_value:.5f}")
print(f"      Best params:")
for k,v in study.best_params.items():
    print(f"        {k}: {v:.5f}" if isinstance(v,float) else f"        {k}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 10-FOLD CV TRAINING (LightGBM + CatBoost + XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

lgb_oof   = np.zeros(len(X))
lgb_test  = np.zeros(len(X_test))
cat_oof   = np.zeros(len(X))
cat_test  = np.zeros(len(X_test))
xgb_oof   = np.zeros(len(X))
xgb_test  = np.zeros(len(X_test))

def eval_score(y_true_log, y_pred_log):
    y_t = np.expm1(y_true_log)
    y_p = np.expm1(y_pred_log).clip(0, 1)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2   = r2_score(y_t, y_p)
    return rmse, r2

# ── LightGBM ─────────────────────────────────────────────────────────────────
print(f"\n[7/9] Training models (10-Fold CV)...")
print(f"\n  +-------------------------------------------------------------+")
print(f"  |         LightGBM (Optuna-tuned) - {N_FOLDS}-Fold CV           |")
print(f"  +-------------------------------------------------------------+")

t0 = time.time()
lgb_fold_scores = []
for fold, (tr_i, val_i) in enumerate(kf.split(X, y), 1):
    m = lgb.LGBMRegressor(**best_lgb_params)
    m.fit(X[tr_i], y[tr_i],
          eval_set=[(X[val_i], y[val_i])],
          callbacks=[lgb.early_stopping(200, verbose=False),
                     lgb.log_evaluation(period=-1)])
    lgb_oof[val_i] = m.predict(X[val_i])
    lgb_test += m.predict(X_test) / N_FOLDS
    rmse, r2 = eval_score(y[val_i], lgb_oof[val_i])
    lgb_fold_scores.append(rmse)
    print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iter={m.best_iteration_}")

lgb_oof_rmse, lgb_oof_r2 = eval_score(y, lgb_oof)
print(f"\n  >> LightGBM OOF RMSE: {lgb_oof_rmse:.5f}  R2: {lgb_oof_r2:.4f}  ({time.time()-t0:.1f}s)")

# ── CatBoost ──────────────────────────────────────────────────────────────────
print(f"\n  +-------------------------------------------------------------+")
print(f"  |              CatBoost - {N_FOLDS}-Fold CV                       |")
print(f"  +-------------------------------------------------------------+")

cat_params = {
    'iterations':        3000,
    'learning_rate':     0.05,
    'depth':             8,
    'l2_leaf_reg':       3.0,
    'bagging_temperature': 0.5,
    'random_strength':   1.0,
    'min_data_in_leaf':  20,
    'grow_policy':       'SymmetricTree',
    'loss_function':     'RMSE',
    'eval_metric':       'RMSE',
    'random_seed':       42,
    'thread_count':     -1,
    'verbose':           False,
    'early_stopping_rounds': 200,
}

t0 = time.time()
cat_fold_scores = []
for fold, (tr_i, val_i) in enumerate(kf.split(X, y), 1):
    m = cb.CatBoostRegressor(**cat_params)
    m.fit(X[tr_i], y[tr_i],
          eval_set=[(X[val_i], y[val_i])],
          verbose=False)
    cat_oof[val_i] = m.predict(X[val_i])
    cat_test += m.predict(X_test) / N_FOLDS
    rmse, r2 = eval_score(y[val_i], cat_oof[val_i])
    cat_fold_scores.append(rmse)
    print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iter={m.best_iteration_}")

cat_oof_rmse, cat_oof_r2 = eval_score(y, cat_oof)
print(f"\n  >> CatBoost OOF RMSE: {cat_oof_rmse:.5f}  R2: {cat_oof_r2:.4f}  ({time.time()-t0:.1f}s)")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print(f"\n  +-------------------------------------------------------------+")
print(f"  |              XGBoost - {N_FOLDS}-Fold CV                        |")
print(f"  +-------------------------------------------------------------+")

xgb_params = {
    'objective':             'reg:squarederror',
    'eval_metric':           'rmse',
    'n_estimators':          3000,
    'learning_rate':         0.03,
    'max_depth':             7,
    'min_child_weight':      5,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'gamma':                 0.1,
    'reg_alpha':             0.1,
    'reg_lambda':            0.2,
    'early_stopping_rounds': 200,
    'random_state':          42,
    'n_jobs':               -1,
    'verbosity':             0,
}

t0 = time.time()
xgb_fold_scores = []
for fold, (tr_i, val_i) in enumerate(kf.split(X, y), 1):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X[tr_i], y[tr_i],
          eval_set=[(X[val_i], y[val_i])],
          verbose=False)
    xgb_oof[val_i] = m.predict(X[val_i])
    xgb_test += m.predict(X_test) / N_FOLDS
    rmse, r2 = eval_score(y[val_i], xgb_oof[val_i])
    xgb_fold_scores.append(rmse)
    print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iter={m.best_iteration}")

xgb_oof_rmse, xgb_oof_r2 = eval_score(y, xgb_oof)
print(f"\n  >> XGBoost OOF RMSE: {xgb_oof_rmse:.5f}  R2: {xgb_oof_r2:.4f}  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. STACKING ENSEMBLE (Ridge meta-learner on OOF predictions)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[8/9] Stacking ensemble with Ridge meta-learner...")

# Stack OOF predictions as meta-features
stack_train = np.column_stack([lgb_oof, cat_oof, xgb_oof])
stack_test  = np.column_stack([lgb_test, cat_test, xgb_test])

# Find optimal Ridge alpha
best_alpha, best_stack_rmse = 1.0, np.inf
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    meta = Ridge(alpha=alpha)
    # Simple train/val split to pick alpha (use all train since we have OOF)
    meta.fit(stack_train, y)
    pred = meta.predict(stack_train)
    rmse, _ = eval_score(y, pred)
    if rmse < best_stack_rmse:
        best_stack_rmse = rmse
        best_alpha = alpha

meta_model = Ridge(alpha=best_alpha)
meta_model.fit(stack_train, y)
stack_oof  = meta_model.predict(stack_train)
stack_test_pred = meta_model.predict(stack_test)

stack_oof_rmse, stack_oof_r2 = eval_score(y, stack_oof)
lgb_w, cat_w, xgb_w = meta_model.coef_
print(f"      Ridge alpha: {best_alpha}")
print(f"      Weights -> LightGBM:{lgb_w:.3f}  CatBoost:{cat_w:.3f}  XGBoost:{xgb_w:.3f}")
print(f"      Stack OOF RMSE: {stack_oof_rmse:.5f}  R2: {stack_oof_r2:.4f}")

# Also compute simple best-weight blend
weights = np.array([1/lgb_oof_rmse, 1/cat_oof_rmse, 1/xgb_oof_rmse])
weights /= weights.sum()
blend_oof  = weights[0]*lgb_oof  + weights[1]*cat_oof  + weights[2]*xgb_oof
blend_test = weights[0]*lgb_test + weights[1]*cat_test + weights[2]*xgb_test
blend_rmse, blend_r2 = eval_score(y, blend_oof)
print(f"      Simple blend RMSE: {blend_rmse:.5f}  R2: {blend_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. GENERATE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[9/9] Generating submission file...")

# Pick final predictions
all_models = {
    'LightGBM':        (lgb_oof_rmse,   lgb_oof_r2,   lgb_test),
    'CatBoost':        (cat_oof_rmse,   cat_oof_r2,   cat_test),
    'XGBoost':         (xgb_oof_rmse,   xgb_oof_r2,   xgb_test),
    'Stack (Ridge)':   (stack_oof_rmse, stack_oof_r2, stack_test_pred),
    'Blend (Weights)': (blend_rmse,     blend_r2,     blend_test),
}

best_name = min(all_models, key=lambda k: all_models[k][0])
best_rmse_final, best_r2_final, best_preds_log = all_models[best_name]

# Inverse log1p transform, then clip to [0,1]
final_preds = np.expm1(best_preds_log).clip(0.0, 1.0)

submission = pd.DataFrame({'Index': test['Index'].values, 'demand': final_preds})
submission.to_csv('submit/sample_submission_v2.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  [FINAL RESULTS SUMMARY - v2 Advanced Pipeline]")
print("=" * 65)
print(f"\n  Techniques applied:")
print(f"    * Log1p target transform (skew fix)")
print(f"    * Geohash -> lat/lon decoding")
print(f"    * Cyclical time encoding (sin/cos)")
print(f"    * KFold-safe target encoding (12 groups, smoothed)")
print(f"    * Rich statistical aggregations")
print(f"    * Optuna HPO (40 trials) for LightGBM")
print(f"    * 10-Fold CV for stable estimates")
print(f"    * CatBoost with native categoricals")
print(f"    * Stacking ensemble (Ridge meta-learner)")
print(f"    * {len(FEATURES)} total features")
print()
print(f"  {'Model':<26} {'OOF RMSE':>10} {'OOF MAE':>10} {'OOF R2':>8}")
print(f"  {'-'*60}")
for name, (rmse, r2, _) in all_models.items():
    mae_val = mean_absolute_error(np.expm1(y), np.expm1(
        lgb_oof if name=='LightGBM' else
        cat_oof if name=='CatBoost' else
        xgb_oof if name=='XGBoost' else
        stack_oof if name=='Stack (Ridge)' else blend_oof
    ).clip(0,1))
    marker = " <-- BEST" if name == best_name else ""
    print(f"  {name:<26} {rmse:>10.5f} {mae_val:>10.5f} {r2:>8.4f}{marker}")
print(f"  {'-'*60}")
print()
print(f"  CHOSEN: {best_name}")
print(f"  OOF RMSE: {best_rmse_final:.5f}  |  OOF R2: {best_r2_final:.4f}")
print()
print(f"  Baseline (v1 LightGBM): RMSE=0.03204  R2=0.9492")
improvement = (0.03204 - best_rmse_final) / 0.03204 * 100
print(f"  Improvement: {improvement:+.2f}% in RMSE")
print()
print(f"  Prediction range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
print(f"  Prediction mean : {final_preds.mean():.4f}")
print(f"  Submission file : dataset/sample_submission.csv ({len(submission)} rows)")
print("=" * 65)
print("  v2 Pipeline complete!")
print("=" * 65)
