import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import config
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'


def decode_geohash(ghash):
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
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


def parse_ts(ts):
    try:
        h, m = str(ts).split(':')
        return int(h) * 60 + int(m)
    except:
        return np.nan


def smoothed_te(grp_mean, grp_count, glob_mean, alpha=20):
    return (grp_count * grp_mean + alpha * glob_mean) / (grp_count + alpha)


def eval_score(y_true_log, y_pred_log):
    y_t = np.expm1(y_true_log)
    y_p = np.expm1(y_pred_log).clip(0, 1)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    return rmse, r2


def run_pipeline(train_path, test_path, submit_dir, v2_submission_path):
    t_start = time.time()

    print("=" * 70)
    print("  FLIPKART GRILOCK 2.0 - PRECISION EDITION (v9)")
    print("  Strategy: v2 proven base + careful lag features + anti-overfitting")
    print("=" * 70)

    print("\n[1/10] Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"       Train: {train.shape}  |  Test: {test.shape}")
    print(f"       Train days: {sorted(train['day'].unique())}  |  Test days: {sorted(test['day'].unique())}")

    y_raw = train['demand'].values
    global_mean_raw = y_raw.mean()

    print("\n[2/10] Applying log1p target transform...")
    y_log = np.log1p(y_raw)
    print(f"       Original skew: {pd.Series(y_raw).skew():.3f} -> log1p skew: {pd.Series(y_log).skew():.3f}")

    print("\n[3/10] Decoding geohash coordinates...")
    all_hashes = pd.concat([train['geohash'], test['geohash']]).unique()
    geo_coords = {}
    for gh in all_hashes:
        try:
            geo_coords[gh] = decode_geohash(gh)
        except:
            geo_coords[gh] = (np.nan, np.nan)

    print("\n[4/10] Building CAREFUL day-48 lag features...")
    day48 = train[train['day'] == 48].copy()
    day48['mins'] = day48['timestamp'].apply(parse_ts)
    day48['geo4'] = day48['geohash'].str[:4]
    day48['geo3'] = day48['geohash'].str[:3]

    geo48_ts_dict = {}
    for g, ts, d in zip(day48['geohash'], day48['timestamp'], day48['demand']):
        geo48_ts_dict[(g, ts)] = d

    geo4_48_ts = day48.groupby(['geo4', 'timestamp'])['demand'].mean()
    geo4_48_ts_dict = geo4_48_ts.to_dict()

    geo3_48_ts = day48.groupby(['geo3', 'timestamp'])['demand'].mean()
    geo3_48_ts_dict = geo3_48_ts.to_dict()

    d48_geo_mean = day48.groupby('geohash')['demand'].mean()
    d48_geo_std = day48.groupby('geohash')['demand'].std().fillna(0)
    d48_geo_max = day48.groupby('geohash')['demand'].max()
    d48_geo_med = day48.groupby('geohash')['demand'].median()
    d48_geo4_mean = day48.groupby('geo4')['demand'].mean()
    d48_geo3_mean = day48.groupby('geo3')['demand'].mean()

    day48['hour'] = day48['mins'] / 60.0
    day48['time_bucket'] = pd.cut(day48['hour'], bins=[-1,5,9,12,16,20,25],
                                  labels=[0,1,2,3,4,5]).astype(float)
    d48_geo_tb = day48.groupby(['geohash', 'time_bucket'])['demand'].mean()
    d48_geo_tb_dict = d48_geo_tb.to_dict()
    d48_geo4_tb = day48.groupby(['geo4', 'time_bucket'])['demand'].mean()
    d48_geo4_tb_dict = d48_geo4_tb.to_dict()

    day48['mins_i'] = day48['mins'].astype(int)
    d48_geo_mins = {}
    for g, m, d in zip(day48['geohash'], day48['mins_i'], day48['demand']):
        d48_geo_mins[(g, m)] = d

    print(f"       Day48 lookup entries: {len(geo48_ts_dict)}")
    print(f"       Day48 geohash stats: {len(d48_geo_mean)}")

    print("\n[5/10] Feature engineering...")
    for df in [train, test]:
        df['mins'] = df['timestamp'].apply(parse_ts)
        df['hour'] = df['mins'] / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_peak'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                        ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(float)
        df['is_night'] = ((df['hour'] < 5) | (df['hour'] > 22)).astype(float)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 10)).astype(float)
        df['is_evening'] = ((df['hour'] >= 17) & (df['hour'] < 21)).astype(float)
        df['time_bucket'] = pd.cut(df['hour'], bins=[-1,5,9,12,16,20,25],
                                  labels=[0,1,2,3,4,5]).astype(float)
        df['hour_sq'] = df['hour'] ** 2
        df['geo3'] = df['geohash'].str[:3]
        df['geo4'] = df['geohash'].str[:4]
        df['geo5'] = df['geohash'].str[:5]
        df['geo6'] = df['geohash'].str[:6]
        df['lat'] = df['geohash'].map(lambda g: geo_coords.get(g, (np.nan, np.nan))[0])
        df['lon'] = df['geohash'].map(lambda g: geo_coords.get(g, (np.nan, np.nan))[1])
        df['LargeVehicles_num'] = (df['LargeVehicles'] == 'Allowed').astype(float)
        df['Landmarks_num'] = (df['Landmarks'] == 'Yes').astype(float)

        df['lag48_exact'] = df.apply(
            lambda r: geo48_ts_dict.get((r['geohash'], r['timestamp']), np.nan), axis=1)
        df['lag48_geo4'] = df.apply(
            lambda r: geo4_48_ts_dict.get((r['geo4'], r['timestamp']), np.nan), axis=1)
        df['lag48_geo3'] = df.apply(
            lambda r: geo3_48_ts_dict.get((r['geo3'], r['timestamp']), np.nan), axis=1)
        df['lag48'] = df['lag48_exact'].fillna(df['lag48_geo4']).fillna(df['lag48_geo3']).fillna(global_mean_raw)

        mins_arr = df['mins'].values.astype(int)
        geos = df['geohash'].values
        lag48_nearby = []
        for g, m in zip(geos, mins_arr):
            vals = []
            for offset in [-15, 0, 15]:
                v = d48_geo_mins.get((g, m + offset), np.nan)
                if not np.isnan(v):
                    vals.append(v)
            lag48_nearby.append(np.mean(vals) if vals else np.nan)
        df['lag48_nearby'] = lag48_nearby
        df['lag48_nearby'] = df['lag48_nearby'].fillna(df['lag48'])

        df['d48_geo_mean'] = df['geohash'].map(d48_geo_mean).fillna(
            df['geo4'].map(d48_geo4_mean).fillna(global_mean_raw))
        df['d48_geo_std'] = df['geohash'].map(d48_geo_std).fillna(0)
        df['d48_geo_max'] = df['geohash'].map(d48_geo_max).fillna(global_mean_raw)
        df['d48_geo_med'] = df['geohash'].map(d48_geo_med).fillna(global_mean_raw)

        tb_vals = []
        for g, g4, tb in zip(df['geohash'], df['geo4'], df['time_bucket']):
            v = d48_geo_tb_dict.get((g, tb), np.nan)
            if np.isnan(v):
                v = d48_geo4_tb_dict.get((g4, tb), np.nan)
            if np.isnan(v):
                v = global_mean_raw
            tb_vals.append(v)
        df['d48_geo_tb'] = tb_vals
        df['lag48_ratio'] = df['lag48'] / (df['d48_geo_mean'] + 1e-9)
        df['lag48_delta'] = df['lag48'] - df['d48_geo_mean']

    for name, ds in [('Train', train), ('Test', test)]:
        cov = ds['lag48_exact'].notna().mean()
        print(f"       {name} exact lag48 coverage: {cov*100:.1f}%")

    print("\n[6/10] Imputing missing values...")
    geo4_rt_mode = train.dropna(subset=['RoadType']).groupby('geo4')['RoadType'].agg(lambda x: x.mode()[0])
    global_rt_mode = train['RoadType'].mode()[0]
    for df in [train, test]:
        mask = df['RoadType'].isna()
        df.loc[mask, 'RoadType'] = df.loc[mask, 'geo4'].map(geo4_rt_mode)
        df['RoadType'].fillna(global_rt_mode, inplace=True)

    temp_map = train.dropna(subset=['Temperature']).groupby(['Weather','day'])['Temperature'].median()
    global_temp = train['Temperature'].median()
    for df in [train, test]:
        mask = df['Temperature'].isna()
        df.loc[mask, 'Temperature'] = df.loc[mask].apply(
            lambda r: temp_map.get((r['Weather'], r['day']), np.nan), axis=1)
        df['Temperature'].fillna(global_temp, inplace=True)

    geo4_wx_mode = train.dropna(subset=['Weather']).groupby('geo4')['Weather'].agg(lambda x: x.mode()[0])
    global_wx = train['Weather'].mode()[0]
    for df in [train, test]:
        mask = df['Weather'].isna()
        df.loc[mask, 'Weather'] = df.loc[mask, 'geo4'].map(geo4_wx_mode)
        df['Weather'].fillna(global_wx, inplace=True)

    temp_std_map = train.groupby('Weather')['Temperature'].std()
    temp_mean_map = train.groupby('Weather')['Temperature'].mean()
    for df in [train, test]:
        df['temp_zscore'] = (df['Temperature'] - df['Weather'].map(temp_mean_map)) / (
            df['Weather'].map(temp_std_map) + 1e-6)

    road_cap = {'Highway': 5, 'Street': 2, 'Residential': 1}
    for df in [train, test]:
        df['road_capacity'] = df['RoadType'].map(road_cap).fillna(1)
        df['capacity_score'] = df['road_capacity'] * df['NumberofLanes']
        df['vehicle_pressure'] = df['capacity_score'] * df['LargeVehicles_num']

    print(f"       NaN remaining: {train[['RoadType','Temperature','Weather']].isna().sum().sum()}")

    print("\n[7/10] Label encoding...")
    le_road = LabelEncoder().fit(pd.concat([train['RoadType'], test['RoadType']]))
    le_weather = LabelEncoder().fit(pd.concat([train['Weather'], test['Weather']]))
    le_geo3 = LabelEncoder().fit(pd.concat([train['geo3'], test['geo3']]))
    le_geo4 = LabelEncoder().fit(pd.concat([train['geo4'], test['geo4']]))
    le_geo5 = LabelEncoder().fit(pd.concat([train['geo5'], test['geo5']]))
    le_geo6 = LabelEncoder().fit(pd.concat([train['geo6'], test['geo6']]))
    for df in [train, test]:
        df['RoadType_enc'] = le_road.transform(df['RoadType'])
        df['Weather_enc'] = le_weather.transform(df['Weather'])
        df['geo3_enc'] = le_geo3.transform(df['geo3'])
        df['geo4_enc'] = le_geo4.transform(df['geo4'])
        df['geo5_enc'] = le_geo5.transform(df['geo5'])
        df['geo6_enc'] = le_geo6.transform(df['geo6'])

    print("\n[8/10] KFold-safe target encodings (14 combos)...")
    N_TE_FOLDS = 5
    SMOOTH_ALPHA = 20
    global_mean_log = np.mean(y_log)
    kf_te = KFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=99)
    TE_COLS = [
        ('geohash',),
        ('geo3',), ('geo4',), ('geo5',), ('geo6',),
        ('RoadType',), ('Weather',), ('NumberofLanes',),
        ('geo4', 'hour'),
        ('RoadType', 'NumberofLanes'),
        ('RoadType', 'Weather'),
        ('geo3', 'RoadType'),
        ('geo4', 'RoadType'),
        ('geo4', 'Weather'),
    ]
    te_train = pd.DataFrame(index=train.index)
    te_test = pd.DataFrame(index=test.index)
    for cols in TE_COLS:
        col_name = 'te_' + '_'.join(cols)
        te_train[col_name] = np.nan
        for tr_idx, val_idx in kf_te.split(train):
            fold_tr = train.iloc[tr_idx].copy()
            fold_val = train.iloc[val_idx].copy()
            fold_tr['_y'] = y_log[tr_idx]
            stats = fold_tr.groupby(list(cols))['_y'].agg(['mean', 'count'])
            stats.columns = ['grp_mean', 'grp_count']
            stats['te'] = smoothed_te(stats['grp_mean'], stats['grp_count'], global_mean_log, SMOOTH_ALPHA)
            val_keys = fold_val[list(cols)]
            if len(cols) == 1:
                mapped = val_keys[cols[0]].map(stats['te'])
            else:
                mapped = pd.Series(val_keys.set_index(list(cols)).index.map(stats['te'].to_dict()).values, index=fold_val.index)
            te_train.loc[fold_val.index, col_name] = pd.to_numeric(mapped, errors='coerce').values
        full = train.copy()
        full['_y'] = y_log
        sf = full.groupby(list(cols))['_y'].agg(['mean', 'count'])
        sf.columns = ['grp_mean', 'grp_count']
        sf['te'] = smoothed_te(sf['grp_mean'], sf['grp_count'], global_mean_log, SMOOTH_ALPHA)
        if len(cols) == 1:
            te_test[col_name] = test[cols[0]].map(sf['te'])
        else:
            te_test[col_name] = pd.Series(test.set_index(list(cols)).index.map(sf['te'].to_dict()).values, index=test.index)
        te_test[col_name] = pd.to_numeric(te_test[col_name], errors='coerce')
        te_train[col_name].fillna(global_mean_log, inplace=True)
        te_test[col_name].fillna(global_mean_log, inplace=True)
    train = pd.concat([train.reset_index(drop=True), te_train.reset_index(drop=True)], axis=1)
    test = pd.concat([test.reset_index(drop=True), te_test.reset_index(drop=True)], axis=1)
    print(f"       Generated {len(TE_COLS)} target encodings.")

    print("\n[9/10] Statistical aggregations + interactions...")
    for stat in ['mean', 'std', 'median', 'max', 'min']:
        m = train.groupby('geo4')['demand'].agg(stat)
        train[f'geo4_demand_{stat}'] = train['geo4'].map(m)
        test[f'geo4_demand_{stat}'] = test['geo4'].map(m).fillna(global_mean_raw)
    for q, qlabel in [(0.1,'p10'), (0.25,'p25'), (0.75,'p75'), (0.90,'p90')]:
        m = train.groupby('geo4')['demand'].quantile(q)
        train[f'geo4_demand_{qlabel}'] = train['geo4'].map(m)
        test[f'geo4_demand_{qlabel}'] = test['geo4'].map(m).fillna(global_mean_raw)
    for stat in ['mean', 'std']:
        m = train.groupby('geo3')['demand'].agg(stat)
        train[f'geo3_demand_{stat}'] = train['geo3'].map(m)
        test[f'geo3_demand_{stat}'] = test['geo3'].map(m).fillna(global_mean_raw)
    for stat in ['mean', 'std', 'max']:
        m = train.groupby('RoadType')['demand'].agg(stat)
        train[f'rt_demand_{stat}'] = train['RoadType'].map(m)
        test[f'rt_demand_{stat}'] = test['RoadType'].map(m).fillna(global_mean_raw)
    g4tb = train.groupby(['geo4', 'time_bucket'])['demand'].mean()
    train['geo4_tb_demand'] = train.set_index(['geo4','time_bucket']).index.map(g4tb).values
    test['geo4_tb_demand'] = test.set_index(['geo4','time_bucket']).index.map(g4tb.to_dict()).values
    train['geo4_tb_demand'] = pd.to_numeric(train['geo4_tb_demand'], errors='coerce').fillna(global_mean_raw)
    test['geo4_tb_demand'] = pd.to_numeric(test['geo4_tb_demand'], errors='coerce').fillna(global_mean_raw)
    gh_freq = train['geohash'].value_counts()
    geo4_freq = train['geo4'].value_counts()
    train['geohash_freq'] = train['geohash'].map(gh_freq)
    test['geohash_freq'] = test['geohash'].map(gh_freq).fillna(0)
    train['geo4_freq'] = train['geo4'].map(geo4_freq)
    test['geo4_freq'] = test['geo4'].map(geo4_freq).fillna(0)
    gh_std = train.groupby('geohash')['demand'].std().fillna(0)
    train['gh_volatility'] = train['geohash'].map(gh_std)
    test['gh_volatility'] = test['geohash'].map(gh_std).fillna(0)
    for df in [train, test]:
        df['lanes_x_rt'] = df['NumberofLanes'] * df['RoadType_enc']
        df['temp_x_weather'] = df['Temperature'] * df['Weather_enc']
        df['large_x_lanes'] = df['LargeVehicles_num'] * df['NumberofLanes']
        df['landmark_x_rt'] = df['Landmarks_num'] * df['RoadType_enc']
        df['hour_x_rt'] = df['hour'] * df['RoadType_enc']
        df['capacity_x_hour'] = df['capacity_score'] * df['hour']
        df['pressure_x_peak'] = df['vehicle_pressure'] * df['is_peak']
        df['lat_lon_product'] = df['lat'].fillna(0) * df['lon'].fillna(0)
        df['lat_rounded'] = (df['lat'].fillna(0) * 100).round() / 100
        df['lon_rounded'] = (df['lon'].fillna(0) * 100).round() / 100
        df['demand_spread'] = df['geo4_demand_p90'].fillna(0) - df['geo4_demand_p10'].fillna(0)
        df['geo4_vs_geo3'] = df['geo4_demand_mean'].fillna(0) - df['geo3_demand_mean'].fillna(0)
    print("       Done.")

    TE_NAMES = [f'te_{"_".join(c)}' for c in TE_COLS]
    GEO_STATS = (
        [f'geo4_demand_{s}' for s in ['mean','std','median','max','min','p10','p25','p75','p90']] +
        [f'geo3_demand_{s}' for s in ['mean','std']] +
        [f'rt_demand_{s}' for s in ['mean','std','max']] +
        ['geo4_tb_demand']
    )
    LAG_FEATS = ['lag48', 'lag48_nearby', 'd48_geo_mean', 'd48_geo_std', 'd48_geo_max',
                 'd48_geo_med', 'd48_geo_tb', 'lag48_ratio', 'lag48_delta']
    FEATURES = (
        ['day', 'mins', 'hour', 'hour_sin', 'hour_cos', 'hour_sq',
         'time_bucket', 'is_peak', 'is_night', 'is_morning', 'is_evening'] +
        ['RoadType_enc', 'NumberofLanes', 'LargeVehicles_num', 'Landmarks_num',
         'road_capacity', 'capacity_score', 'vehicle_pressure'] +
        ['Temperature', 'temp_zscore', 'Weather_enc'] +
        ['geo3_enc', 'geo4_enc', 'geo5_enc', 'geo6_enc',
         'lat', 'lon', 'lat_rounded', 'lon_rounded', 'lat_lon_product'] +
        ['geohash_freq', 'geo4_freq', 'gh_volatility'] +
        ['lanes_x_rt', 'temp_x_weather', 'large_x_lanes', 'landmark_x_rt',
         'hour_x_rt', 'capacity_x_hour', 'pressure_x_peak',
         'demand_spread', 'geo4_vs_geo3'] +
        LAG_FEATS +
        TE_NAMES +
        GEO_STATS
    )
    FEATURES = list(dict.fromkeys(FEATURES))
    for f in FEATURES:
        for df in [train, test]:
            if f not in df.columns:
                df[f] = 0.0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0.0)

    X = train[FEATURES].values
    y = y_log
    X_test = test[FEATURES].values
    print(f"\n       Total features   : {len(FEATURES)}")
    print(f"       Training rows    : {len(X)}")
    print(f"       Test rows        : {len(X_test)}")

    N_HPO_FOLDS = 5
    print(f"\n[10/10] Optuna HPO - LightGBM ({OPTUNA_LGB_TRIALS} trials, {N_HPO_FOLDS}-fold)...")
    def lgb_objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('nl', 63, 300),
            'max_depth': trial.suggest_int('md', 5, 10),
            'min_child_samples': trial.suggest_int('mcs', 10, 60),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('ra', 1e-4, 2.0, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-4, 2.0, log=True),
            'min_split_gain': trial.suggest_float('msg', 0.0, 0.15),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        kf_h = KFold(n_splits=N_HPO_FOLDS, shuffle=True, random_state=42)
        scores = []
        for ti, vi in kf_h.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[ti], y[ti],
                  eval_set=[(X[vi], y[vi])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            scores.append(eval_score(y[vi], m.predict(X[vi]))[0])
        return np.mean(scores)

    t0 = time.time()
    study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=OPTUNA_LGB_TRIALS, show_progress_bar=False)
    best_lgb = study_lgb.best_params
    lgb_final = {
        'objective': 'regression', 'metric': 'rmse', 'n_estimators': 4000,
        'verbose': -1, 'n_jobs': -1, 'random_state': 42,
        'learning_rate': best_lgb['lr'],
        'num_leaves': best_lgb['nl'],
        'max_depth': best_lgb['md'],
        'min_child_samples': best_lgb['mcs'],
        'subsample': best_lgb['ss'],
        'colsample_bytree': best_lgb['cs'],
        'reg_alpha': best_lgb['ra'],
        'reg_lambda': best_lgb['rl'],
        'min_split_gain': best_lgb['msg'],
    }
    print(f"       LGB HPO done: {time.time()-t0:.1f}s | Best RMSE: {study_lgb.best_value:.5f}")
    for k, v in study_lgb.best_params.items():
        print(f"         {k}: {v:.5f}" if isinstance(v, float) else f"         {k}: {v}")

    print(f"       Optuna HPO - CatBoost ({OPTUNA_CAT_TRIALS} trials, GPU)...")
    def cat_objective(trial):
        params = {
            'iterations': 2000,
            'learning_rate': trial.suggest_float('lr', 0.01, 0.12, log=True),
            'depth': trial.suggest_int('depth', 5, 9),
            'l2_leaf_reg': trial.suggest_float('l2', 0.5, 12.0),
            'bagging_temperature': trial.suggest_float('bt', 0.0, 1.5),
            'random_strength': trial.suggest_float('rs', 0.0, 2.5),
            'min_data_in_leaf': trial.suggest_int('mdl', 10, 50),
            'grow_policy': trial.suggest_categorical('gp', ['SymmetricTree', 'Depthwise']),
            'task_type': 'GPU',
            'loss_function': 'RMSE', 'eval_metric': 'RMSE',
            'random_seed': 42, 'verbose': False, 'early_stopping_rounds': 100,
        }
        kf_h = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for ti, vi in kf_h.split(X):
            m = cb.CatBoostRegressor(**params)
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
            scores.append(eval_score(y[vi], m.predict(X[vi]))[0])
        return np.mean(scores)

    t0 = time.time()
    study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(cat_objective, n_trials=OPTUNA_CAT_TRIALS, show_progress_bar=False)
    best_cat = study_cat.best_params
    cat_final = {
        'iterations': 5000, 'task_type': 'GPU',
        'loss_function': 'RMSE', 'eval_metric': 'RMSE',
        'random_seed': 42, 'verbose': False, 'early_stopping_rounds': 200,
        'learning_rate': best_cat['lr'],
        'depth': best_cat['depth'],
        'l2_leaf_reg': best_cat['l2'],
        'bagging_temperature': best_cat['bt'],
        'random_strength': best_cat['rs'],
        'min_data_in_leaf': best_cat['mdl'],
        'grow_policy': best_cat['gp'],
    }
    print(f"       CAT HPO done: {time.time()-t0:.1f}s | Best RMSE: {study_cat.best_value:.5f}")
    for k, v in study_cat.best_params.items():
        print(f"         {k}: {v:.5f}" if isinstance(v, float) else f"         {k}: {v}")

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 4000,
        'learning_rate': 0.03,
        'max_depth': 7,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        'early_stopping_rounds': 200,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',
        'device': 'cuda',
    }

    N_FOLDS = 10
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    lgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))
    cat_oof = np.zeros(len(X))
    cat_test = np.zeros(len(X_test))
    xgb_oof = np.zeros(len(X))
    xgb_test = np.zeros(len(X_test))

    print(f"\n  === Training {N_FOLDS}-Fold CV (LightGBM + CatBoost + XGBoost) ===")
    print(f"\n  +- LightGBM (Optuna-tuned) -----------------------------------+")
    t0 = time.time()
    for fold, (ti, vi) in enumerate(kf.split(X, y), 1):
        m = lgb.LGBMRegressor(**lgb_final)
        m.fit(X[ti], y[ti],
              eval_set=[(X[vi], y[vi])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
        lgb_oof[vi] = m.predict(X[vi])
        lgb_test += m.predict(X_test) / N_FOLDS
        rmse, r2 = eval_score(y[vi], lgb_oof[vi])
        print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iters={m.best_iteration_}")
    lgb_rmse, lgb_r2 = eval_score(y, lgb_oof)
    print(f"\n  >> LightGBM OOF: RMSE={lgb_rmse:.5f}  R2={lgb_r2:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n  +- CatBoost (Optuna-tuned + GPU) -----------------------------+")
    t0 = time.time()
    for fold, (ti, vi) in enumerate(kf.split(X, y), 1):
        m = cb.CatBoostRegressor(**cat_final)
        m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
        cat_oof[vi] = m.predict(X[vi])
        cat_test += m.predict(X_test) / N_FOLDS
        rmse, r2 = eval_score(y[vi], cat_oof[vi])
        print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iters={m.best_iteration_}")
    cat_rmse, cat_r2 = eval_score(y, cat_oof)
    print(f"\n  >> CatBoost OOF: RMSE={cat_rmse:.5f}  R2={cat_r2:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n  +- XGBoost (GPU) ---------------------------------------------+")
    t0 = time.time()
    try:
        for fold, (ti, vi) in enumerate(kf.split(X, y), 1):
            m = xgb.XGBRegressor(**xgb_params)
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
            xgb_oof[vi] = m.predict(X[vi])
            xgb_test += m.predict(X_test) / N_FOLDS
            rmse, r2 = eval_score(y[vi], xgb_oof[vi])
            print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iters={m.best_iteration}")
    except Exception as e:
        print(f"  XGBoost GPU failed ({e}), retrying with CPU...")
        xgb_params_cpu = xgb_params.copy()
        xgb_params_cpu['tree_method'] = 'hist'
        xgb_params_cpu.pop('device', None)
        xgb_oof[:] = 0
        xgb_test[:] = 0
        for fold, (ti, vi) in enumerate(kf.split(X, y), 1):
            m = xgb.XGBRegressor(**xgb_params_cpu)
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
            xgb_oof[vi] = m.predict(X[vi])
            xgb_test += m.predict(X_test) / N_FOLDS
            rmse, r2 = eval_score(y[vi], xgb_oof[vi])
            print(f"  Fold {fold:02d}/{N_FOLDS} -> RMSE: {rmse:.5f}  R2: {r2:.4f}  iters={m.best_iteration}")
    xgb_rmse, xgb_r2 = eval_score(y, xgb_oof)
    print(f"\n  >> XGBoost OOF: RMSE={xgb_rmse:.5f}  R2={xgb_r2:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n  +- Stacking Ensemble (Ridge meta-learner) --------------------+")
    stack_train = np.column_stack([lgb_oof, cat_oof, xgb_oof])
    stack_test = np.column_stack([lgb_test, cat_test, xgb_test])
    best_alpha, best_stack_rmse = 1.0, np.inf
    for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        meta = Ridge(alpha=alpha)
        meta.fit(stack_train, y)
        pred = meta.predict(stack_train)
        rmse, _ = eval_score(y, pred)
        if rmse < best_stack_rmse:
            best_stack_rmse = rmse
            best_alpha = alpha
    meta_model = Ridge(alpha=best_alpha)
    meta_model.fit(stack_train, y)
    stack_oof = meta_model.predict(stack_train)
    stack_test_pred = meta_model.predict(stack_test)
    stack_rmse, stack_r2 = eval_score(y, stack_oof)
    lgb_w, cat_w, xgb_w = meta_model.coef_
    print(f"  Ridge alpha: {best_alpha}")
    print(f"  Weights -> LGB:{lgb_w:.3f}  CAT:{cat_w:.3f}  XGB:{xgb_w:.3f}")
    print(f"  Stack OOF: RMSE={stack_rmse:.5f}  R2={stack_r2:.4f}")

    weights = np.array([1/lgb_rmse, 1/cat_rmse, 1/xgb_rmse])
    weights /= weights.sum()
    blend_oof = weights[0]*lgb_oof + weights[1]*cat_oof + weights[2]*xgb_oof
    blend_test = weights[0]*lgb_test + weights[1]*cat_test + weights[2]*xgb_test
    blend_rmse, blend_r2 = eval_score(y, blend_oof)
    print(f"  Blend OOF: RMSE={blend_rmse:.5f}  R2={blend_r2:.4f}")
    print(f"  Blend weights: LGB={weights[0]:.3f}  CAT={weights[1]:.3f}  XGB={weights[2]:.3f}")

    os.makedirs(submit_dir, exist_ok=True)
    all_models = {
        'LightGBM': (lgb_rmse, lgb_r2, lgb_test),
        'CatBoost': (cat_rmse, cat_r2, cat_test),
        'XGBoost': (xgb_rmse, xgb_r2, xgb_test),
        'Stack (Ridge)': (stack_rmse, stack_r2, stack_test_pred),
        'Blend': (blend_rmse, blend_r2, blend_test),
    }
    best_name = min(all_models, key=lambda k: all_models[k][0])
    best_rmse_final, best_r2_final, best_preds_log = all_models[best_name]
    final_preds = np.expm1(best_preds_log).clip(0.0, 1.0)

    index_col = 'Index' if 'Index' in test.columns else None
    if index_col is not None:
        output_df = pd.DataFrame({index_col: test[index_col].values, 'demand': final_preds})
    else:
        output_df = test[['geohash', 'timestamp']].copy()
        output_df['demand'] = final_preds
    output_df.to_csv(os.path.join(submit_dir, 'sample_submission_v9.csv'), index=False)

    for name, (rmse, r2, preds_log) in all_models.items():
        preds = np.expm1(preds_log).clip(0.0, 1.0)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        if index_col is not None:
            pd.DataFrame({index_col: test[index_col], 'demand': preds}).to_csv(
                os.path.join(submit_dir, f'sample_submission_v9_{safe_name}.csv'), index=False)
        else:
            pd.DataFrame({'Index': test.index, 'demand': preds}).to_csv(
                os.path.join(submit_dir, f'sample_submission_v9_{safe_name}.csv'), index=False)

    if os.path.exists(v2_submission_path):
        v2_preds = pd.read_csv(v2_submission_path)['demand'].values
        for w_v9 in [0.3, 0.5, 0.7]:
            blended = w_v9 * final_preds + (1 - w_v9) * v2_preds
            if index_col is not None:
                pd.DataFrame({index_col: test[index_col], 'demand': blended}).to_csv(
                    os.path.join(submit_dir, f'sample_submission_v9_v2_blend{int(w_v9*100)}.csv'), index=False)
            else:
                pd.DataFrame({'Index': test.index, 'demand': blended}).to_csv(
                    os.path.join(submit_dir, f'sample_submission_v9_v2_blend{int(w_v9*100)}.csv'), index=False)
            if len(v2_preds) >= len(y):
                _ = r2_score(np.expm1(y), np.expm1(w_v9 * best_preds_log + (1-w_v9) * np.log1p(v2_preds[:len(y)])).clip(0,1))
            print(f"  Saved v9({int(w_v9*100)}%)+v2({int((1-w_v9)*100)}%) blend")

    total_time = time.time() - t_start
    print("\n")
    print("=" * 70)
    print("  [FINAL RESULTS - v9 PRECISION EDITION]")
    print("=" * 70)
    print("\n  {'Model':<28} {'OOF RMSE':>10} {'OOF R2':>8}  {'Score':>8}")
    print("  {'-'*58}")
    for name, (rmse, r2, _) in all_models.items():
        score = max(0, 100 * r2)
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<28} {rmse:>10.5f} {r2:>8.4f}  {score:>8.2f}{marker}")
    print("  {'-'*58}")
    print()
    print("  BASELINES: v2=91.11 (online) | v8=89.18 (online)")
    print()
    print(f"  Prediction range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    print(f"  Prediction mean : {final_preds.mean():.4f}")
    print(f"  Submission file : {os.path.join(submit_dir, 'sample_submission_v9.csv')} ({len(output_df)} rows)")
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print()
    print("  V9 ANTI-OVERFITTING strategy:")
    print(f"    [FIX] Only {len(FEATURES)} features (vs 135 in v8)")
    print("    [FIX] NO day49 anchor features (midnight != daytime)")
    print("    [FIX] NO pseudo-labeling (compounds errors)")
    print("    [FIX] NO sample weighting (biases toward midnight)")
    print("    [FIX] Careful lag features: 9 (vs 50+ in v8)")
    print("    [ADD] Optuna HPO for both LGB and CatBoost (GPU)")
    print(f"    [ADD] {len(TE_COLS)} target encodings (geo6, geo5, geo4xWeather)")
    print("    [ADD] Quantile features for geo4 demand distribution")
    print("    [ADD] v9+v2 blended submissions for safety")
    print("=" * 70)
    print("  v9 Pipeline complete!")
    print("=" * 70)

    return {
        'best_model_name': best_name,
        'best_rmse': best_rmse_final,
        'best_r2': best_r2_final,
        'features': FEATURES,
        'num_features': len(FEATURES)
    }
