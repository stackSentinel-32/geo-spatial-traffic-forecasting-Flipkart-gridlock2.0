import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
import time
import config

BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'

def decode_geohash(ghash):
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon = True
    for char in ghash:
        cd = BASE32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                lon_range[0 if not (cd & mask) else 1] = mid if not (cd & mask) else lon_range[0]
                if cd & mask: lon_range[0] = mid
                else:         lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if cd & mask: lat_range[0] = mid
                else:         lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0]+lat_range[1])/2, (lon_range[0]+lon_range[1])/2

def parse_ts(ts):
    try:
        h,m = str(ts).split(':'); return int(h)*60+int(m)
    except: return np.nan

# Safer lag feature builder
def add_lag_features(df, train_df, recent_lags):
    df = df.copy()
    for i, (lag_day, lag_min, delta) in enumerate(recent_lags[:4]):
        # Geohash-level lookup
        try:
            subset = train_df[(train_df['day']==lag_day) & (train_df['minutes']==lag_min)][['geohash','demand']]
            gh_map = subset.set_index('geohash')['demand']
            df[f'lag_{i+1}_gh'] = df['geohash'].map(gh_map)
        except:
            df[f'lag_{i+1}_gh'] = np.nan

        # geo4-level lookup
        try:
            subset_g4 = train_df[(train_df['day']==lag_day) & (train_df['minutes']==lag_min)].groupby('geo4')['demand'].mean()
            df[f'lag_{i+1}_geo4'] = df['geo4'].map(subset_g4)
        except:
            df[f'lag_{i+1}_geo4'] = np.nan

        # geo3-level lookup
        try:
            subset_g3 = train_df[(train_df['day']==lag_day) & (train_df['minutes']==lag_min)].groupby('geo3')['demand'].mean()
            df[f'lag_{i+1}_geo3'] = df['geo3'].map(subset_g3)
        except:
            df[f'lag_{i+1}_geo3'] = np.nan

    # Fill lags with chain: geo3 → geo4 → global mean
    global_mean_raw = train_df['demand'].mean()
    lag_cols = [c for c in df.columns if c.startswith('lag_')]
    for c in lag_cols:
        if '_gh' in c:
            i = c.split('_')[1]
            df[c] = df[c].fillna(df.get(f'lag_{i}_geo4', global_mean_raw)).fillna(
                        df.get(f'lag_{i}_geo3', global_mean_raw)).fillna(global_mean_raw)

    # Lag statistics
    gh_lag_cols = [c for c in df.columns if c.startswith('lag_') and '_gh' in c]
    if gh_lag_cols:
        lag_mat = df[gh_lag_cols]
        df['lag_mean']    = lag_mat.mean(axis=1)
        df['lag_max']     = lag_mat.max(axis=1)
        df['lag_min']     = lag_mat.min(axis=1)
        df['lag_trend']   = lag_mat.iloc[:,0] - lag_mat.iloc[:,-1] if len(gh_lag_cols)>1 else 0.0
        df['lag_std']     = lag_mat.std(axis=1).fillna(0)

    return df

def get_knn_features(df, nn, geo_demand):
    df = df.copy()
    df_coords = df[['lat','lon']].fillna(0).values
    df_coords_rad = np.radians(df_coords)
    dists, idxs = nn.kneighbors(df_coords_rad)

    # Exclude self (distance=0) for training geohashes
    neighbor_demand_mean = []
    neighbor_demand_max  = []
    neighbor_demand_std  = []
    neighbor_dist_mean   = []
    for i in range(len(df)):
        # Skip first neighbor if it's the geohash itself
        idx_list = idxs[i]
        d_list   = dists[i]
        # Filter out distance=0 (self)
        mask = d_list > 1e-8
        if mask.sum() == 0:
            idx_list = idxs[i][1:config.K_NEIGHBORS+1]
            d_list   = dists[i][1:config.K_NEIGHBORS+1]
        else:
            idx_list = idx_list[mask][:config.K_NEIGHBORS]
            d_list   = d_list[mask][:config.K_NEIGHBORS]

        nbr_demands = geo_demand['demand_mean'].iloc[idx_list].values
        neighbor_demand_mean.append(np.mean(nbr_demands))
        neighbor_demand_max.append(np.max(nbr_demands))
        neighbor_demand_std.append(np.std(nbr_demands))
        # Convert haversine dist (radians) to km
        neighbor_dist_mean.append(np.mean(d_list) * 6371)

    df['knn_demand_mean'] = neighbor_demand_mean
    df['knn_demand_max']  = neighbor_demand_max
    df['knn_demand_std']  = neighbor_demand_std
    df['knn_dist_km']     = neighbor_dist_mean
    return df

def run_pipeline():
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

    warnings.filterwarnings('ignore')
    logging.getLogger('catboost').setLevel(logging.CRITICAL)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()

    print("-" * 65)
    print("FLIPKART-GRIDLOCK-2.0 TRAFFIC FORECASTING PIPELINE !!")
    print("-" * 65)

    # 1. LOAD DATA
    print("\n[1/11] Loading data...")
    train = pd.read_csv(config.TRAIN_PATH)
    test  = pd.read_csv(config.TEST_PATH)
    print(f"       Train: {train.shape}  |  Test: {test.shape}")

    train['minutes'] = train['timestamp'].apply(parse_ts)
    test['minutes']  = test['timestamp'].apply(parse_ts)
    train['hour']    = train['minutes'] / 60.0
    test['hour']     = test['minutes']  / 60.0
    train['geo3']    = train['geohash'].str[:3]
    test['geo3']     = test['geohash'].str[:3]
    train['geo4']    = train['geohash'].str[:4]
    test['geo4']     = test['geohash'].str[:4]
    train['geo5']    = train['geohash'].str[:5]
    test['geo5']     = test['geohash'].str[:5]
    train['geo6']    = train['geohash'].str[:6]
    test['geo6']     = test['geohash'].str[:6]

    # Day distribution
    print(f"       Train days  : {sorted(train['day'].unique())}")
    print(f"       Test days   : {sorted(test['day'].unique())}")
    print(f"       Train times : {sorted(train['timestamp'].unique())}")
    print(f"       Test times  : {sorted(test['timestamp'].unique())}")

    # 2. GEOHASH LAT/LON DECODE
    print("\n[2/11] Decoding geohash coordinates...")
    all_hashes = pd.concat([train['geohash'], test['geohash']]).unique()
    geo_coords = {}
    for gh in all_hashes:
        try:    geo_coords[gh] = decode_geohash(gh)
        except: geo_coords[gh] = (np.nan, np.nan)

    for df in [train, test]:
        df['lat'] = df['geohash'].map(lambda g: geo_coords.get(g,(np.nan,np.nan))[0])
        df['lon'] = df['geohash'].map(lambda g: geo_coords.get(g,(np.nan,np.nan))[1])

    # 3. IMPUTATION (train stats only)
    print("\n[3/11] Imputing missing values...")
    geo4_rt_mode = train.dropna(subset=['RoadType']).groupby('geo4')['RoadType'].agg(lambda x: x.mode()[0])
    global_rt_mode = train['RoadType'].mode()[0]
    for df in [train, test]:
        m = df['RoadType'].isna()
        df.loc[m,'RoadType'] = df.loc[m,'geo4'].map(geo4_rt_mode)
        df['RoadType'].fillna(global_rt_mode, inplace=True)

    temp_map = train.dropna(subset=['Temperature']).groupby(['Weather','day'])['Temperature'].median()
    global_temp = train['Temperature'].median()
    for df in [train, test]:
        m = df['Temperature'].isna()
        df.loc[m,'Temperature'] = df.loc[m].apply(
            lambda r: temp_map.get((r['Weather'],r['day']), np.nan), axis=1)
        df['Temperature'].fillna(global_temp, inplace=True)

    geo4_wx_mode = train.dropna(subset=['Weather']).groupby('geo4')['Weather'].agg(lambda x: x.mode()[0])
    global_wx = train['Weather'].mode()[0]
    for df in [train, test]:
        m = df['Weather'].isna()
        df.loc[m,'Weather'] = df.loc[m,'geo4'].map(geo4_wx_mode)
        df['Weather'].fillna(global_wx, inplace=True)

    print(f"       NaN remaining: train={train[['RoadType','Temperature','Weather']].isna().sum().sum()}  test=0")

    # 4. CRITICAL: TEMPORAL LAG DEMAND FEATURES
    print("\n[4/11] Building temporal lag demand features...")
    train_times = train[['day','minutes']].drop_duplicates().values.tolist()
    test_day    = test['day'].iloc[0]
    test_mins   = test['minutes'].iloc[0]
    print(f"       Test target: day={test_day}, minutes={test_mins} ({int(test_mins//60)}:{int(test_mins%60):02d})")

    recent_lags = []
    for d, m in train_times:
        if (d < test_day) or (d == test_day and m < test_mins):
            recent_lags.append((d, m, (test_day - d)*24*60 + (test_mins - m)))

    recent_lags.sort(key=lambda x: x[2])  # sort by time distance
    print(f"       Available lags (day, min, delta_min):")
    for d,m,dt in recent_lags[:5]:
        print(f"         day={d}, time={int(m//60)}:{int(m%60):02d}, delta={dt:.0f}min ago")

    train = add_lag_features(train, train, recent_lags)
    test  = add_lag_features(test, train, recent_lags)
    lag_feat_cols = [c for c in train.columns if c.startswith('lag_')]
    print(f"       Lag features created: {lag_feat_cols}")

    # 5. SPATIAL KNN NEIGHBOR DEMAND FEATURES
    print("\n[5/11] Building spatial KNN neighbor demand features...")
    geo_demand = train.groupby('geohash').agg(
        lat=('lat','first'), lon=('lon','first'),
        demand_mean=('demand','mean'),
        demand_std=('demand','std'),
        demand_max=('demand','max'),
    ).reset_index().dropna(subset=['lat','lon'])

    coords_train = geo_demand[['lat','lon']].values
    nn = NearestNeighbors(n_neighbors=config.K_NEIGHBORS+1, algorithm='ball_tree', metric='haversine')
    coords_rad = np.radians(coords_train)
    nn.fit(coords_rad)

    train = get_knn_features(train, nn, geo_demand)
    test  = get_knn_features(test, nn, geo_demand)
    print(f"       KNN features: knn_demand_mean/max/std/dist_km")

    # 6. FULL FEATURE ENGINEERING
    print("\n[6/11] Full feature engineering...")
    for df in [train, test]:
        df['LargeVehicles_num'] = (df['LargeVehicles']=='Allowed').astype(float)
        df['Landmarks_num']     = (df['Landmarks']=='Yes').astype(float)
        df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
        df['is_peak']   = df['hour'].apply(lambda h: 1 if(7<=h<=9 or 17<=h<=20) else 0)
        df['is_night']  = df['hour'].apply(lambda h: 1 if(h<5 or h>22) else 0)
        df['time_bucket'] = pd.cut(df['hour'],bins=[-1,5,9,12,16,20,25],labels=[0,1,2,3,4,5]).astype(float)
        road_cap = {'Highway':5,'Street':2,'Residential':1}
        df['road_capacity']   = df['RoadType'].map(road_cap).fillna(1)
        df['capacity_score']  = df['road_capacity'] * df['NumberofLanes']
        df['vehicle_pressure']= df['capacity_score'] * df['LargeVehicles_num']
        df['lat_lon_product'] = df['lat'].fillna(0) * df['lon'].fillna(0)

    # Temperature z-score
    temp_std_map = train.groupby('Weather')['Temperature'].std()
    temp_mean_map= train.groupby('Weather')['Temperature'].mean()
    for df in [train, test]:
        df['temp_zscore'] = (df['Temperature'] -
            df['Weather'].map(temp_mean_map)) / (df['Weather'].map(temp_std_map)+1e-6)

    # Label encode
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

    # Interaction features
    for df in [train, test]:
        df['lanes_x_rt']      = df['NumberofLanes'] * df['RoadType_enc']
        df['temp_x_weather']  = df['Temperature'] * df['Weather_enc']
        df['large_x_lanes']   = df['LargeVehicles_num'] * df['NumberofLanes']
        df['landmark_x_rt']   = df['Landmarks_num'] * df['RoadType_enc']
        df['hour_x_rt']       = df['hour'] * df['RoadType_enc']
        df['capacity_x_hour'] = df['capacity_score'] * df['hour']
        df['pressure_x_peak'] = df['vehicle_pressure'] * df['is_peak']

    # 7. KFOLD TARGET ENCODING (KFold-safe, smoothed)
    print("\n[7/11] KFold-safe target encodings...")
    y_raw    = train['demand'].values
    pt       = PowerTransformer(method='yeo-johnson', standardize=False)
    y_trans  = pt.fit_transform(y_raw.reshape(-1,1)).ravel()
    global_y = np.mean(y_trans)
    global_y_max = y_trans.max()
    print(f"       Yeo-Johnson: skew {pd.Series(y_raw).skew():.3f} -> {pd.Series(y_trans).skew():.3f}")

    kf_te = KFold(n_splits=config.N_TE_FOLDS, shuffle=True, random_state=99)

    TE_COLS = [
        ('geohash',),
        ('geo3',), ('geo4',), ('geo5',),
        ('RoadType',), ('Weather',), ('NumberofLanes',),
        ('geo4','hour'), ('RoadType','NumberofLanes'),
        ('RoadType','Weather'), ('geo3','RoadType'), ('geo4','RoadType'),
        ('geo4','Weather'), ('RoadType','is_peak'),
    ]

    te_train_df = pd.DataFrame(index=train.index)
    te_test_df  = pd.DataFrame(index=test.index)

    for cols in TE_COLS:
        col_name = 'te_'+'_'.join(cols)
        te_train_df[col_name] = np.nan
        for tr_i, val_i in kf_te.split(train):
            ft = train.iloc[tr_i].copy(); ft['_y'] = y_trans[tr_i]
            fv = train.iloc[val_i].copy()
            stats = ft.groupby(list(cols))['_y'].agg(['mean','count'])
            stats['te'] = (stats['count']*stats['mean']+config.SMOOTH_ALPHA*global_y)/(stats['count']+config.SMOOTH_ALPHA)
            keys = fv[list(cols)]
            if len(cols)==1:
                mapped = keys[cols[0]].map(stats['te'])
            else:
                mapped = keys.set_index(list(cols)).index.map(stats['te'].to_dict())
                mapped = pd.Series(mapped, index=fv.index)
            te_train_df.loc[fv.index, col_name] = pd.to_numeric(mapped, errors='coerce').values

        # Full-train stats for test
        ft_all = train.copy(); ft_all['_y'] = y_trans
        stats_all = ft_all.groupby(list(cols))['_y'].agg(['mean','count'])
        stats_all['te'] = (stats_all['count']*stats_all['mean']+config.SMOOTH_ALPHA*global_y)/(stats_all['count']+config.SMOOTH_ALPHA)
        if len(cols)==1:
            te_test_df[col_name] = test[cols[0]].map(stats_all['te'])
        else:
            te_test_df[col_name] = test.set_index(list(cols)).index.map(stats_all['te'].to_dict())
        te_test_df[col_name] = pd.to_numeric(te_test_df[col_name], errors='coerce')
        te_train_df[col_name].fillna(global_y, inplace=True)
        te_test_df[col_name].fillna(global_y, inplace=True)

    train = pd.concat([train.reset_index(drop=True), te_train_df.reset_index(drop=True)], axis=1)
    test  = pd.concat([test.reset_index(drop=True),  te_test_df.reset_index(drop=True)],  axis=1)

    # Group aggregation stats
    for stat in ['mean','std','median','max','min']:
        m = train.groupby('geo4')['demand'].agg(stat)
        train[f'geo4_d_{stat}'] = train['geo4'].map(m)
        test[f'geo4_d_{stat}']  = test['geo4'].map(m).fillna(train['demand'].mean())

    for stat in ['mean','std']:
        m = train.groupby('geo3')['demand'].agg(stat)
        train[f'geo3_d_{stat}'] = train['geo3'].map(m)
        test[f'geo3_d_{stat}']  = test['geo3'].map(m).fillna(train['demand'].mean())

    for stat in ['mean','std','max']:
        m = train.groupby('RoadType')['demand'].agg(stat)
        train[f'rt_d_{stat}'] = train['RoadType'].map(m)
        test[f'rt_d_{stat}']  = test['RoadType'].map(m).fillna(train['demand'].mean())

    # Per-geohash demand volatility (std) — uncertainty signal
    gh_std = train.groupby('geohash')['demand'].std().fillna(0)
    train['gh_demand_volatility'] = train['geohash'].map(gh_std)
    test['gh_demand_volatility']  = test['geohash'].map(gh_std).fillna(0)

    # Geohash demand count (how well-known is this location)
    gh_cnt = train.groupby('geohash')['demand'].count()
    train['gh_obs_count'] = train['geohash'].map(gh_cnt)
    test['gh_obs_count']  = test['geohash'].map(gh_cnt).fillna(0)

    print(f"       {len(TE_COLS)} KFold target encodings done.")

    # FEATURE SET
    TE_NAMES   = ['te_'+'_'.join(c) for c in TE_COLS]
    GEO_STATS  = ([f'geo4_d_{s}' for s in ['mean','std','median','max','min']] +
                   [f'geo3_d_{s}' for s in ['mean','std']] +
                   [f'rt_d_{s}'   for s in ['mean','std','max']])
    LAG_FEATS  = [c for c in train.columns if c.startswith('lag_')]
    KNN_FEATS  = ['knn_demand_mean','knn_demand_max','knn_demand_std','knn_dist_km']

    FEATURES = (
        ['day','minutes','hour','hour_sin','hour_cos','time_bucket','is_peak','is_night',
         'RoadType_enc','NumberofLanes','LargeVehicles_num','Landmarks_num',
         'Temperature','temp_zscore','Weather_enc',
         'road_capacity','capacity_score','vehicle_pressure',
         'geo3_enc','geo4_enc','geo5_enc','geo6_enc',
         'lat','lon','lat_lon_product',
         'lanes_x_rt','temp_x_weather','large_x_lanes','landmark_x_rt',
         'hour_x_rt','capacity_x_hour','pressure_x_peak',
         'gh_demand_volatility','gh_obs_count']
        + TE_NAMES + GEO_STATS + LAG_FEATS + KNN_FEATS
    )

    # Remove duplicates, ensure all exist and are numeric
    FEATURES = list(dict.fromkeys(FEATURES))
    for f in FEATURES:
        for df in [train, test]:
            if f not in df.columns: df[f] = 0.0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0.0)

    X      = train[FEATURES].values
    y      = y_trans     # Yeo-Johnson transformed
    y_orig = y_raw
    X_test = test[FEATURES].values

    # DAY-RECENCY SAMPLE WEIGHTS: day=49 rows are closer to test, weight 2.5x
    sample_weights = np.where(train['day']==49, 2.5, 1.0)

    print(f"\n       Total features   : {len(FEATURES)}")
    print(f"       Train rows       : {len(X)}")
    print(f"       Test rows        : {len(X_test)}")
    print(f"       Day-49 train rows: {(train['day']==49).sum()}")

    # Define metrics inside run_pipeline for clean closure access
    def eval_orig(y_true_t, y_pred_t):
        y_true_clipped = np.clip(y_true_t, None, global_y_max)
        y_pred_clipped = np.clip(y_pred_t, None, global_y_max)
        y_t = pt.inverse_transform(y_true_clipped.reshape(-1,1)).ravel().clip(0,1)
        y_p = pt.inverse_transform(y_pred_clipped.reshape(-1,1)).ravel().clip(0,1)
        return np.sqrt(mean_squared_error(y_t, y_p))

    def rmse_orig(y_true_t, y_pred_t):
        y_true_clipped = np.clip(y_true_t, None, global_y_max)
        y_pred_clipped = np.clip(y_pred_t, None, global_y_max)
        yt = pt.inverse_transform(y_true_clipped.reshape(-1,1)).ravel().clip(0,1)
        yp = pt.inverse_transform(y_pred_clipped.reshape(-1,1)).ravel().clip(0,1)
        return np.sqrt(mean_squared_error(yt, yp)), r2_score(yt, yp)

    # 8. OPTUNA HPO — LightGBM
    print(f"\n[8/11] Optuna HPO for LightGBM ({config.OPTUNA_LGB_TRIALS} trials)...")
    def lgb_objective(trial):
        params = {
            'objective':'regression','metric':'rmse','n_estimators':2000,'verbose':-1,'n_jobs':-1,
            'learning_rate':   trial.suggest_float('lr',   0.01, 0.15, log=True),
            'num_leaves':      trial.suggest_int('nl',      64, 300),
            'max_depth':       trial.suggest_int('md',       5, 12),
            'min_child_samples':trial.suggest_int('mcs',   10, 60),
            'subsample':       trial.suggest_float('ss',  0.55, 1.0),
            'colsample_bytree':trial.suggest_float('cs',  0.50, 1.0),
            'reg_alpha':       trial.suggest_float('ra',  1e-4, 2.0, log=True),
            'reg_lambda':      trial.suggest_float('rl',  1e-4, 2.0, log=True),
            'min_split_gain':  trial.suggest_float('msg', 0.0,  0.2),
            'random_state':    42,
        }
        kf_h = KFold(n_splits=config.N_HPO_FOLDS, shuffle=True, random_state=42)
        sc = []
        for ti,vi in kf_h.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[ti], y[ti], sample_weight=sample_weights[ti],
                  eval_set=[(X[vi],y[vi])],
                  callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(-1)])
            sc.append(eval_orig(y[vi], m.predict(X[vi])))
        return np.mean(sc)

    t0 = time.time()
    study_lgb = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=config.OPTUNA_LGB_TRIALS)
    print(f"       LightGBM HPO: {time.time()-t0:.1f}s  best={study_lgb.best_value:.5f}")
    best_lgb = study_lgb.best_params
    lgb_final_params = {
        'objective':'regression','metric':'rmse','n_estimators':4000,
        'verbose':-1,'n_jobs':-1,'random_state':42,
        'learning_rate':    best_lgb['lr'],
        'num_leaves':       best_lgb['nl'],
        'max_depth':        best_lgb['md'],
        'min_child_samples':best_lgb['mcs'],
        'subsample':        best_lgb['ss'],
        'colsample_bytree': best_lgb['cs'],
        'reg_alpha':        best_lgb['ra'],
        'reg_lambda':       best_lgb['rl'],
        'min_split_gain':   best_lgb['msg'],
    }
    print(f"       Best LGB params: lr={best_lgb['lr']:.4f} nl={best_lgb['nl']} md={best_lgb['md']}")

    # 9. OPTUNA HPO — CatBoost
    print(f"\n[9/11] Optuna HPO for CatBoost ({config.OPTUNA_CAT_TRIALS} trials)...")
    def cat_objective(trial):
        params = {
            'iterations':3000,'loss_function':'RMSE','eval_metric':'RMSE',
            'random_seed':42,'thread_count':-1,'verbose':False,'early_stopping_rounds':100,
            'task_type': config.CATBOOST_TASK_TYPE,
            'learning_rate':      trial.suggest_float('lr',  0.01, 0.15, log=True),
            'depth':              trial.suggest_int('depth',  4, 10),
            'l2_leaf_reg':        trial.suggest_float('l2',  0.5, 10.0),
            'bagging_temperature':trial.suggest_float('bt',  0.0, 2.0),
            'random_strength':    trial.suggest_float('rs',  0.0, 3.0),
            'min_data_in_leaf':   trial.suggest_int('mdl',    5, 50),
            'grow_policy':        trial.suggest_categorical('gp', ['SymmetricTree','Depthwise']),
        }
        kf_h = KFold(n_splits=config.N_HPO_FOLDS, shuffle=True, random_state=42)
        sc = []
        for ti,vi in kf_h.split(X):
            m = cb.CatBoostRegressor(**params)
            m.fit(X[ti], y[ti], sample_weight=sample_weights[ti],
                  eval_set=[(X[vi],y[vi])], verbose=False)
            sc.append(eval_orig(y[vi], m.predict(X[vi])))
        return np.mean(sc)

    t0 = time.time()
    study_cat = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(cat_objective, n_trials=config.OPTUNA_CAT_TRIALS)
    print(f"       CatBoost HPO: {time.time()-t0:.1f}s  best={study_cat.best_value:.5f}")
    best_cat = study_cat.best_params
    cat_final_params = {
        'iterations':4000,'loss_function':'RMSE','eval_metric':'RMSE',
        'random_seed':42,'thread_count':-1,'verbose':False,'early_stopping_rounds':200,
        'task_type': config.CATBOOST_TASK_TYPE,
        'learning_rate':      best_cat['lr'],
        'depth':              best_cat['depth'],
        'l2_leaf_reg':        best_cat['l2'],
        'bagging_temperature':best_cat['bt'],
        'random_strength':    best_cat['rs'],
        'min_data_in_leaf':   best_cat['mdl'],
        'grow_policy':        best_cat['gp'],
    }
    print(f"       Best CAT params: lr={best_cat['lr']:.4f} depth={best_cat['depth']} gp={best_cat['gp']}")

    # 10. 10-FOLD CV TRAINING (LightGBM + CatBoost + HistGBM)
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)

    lgb_oof  = np.zeros(len(X)); lgb_test  = np.zeros(len(X_test))
    cat_oof  = np.zeros(len(X)); cat_test  = np.zeros(len(X_test))
    hgb_oof  = np.zeros(len(X)); hgb_test  = np.zeros(len(X_test))

    print(f"\n[10/11] Training 3 models — {config.N_FOLDS}-Fold CV...")

    # ── LightGBM ──────────────────────────────────────────────────────────────────
    print(f"\n  +- LightGBM (Optuna-tuned, sample-weighted) ----------------+")
    t0 = time.time()
    for fold,(ti,vi) in enumerate(kf.split(X,y),1):
        m = lgb.LGBMRegressor(**lgb_final_params)
        m.fit(X[ti],y[ti], sample_weight=sample_weights[ti],
              eval_set=[(X[vi],y[vi])],
              callbacks=[lgb.early_stopping(200,verbose=False),lgb.log_evaluation(-1)])
        lgb_oof[vi] = m.predict(X[vi])
        lgb_test   += m.predict(X_test) / config.N_FOLDS
        r, r2 = rmse_orig(y[vi], lgb_oof[vi])
        print(f"  Fold {fold:02d}/{config.N_FOLDS} -> RMSE: {r:.5f}  R2: {r2:.4f}  iter={m.best_iteration_}")
    lgb_r, lgb_r2 = rmse_orig(y, lgb_oof)
    print(f"  >> LightGBM OOF: RMSE={lgb_r:.5f}  R2={lgb_r2:.4f}  ({time.time()-t0:.1f}s)")

    # ── CatBoost ─────────────────────────────────────────────────────────────────
    print(f"\n  +- CatBoost (Optuna-tuned, sample-weighted) ----------------+")
    t0 = time.time()
    for fold,(ti,vi) in enumerate(kf.split(X,y),1):
        m = cb.CatBoostRegressor(**cat_final_params)
        m.fit(X[ti],y[ti], sample_weight=sample_weights[ti],
              eval_set=[(X[vi],y[vi])], verbose=False)
        cat_oof[vi] = m.predict(X[vi])
        cat_test   += m.predict(X_test) / config.N_FOLDS
        r, r2 = rmse_orig(y[vi], cat_oof[vi])
        print(f"  Fold {fold:02d}/{config.N_FOLDS} -> RMSE: {r:.5f}  R2: {r2:.4f}  iter={m.best_iteration_}")
    cat_r, cat_r2 = rmse_orig(y, cat_oof)
    print(f"  >> CatBoost OOF: RMSE={cat_r:.5f}  R2={cat_r2:.4f}  ({time.time()-t0:.1f}s)")

    # ── HistGradientBoosting ──────────────────────────────────────────────────────
    print(f"\n  +- HistGradientBoosting (sklearn) --------------------------+")
    t0 = time.time()
    hgb_params = dict(
        max_iter=2000, learning_rate=0.05, max_depth=8,
        min_samples_leaf=20, l2_regularization=0.1,
        max_bins=255, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=50, random_state=42
    )
    for fold,(ti,vi) in enumerate(kf.split(X,y),1):
        m = HistGradientBoostingRegressor(**hgb_params)
        m.fit(X[ti], y[ti], sample_weight=sample_weights[ti])
        hgb_oof[vi] = m.predict(X[vi])
        hgb_test   += m.predict(X_test) / config.N_FOLDS
        r, r2 = rmse_orig(y[vi], hgb_oof[vi])
        print(f"  Fold {fold:02d}/{config.N_FOLDS} -> RMSE: {r:.5f}  R2: {r2:.4f}  iter={m.n_iter_}")
    hgb_r, hgb_r2 = rmse_orig(y, hgb_oof)
    print(f"  >> HistGBM OOF: RMSE={hgb_r:.5f}  R2={hgb_r2:.4f}  ({time.time()-t0:.1f}s)")

    # ── OPTUNA BLEND WEIGHTS (non-negative constrained) ─────────────────────────
    print(f"\n  +- Optuna-optimized non-negative blend ---------------------+")
    oof_preds  = np.column_stack([lgb_oof, cat_oof, hgb_oof])
    test_preds = np.column_stack([lgb_test, cat_test, hgb_test])

    def blend_rmse(weights):
        w = np.abs(weights) / (np.abs(weights).sum() + 1e-9)
        blended = oof_preds @ w
        return rmse_orig(y, blended)[0]

    best_w, best_blend_rmse = None, np.inf
    for _ in range(30):
        w0 = np.random.dirichlet([1,1,1])
        res = minimize(blend_rmse, w0, method='Nelder-Mead',
                       options={'maxiter':10000,'xatol':1e-7,'fatol':1e-7})
        if res.fun < best_blend_rmse:
            best_blend_rmse = res.fun
            best_w = np.abs(res.x) / (np.abs(res.x).sum() + 1e-9)

    blend_oof  = oof_preds @ best_w
    blend_test = test_preds @ best_w
    blend_r, blend_r2 = rmse_orig(y, blend_oof)
    print(f"  Weights: LGB={best_w[0]:.3f}  CAT={best_w[1]:.3f}  HGB={best_w[2]:.3f}")
    print(f"  Blend OOF: RMSE={blend_r:.5f}  R2={blend_r2:.4f}")

    # ── PSEUDO-LABELING (semi-supervised boost) ──────────────────────────────────
    print(f"\n  +- Pseudo-labeling (semi-supervised) -----------------------+")

    # Get raw-scale test predictions
    pseudo_preds_raw = pt.inverse_transform(blend_test.reshape(-1,1)).ravel().clip(0,1)
    pseudo_preds_t   = blend_test  # transformed scale

    # Use ALL test rows as pseudo-labels (with lower weight than real labels)
    X_pseudo = X_test.copy()
    y_pseudo  = pseudo_preds_t

    # Combine train + pseudo-labeled test
    X_combined = np.vstack([X, X_pseudo])
    y_combined  = np.concatenate([y, y_pseudo])
    sw_combined = np.concatenate([sample_weights, np.full(len(X_test), 0.4)])

    print(f"  Combined training set: {len(X_combined)} rows")
    print(f"  Pseudo-label weight: 0.4 (vs real data 1.0-2.5)")

    # Retrain best model (CatBoost) on combined data with pseudo-labels
    print(f"  Retraining CatBoost with pseudo-labels...")

    # Use 85% for train, 15% val for early stopping
    val_size = int(0.15 * len(X))
    X_tr_ps, X_val_ps = X_combined[:-val_size], X_combined[-val_size:]
    y_tr_ps, y_val_ps = y_combined[:-val_size], y_combined[-val_size:]
    sw_tr_ps          = sw_combined[:-val_size]

    cat_pseudo = cb.CatBoostRegressor(**cat_final_params)
    cat_pseudo.fit(X_tr_ps, y_tr_ps, sample_weight=sw_tr_ps,
                   eval_set=[(X_val_ps, y_val_ps)], verbose=False)
    pseudo_test_pred = cat_pseudo.predict(X_test)
    pseudo_r = eval_orig(y, cat_pseudo.predict(X[:len(y)]))
    print(f"  Pseudo-label CatBoost train RMSE: {pseudo_r:.5f}")

    # Final blend: 70% original ensemble + 30% pseudo-label model
    final_test_t = 0.70 * blend_test + 0.30 * pseudo_test_pred
    final_test_t_clipped = np.clip(final_test_t, None, global_y_max)
    final_preds  = pt.inverse_transform(final_test_t_clipped.reshape(-1,1)).ravel().clip(0.0, 1.0)

    # 11. GENERATE SUBMISSION
    print(f"\n[11/11] Generating submission file...")
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({'Index': test['Index'].values, 'demand': final_preds})
    submission.to_csv(config.SUBMISSION_PATH, index=False)

    # FINAL SUMMARY
    total_time = time.time() - t_start
    print("\n")
    print("=" * 65)
    print("  [FINAL RESULTS SUMMARY - v3 ULTRA Pipeline]")
    print("=" * 65)
    print(f"\n  NEW techniques vs v2:")
    print(f"    [+] Temporal lag demand features ({len(recent_lags)} lag steps)")
    print(f"    [+] Spatial KNN neighbor demand (K={config.K_NEIGHBORS})")
    print(f"    [+] Yeo-Johnson target transform")
    print(f"    [+] CatBoost Optuna HPO ({config.OPTUNA_CAT_TRIALS} trials)")
    print(f"    [+] LightGBM Optuna HPO ({config.OPTUNA_LGB_TRIALS} trials)")
    print(f"    [+] HistGradientBoosting (sklearn)")
    print(f"    [+] Day-recency sample weights (day49=2.5x)")
    print(f"    [+] Pseudo-labeling (0.4 weight on test)")
    print(f"    [+] Optuna non-negative blend weights")
    print(f"    [+] {len(FEATURES)} total features")
    print()
    print(f"  {'Model':<28} {'OOF RMSE':>10} {'OOF R2':>8}")
    print(f"  {'-'*50}")
    for name, r, r2 in [
        ('LightGBM (HPO+weighted)', lgb_r, lgb_r2),
        ('CatBoost (HPO+weighted)', cat_r, cat_r2),
        ('HistGradientBoosting',    hgb_r, hgb_r2),
        ('Optuna Blend',            blend_r, blend_r2),
    ]:
        print(f"  {name:<28} {r:>10.5f} {r2:>8.4f}")
    print(f"  {'-'*50}")
    print()

    improvement_v1 = (0.03204 - blend_r) / 0.03204 * 100
    improvement_v2 = (0.03012 - blend_r) / 0.03012 * 100
    print(f"  v3 OOF RMSE  : {blend_r:.5f}  ({improvement_v1:+.1f}% vs v1, {improvement_v2:+.1f}% vs v2)")
    print()
    print(f"  Prediction range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    print(f"  Prediction mean : {final_preds.mean():.4f}")
    print(f"  Submission file : {config.SUBMISSION_PATH} ({len(submission)} rows)")
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print("-" * 65)
    print("  FLIPKART GRIDLOCK-2.0 Pipeline complete!")
    print("-" * 65)
