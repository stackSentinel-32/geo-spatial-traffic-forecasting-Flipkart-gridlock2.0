# pipeline.py

import pandas as pd
import numpy as np

def base_preprocessing(df):
    df = df.copy()
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_of_day_minutes'] = df['hour'] * 60 + df['minute']
    
    df['sin_time'] = np.sin(2 * np.pi * df['time_of_day_minutes'] / 1440)
    df['cos_time'] = np.cos(2 * np.pi * df['time_of_day_minutes'] / 1440)
    
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    
    df['RoadType'] = df['RoadType'].fillna('Missing')
    df['Weather'] = df['Weather'].fillna('Missing')
    return df

def fit_historical_lookups(train_df):
    day48_source = train_df[train_df['day'] == 48]
    
    # 1. Exact History Match
    lag_lookup = day48_source.groupby(['geohash', 'timestamp'])['demand'].mean().reset_index()
    lag_lookup.rename(columns={'demand': 'lag_demand_day48'}, inplace=True)
    
    # 2. Neighborhood Smoothing (Micro-region)
    geo5_lookup = day48_source.groupby(['geohash_5', 'timestamp'])['demand'].mean().reset_index()
    geo5_lookup.rename(columns={'demand': 'lag_demand_geo5_day48'}, inplace=True)
    
    # 3. District Smoothing (Macro-region)
    geo4_lookup = day48_source.groupby(['geohash_4', 'timestamp'])['demand'].mean().reset_index()
    geo4_lookup.rename(columns={'demand': 'lag_demand_geo4_day48'}, inplace=True)
    
    # 4. Long-term static spatial baseline
    geohash_mean_lookup = day48_source.groupby('geohash')['demand'].mean().reset_index()
    geohash_mean_lookup.rename(columns={'demand': 'geohash_overall_mean'}, inplace=True)
    
    global_demand_mean = day48_source['demand'].mean()
    
    # Temperature profile mapping
    train_temp_medians = train_df.groupby('time_of_day_minutes')['Temperature'].median().to_dict()
    global_temp_median = train_df['Temperature'].median()
    
    return {
        'lag_lookup': lag_lookup,
        'geo5_lookup': geo5_lookup,
        'geo4_lookup': geo4_lookup,
        'geohash_mean_lookup': geohash_mean_lookup,
        'global_demand_mean': global_demand_mean,
        'train_temp_medians': train_temp_medians,
        'global_temp_median': global_temp_median
    }

def apply_historical_lookups(df, lookups):
    df = df.copy()
    
    df = df.merge(lookups['lag_lookup'], on=['geohash', 'timestamp'], how='left')
    df = df.merge(lookups['geo5_lookup'], on=['geohash_5', 'timestamp'], how='left')
    df = df.merge(lookups['geo4_lookup'], on=['geohash_4', 'timestamp'], how='left')
    df = df.merge(lookups['geohash_mean_lookup'], on='geohash', how='left')
    
    # Cascade back-fills
    df['geohash_overall_mean'] = df['geohash_overall_mean'].fillna(lookups['global_demand_mean'])
    df['lag_demand_day48'] = df['lag_demand_day48'].fillna(df['geohash_overall_mean'])
    df['lag_demand_geo5_day48'] = df['lag_demand_geo5_day48'].fillna(df['lag_demand_day48'])
    df['lag_demand_geo4_day48'] = df['lag_demand_geo4_day48'].fillna(df['lag_demand_geo5_day48'])
    
    # Temperature Imputation
    df['Temperature'] = df['Temperature'].fillna(df['time_of_day_minutes'].map(lookups['train_temp_medians']))
    df['Temperature'] = df['Temperature'].fillna(lookups['global_temp_median'])
    
    return df