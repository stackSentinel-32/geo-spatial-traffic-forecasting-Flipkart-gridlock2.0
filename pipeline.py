# pipeline.py

import pandas as pd
import numpy as np

def base_preprocessing(df):
    """Parses structural columns, handles categorical text defaults, and adds cyclical attributes."""
    df = df.copy()
    
    # 1. Parse string timestamps into linear continuous timelines
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_of_day_minutes'] = df['hour'] * 60 + df['minute']
    
    # 2. Build Cyclical Mathematical Triggers
    df['sin_time'] = np.sin(2 * np.pi * df['time_of_day_minutes'] / 1440)
    df['cos_time'] = np.cos(2 * np.pi * df['time_of_day_minutes'] / 1440)
    
    # 3. Structural Geohash Resolution Hierarchies
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    
    # 4. Neutralize Missing Values for Categories
    df['RoadType'] = df['RoadType'].fillna('Missing')
    df['Weather'] = df['Weather'].fillna('Missing')
    
    return df

def fit_historical_lookups(train_df):
    """Computes fixed spatial and temporal lookups based strictly on Ground-Truth Day 48."""
    # Isolate complete Day 48 baseline records
    day48_source = train_df[train_df['day'] == 48]
    
    # Target 1: Micro slice lookup
    lag_lookup = day48_source.groupby(['geohash', 'timestamp'])['demand'].mean().reset_index()
    lag_lookup.rename(columns={'demand': 'lag_demand_day48'}, inplace=True)
    
    # Target 2: Macro regional fallback anchor
    geohash_mean_lookup = day48_source.groupby('geohash')['demand'].mean().reset_index()
    geohash_mean_lookup.rename(columns={'demand': 'geohash_overall_mean'}, inplace=True)
    
    # Global training average baseline 
    global_demand_mean = day48_source['demand'].mean()
    
    # Calculate a mapping matrix for Temperature based on time-of-day medians
    train_temp_medians = train_df.groupby('time_of_day_minutes')['Temperature'].median().to_dict()
    global_temp_median = train_df['Temperature'].median()
    
    return {
        'lag_lookup': lag_lookup,
        'geohash_mean_lookup': geohash_mean_lookup,
        'global_demand_mean': global_demand_mean,
        'train_temp_medians': train_temp_medians,
        'global_temp_median': global_temp_median
    }

def apply_historical_lookups(df, lookups):
    """Symmetrically maps train lookups onto incoming dataframes (Train slices or separate Test files)."""
    df = df.copy()
    
    # Merge mappings
    df = df.merge(lookups['lag_lookup'], on=['geohash', 'timestamp'], how='left')
    df = df.merge(lookups['geohash_mean_lookup'], on='geohash', how='left')
    
    # Clean up empty slots using fallback sequence rules
    df['geohash_overall_mean'] = df['geohash_overall_mean'].fillna(lookups['global_demand_mean'])
    df['lag_demand_day48'] = df['lag_demand_day48'].fillna(df['geohash_overall_mean'])
    
    # Apply Time-of-Day Temperature Imputations
    df['Temperature'] = df['Temperature'].fillna(df['time_of_day_minutes'].map(lookups['train_temp_medians']))
    df['Temperature'] = df['Temperature'].fillna(lookups['global_temp_median'])
    
    return df