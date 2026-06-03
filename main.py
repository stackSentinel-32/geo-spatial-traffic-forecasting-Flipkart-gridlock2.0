import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

# ==========================================
# STEP 4.1: LOAD DATA & DEFINE PREPROCESSING
# ==========================================
print("Loading training dataset...")
df = pd.read_csv('dataset/train.csv')

def pipeline_preprocessing(dataframe):
    # 1. Temporal Feature Engineering
    dataframe['hour'] = dataframe['timestamp'].apply(lambda x: int(x.split(':')[0]))
    dataframe['minute'] = dataframe['timestamp'].apply(lambda x: int(x.split(':')[1]))
    dataframe['time_of_day_minutes'] = dataframe['hour'] * 60 + dataframe['minute']
    
    # Cyclical Transformations
    dataframe['sin_time'] = np.sin(2 * np.pi * dataframe['time_of_day_minutes'] / 1440)
    dataframe['cos_time'] = np.cos(2 * np.pi * dataframe['time_of_day_minutes'] / 1440)
    
    # 2. Hierarchical Spatial Engineering
    dataframe['geohash_4'] = dataframe['geohash'].str[:4]
    dataframe['geohash_5'] = dataframe['geohash'].str[:5]
    
    # 3. Categorical Missing Value Handling (Neutralization)
    dataframe['RoadType'] = dataframe['RoadType'].fillna('Missing')
    dataframe['Weather'] = dataframe['Weather'].fillna('Missing')
    
    return dataframe

# Run base transformations
df = pipeline_preprocessing(df)

# ==========================================
# STEP 4.2: GENERATE HISTORICAL LOOKUPS
# ==========================================
print("Generating golden historical features from Day 48...")
day48_data = df[df['day'] == 48]

# Target 1: Exact day-over-day slice lag
lag_lookup = day48_data.groupby(['geohash', 'timestamp'])['demand'].mean().reset_index()
lag_lookup.rename(columns={'demand': 'lag_demand_day48'}, inplace=True)

# Target 2: Macro geographic fallback anchor
geohash_mean_lookup = day48_data.groupby('geohash')['demand'].mean().reset_index()
geohash_mean_lookup.rename(columns={'demand': 'geohash_overall_mean'}, inplace=True)

# Global baseline backup
global_demand_mean = day48_data['demand'].mean()

# Merge lookups back into our main working dataframe
df = df.merge(lag_lookup, on=['geohash', 'timestamp'], how='left')
df = df.merge(geohash_mean_lookup, on='geohash', how='left')

# Fill missing slices gracefully using our hierarchical backups
df['geohash_overall_mean'] = df['geohash_overall_mean'].fillna(global_demand_mean)
df['lag_demand_day48'] = df['lag_demand_day48'].fillna(df['geohash_overall_mean'])

# Option B for Temperature: Fill using time-of-day medians to preserve native patterns
time_temp_medians = df.groupby('time_of_day_minutes')['Temperature'].transform('median')
df['Temperature'] = df['Temperature'].fillna(time_temp_medians).fillna(df['Temperature'].median())

# ==========================================
# STEP 4.3: SETUP LOCAL TIME-BASED VALIDATION
# ==========================================
# Features to train on
features = [
    'geohash', 'geohash_4', 'geohash_5', 'day', 'hour', 'minute', 'time_of_day_minutes',
    'sin_time', 'cos_time', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'lag_demand_day48', 'geohash_overall_mean'
]
cat_features = ['geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
target = 'demand'

# Validation Strategy: Train on Day 48, validate on the training set's Day 49 entries
X_train = df[df['day'] == 48][features]
y_train = df[df['day'] == 48][target]

X_val = df[df['day'] == 49][features]
y_val = df[df['day'] == 49][target]

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# ==========================================
# STEP 4.4: INITIALIZE & RUN CATBOOST
# ==========================================
print("Training CatBoost Regressor...")
model = CatBoostRegressor(
    iterations=2500,
    learning_rate=0.04,       # Low learning rate splits the gradient updates accurately
    depth=7,                  # Depth 7 effortlessly pairs time-of-day with spatial layout
    l2_leaf_reg=5,            # Regularization stabilizes high cardinality noise
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, use_best_model=True)

# ==========================================
# STEP 4.5: APPLY PIPELINE TO THE SEPARATE TEST FILE
# ==========================================
print("\n--- Pipeline ready for Separate Test Dataset ---")
print("""
To predict your final hidden test dataset file, run these lines:
test_df = pd.read_csv('dataset/test.csv')
test_df = pipeline_preprocessing(test_df)
test_df = test_df.merge(lag_lookup, on=['geohash', 'timestamp'], how='left')
test_df = test_df.merge(geohash_mean_lookup, on='geohash', how='left')
test_df['geohash_overall_mean'] = test_df['geohash_overall_mean'].fillna(global_demand_mean)
test_df['lag_demand_day48'] = test_df['lag_demand_day48'].fillna(test_df['geohash_overall_mean'])
test_df['Temperature'] = test_df['Temperature'].fillna(test_df['time_of_day_minutes'].map(df.groupby('time_of_day_minutes')['Temperature'].median()))

predictions = model.predict(test_df[features])
""")