# pipeline_config.py

# File Paths
TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
SUBMISSION_PATH = 'catboost_submission.csv'

# Column Configuration
TARGET = 'demand'

# Clear, uncorrupted feature matrix
FEATURES = [
    'geohash', 'geohash_4', 'geohash_5', 'day', 'hour', 'minute', 'time_of_day_minutes',
    'sin_time', 'cos_time', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 
    'lag_demand_day48', 'lag_demand_geo5_day48', 'lag_demand_geo4_day48',
    'geohash_overall_mean'
]

CAT_FEATURES = [
    'geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather'
]

# Slower learning rate paired with deep learning capacity to squeeze out raw accuracy
CATBOOST_PARAMS = {
    'iterations': 4000,
    'learning_rate': 0.01,       # Lower learning rate avoids early local minima
    'depth': 6,                  # Increased to depth 6 for higher complexity mapping
    'l2_leaf_reg': 10,
    'random_strength': 2.0,
    'bagging_temperature': 0.8,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 100
}