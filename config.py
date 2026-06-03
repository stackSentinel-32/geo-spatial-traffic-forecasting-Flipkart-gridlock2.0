# config.py

# File Paths
TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
SUBMISSION_PATH = 'catboost_submission.csv'

# Column Configuration
TARGET = 'demand'

# Cleaned features list (No future-bias trend factors inside the tree splits)
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

# Optimal Regularization Hyperparameters
CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 5,
    'l2_leaf_reg': 15,
    'random_strength': 2.0,
    'bagging_temperature': 0.8,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 100
}