# config.py

# File Paths
TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
SUBMISSION_PATH = 'catboost_submission.csv'

# Column Configuration
TARGET = 'demand'

FEATURES = [
    'geohash', 'geohash_4', 'geohash_5', 'day', 'hour', 'minute', 'time_of_day_minutes',
    'sin_time', 'cos_time', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks',
    'Temperature', 'Weather', 'lag_demand_day48', 'geohash_overall_mean'
]

CAT_FEATURES = [
    'geohash', 'geohash_4', 'geohash_5', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather'
]

# Model Parameters Tuned to Aggressively Fight Overfitting
CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.03,        # Smooth gradient convergence
    'depth': 5,                   # Shallow depth reduces spatial overfitting
    'l2_leaf_reg': 15,            # High regularization stabilizes high-cardinality splits
    'random_strength': 2.0,       # Introduces diversity to tree structures 
    'bagging_temperature': 0.8,    # Regularizes tree construction via bootstrap sampling
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 100
}