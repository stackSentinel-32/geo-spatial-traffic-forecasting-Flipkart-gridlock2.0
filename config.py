# Flipkart GriLock 2.0 - Configuration Module

# Paths
TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
SUBMISSION_PATH = 'submit/sample_submission_v3.csv'

# Tuning and validation settings
OPTUNA_LGB_TRIALS = 50
OPTUNA_CAT_TRIALS = 30
N_HPO_FOLDS = 3
N_FOLDS = 10

# Hyperparameters
K_NEIGHBORS = 8
SMOOTH_ALPHA = 15
N_TE_FOLDS = 5

# GPU Settings
CATBOOST_TASK_TYPE = 'GPU'
