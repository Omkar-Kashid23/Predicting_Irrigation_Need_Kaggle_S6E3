import os

# Paths (adjust if running locally vs Kaggle)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(DATA_DIR, "submission.csv")

# Constants
RANDOM_STATE = 42
TARGET_COL = 'Irrigation_Need'
ID_COL = 'id'
CLASS_ORDER = ['Low', 'Medium', 'High']
PALETTE = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
TEST_SIZE = 0.05
N_FOLDS = 5
N_TRIALS_OPTUNA = 50  # Adjust based on compute budget

# Base Model Hyperparameters (replace with Optuna best if tuning)
MODEL_PARAMS = {
    'LGBM': {
        'n_estimators': 600, 'learning_rate': 0.05, 'max_depth': 6,
        'colsample_bytree': 0.8, 'subsample': 1.0, 'reg_alpha': 0.1,
        'reg_lambda': 0.1, 'class_weight': 'balanced', 'n_jobs': -1, 
        'verbose': -1, 'random_state': RANDOM_STATE
    },
    'XGB': {
        'n_estimators': 600, 'learning_rate': 0.05, 'max_depth': 6,
        'colsample_bytree': 0.8, 'subsample': 1.0, 'reg_alpha': 1.0,
        'reg_lambda': 1.0, 'eval_metric': 'auc', 'use_label_encoder': False,
        'n_jobs': -1, 'verbosity': 0, 'random_state': RANDOM_STATE
    },
    'CAT': {
        'iterations': 600, 'learning_rate': 0.05, 'depth': 6,
        'l2_leaf_reg': 3.0, 'bagging_temperature': 1.0, 'random_strength': 1.0,
        'border_count': 64, 'verbose': 0, 'random_state': RANDOM_STATE
    }
}

# Plotting
PLOT_CONFIG = {
    'style': 'whitegrid',
    'font_scale': 1.05,
    'dpi': 130,
    'palette': PALETTE,
    'order': CLASS_ORDER
}
