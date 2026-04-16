import os
import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_PATH, TEST_PATH, SUB_PATH, TARGET_COL, ID_COL, CLASS_ORDER, PALETTE, N_FOLDS, MODEL_PARAMS, RANDOM_STATE
from utils import setup_plotting, load_data, preprocess_data, plot_target_distribution, train_cv, generate_submission

def main():
    print("🚀 Starting Pipeline...")
    setup_plotting()
    
    # 1. Load Data
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    
    # 2. Preprocess
    X_train, X_test, y, le = preprocess_data(train, test, TARGET_COL, ID_COL)
    
    # 3. EDA (Optional)
    # plot_target_distribution(y, le, PALETTE, CLASS_ORDER)
    
    # 4. Train & Cross-Validate
    oof_preds, test_preds, cv_scores = train_cv(X_train, y, X_test, MODEL_PARAMS, N_FOLDS, RANDOM_STATE)
    
    # 5. Generate Submission (Equal weights, or pass custom weights dict)
    generate_submission(test[ID_COL], test_preds, filepath=SUB_PATH)
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
