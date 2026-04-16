import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==================== PLOTTING ====================
def setup_plotting(style='whitegrid', font_scale=1.05, dpi=130):
    sns.set_theme(style=style, font_scale=font_scale)
    plt.rcParams.update({
        'figure.dpi': dpi,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

def plot_target_distribution(y, le, palette, order):
    labels = le.inverse_transform(y)
    counts = pd.Series(labels).value_counts().loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(counts.index, counts.values, color=[palette[c] for c in order], edgecolor='black')
    axes[0].set_title('Class Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5000, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
    axes[1].pie(counts.values, labels=order, colors=[palette[c] for c in order],
                autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.5, edgecolor='white'))
    axes[1].set_title('Class Share (Donut)')
    plt.tight_layout()
    plt.show()

# ==================== DATA ====================
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"✅ Loaded | Train: {train.shape} | Test: {test.shape}")
    return train, test

def preprocess_data(train, test, target_col, id_col, le=None):
    y = train[target_col]
    if le is None:
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    X_train = train.drop(columns=[id_col, target_col])
    X_test = test.drop(columns=[id_col])
    
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    return X_train, X_test, y, le

# ==================== TRAINING & CV ====================
def train_cv(X_train, y_train, X_test, model_configs, n_folds=5, random_state=42):
    oof_preds = {name: np.zeros((len(y_train), len(np.unique(y_train)))) for name in model_configs}
    test_preds = {name: np.zeros((len(X_test), len(np.unique(y_train)))) for name in model_configs}
    cv_scores = {}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for name, params in model_configs.items():
        print(f"\n{'='*40}\n🚀 Training: {name}\n{'='*40}")
        ModelClass = {'LGBM': 'lgb', 'XGB': 'xgb', 'CAT': 'cbt'}[name]
        model_oof = np.zeros((len(y_train), len(np.unique(y_train))))
        model_test = np.zeros((len(X_test), len(np.unique(y_train))))
        fold_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            if ModelClass == 'CAT':
                model = catboost.CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif ModelClass == 'XGB':
                model = xgboost.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model = lightgbm.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            model_oof[val_idx] = model.predict_proba(X_val)
            model_test += model.predict_proba(X_test)
            
            fold_aucs.append(roc_auc_score(y_val, model_oof[val_idx], multi_class='ovr'))
            print(f"  📊 Fold {fold+1} AUC: {fold_aucs[-1]:.4f}")
            
        model_test /= n_folds
        mean_auc = np.mean(fold_aucs)
        print(f"  ✅ {name} CV AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}\n")
        oof_preds[name] = model_oof
        test_preds[name] = model_test
        cv_scores[name] = mean_auc
        
    return oof_preds, test_preds, cv_scores

# ==================== OPTUNA TUNING (Optional) ====================
def tune_model(X_train, y_train, model_name, n_trials=50, n_folds=5, random_state=42):
    def objective(trial):
        if model_name == 'LGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1200, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'class_weight': 'balanced', 'n_jobs': -1, 'verbose': -1, 'random_state': random_state
            }
            model = lightgbm.LGBMClassifier(**params)
        elif model_name == 'XGB':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1200, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'eval_metric': 'auc', 'random_state': random_state, 'n_jobs': -1, 'verbosity': 0
            }
            model = xgboost.XGBClassifier(**params)
        elif model_name == 'CAT':
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1200, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 3),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': random_state, 'verbose': 0
            }
            model = catboost.CatBoostClassifier(**params)
            
        oof = np.zeros(len(y_train))
        for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state).split(X_train, y_train)):
            model.fit(X_train.iloc[train_idx], y_train[train_idx], 
                      eval_set=[(X_train.iloc[val_idx], y_train[val_idx])], verbose=False)
            oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1] if hasattr(model, 'predict_proba') else 0
            
        return roc_auc_score(y_train, oof, multi_class='ovr', average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"✅ Best {model_name} AUC: {study.best_value:.5f}")
    return study.best_params

# ==================== SUBMISSION ====================
def generate_submission(test_ids, oof_preds, weights=None, filepath="submission.csv"):
    model_names = list(oof_preds.keys())
    weights = np.array(weights) / np.sum(weights) if weights is not None else np.ones(len(model_names)) / len(model_names)
    
    blended_test = np.zeros_like(oof_preds[model_names[0]])
    for name, preds in oof_preds.items():
        blended_test += preds * weights[model_names.index(name)]
        
    decoded_preds = LabelEncoder().fit_transform(['Low', 'Medium', 'High'])
    final_preds = np.argmax(blended_test, axis=1)
    decoded_preds = ['Low', 'Medium', 'High'][np.argmax(blended_test, axis=1)]
    
    sub = pd.DataFrame({'id': test_ids, 'Irrigation_Need': decoded_preds})
    sub.to_csv(filepath, index=False)
    print(f"✅ Submission saved to {filepath}")
    print("\nPrediction Distribution:")
    print(sub['Irrigation_Need'].value_counts())
    print("\nSample:")
    print(sub.head())
    return sub
