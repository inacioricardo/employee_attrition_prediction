"""
Train the most effective model to predict employee attrition.
This script can be imported and used in a notebook or as a standalone module.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_best_model(X, y, random_state=42, verbose=True):
    """
    Trains and selects the best model for attrition prediction using cross-validation and grid search.
    Returns the best model and its evaluation metrics on a holdout set.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

    # Define candidate models and parameter grids
    models = {
        'RandomForest': (RandomForestClassifier(random_state=random_state), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.3]
        }),
        'LightGBM': (LGBMClassifier(random_state=random_state), {
            'n_estimators': [100, 200],
            'max_depth': [-1, 10],
            'learning_rate': [0.1, 0.3]
        })
    }

    best_score = 0
    best_model = None
    best_name = None
    best_metrics = None

    for name, (model, param_grid) in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        grid = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, 'predict_proba') else None
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            print(f"Best params: {grid.best_params_}")
            print(f"Accuracy: {acc:.4f}")
            if auc is not None:
                print(f"ROC AUC: {auc:.4f}")
            print(classification_report(y_test, y_pred))
        if auc is not None and auc > best_score:
            best_score = auc
            best_model = grid.best_estimator_
            best_name = name
            best_metrics = {
                'accuracy': acc,
                'roc_auc': auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
    if verbose and best_model is not None:
        print(f"\nBest model: {best_name} (ROC AUC: {best_score:.4f})")
    return best_model, best_metrics
