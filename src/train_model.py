"""
Train models to predict employee attrition with visualization capabilities.
This script can be imported and used in a notebook or as a standalone module.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_logistic_regression(X_train, X_test, y_train, y_test, max_iter=1000):
    """
    Train a logistic regression model and return predictions and model.
    
    Returns:
        model: Trained LogisticRegression model
        y_pred: Predictions on test set
        y_proba: Prediction probabilities
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_proba

def visualize_feature_importance(model, X_train, top_n=15, figsize=(10, 6)):
    """
    Visualize the most important features from a trained model.
    
    Args:
        model: Trained model with coef_ attribute (e.g., LogisticRegression)
        X_train: Training features DataFrame
        top_n: Number of top features to display
        figsize: Figure size tuple
    """
    # Get feature importance (coefficients for logistic regression)
    if hasattr(model, 'coef_'):
        feature_importance = pd.Series(model.coef_[0], index=X_train.columns)
    elif hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    else:
        raise ValueError("Model must have either coef_ or feature_importances_ attribute")
    
    # Get top features by absolute importance
    top_features = feature_importance.abs().sort_values(ascending=False).head(top_n)
    
    # Create visualization
    plt.figure(figsize=figsize)
    sns.barplot(x=top_features.values, y=top_features.index, palette='coolwarm')
    plt.title('Top Features Predicting Attrition (Model Coefficients)')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    return top_features

def visualize_correlation_matrix(X_train, figsize=(16, 12)):
    """
    Create a correlation heatmap for the training features.
    
    Args:
        X_train: Training features DataFrame
        figsize: Figure size tuple
    """
    corr_matrix = X_train.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Feature Correlation Matrix (Processed Features)')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def analyze_top_correlations(X_train, top_n=10, figsize=(10, 6)):
    """
    Analyze and visualize the top correlated feature pairs.
    
    Args:
        X_train: Training features DataFrame
        top_n: Number of top correlation pairs to show
        figsize: Figure size tuple
    """
    corr_matrix = X_train.corr()
    
    # Extract all correlation pairs
    pairs = []
    for i, feat1 in enumerate(corr_matrix.columns):
        for j, feat2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates and self-correlation
                pair_corr = corr_matrix.loc[feat1, feat2]
                pairs.append(((feat1, feat2), abs(pair_corr)))
    
    # Sort by correlation strength
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top_pairs = pairs_sorted[:top_n]
    
    # Prepare data for visualization
    pair_labels = [f'{a} & {b}' for (a, b), _ in top_pairs]
    pair_values = [v for (_, v) in top_pairs]
    
    # Create visualization
    plt.figure(figsize=figsize)
    sns.barplot(x=pair_values, y=pair_labels, palette='coolwarm')
    plt.title(f'Top {top_n} Feature Pairs Most Correlated')
    plt.xlabel('Absolute Correlation Value')
    plt.ylabel('Feature Pair')
    plt.tight_layout()
    plt.show()
    
    return top_pairs

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

def complete_analysis_workflow(X, y, test_size=0.2, random_state=42, visualize=True):
    """
    Complete workflow for training a model and analyzing results.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        visualize: Whether to create visualizations
    
    Returns:
        dict: Contains model, predictions, and analysis results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
    
    # Train logistic regression model
    model, y_pred, y_proba = train_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Print basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create visualizations if requested
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    if visualize:
        print("\nGenerating visualizations...")
        
        # Feature importance
        top_features = visualize_feature_importance(model, X_train)
        results['top_features'] = top_features
        
        # Correlation matrix
        corr_matrix = visualize_correlation_matrix(X_train)
        results['correlation_matrix'] = corr_matrix
        
        # Top correlations
        top_correlations = analyze_top_correlations(X_train)
        results['top_correlations'] = top_correlations
    
    return results
