"""
Imbalance handling script for employee attrition data.
This script can be imported and used in a notebook or as a standalone module.
"""
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

def balance_with_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance the dataset.
    Args:
        X: Features DataFrame
        y: Target Series
        random_state: Random seed for reproducibility
    Returns:
        X_res, y_res: Balanced features and target
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def print_class_distribution(y, label='Class distribution'):
    counter = Counter(y)
    print(f"{label}: {dict(counter)}")
