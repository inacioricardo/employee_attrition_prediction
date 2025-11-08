"""
Employee Attrition Prediction Package

This package contains utilities for loading, preprocessing, and analyzing
employee attrition data.
"""

__version__ = "0.1.0"

# Import main functions to make them easily accessible
from .load_data import load_attrition_data
from .preprocessing import preprocess_attrition_data, binary_encode_columns, one_hot_encode_columns, preprocess_pipeline
from .imbalance_handling import balance_with_smote, print_class_distribution
from .train_model import (
    train_logistic_regression,
    visualize_feature_importance,
    visualize_correlation_matrix,
    analyze_top_correlations,
    complete_analysis_workflow,
    train_best_model
)

# Define what gets imported with "from src import *"
__all__ = [
    # Data loading
    'load_attrition_data',
    
    # Preprocessing
    'preprocess_attrition_data',
    'binary_encode_columns',
    'one_hot_encode_columns', 
    'preprocess_pipeline',
    
    # Imbalance handling
    'balance_with_smote',
    'print_class_distribution',
    
    # Model training and analysis
    'train_logistic_regression',
    'visualize_feature_importance',
    'visualize_correlation_matrix',
    'analyze_top_correlations',
    'complete_analysis_workflow',
    'train_best_model'
]
