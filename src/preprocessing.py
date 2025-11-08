"""
Preprocessing script for employee attrition data.
This script can be imported and used in a notebook or as a standalone module.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def binary_encode_columns(df, columns):
    """
    Encode binary categorical columns to 0/1 values.
    Creates new columns with '_bin' suffix.
    """
    df_encoded = df.copy()
    for col in columns:
        if col in df_encoded.columns:
            unique_vals = df_encoded[col].dropna().unique()
            if len(unique_vals) == 2:
                val_map = None
                if set(unique_vals) == set(['Yes', 'No']):
                    val_map = {'No': 0, 'Yes': 1}
                elif set(unique_vals) == set(['Y', 'N']):
                    val_map = {'N': 0, 'Y': 1}
                elif set(unique_vals) == set(['Male', 'Female']):
                    val_map = {'Female': 0, 'Male': 1}
                else:
                    sorted_vals = sorted(unique_vals)
                    val_map = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                df_encoded[col + '_bin'] = df_encoded[col].map(val_map)
    return df_encoded

def one_hot_encode_columns(df, columns):
    """
    One-hot encode categorical columns with descriptive names.
    Uses column prefixes to maintain meaningful feature names.
    """
    df_encoded = df.copy()
    cols_to_encode = [col for col in columns if col in df_encoded.columns]
    if cols_to_encode:
        df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=False, prefix=cols_to_encode)
    return df_encoded

def handle_missing_values(df, numerical_cols, categorical_cols, strategy='mean'):
    df_clean = df.copy()
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'mean':
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    return df_clean

def scale_numerical_features(df, columns, fit=True, scaler=None):
    df_scaled = df.copy()
    if scaler is None:
        scaler = StandardScaler()
    if fit and columns:
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    elif columns:
        df_scaled[columns] = scaler.transform(df_scaled[columns])
    return df_scaled, scaler

def remove_constant_columns(df):
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df

def preprocess_pipeline(df, target_col='Attrition', scale_features=True, remove_constants=True):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_processed = handle_missing_values(df, numerical_cols, categorical_cols, strategy='mean')
    if remove_constants:
        df_processed = remove_constant_columns(df_processed)
    # Label encode all categorical variables
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    if scale_features:
        cols_to_scale = [col for col in numerical_cols if col in df_processed.columns and col != target_col]
        if cols_to_scale:
            df_processed, _ = scale_numerical_features(df_processed, cols_to_scale, fit=True)
    return df_processed

def preprocess_attrition_data(df):
    """
    Complete preprocessing pipeline specific to attrition data.
    Handles categorical encoding with meaningful names for visualisation.
    Education is kept as numeric (1-5 ordinal scale).
    
    Returns:
        df_processed: Preprocessed DataFrame ready for modeling
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Keep Education as numeric (1-5 represents education level hierarchy)
    # 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor
    
    # Define column groups based on proper categorical analysis
    # ONE-HOT ENCODING: Categorical text columns with no inherent order
    one_hot_cols = [
        "BusinessTravel",      # 3 categories: Travel_Rarely, Travel_Frequently, Non-Travel  
        "Department",          # 3 categories: Sales, Research & Development, Human Resources
        "EducationField",      # 6 categories: Life Sciences, Medical, Marketing, etc.
        "JobRole",            # 9 categories: Sales Executive, Research Scientist, etc.
        "MaritalStatus"       # 3 categories: Single, Married, Divorced
    ]
    
    # BINARY ENCODING: Yes/No, Male/Female type columns
    binary_cols = ["Attrition", "OverTime", "Gender", "Over18"]
    
    # ORDINAL/NUMERIC: Keep as-is (Education, satisfaction ratings, etc.)
    # These include: Education, EnvironmentSatisfaction, JobInvolvement, JobLevel,
    # JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance, etc.
    
    # Apply one-hot encoding with descriptive names
    df_encoded = pd.get_dummies(df_copy, columns=one_hot_cols, prefix=one_hot_cols, drop_first=False)
    
    # Apply binary encoding
    df_encoded = binary_encode_columns(df_encoded, binary_cols)
    
    # Drop original binary columns after encoding
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[col])
    
    # Apply final preprocessing pipeline but exclude Education from scaling
    # since we want to keep it as ordinal 1-5 values
    df_processed = preprocess_pipeline_custom(df_encoded, target_col="Attrition_bin", 
                                            scale_features=True, remove_constants=True,
                                            exclude_from_scaling=['Education'])
    
    return df_processed

def preprocess_pipeline_custom(df, target_col='Attrition', scale_features=True, remove_constants=True, exclude_from_scaling=None):
    """
    Custom preprocessing pipeline that allows excluding specific columns from scaling.
    """
    if exclude_from_scaling is None:
        exclude_from_scaling = []
        
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_processed = handle_missing_values(df, numerical_cols, categorical_cols, strategy='mean')
    if remove_constants:
        df_processed = remove_constant_columns(df_processed)
    # Label encode all categorical variables
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    if scale_features:
        cols_to_scale = [col for col in numerical_cols 
                        if col in df_processed.columns 
                        and col != target_col 
                        and col not in exclude_from_scaling]
        if cols_to_scale:
            df_processed, _ = scale_numerical_features(df_processed, cols_to_scale, fit=True)
    return df_processed
