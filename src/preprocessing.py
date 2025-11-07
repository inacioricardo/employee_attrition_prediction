"""
Data preprocessing utilities for the employee attrition prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


class DataPreprocessor:
    """Class to handle data preprocessing operations."""
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return self.numerical_cols, self.categorical_cols
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            for col in self.numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numerical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                     target_col: str = 'Attrition') -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df: Input dataframe
            target_col: Name of the target column
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                                 columns: Optional[List[str]] = None,
                                 fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input dataframe
            columns: List of columns to scale (if None, uses self.numerical_cols)
            fit: Whether to fit the scaler or use existing fit
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_scaled = df.copy()
        cols_to_scale = columns if columns else self.numerical_cols
        
        # Ensure columns exist in dataframe
        cols_to_scale = [col for col in cols_to_scale if col in df_scaled.columns]
        
        if fit:
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
        else:
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])
        
        return df_scaled
    
    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with constant values.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with constant columns removed
        """
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        
        if constant_cols:
            print(f"Removing constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                           target_col: str = 'Attrition',
                           scale_features: bool = True,
                           remove_constants: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
            target_col: Name of the target column
            scale_features: Whether to scale numerical features
            remove_constants: Whether to remove constant columns
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Starting preprocessing pipeline...")
        
        # Identify column types
        self.identify_column_types(df)
        print(f"Identified {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical columns")
        
        # Handle missing values
        df_processed = self.handle_missing_values(df, strategy='mean')
        print(f"Missing values handled")
        
        # Remove constant columns
        if remove_constants:
            df_processed = self.remove_constant_columns(df_processed)
        
        # Encode categorical variables
        df_processed = self.encode_categorical_variables(df_processed, target_col)
        print("Categorical variables encoded")
        
        # Scale numerical features (excluding target if it's numerical)
        if scale_features:
            cols_to_scale = [col for col in self.numerical_cols 
                           if col in df_processed.columns and col != target_col]
            if cols_to_scale:
                df_processed = self.scale_numerical_features(df_processed, columns=cols_to_scale)
                print("Numerical features scaled")
        
        print("Preprocessing complete!")
        return df_processed


def prepare_features_and_target(df: pd.DataFrame, 
                               target_col: str = 'Attrition') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y
