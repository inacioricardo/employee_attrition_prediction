"""
Data loading utilities for the employee attrition prediction project.
"""

import pandas as pd
import os
from typing import Optional


class DataLoader:
    """Class to handle data loading operations."""
    
    def __init__(self, data_path: str = "../WA_Fn-UseC_-HR-Employee-Attrition.csv"):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the employee attrition dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        return info
    
    def quick_summary(self, df: pd.DataFrame) -> None:
        """
        Print a quick summary of the dataset.
        
        Args:
            df: Input dataframe
        """
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Number of rows: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print(f"\nMissing values: {df.isnull().sum().sum()}")
        
        if 'Attrition' in df.columns:
            attrition_rate = (df['Attrition'].value_counts()['Yes'] / len(df)) * 100
            print(f"Attrition rate: {attrition_rate:.2f}%")
        
        print("\nColumn types:")
        print(df.dtypes.value_counts())
        print("=" * 60)


def load_employee_data(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load employee attrition data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    loader = DataLoader(file_path)
    return loader.load_data()
