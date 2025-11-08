"""
Script to load employee attrition data.
Can be imported and used in notebooks or other scripts.
"""
import pandas as pd

def load_attrition_data(path='data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    """
    Loads the employee attrition dataset from the given path.
    Returns:
        df: pandas DataFrame
    """
    df = pd.read_csv(path)
    return df
