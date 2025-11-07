"""
Feature engineering utilities for the employee attrition prediction project.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Class to handle feature engineering operations."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        pass
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-related features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new tenure features
        """
        df_new = df.copy()
        
        # Years since last promotion ratio
        if 'YearsSinceLastPromotion' in df_new.columns and 'YearsAtCompany' in df_new.columns:
            df_new['PromotionRatio'] = df_new['YearsSinceLastPromotion'] / (df_new['YearsAtCompany'] + 1)
        
        # Years with current manager ratio
        if 'YearsWithCurrManager' in df_new.columns and 'YearsAtCompany' in df_new.columns:
            df_new['ManagerTenureRatio'] = df_new['YearsWithCurrManager'] / (df_new['YearsAtCompany'] + 1)
        
        # Total experience vs company tenure
        if 'TotalWorkingYears' in df_new.columns and 'YearsAtCompany' in df_new.columns:
            df_new['CompanyTenureRatio'] = df_new['YearsAtCompany'] / (df_new['TotalWorkingYears'] + 1)
        
        return df_new
    
    def create_compensation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create compensation-related features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new compensation features
        """
        df_new = df.copy()
        
        # Monthly income per year at company
        if 'MonthlyIncome' in df_new.columns and 'YearsAtCompany' in df_new.columns:
            df_new['IncomePerYearAtCompany'] = df_new['MonthlyIncome'] / (df_new['YearsAtCompany'] + 1)
        
        # Income growth rate
        if 'MonthlyIncome' in df_new.columns and 'PercentSalaryHike' in df_new.columns:
            df_new['IncomeGrowthRate'] = df_new['MonthlyIncome'] * (df_new['PercentSalaryHike'] / 100)
        
        # Hourly rate vs monthly income ratio
        if 'HourlyRate' in df_new.columns and 'MonthlyIncome' in df_new.columns:
            df_new['HourlyToMonthlyRatio'] = df_new['HourlyRate'] / (df_new['MonthlyIncome'] + 1)
        
        return df_new
    
    def create_satisfaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create satisfaction-related composite features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new satisfaction features
        """
        df_new = df.copy()
        
        satisfaction_cols = []
        if 'JobSatisfaction' in df_new.columns:
            satisfaction_cols.append('JobSatisfaction')
        if 'EnvironmentSatisfaction' in df_new.columns:
            satisfaction_cols.append('EnvironmentSatisfaction')
        if 'RelationshipSatisfaction' in df_new.columns:
            satisfaction_cols.append('RelationshipSatisfaction')
        
        if len(satisfaction_cols) > 1:
            df_new['TotalSatisfaction'] = df_new[satisfaction_cols].sum(axis=1)
            df_new['AvgSatisfaction'] = df_new[satisfaction_cols].mean(axis=1)
        
        # Work-life balance score
        if 'WorkLifeBalance' in df_new.columns and 'JobSatisfaction' in df_new.columns:
            df_new['WorkLifeSatisfaction'] = df_new['WorkLifeBalance'] * df_new['JobSatisfaction']
        
        return df_new
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create employee engagement features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new engagement features
        """
        df_new = df.copy()
        
        # Training to years ratio
        if 'TrainingTimesLastYear' in df_new.columns and 'YearsAtCompany' in df_new.columns:
            df_new['TrainingPerYear'] = df_new['TrainingTimesLastYear'] / (df_new['YearsAtCompany'] + 1)
        
        # Job involvement and satisfaction
        if 'JobInvolvement' in df_new.columns and 'JobSatisfaction' in df_new.columns:
            df_new['InvolvementSatisfactionScore'] = df_new['JobInvolvement'] * df_new['JobSatisfaction']
        
        # Performance and satisfaction
        if 'PerformanceRating' in df_new.columns and 'JobSatisfaction' in df_new.columns:
            df_new['PerformanceSatisfactionScore'] = df_new['PerformanceRating'] * df_new['JobSatisfaction']
        
        return df_new
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new age features
        """
        df_new = df.copy()
        
        if 'Age' in df_new.columns:
            # Age groups
            df_new['AgeGroup'] = pd.cut(df_new['Age'], 
                                        bins=[0, 25, 35, 45, 55, 100],
                                        labels=['<25', '25-35', '35-45', '45-55', '55+'])
            
            # Experience vs Age ratio
            if 'TotalWorkingYears' in df_new.columns:
                df_new['ExperienceAgeRatio'] = df_new['TotalWorkingYears'] / df_new['Age']
        
        return df_new
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        print("Creating engineered features...")
        
        df_engineered = df.copy()
        
        df_engineered = self.create_tenure_features(df_engineered)
        print("✓ Tenure features created")
        
        df_engineered = self.create_compensation_features(df_engineered)
        print("✓ Compensation features created")
        
        df_engineered = self.create_satisfaction_features(df_engineered)
        print("✓ Satisfaction features created")
        
        df_engineered = self.create_engagement_features(df_engineered)
        print("✓ Engagement features created")
        
        df_engineered = self.create_age_features(df_engineered)
        print("✓ Age features created")
        
        print(f"Feature engineering complete! New shape: {df_engineered.shape}")
        
        return df_engineered


def create_interaction_features(df: pd.DataFrame, 
                                feature_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create interaction features from pairs of columns.
    
    Args:
        df: Input dataframe
        feature_pairs: List of tuples containing feature pairs
        
    Returns:
        pd.DataFrame: Dataframe with interaction features
    """
    df_new = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df_new.columns and feat2 in df_new.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df_new[interaction_name] = df_new[feat1] * df_new[feat2]
    
    return df_new
