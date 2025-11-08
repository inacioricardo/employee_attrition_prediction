#!/usr/bin/env python3
"""
Command-line interface for Employee Attrition Prediction.

This script demonstrates how to use the package for end-to-end analysis.
"""
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src import (
    load_attrition_data,
    preprocess_attrition_data,
    balance_with_smote,
    complete_analysis_workflow
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Employee Attrition Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                              # Use default data path
  python run_analysis.py --data data/my_data.csv     # Custom data path
  python run_analysis.py --no-visualize              # Skip visualizations
        """
    )
    
    parser.add_argument(
        "--data",
        default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        help="Path to the employee data CSV file"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating visualizations"
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    print("🚀 Employee Attrition Prediction Analysis")
    print("=" * 50)
    
    try:
        # Load data
        print(f"📊 Loading data from: {args.data}")
        df = load_attrition_data(args.data)
        print(f"✅ Loaded {df.shape[0]} employees with {df.shape[1]} features")
        
        # Preprocess data
        print("\n🔄 Preprocessing data...")
        df_processed = preprocess_attrition_data(df)
        print(f"✅ Processed data shape: {df_processed.shape}")
        
        # Prepare features and target
        X = df_processed.drop(columns=['Attrition_bin'])
        y = df_processed['Attrition_bin']
        
        # Apply SMOTE for class balance
        print("\n⚖️ Balancing classes with SMOTE...")
        X_bal, y_bal = balance_with_smote(X, y)
        print(f"✅ Balanced dataset: {X_bal.shape[0]} samples")
        
        # Run complete analysis
        print("\n🤖 Running complete analysis workflow...")
        visualize = not args.no_visualize
        results = complete_analysis_workflow(
            X_bal, y_bal, 
            test_size=0.2, 
            random_state=42, 
            visualize=visualize
        )
        
        print(f"\n🎯 Final Results:")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   ROC AUC: {results['roc_auc']:.4f}")
        
        if visualize:
            print(f"   Top {len(results['top_features'])} features identified")
            print("   Visualizations generated")
        
        print("\n🎉 Analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Error: Data file not found: {args.data}")
        print("Please check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()