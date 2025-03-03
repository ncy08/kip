#!/usr/bin/env python3
"""
Simplified script to run SHAP analysis on the latest run using the XGBoost model.
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Define the paths directly
MODEL_PATH = "models/xgboost_optuna/model.json"
DATA_PATH = "data/runs.csv"  # Update this if your data is in a different location

def extract_latest_run(df, date_column='date'):
    """Extract the latest run from the dataset."""
    # Ensure date column is datetime type
    if pd.api.types.is_string_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Get the latest date
    latest_date = df[date_column].max()
    print(f"Latest run date: {latest_date}")
    
    # Filter for the latest run
    latest_run = df[df[date_column] == latest_date]
    return latest_run

def analyze_run():
    """Main function to analyze the latest run."""
    print(f"Starting SHAP analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify and load the model
    model_path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading XGBoost model from {model_path}")
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Verify and load the data
    data_path = os.path.abspath(DATA_PATH)
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please update the DATA_PATH variable in this script.")
        sys.exit(1)
    
    print(f"Loading run data from {data_path}")
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)
    
    # Extract the latest run
    try:
        date_column = 'date'  # Update this if your date column has a different name
        if date_column not in data.columns:
            potential_date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if potential_date_cols:
                date_column = potential_date_cols[0]
                print(f"Using '{date_column}' as date column")
            else:
                print("No date column found. Using all data.")
                latest_run = data
        else:
            latest_run = extract_latest_run(data, date_column)
        
        print(f"Latest run has {latest_run.shape[0]} rows")
        
        # Prepare features (excluding date column and any target)
        X = latest_run.drop([date_column], axis=1, errors='ignore')
        target_cols = ['pace', 'target_pace', 'time']
        for col in target_cols:
            if col in X.columns:
                X = X.drop([col], axis=1, errors='ignore')
        
        print(f"Using {X.shape[1]} features for SHAP analysis")
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        sys.exit(1)
    
    # Calculate SHAP values
    try:
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        expected_value = explainer.expected_value
        print(f"Base value (expected pace): {expected_value}")
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        sys.exit(1)
    
    # Analyze feature importance
    try:
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values.values).mean(0),
            'mean_shap_value': shap_values.values.mean(0)
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Generate report
        print("\n" + "=" * 50)
        print("RUN ANALYSIS REPORT")
        print("=" * 50)
        
        print("\nTop Contributing Factors:")
        top_n = 5  # Show top 5 features
        for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
            feature = row['feature']
            importance = row['importance']
            mean_shap = row['mean_shap_value']
            
            # Determine direction of impact
            impact = "increasing" if mean_shap > 0 else "decreasing"
            direction = "↑" if mean_shap > 0 else "↓"
            
            # Get feature value for this run
            feature_value = X[feature].iloc[0]
            
            print(f"{i}. {feature} ({direction}): Importance score {importance:.4f}")
            print(f"   Current value: {feature_value}")
            print(f"   Impact: {impact} pace by {abs(mean_shap):.4f}")
        
        # Generate recommendations
        print("\nRecommendations:")
        for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
            feature = row['feature']
            mean_shap = row['mean_shap_value']
            
            if mean_shap > 0:
                print(f"• Consider decreasing '{feature}' to improve pace")
            else:
                print(f"• Consider increasing '{feature}' to improve pace")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        sys.exit(1)
    
    # Create visualizations
    try:
        print("\nGenerating visualizations...")
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.title("Feature Importance")
        
        # Detailed plot of top features
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="dot", max_display=10, show=False)
        plt.tight_layout()
        plt.title("Feature Impact")
        
        print("Displaying plots. Close the plot window to continue.")
        plt.show()
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        # Continue even if visualization fails
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_run() 