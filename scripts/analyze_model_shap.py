#!/usr/bin/env python3
"""
Analyze XGBoost model using SHAP values.
Can analyze either the entire dataset or just the latest run.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# Define the paths
MODEL_PATH = "models/xgboost_optuna/model.json"
DATA_PATH = "data/processed_data/activities_tabular.csv"
LATEST_OUTPUT_DIR = "results/shap_analysis/latest"
ALL_OUTPUT_DIR = "results/shap_analysis/all"

def extract_latest_run(df, date_column='date'):
    """Extract the latest run from the dataset."""
    # Ensure date column is datetime type
    if date_column in df.columns:
        if pd.api.types.is_string_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Get the latest date
        latest_date = df[date_column].max()
        print(f"Latest run date: {latest_date}")
        
        # Filter for the latest run
        latest_run = df[df[date_column] == latest_date]
        return latest_run
    else:
        print(f"Warning: Date column '{date_column}' not found. Using all data.")
        return df

def load_model_and_data(model_path=MODEL_PATH, data_path=DATA_PATH):
    """Load the XGBoost model and dataset."""
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Try both absolute and relative paths for model
    model_paths_to_try = [
        model_path,
        os.path.join(root_dir, model_path)
    ]
    
    model = None
    for path in model_paths_to_try:
        if os.path.exists(path):
            try:
                model = xgb.Booster()
                model.load_model(path)
                print(f"Model loaded from: {path}")
                break
            except Exception as e:
                print(f"Error loading model from {path}: {str(e)}")
    
    if model is None:
        raise FileNotFoundError(f"Could not load model from any of the paths: {model_paths_to_try}")
    
    # Try both absolute and relative paths for data
    data_paths_to_try = [
        data_path,
        os.path.join(root_dir, data_path)
    ]
    
    data = None
    for path in data_paths_to_try:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path)
                print(f"Data loaded from: {path}")
                print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
                break
            except Exception as e:
                print(f"Error loading data from {path}: {str(e)}")
    
    if data is None:
        raise FileNotFoundError(f"Could not load data from any of the paths: {data_paths_to_try}")
    
    return model, data

def prepare_features(df, date_column='date'):
    """Prepare features for SHAP analysis by removing date and target columns."""
    # Create a copy to avoid modifying the original
    X = df.copy()
    
    # Remove date column
    if date_column in X.columns:
        X = X.drop([date_column], axis=1, errors='ignore')
    
    # Remove potential target columns
    target_cols = ['pace', 'target_pace', 'time', 'moving_time', 'elapsed_time']
    for col in target_cols:
        if col in X.columns:
            X = X.drop([col], axis=1, errors='ignore')
    
    # Get feature names from the model
    model_features = load_feature_names()
    
    # Keep only the features that the model was trained on
    if model_features:
        # Keep only features that are in both X and model_features
        common_features = [f for f in model_features if f in X.columns]
        X = X[common_features]
        # If any model features are missing, add them with zeros
        missing_features = [f for f in model_features if f not in X.columns]
        for f in missing_features:
            X[f] = 0
        print(f"Using {len(common_features)} features from model, added {len(missing_features)} missing features")
    else:
        # Remove any non-numeric columns (SHAP requires numeric input)
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            X = X.drop(non_numeric_cols, axis=1, errors='ignore')
    
    # Fill any missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    
    return X

def load_feature_names():
    """Load feature names from the model metadata."""
    # Try to load from metadata.json if it exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    metadata_path = os.path.join(root_dir, "models/xgboost_optuna/metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'feature_names' in metadata:
                print(f"Loaded {len(metadata['feature_names'])} feature names from metadata")
                return metadata['feature_names']
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
    
    # If that fails, try using DMatrix to get feature names
    try:
        model_path = os.path.join(root_dir, MODEL_PATH)
        if os.path.exists(model_path):
            # Load a sample data
            data_path = os.path.join(root_dir, DATA_PATH)
            sample_data = pd.read_csv(data_path).head(1)
            # Remove target and non-numeric columns
            for col in ['pace', 'target_pace', 'time', 'moving_time', 'elapsed_time', 'date']:
                if col in sample_data.columns:
                    sample_data = sample_data.drop([col], axis=1)
            non_numeric_cols = sample_data.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                sample_data = sample_data.drop(non_numeric_cols, axis=1)
            # Create DMatrix and get feature names
            dmatrix = xgb.DMatrix(sample_data)
            feature_names = dmatrix.feature_names
            if feature_names:
                print(f"Loaded {len(feature_names)} feature names from DMatrix")
                return feature_names
    except Exception as e:
        print(f"Error getting feature names from DMatrix: {str(e)}")
    
    return []

def analyze_dataset(model, data, date_column='date', do_display=True):
    """Analyze the entire dataset."""
    
    # Create output directory
    output_path = ALL_OUTPUT_DIR
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare data for SHAP analysis
    X = data.drop('pace', axis=1)
    y = data['pace']
    
    # Add missing features 
    model_features = model.feature_names
    if model_features is None:
        print("Warning: Model has no feature names, using data columns")
        model_features = X.columns.tolist()
        
    missing_features = [f for f in model_features if f not in X.columns]
    print(f"Using {len(model_features) - len(missing_features)} features from model, added {len(missing_features)} missing features")
    
    for f in missing_features:
        X[f] = 0
        
    # Ensure X has all the features in the right order
    X = X[model_features]
    print(f"Using {X.shape[1]} features for SHAP analysis")
    
    # Run SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Create visualizations
    print("Generating visualizations for entire dataset...")
    
    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Feature Importance (All Runs)")
    plt.tight_layout()
    bar_plot_path = os.path.join(output_path, "feature_importance_bar_all_runs.png")
    plt.savefig(bar_plot_path)
    plt.close()
    print(f"Saved bar plot to {bar_plot_path}")
    
    # Dot plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=15, show=False)
    plt.title("Feature Impact (All Runs)")
    plt.tight_layout()
    dot_plot_path = os.path.join(output_path, "feature_impact_dot_all_runs.png")
    plt.savefig(dot_plot_path)
    plt.close()
    print(f"Saved dot plot to {dot_plot_path}")
    
    # Save feature importance to CSV
    feature_importance = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': np.abs(shap_values).mean(0),
        'mean_shap_value': shap_values.mean(0)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    csv_path = os.path.join(output_path, "feature_importance_all_runs.csv")
    feature_importance.to_csv(csv_path, index=False)
    print(f"Saved feature importance to {csv_path}")
    
    # Generate report for all runs analysis
    report_text = "\n" + "=" * 50 + "\n"
    report_text += "ALL RUNS ANALYSIS REPORT\n"
    report_text += "=" * 50 + "\n\n"
    
    report_text += "Date of Analysis: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
    report_text += f"Number of Runs Analyzed: {data.shape[0]}\n\n"
    
    report_text += "Top Contributing Factors:\n"
    top_n = 10  # Show top 10 features
    for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
        feature = row['feature']
        importance = row['importance']
        mean_shap = row['mean_shap_value']
        
        # Determine direction of impact
        impact = "increasing" if mean_shap > 0 else "decreasing"
        direction = "↑" if mean_shap > 0 else "↓"
        
        # Get feature statistics
        feature_mean = X[feature].mean()
        feature_std = X[feature].std() 
        feature_min = X[feature].min()
        feature_max = X[feature].max()
        
        report_text += f"{i}. {feature} ({direction}): Importance score {importance:.4f}\n"
        report_text += f"   Average value: {feature_mean:.4f} (std: {feature_std:.4f})\n"
        report_text += f"   Range: {feature_min:.4f} to {feature_max:.4f}\n"
        report_text += f"   Impact: {impact} pace by {abs(mean_shap):.4f} on average\n\n"
    
    # Save report
    report_path = os.path.join(output_path, "all_runs_analysis.md")
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved all runs analysis report to {report_path}")
    
    # Also display plots if requested
    if do_display:
        print("Displaying plots. Close the plot windows to continue.")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=True)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="dot", max_display=15, show=True)

def analyze_latest_run(model, data, date_column='date', do_display=True):
    """Analyze just the latest run using SHAP."""
    print(f"Analyzing latest run")
    
    # Create output directory
    output_path = LATEST_OUTPUT_DIR
    os.makedirs(output_path, exist_ok=True)
    
    # Extract the latest run
    latest_run = extract_latest_run(data, date_column)
    if latest_run is None or len(latest_run) == 0:
        print("No valid runs found with date information. Using the first run in the dataset.")
        latest_run = data.iloc[0:1]
    
    print(f"Found latest run: {latest_run.shape[0]} records")
    
    # Prepare features for the latest run
    X = latest_run.drop('pace', axis=1)
    
    # Add missing features
    model_features = model.feature_names
    if model_features is None:
        print("Warning: Model has no feature names, using data columns")
        model_features = X.columns.tolist()
    
    missing_features = [f for f in model_features if f not in X.columns]
    print(f"Using {len(model_features) - len(missing_features)} features from model, added {len(missing_features)} missing features")
    
    for f in missing_features:
        X[f] = 0
    
    # Ensure the features are in the correct order
    X = X[model_features]
    feature_names = X.columns.tolist()
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    print(f"Base value (expected pace): {expected_value}")
    
    # Create the feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(0),
        'mean_shap_value': shap_values.mean(0)
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Generate report
    report_text = "\n" + "=" * 50 + "\n"
    report_text += "LATEST RUN ANALYSIS REPORT\n"
    report_text += "=" * 50 + "\n\n"
    
    report_text += "Date of Analysis: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
    if date_column in latest_run.columns:
        report_text += "Run Date: " + str(latest_run[date_column].iloc[0]) + "\n\n"
    
    report_text += "Top Contributing Factors:\n"
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
        
        report_text += f"{i}. {feature} ({direction}): Importance score {importance:.4f}\n"
        report_text += f"   Current value: {feature_value}\n"
        report_text += f"   Impact: {impact} pace by {abs(mean_shap):.4f}\n\n"
    
    # Generate recommendations
    report_text += "Recommendations:\n"
    for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
        feature = row['feature']
        mean_shap = row['mean_shap_value']
        
        if mean_shap > 0:
            report_text += f"• Consider decreasing '{feature}' to improve pace\n"
        else:
            report_text += f"• Consider increasing '{feature}' to improve pace\n"
    
    # Save report to file
    report_path = os.path.join(output_path, "latest_run_analysis.md")
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved analysis report to {report_path}")
    
    # Save feature importance to CSV
    csv_path = os.path.join(output_path, "feature_importance_latest_run.csv")
    feature_importance.to_csv(csv_path, index=False)
    print(f"Saved feature importance to {csv_path}")
    
    # Create visualizations
    print("Generating visualizations for latest run...")
    
    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Feature Importance (Latest Run)")
    plt.tight_layout()
    bar_plot_path = os.path.join(output_path, "feature_importance_bar_latest.png")
    plt.savefig(bar_plot_path)
    plt.close()
    print(f"Saved bar plot to {bar_plot_path}")
    
    # Dot plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=10, show=False)
    plt.title("Feature Impact (Latest Run)")
    plt.tight_layout()
    dot_plot_path = os.path.join(output_path, "feature_impact_dot_latest.png")
    plt.savefig(dot_plot_path)
    plt.close()
    print(f"Saved dot plot to {dot_plot_path}")
    
    # Also display plots if requested
    if do_display:
        print("Displaying plots. Close the plot windows to continue.")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=True)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="dot", max_display=10, show=True)

def main():
    """Main function to parse arguments and run analysis."""
    global LATEST_OUTPUT_DIR, ALL_OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description='Analyze XGBoost model using SHAP values')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the model file')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to the data file')
    parser.add_argument('--date-column', type=str, default='date', help='Name of the date column')
    parser.add_argument('--output-dir', type=str, help='Output directory for analysis results')
    parser.add_argument('--latest', action='store_true', help='Analyze only the latest run')
    parser.add_argument('--all', action='store_true', help='Analyze the entire dataset')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots (save only)')
    args = parser.parse_args()
    
    # Update output directory if provided by the user
    if args.output_dir:
        LATEST_OUTPUT_DIR = args.output_dir
    
    print(f"Starting SHAP analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load model and data
        model, data = load_model_and_data(args.model, args.data)
        
        # Analyze based on provided flags
        if args.latest:
            print(f"Analyzing latest run... Results will be saved to {LATEST_OUTPUT_DIR}")
            analyze_latest_run(model, data, args.date_column, not args.no_display)
            
        if args.all:
            print(f"Analyzing all runs... Results will be saved to {ALL_OUTPUT_DIR}")
            analyze_dataset(model, data, args.date_column, not args.no_display)
            
        if not args.latest and not args.all:
            # Default behavior if no flags specified
            print("No specific analysis type requested. Analyzing both latest run and all runs...")
            print(f"All runs analysis will be saved to {ALL_OUTPUT_DIR}")
            analyze_dataset(model, data, args.date_column, not args.no_display)
            
            print(f"Latest run analysis will be saved to {LATEST_OUTPUT_DIR}")
            analyze_latest_run(model, data, args.date_column, not args.no_display)
        
        print(f"\nAnalysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        if 'TreeExplainer' in str(e):
            print("If this is a path error, try using absolute paths with --model and --data")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 