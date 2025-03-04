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
import math

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

def generate_enhanced_shap_report(shap_df, 
                                 negative_label="speeds up pace", 
                                 positive_label="slows pace", 
                                 metric_name="pace",
                                 top_n=10,
                                 is_latest_run=False):
    """
    Generates a textual SHAP-based report for a set of features/metrics.
    :param shap_df: A pandas DataFrame with at least columns:
        - 'feature': the name of the feature
        - 'mean_shap_value': average SHAP (negative -> speed up, positive -> slow down)
        - 'importance': absolute SHAP magnitude
        - 'avg_value': average of that feature across all runs (optional, for context)
        - 'std_value': standard deviation of that feature (optional)
    :param negative_label: Phrase used when SHAP is negative
    :param positive_label: Phrase used when SHAP is positive
    :param metric_name: Name of the metric being predicted (e.g. "pace", "time")
    :param top_n: How many top features to highlight.
    :param is_latest_run: If True, formats report for latest run; if False, for all runs.
    :return: A string containing a structured, improved SHAP summary.
    """
    
    # Copy to avoid modifying the original
    df = shap_df.copy()
    
    # Sort by absolute SHAP importance
    df.sort_values("importance", ascending=False, inplace=True)
    
    # Create a header
    report_lines = []
    report_lines.append("\n" + "=" * 50)
    if is_latest_run:
        report_lines.append(f"LATEST RUN ANALYSIS REPORT")
    else:
        report_lines.append(f"ALL RUNS ANALYSIS REPORT") 
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Add date information
    report_lines.append(f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if "run_date" in df.columns and not df["run_date"].isna().all():
        report_lines.append(f"Run Date: {df['run_date'].iloc[0]}")
    if not is_latest_run and "num_runs" in df.columns:
        report_lines.append(f"Number of Runs Analyzed: {df['num_runs'].iloc[0]}")
    report_lines.append("")
    
    # Provide an introduction / disclaimers if it's not a latest run report
    if not is_latest_run:
        report_lines.append("INTRODUCTION")
        report_lines.append("----------------------------------------------")
        report_lines.append(
            "This report summarizes how each feature influences your predicted "
            f"{metric_name}. Positive values generally mean the feature is associated "
            f"with a higher (slower) {metric_name}, and negative values mean the feature is "
            f"associated with a lower (faster) {metric_name}.\n"
            "Note: SHAP shows correlations the model has learned, not guaranteed causation. "
        )
        report_lines.append("")
    
    # Summarize top features
    report_lines.append("Top Contributing Factors:")

    # List only top_n features
    top_features = df.head(top_n)
    
    # Lists to store positive and negative factors
    positive_factors = []  # slowing pace
    negative_factors = []  # speeding up pace
    total_positive_impact = 0
    total_negative_impact = 0
    
    for i, row in enumerate(top_features.itertuples(), start=1):
        feat = row.feature
        mean_shap = getattr(row, "mean_shap_value", 0.0)
        importance = getattr(row, "importance", 0.0)
        
        # Try to get optional stats
        avg_val = getattr(row, "avg_value", None)
        std_val = getattr(row, "std_value", None)
        min_val = getattr(row, "min_value", None)
        max_val = getattr(row, "max_value", None)
        
        # Convert to seconds for interpretation
        seconds_impact = mean_shap * 60
        
        # Apply rounding logic - round positive values up, negative values down
        if mean_shap > 0:
            seconds_rounded = math.ceil(seconds_impact)
            direction_word = positive_label
            direction_symbol = "↑"
            positive_factors.append(feat)
            total_positive_impact += seconds_rounded
        else:
            seconds_rounded = math.floor(seconds_impact)
            direction_word = negative_label
            direction_symbol = "↓"
            negative_factors.append(feat)
            total_negative_impact += abs(seconds_rounded)
        
        # Create a detailed feature summary
        feature_summary = [
            f"{i}. {feat} ({direction_symbol}): Importance score {importance:.4f}"
        ]
        
        # If we have average stats, mention them
        if is_latest_run:
            if avg_val is not None:
                feature_summary.append(f"   Current value: {avg_val}")
            feature_summary.append(f"   Impact: {direction_word.split(' ')[0]} pace by {abs(mean_shap):.4f}")
        else:
            if avg_val is not None and std_val is not None:
                feature_summary.append(f"   Average value: {avg_val:.4f} (std: {std_val:.4f})")
                if min_val is not None and max_val is not None:
                    feature_summary.append(f"   Range: {min_val:.4f} to {max_val:.4f}")
            feature_summary.append(f"   Impact: {direction_word} by {abs(mean_shap):.4f} on average")
        
        report_lines.append("\n".join(feature_summary))
        report_lines.append("")  # blank line for spacing

    # Add section header for interpretations
    if is_latest_run:
        report_lines.append("\nLocal SHAP Analysis (Per-Run Explanations)")
    else:
        report_lines.append("\nGlobal SHAP Analysis (Across All Runs)")
    report_lines.append("=" * 40)
    
    if is_latest_run:
        report_lines.append("This analyzes SHAP values for a single run to see what impacted its predicted pace.\n")
    else:
        report_lines.append("This analyzes SHAP values across your entire dataset to find patterns that hold consistently over time.\n")
    
    # Add interpretations
    report_lines.append("Interpretations:")
    
    for i, row in enumerate(top_features.itertuples(), start=1):
        feat = row.feature
        mean_shap = getattr(row, "mean_shap_value", 0.0)
        
        # Convert to seconds for interpretation
        seconds_impact = mean_shap * 60
        
        # Apply rounding logic - round positive values up, negative values down
        if mean_shap > 0:
            seconds_rounded = math.ceil(seconds_impact)
            if is_latest_run:
                report_lines.append(f"• {feat} slowed this run by +{seconds_rounded} seconds/mile.")
            else:
                report_lines.append(f"• {feat} typically slows pace by +{seconds_rounded} seconds/mile on average.")
        else:
            seconds_rounded = abs(math.floor(seconds_impact))
            if is_latest_run:
                report_lines.append(f"• {feat} sped up this run by -{seconds_rounded} seconds/mile.")
            else:
                report_lines.append(f"• {feat} typically speeds up runs by -{seconds_rounded} seconds/mile on average.")
    
    # Add summary statements
    report_lines.append("")
    
    if negative_factors:
        negative_factors_text = ", ".join(negative_factors[:-1])
        if len(negative_factors) > 1:
            negative_factors_text += f" & {negative_factors[-1]}"
        else:
            negative_factors_text = negative_factors[0]
            
        if is_latest_run:
            report_lines.append(f"• {negative_factors_text} sped up this run by {total_negative_impact} seconds total.")
        else:
            report_lines.append(f"• {negative_factors_text} typically speed up your runs by {total_negative_impact} seconds on average.")
    
    if positive_factors:
        positive_factors_text = ", ".join(positive_factors[:-1])
        if len(positive_factors) > 1:
            positive_factors_text += f" & {positive_factors[-1]}"
        else:
            positive_factors_text = positive_factors[0]
            
        if is_latest_run:
            report_lines.append(f"• {positive_factors_text} slowed down this run by {total_positive_impact} seconds total.")
        else:
            report_lines.append(f"• {positive_factors_text} typically slow down your runs by {total_positive_impact} seconds on average.")
    
    report_lines.append("")
    
    # Add explanatory note about SHAP values
    if is_latest_run:
        report_lines.append("• SHAP values quantify exactly how much each factor is contributing to your pace for this specific run.")
    else:
        report_lines.append("• The top features here are the ones that matter most across all runs.")
    report_lines.append("• Positive SHAP values indicate factors that slow you down, negative values indicate factors that speed you up.")
    
    # Add recommendations for latest run
    if is_latest_run:
        report_lines.append("\nRecommendations:")
        for i, row in enumerate(top_features.itertuples(), start=1):
            feat = row.feature
            mean_shap = getattr(row, "mean_shap_value", 0.0)
            
            if mean_shap > 0:
                report_lines.append(f"• Consider decreasing '{feat}' to improve pace")
            else:
                report_lines.append(f"• Consider increasing '{feat}' to improve pace")
    else:
        # Add actionable suggestions for all runs analysis
        report_lines.append("\nACTIONABLE SUGGESTIONS")
        report_lines.append("----------------------------------------------")
        suggestions = [
            "• Focus on the high-impact positive SHAP features (which slow you down) if you want to improve speed.",
            "• Reinforce or maintain the negative SHAP features (which help you run faster).",
            "• Track changes over time to see if modifications align with improved performance.",
            "• Remember that correlation does not imply causation—use training experiments to validate these findings."
        ]
        report_lines.extend(suggestions)
    
    # Join lines and return the report
    return "\n".join(report_lines)

def analyze_dataset(model, data, date_column='date', do_display=True):
    """Analyze the entire dataset using SHAP."""
    print(f"Analyzing entire dataset")
    
    # Create output directory
    output_path = ALL_OUTPUT_DIR
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare features
    X, y = prepare_features(data, date_column)
    feature_names = X.columns.tolist()
    
    print(f"Using {len(feature_names)} features for SHAP analysis")
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    print(f"Base value (expected pace): {expected_value}")
    
    # Create visualizations
    print("Generating visualizations for entire dataset...")
    
    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Feature Importance (All Runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "feature_importance_bar_all_runs.png"))
    plt.close()
    print(f"Saved bar plot to {os.path.join(output_path, 'feature_importance_bar_all_runs.png')}")
    
    # Dot plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.title("Feature Impact (All Runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "feature_impact_dot_all_runs.png"))
    plt.close()
    print(f"Saved dot plot to {os.path.join(output_path, 'feature_impact_dot_all_runs.png')}")
    
    # Create feature importance DataFrame
    feature_importance = {
        'feature': [],
        'importance': [],
        'mean_shap_value': [],
        'avg_value': [],
        'std_value': [],
        'min_value': [],
        'max_value': [],
        'num_runs': []
    }
    
    for i, name in enumerate(feature_names):
        feature_importance['feature'].append(name)
        feature_importance['importance'].append(np.abs(shap_values[:, i]).mean())
        feature_importance['mean_shap_value'].append(shap_values[:, i].mean())
        feature_importance['avg_value'].append(X[name].mean())
        feature_importance['std_value'].append(X[name].std())
        feature_importance['min_value'].append(X[name].min())
        feature_importance['max_value'].append(X[name].max())
        feature_importance['num_runs'].append(len(X))
    
    feature_importance_df = pd.DataFrame(feature_importance)
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Save to CSV
    feature_importance_df.to_csv(os.path.join(output_path, "feature_importance_all_runs.csv"), index=False)
    print(f"Saved feature importance to {os.path.join(output_path, 'feature_importance_all_runs.csv')}")
    
    # Generate enhanced report using the new function
    report_text = generate_enhanced_shap_report(
        feature_importance_df,
        negative_label="speeds up runs",
        positive_label="slows pace",
        metric_name="pace",
        top_n=10,
        is_latest_run=False
    )
    
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
    
    # Create feature importance DataFrame
    feature_importance = {
        'feature': [],
        'importance': [],
        'mean_shap_value': [],
        'avg_value': [],
        'run_date': []
    }
    
    for i, name in enumerate(feature_names):
        feature_importance['feature'].append(name)
        feature_importance['importance'].append(abs(shap_values[0][i]))
        feature_importance['mean_shap_value'].append(shap_values[0][i])
        feature_importance['avg_value'].append(X[name].iloc[0])
        if date_column in latest_run.columns:
            feature_importance['run_date'].append(latest_run[date_column].iloc[0])
        else:
            feature_importance['run_date'].append(None)
    
    feature_importance_df = pd.DataFrame(feature_importance)
    
    # Generate enhanced report using the new function
    report_text = generate_enhanced_shap_report(
        feature_importance_df,
        negative_label="speeds up pace",
        positive_label="slows pace",
        metric_name="pace",
        top_n=5,
        is_latest_run=True
    )
    
    # Save report to file
    report_path = os.path.join(output_path, "latest_run_analysis.md")
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved analysis report to {report_path}")
    
    # Save feature importance to CSV
    csv_path = os.path.join(output_path, "feature_importance_latest_run.csv")
    feature_importance_df.to_csv(csv_path, index=False)
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