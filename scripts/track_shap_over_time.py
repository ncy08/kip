#!/usr/bin/env python3
"""
Track SHAP value changes over time to visualize adaptation patterns and training improvements.
This script shows how the impact of different features changes across months or training cycles.
It can also analyze the trailing 3 months with month-over-month and 90-day trend analyses.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

# Define paths
MODEL_PATH = "models/xgboost_optuna/model.json"
DATA_PATH = "data/processed_data/activities_tabular.csv"
OUTPUT_DIR = "results/shap_analysis/trailing"
TRAILING_OUTPUT_DIR = "results/shap_analysis/trailing"

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
    """Prepare features for SHAP analysis while preserving date information."""
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert date to datetime if it's not already
    if date_column in df_copy.columns:
        if pd.api.types.is_string_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    else:
        print(f"Warning: Date column '{date_column}' not found. Temporal analysis may not be possible.")
        # Add a placeholder date column
        df_copy[date_column] = pd.NaT
    
    # Extract date components for grouping
    date_columns = []
    if not df_copy[date_column].isna().all():
        df_copy['year'] = df_copy[date_column].dt.year
        df_copy['month'] = df_copy[date_column].dt.month
        df_copy['month_name'] = df_copy[date_column].dt.month_name()
        df_copy['quarter'] = df_copy[date_column].dt.quarter
        df_copy['year_month'] = df_copy[date_column].dt.strftime('%Y-%m')
        df_copy['week'] = df_copy[date_column].dt.isocalendar().week
        date_columns = ['date', 'year', 'month', 'month_name', 'quarter', 'year_month', 'week']
    
    # Save the date information
    date_df = df_copy[[c for c in df_copy.columns if c in date_columns]]
    
    # Remove date-related columns for SHAP analysis
    X = df_copy.drop([c for c in date_columns if c in df_copy.columns], axis=1, errors='ignore')
    
    # Remove potential target columns
    target_cols = ['pace', 'target_pace', 'time', 'moving_time', 'elapsed_time']
    for col in target_cols:
        if col in X.columns:
            X = X.drop([col], axis=1, errors='ignore')
    
    # Load feature names
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        metadata_path = os.path.join(root_dir, "models/xgboost_optuna/metadata.json")
        
        model_features = []
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'feature_names' in metadata:
                model_features = metadata['feature_names']
                print(f"Loaded {len(model_features)} feature names from metadata")
    except Exception as e:
        print(f"Error loading feature names: {str(e)}")
        model_features = []
    
    # Keep only model features if available
    if model_features:
        # Keep only features that are in both X and model_features
        common_features = [f for f in model_features if f in X.columns]
        X = X[common_features]
        # If any model features are missing, add them with zeros
        missing_features = [f for f in model_features if f not in X.columns]
        if missing_features:
            missing_df = pd.DataFrame(0, index=X.index, columns=missing_features)
            X = pd.concat([X, missing_df], axis=1)
        print(f"Using {len(common_features)} features from model, added {len(missing_features)} missing features")
    else:
        # Remove any non-numeric columns (SHAP requires numeric input)
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            X = X.drop(non_numeric_cols, axis=1, errors='ignore')
    
    # Fill any missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    
    return X, date_df

def calculate_shap_values(model, X):
    """Calculate SHAP values for the given data."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    print(f"Base value (expected pace): {expected_value}")
    return shap_values, expected_value

def create_shap_time_series(X, shap_values, date_df, top_n=5, time_unit='month'):
    """Create time series of SHAP values for top features."""
    # Get feature importance (mean absolute SHAP value)
    feature_names = X.columns.tolist()
    feature_importance = {}
    
    for i, name in enumerate(feature_names):
        feature_importance[name] = np.abs(shap_values[:, i]).mean()
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:top_n]]
    
    print(f"Top {top_n} features: {', '.join(top_features)}")
    
    # Create DataFrame with SHAP values for each feature - optimized to avoid fragmentation
    # First create all the SHAP columns in a separate DataFrame
    shap_columns = {}
    for i, feature in enumerate(feature_names):
        shap_columns[f"shap_{feature}"] = shap_values[:, i]
    
    # Create the DataFrame at once to avoid fragmentation
    shap_df = pd.DataFrame(shap_columns)
    
    # Add date information
    shap_df = pd.concat([shap_df.reset_index(drop=True), date_df.reset_index(drop=True)], axis=1)
    
    # Add original feature values for the top features
    feature_values = {}
    for feature in top_features:
        feature_values[feature] = X[feature].values
    
    feature_df = pd.DataFrame(feature_values)
    shap_df = pd.concat([shap_df, feature_df], axis=1)
    
    return shap_df, top_features

def plot_monthly_trends(shap_df, top_features, output_dir, time_grouping='year_month'):
    """Plot trends of SHAP values for top features over time."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot changes over time for top features
    plt.figure(figsize=(14, 8))
    
    # Group by time unit and calculate average SHAP values
    grouped_avg = shap_df.groupby(time_grouping)[[f"shap_{f}" for f in top_features]].mean()
    
    # Sort by chronological order
    grouped_avg = grouped_avg.sort_index()
    
    # Plot
    ax = grouped_avg.plot(kind='line', marker='o')
    
    # Add horizontal line at y=0 to indicate neutral impact
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add annotations to explain what the values mean
    plt.figtext(0.5, 0.01, 
        'Negative SHAP values (below dashed line) → Feature is helping to DECREASE pace (make runs faster)\n'
        'Positive SHAP values (above dashed line) → Feature is causing to INCREASE pace (make runs slower)',
        ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    
    time_label = "Month" if time_grouping == 'year_month' else time_grouping.replace('_', ' ').title()
    plt.title(f'How Feature Impact Changes Over Time', fontsize=16)
    plt.ylabel('Average SHAP Value (Impact on Pace)', fontsize=12)
    plt.xlabel(time_label, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"feature_impact_over_time_line.png"), dpi=150, bbox_inches='tight')
    print(f"Saved line plot to {os.path.join(output_dir, 'feature_impact_over_time_line.png')}")
    
    # Also create a bar plot version
    plt.figure(figsize=(14, 8))
    grouped_avg.plot(kind='bar')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.figtext(0.5, 0.01, 
        'Negative SHAP values (below dashed line) → Feature is helping to DECREASE pace (make runs faster)\n'
        'Positive SHAP values (above dashed line) → Feature is causing to INCREASE pace (make runs slower)',
        ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    
    plt.title(f'{time_label}ly Changes in Feature Impact', fontsize=16)
    plt.ylabel('Average SHAP Value (Impact on Pace)', fontsize=12)
    plt.xlabel(time_label, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"feature_impact_over_time_bar.png"), dpi=150, bbox_inches='tight')
    print(f"Saved bar plot to {os.path.join(output_dir, 'feature_impact_over_time_bar.png')}")
    
    # Create individual plots for each feature to see clearer trends
    for feature in top_features:
        plt.figure(figsize=(12, 6))
        
        # Plot SHAP values
        ax = grouped_avg[f"shap_{feature}"].plot(kind='line', marker='o', label=f"Impact on Pace", color='blue')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Format y-axis to show clearer values
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        
        plt.title(f'How {feature} Impact Changes Over Time', fontsize=16)
        plt.ylabel('SHAP Value', fontsize=12)
        plt.xlabel(time_label, fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add interpretation explanation
        if grouped_avg[f"shap_{feature}"].mean() < 0:
            impact_type = "DECREASE (improve)"
            trend_text = "becoming more negative → bigger improvement in pace"
        else:
            impact_type = "INCREASE (slow down)"
            trend_text = "becoming less positive → less pace slowdown"
        
        plt.figtext(0.5, 0.01, 
            f'This feature tends to {impact_type} pace\n'
            f'If the line is {trend_text}',
            ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"impact_over_time_{feature}.png"), dpi=150, bbox_inches='tight')
        print(f"Saved {feature} plot to {os.path.join(output_dir, f'impact_over_time_{feature}.png')}")
    
    # Create a correlation plot between original feature values and their SHAP impact
    for feature in top_features:
        # Group by time unit
        feature_over_time = shap_df.groupby(time_grouping).agg({
            feature: 'mean',
            f"shap_{feature}": 'mean'
        }).sort_index()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(feature_over_time[feature], feature_over_time[f"shap_{feature}"], 
                    alpha=0.7, s=50)
        
        # Add trendline
        if len(feature_over_time) > 1:
            try:
                z = np.polyfit(feature_over_time[feature], feature_over_time[f"shap_{feature}"], 1)
                p = np.poly1d(z)
                plt.plot(feature_over_time[feature], p(feature_over_time[feature]), 
                        "r--", alpha=0.7, label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}")
                plt.legend()
            except np.linalg.LinAlgError:
                print(f"Warning: Could not calculate trend line for {feature} due to numerical issues.")
            except Exception as e:
                print(f"Warning: Error calculating trend line for {feature}: {str(e)}")
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.title(f'Relationship Between {feature} Value and Its Impact', fontsize=14)
        plt.xlabel(f'Average {feature}', fontsize=12)
        plt.ylabel(f'SHAP Impact on Pace', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"correlation_{feature}.png"), dpi=150)
        print(f"Saved correlation plot to {os.path.join(output_dir, f'correlation_{feature}.png')}")
    
    return grouped_avg

def generate_interpretation_report(shap_df, top_features, grouped_avg, output_dir, time_grouping='year_month'):
    """Generate a textual report interpreting the time-series SHAP values."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate trends (is feature impact getting better or worse over time?)
    # We'll use simple linear regression to determine the trend
    
    report_lines = []
    report_lines.append("\n" + "=" * 50)
    report_lines.append("FEATURE IMPACT OVER TIME ANALYSIS")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append(f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Runs Analyzed: {len(shap_df)}")
    report_lines.append(f"Time Period: {grouped_avg.index[0]} to {grouped_avg.index[-1]}")
    report_lines.append(f"Time Grouping: {time_grouping.replace('_', ' ').title()}")
    report_lines.append("")
    
    report_lines.append("TREND ANALYSIS")
    report_lines.append("====================")
    report_lines.append("")
    
    # Calculate trends for each feature
    for feature in top_features:
        values = grouped_avg[f"shap_{feature}"]
        x = np.arange(len(values))
        
        # Skip if we don't have enough data points
        if len(values) < 2:
            report_lines.append(f"Not enough data to analyze trends for {feature}")
            continue
            
        # Calculate slope using linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine if impact is improving over time
            avg_shap = values.mean()
            
            # Calculate trend confidence
            r_squared = r_value**2
            conf_level = "High" if r_squared > 0.5 else "Medium" if r_squared > 0.2 else "Low"
            
            # Interpret based on average SHAP value
            if avg_shap < 0:
                # Negative SHAP = improving pace
                if slope < 0:
                    trend = "IMPROVING (having greater positive impact)"
                    explanation = f"This feature is becoming more effective at reducing your pace over time."
                else:
                    trend = "DIMINISHING (having less positive impact)"
                    explanation = f"This feature is becoming less effective at reducing your pace over time."
            else:
                # Positive SHAP = slowing pace
                if slope < 0:
                    trend = "IMPROVING (having less negative impact)"
                    explanation = f"This feature is slowing your pace less over time."
                else:
                    trend = "WORSENING (having greater negative impact)"
                    explanation = f"This feature is slowing your pace more over time."
            
            # Calculate percent change from first to last month
            if len(values) >= 2 and abs(values.iloc[0]) > 0:
                pct_change = ((values.iloc[-1] - values.iloc[0]) / abs(values.iloc[0])) * 100
                change_str = f"{pct_change:.1f}% change"
            else:
                change_str = "not enough data for percent change"
            
            report_lines.append(f"Feature: {feature}")
            report_lines.append(f"  Trend: {trend} ({change_str})")
            report_lines.append(f"  Confidence: {conf_level} (R² = {r_squared:.2f})")
            report_lines.append(f"  Explanation: {explanation}")
            
            # Add more detailed insights
            first_month = values.index[0]
            last_month = values.index[-1]
            report_lines.append(f"  First period ({first_month}): {values.iloc[0]:.4f}")
            report_lines.append(f"  Last period ({last_month}): {values.iloc[-1]:.4f}")
            report_lines.append(f"  Overall mean: {avg_shap:.4f}")
            report_lines.append("  ")
        except Exception as e:
            report_lines.append(f"Feature: {feature}")
            report_lines.append(f"  Trend: Could not calculate due to insufficient data or numerical issues.")
            report_lines.append(f"  Error: {str(e)}")
            report_lines.append("  ")
    
    report_lines.append("\nINTERPRETATION GUIDE")
    report_lines.append("====================")
    report_lines.append("")
    report_lines.append("• Negative SHAP values indicate a feature is helping to DECREASE pace (make runs faster)")
    report_lines.append("• Positive SHAP values indicate a feature is causing to INCREASE pace (make runs slower)")
    report_lines.append("• For features with negative SHAP values, a downward trend is good (more improvement)")
    report_lines.append("• For features with positive SHAP values, a downward trend is good (less slowdown)")
    report_lines.append("")
    report_lines.append("TRAINING INSIGHTS")
    report_lines.append("====================")
    report_lines.append("")
    report_lines.append("Look for evidence of training adaptations in these trends:")
    report_lines.append("• If temperature impact decreases over time, you may be adapting to heat better")
    report_lines.append("• If cadence impact improves over time, your running form may be becoming more efficient")
    report_lines.append("• If recovery metrics (e.g., days_since_last_run) show decreasing impact, your recovery may be improving")
    report_lines.append("• If grade/elevation impacts decrease, your hill strength may be improving")
    report_lines.append("• If distance impact becomes more negative over time, your endurance may be improving")
    
    # Add feature correlations section
    report_lines.append("\nFEATURE VALUE CORRELATIONS")
    report_lines.append("====================")
    report_lines.append("")
    report_lines.append("Relationships between average feature values and their impact over time:")
    
    for feature in top_features:
        # Group by time unit
        feature_over_time = shap_df.groupby(time_grouping).agg({
            feature: 'mean',
            f"shap_{feature}": 'mean'
        }).sort_index()
        
        if len(feature_over_time) > 1:
            # Calculate correlation
            try:
                corr = feature_over_time[feature].corr(feature_over_time[f"shap_{feature}"])
                corr_strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                corr_direction = "positive" if corr > 0 else "negative"
                
                report_lines.append(f"• {feature}: {corr_strength} {corr_direction} correlation ({corr:.2f})")
                
                if corr > 0.3:
                    report_lines.append(f"  → As {feature} increases, it tends to slow down your pace more")
                elif corr < -0.3:
                    report_lines.append(f"  → As {feature} increases, it tends to help your pace more")
            except Exception as e:
                report_lines.append(f"• {feature}: Could not calculate correlation (Error: {str(e)})")
    
    # Save report
    report_path = os.path.join(output_dir, "feature_impact_trends.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"Saved interpretation report to {report_path}")
    return "\n".join(report_lines)

def filter_trailing_months(df, date_column='date', months=3):
    """Filter data to only include the trailing N months."""
    if date_column not in df.columns:
        print(f"Warning: Date column '{date_column}' not found. Cannot filter trailing months.")
        return df
    
    # Ensure date column is datetime type
    if pd.api.types.is_string_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Get the latest date
    latest_date = df[date_column].max()
    if pd.isna(latest_date):
        print("Warning: No valid dates found. Cannot filter trailing months.")
        return df
    
    # Calculate cutoff date (N months ago)
    cutoff_date = latest_date - pd.DateOffset(months=months)
    print(f"Filtering data from {cutoff_date} to {latest_date} (trailing {months} months)")
    
    # Filter data
    filtered_df = df[df[date_column] >= cutoff_date]
    print(f"Filtered from {len(df)} to {len(filtered_df)} rows")
    
    return filtered_df

def analyze_trailing_months(model, data, date_column='date', months=3, top_n=5, output_dir=TRAILING_OUTPUT_DIR):
    """Analyze SHAP values for the trailing N months."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data to trailing months
    trailing_data = filter_trailing_months(data, date_column, months)
    
    # Prepare features
    X, date_df = prepare_features(trailing_data, date_column)
    
    # Calculate SHAP values
    shap_values, expected_value = calculate_shap_values(model, X)
    
    # Create SHAP time series DataFrame
    shap_df, top_features = create_shap_time_series(X, shap_values, date_df, top_n)
    
    # Create month-over-month analysis
    mom_output_dir = os.path.join(output_dir, "month_over_month")
    os.makedirs(mom_output_dir, exist_ok=True)
    
    # Group by month for month-over-month analysis
    grouped_avg = plot_monthly_trends(shap_df, top_features, mom_output_dir, 'year_month')
    
    # Generate month-over-month report
    mom_report = generate_interpretation_report(
        shap_df, top_features, grouped_avg, mom_output_dir, 'year_month'
    )
    
    # Create 90-day trend analysis
    trend_output_dir = os.path.join(output_dir, "90day_trend")
    os.makedirs(trend_output_dir, exist_ok=True)
    
    # Add a 'days_ago' column for 90-day trend analysis
    if 'date' in date_df.columns:
        latest_date = date_df['date'].max()
        date_df['days_ago'] = (latest_date - date_df['date']).dt.days
        shap_df['days_ago'] = date_df['days_ago']
        
        # Create weekly bins for the 90-day trend
        # Calculate the number of weeks needed based on the actual data range
        max_days = shap_df['days_ago'].max()
        num_weeks = min(13, (max_days // 7) + 1)  # Cap at 13 weeks (90 days)
        
        # Create bins with appropriate number of labels
        bins = range(0, (num_weeks * 7) + 1, 7)
        labels = [f"Week {i+1}" for i in range(len(bins)-1)]
        
        shap_df['week_bin'] = pd.cut(
            shap_df['days_ago'], 
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Plot 90-day trend
        weekly_avg = plot_monthly_trends(shap_df, top_features, trend_output_dir, 'week_bin')
        
        # Generate 90-day trend report
        trend_report = generate_interpretation_report(
            shap_df, top_features, weekly_avg, trend_output_dir, 'week_bin'
        )
    else:
        print(f"Warning: Date column not found in processed data. Cannot create 90-day trend analysis.")
    
    # Create summary report
    summary_lines = []
    summary_lines.append("# Trailing 3-Month SHAP Analysis")
    summary_lines.append(f"\nDate of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"\nAnalysis Period: {date_df['date'].min().strftime('%Y-%m-%d')} to {date_df['date'].max().strftime('%Y-%m-%d')}")
    summary_lines.append(f"\nTotal Runs Analyzed: {len(shap_df)}")
    summary_lines.append("\n## Key Findings")
    
    # Extract key findings from both reports
    if 'year_month' in grouped_avg.index.names:
        months_analyzed = len(grouped_avg.index)
        summary_lines.append(f"\n### Month-over-Month Analysis ({months_analyzed} months)")
        
        # Add top 3 features and their trends
        for i, feature in enumerate(top_features[:3]):
            values = grouped_avg[f"shap_{feature}"]
            if len(values) >= 2:
                change = values.iloc[-1] - values.iloc[0]
                direction = "improved" if (values.mean() < 0 and change < 0) or (values.mean() > 0 and change < 0) else "worsened"
                summary_lines.append(f"- {feature}: Impact has {direction} over the last {months_analyzed} months")
    
    summary_lines.append("\n### 90-Day Trend Analysis")
    summary_lines.append("\nSee detailed reports in the respective folders for complete analysis.")
    
    # Save summary report
    summary_path = os.path.join(output_dir, "trailing_analysis_summary.md")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    
    print(f"Saved trailing analysis summary to {summary_path}")
    return shap_df, top_features

def main():
    """Main function to track SHAP values over time with focus on trailing analysis."""
    parser = argparse.ArgumentParser(description='Track SHAP value changes over time with focus on trailing analysis')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the model file')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to the data file')
    parser.add_argument('--date-column', type=str, default='date', help='Name of the date column')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output directory for results')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top features to analyze')
    parser.add_argument('--trailing-months', type=int, default=3, help='Number of trailing months to analyze')
    parser.add_argument('--custom-features', type=str, help='Comma-separated list of specific features to analyze instead of top-n')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots (save only)')
    parser.add_argument('--full-history', action='store_true', help='Analyze full history instead of just trailing months')
    args = parser.parse_args()
    
    # Update output directory if provided
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting SHAP time series analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load model and data
        model, data = load_model_and_data(args.model, args.data)
        
        # By default, do trailing analysis
        if not args.full_history:
            print(f"Analyzing trailing {args.trailing_months} months...")
            
            # Handle custom features if provided
            custom_features = None
            if args.custom_features:
                custom_features = [f.strip() for f in args.custom_features.split(',')]
                print(f"Using custom features: {', '.join(custom_features)}")
            
            # Perform trailing analysis
            shap_df, top_features = analyze_trailing_months(
                model, data, args.date_column, args.trailing_months, 
                args.top_n, output_dir
            )
            
            # If custom features were provided, update the analysis with those features
            if custom_features:
                # Filter data to trailing months
                trailing_data = filter_trailing_months(data, args.date_column, args.trailing_months)
                
                # Prepare features
                X, date_df = prepare_features(trailing_data, args.date_column)
                
                # Calculate SHAP values
                shap_values, _ = calculate_shap_values(model, X)
                
                # Verify all requested features exist
                missing = [f for f in custom_features if f not in X.columns]
                if missing:
                    print(f"Warning: The following requested features do not exist in the data: {', '.join(missing)}")
                    custom_features = [f for f in custom_features if f in X.columns]
                
                if custom_features:
                    # Create SHAP time series DataFrame with custom features
                    shap_df, _ = create_shap_time_series(X, shap_values, date_df, args.top_n)
                    
                    # Create month-over-month analysis with custom features
                    mom_output_dir = os.path.join(output_dir, "month_over_month")
                    os.makedirs(mom_output_dir, exist_ok=True)
                    
                    # Group by month for month-over-month analysis
                    grouped_avg = plot_monthly_trends(shap_df, custom_features, mom_output_dir, 'year_month')
                    
                    # Generate month-over-month report
                    generate_interpretation_report(
                        shap_df, custom_features, grouped_avg, mom_output_dir, 'year_month'
                    )
                    
                    # Create 90-day trend analysis with custom features
                    trend_output_dir = os.path.join(output_dir, "90day_trend")
                    os.makedirs(trend_output_dir, exist_ok=True)
                    
                    # Add a 'days_ago' column for 90-day trend analysis if not already present
                    if 'days_ago' not in shap_df.columns and 'date' in date_df.columns:
                        latest_date = date_df['date'].max()
                        date_df['days_ago'] = (latest_date - date_df['date']).dt.days
                        shap_df['days_ago'] = date_df['days_ago']
                        
                        # Create weekly bins for the 90-day trend if not already present
                        if 'week_bin' not in shap_df.columns:
                            # Calculate the number of weeks needed based on the actual data range
                            max_days = shap_df['days_ago'].max()
                            num_weeks = min(13, (max_days // 7) + 1)  # Cap at 13 weeks (90 days)
                            
                            # Create bins with appropriate number of labels
                            bins = range(0, (num_weeks * 7) + 1, 7)
                            labels = [f"Week {i+1}" for i in range(len(bins)-1)]
                            
                            shap_df['week_bin'] = pd.cut(
                                shap_df['days_ago'], 
                                bins=bins,
                                labels=labels,
                                include_lowest=True
                            )
                    
                    # Plot 90-day trend with custom features
                    weekly_avg = plot_monthly_trends(shap_df, custom_features, trend_output_dir, 'week_bin')
                    
                    # Generate 90-day trend report with custom features
                    generate_interpretation_report(
                        shap_df, custom_features, weekly_avg, trend_output_dir, 'week_bin'
                    )
            
            print(f"Trailing analysis complete! Results saved to {output_dir}")
            
            # Print usage examples
            print("\nUSAGE EXAMPLES:")
            print("====================")
            print("• Analyze different trailing period: python scripts/track_shap_over_time.py --trailing-months 6")
            print("• Compare specific features: python scripts/track_shap_over_time.py --custom-features temperature,average_cadence,days_since_last_run")
            print("• Analyze full history instead: python scripts/track_shap_over_time.py --full-history")
            
        else:
            # If full history analysis is requested, use the original functionality
            print("Analyzing full history...")
            
            # Create a subdirectory for full history analysis
            full_history_dir = os.path.join(output_dir, "full_history")
            os.makedirs(full_history_dir, exist_ok=True)
            
            # Prepare features, preserving date information
            X, date_df = prepare_features(data, args.date_column)
            
            # Calculate SHAP values
            shap_values, expected_value = calculate_shap_values(model, X)
            
            # Handle custom features if provided
            top_n = args.top_n
            custom_features = None
            if args.custom_features:
                custom_features = [f.strip() for f in args.custom_features.split(',')]
                print(f"Using custom features: {', '.join(custom_features)}")
                # Verify all requested features exist
                missing = [f for f in custom_features if f not in X.columns]
                if missing:
                    print(f"Warning: The following requested features do not exist in the data: {', '.join(missing)}")
                    custom_features = [f for f in custom_features if f in X.columns]
                    if not custom_features:
                        print("No valid custom features found. Reverting to top-n features.")
                        custom_features = None
            
            # Create SHAP time series DataFrame
            shap_df, top_features = create_shap_time_series(X, shap_values, date_df, top_n, 'year_month')
            
            # Replace with custom features if provided
            if custom_features:
                top_features = custom_features
            
            # Plot trends
            grouped_avg = plot_monthly_trends(shap_df, top_features, full_history_dir, 'year_month')
            
            # Generate interpretation report
            report = generate_interpretation_report(shap_df, top_features, grouped_avg, full_history_dir, 'year_month')
            
            # Display final message
            print("\nFull history analysis complete!")
            print(f"Results saved to {full_history_dir}")
        
        # Show plots if requested
        if not args.no_display:
            print("\nDisplaying plots...")
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 