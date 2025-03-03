#!/usr/bin/env python3
"""
Runner script for SHAP analysis on the latest run.
This script loads the model and run data then performs SHAP analysis.
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Add proper import path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    # Try importing using direct path
    from scripts.shap_analysis import run_shap_analysis
except ImportError:
    try:
        # Try alternative import if running from scripts directory
        from shap_analysis import run_shap_analysis
    except ImportError:
        print("Error: Could not import run_shap_analysis function.")
        print("Make sure shap_analysis.py is in the scripts directory.")
        sys.exit(1)

def main():
    """Run SHAP analysis on the latest run."""
    try:
        print("Running SHAP analysis on latest run...")
        
        # Load your trained model
        # Update the path to wherever your model is saved
        model_path = os.path.join(parent_dir, "models", "xgb_model.json")  # Adjust this path to your model
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please update the model_path in this script to point to your trained model.")
            # Try alternative location
            alternative_path = os.path.join(parent_dir, "model.json")
            if os.path.exists(alternative_path):
                print(f"Found model at alternative location: {alternative_path}")
                model_path = alternative_path
            else:
                print("Available directories in parent folder:")
                for item in os.listdir(parent_dir):
                    if os.path.isdir(os.path.join(parent_dir, item)):
                        print(f" - {item}/")
                sys.exit(1)
        
        print(f"Loading model from {model_path}...")
        try:
            model = xgb.Booster()
            model.load_model(model_path)
        except Exception as e:
            print(f"Error loading XGBoost model: {str(e)}")
            print("Make sure the model file is a valid XGBoost model.")
            sys.exit(1)
        
        # Load your running data
        # Update the path to your data file
        data_path = os.path.join(parent_dir, "data", "runs.csv")  # Adjust this path to your data
        
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            print("Please update the data_path in this script to point to your running data.")
            
            # Try to find CSV files in the parent directory
            csv_files = []
            for root, dirs, files in os.walk(parent_dir):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if csv_files:
                print("Found the following CSV files:")
                for i, file in enumerate(csv_files):
                    print(f" {i+1}. {file}")
                print("Update the data_path variable to use one of these files.")
            sys.exit(1)
        
        print(f"Loading run data from {data_path}...")
        try:
            data = pd.read_csv(data_path)
            print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
            print("Columns found in data:")
            for col in data.columns:
                print(f" - {col}: {data[col].dtype}")
            
            # Check if date column exists
            date_column = 'date'
            if date_column not in data.columns:
                print(f"Warning: '{date_column}' column not found in data.")
                date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
                if date_columns:
                    print(f"Found possible date columns: {date_columns}")
                    date_column = date_columns[0]
                    print(f"Using '{date_column}' as date column")
                else:
                    print("No date column found. Will use all data for analysis.")
                    # Create a dummy date column with the most recent date at the last row
                    data['date'] = pd.date_range(start='2023-01-01', periods=data.shape[0])
                    date_column = 'date'
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
        
        # Run the SHAP analysis
        print("Calculating SHAP values and generating report...")
        try:
            results = run_shap_analysis(
                model=model,
                data=data,
                baseline_pace=None,  # Will use model's expected value if None
                date_column=date_column,
                top_n=5,  # Show top 5 features
                visualize=True
            )
        except Exception as e:
            print(f"Error during SHAP analysis: {str(e)}")
            traceback.print_exc()
            
            # Provide more debugging information
            print("\nDebugging information:")
            print(f"Python version: {sys.version}")
            print(f"XGBoost version: {xgb.__version__}")
            try:
                import shap
                print(f"SHAP version: {shap.__version__}")
            except ImportError:
                print("SHAP package not installed. Install with: pip install shap")
            
            sys.exit(1)
        
        # Print the summary report
        print("\n" + "=" * 50)
        print("RUN IMPROVEMENT REPORT")
        print("=" * 50)
        print(results['report']['summary'])
        
        print("\nTop Contributing Factors:")
        for i, feature in enumerate(results['report']['features'], 1):
            impact_direction = "↑" if feature['impact'] == 'increasing' else "↓"
            print(f"{i}. {feature['name']} ({impact_direction}): Importance score {feature['importance']:.4f}")
        
        print("\nRecommendations:")
        for rec in results['report']['recommendations']:
            print(f"• {rec}")
        
        # Display visualizations
        if 'visualization' in results:
            print("\nDisplaying SHAP visualizations...")
            plt.tight_layout()
            plt.show()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 