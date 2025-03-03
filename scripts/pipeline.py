import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from data_cleaning import clean_data
from feature_generation import generate_features
from transformations import get_tabular_features, get_sequence_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline')

def run_pipeline(input_csv, cleaned_csv=None, features_csv=None, tabular_csv=None, sequence_npy=None):
    """
    Runs the complete data pipeline: cleaning, feature generation, and transformations.
    
    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file
    cleaned_csv : str, optional
        Path where the cleaned CSV will be saved, defaults to data/processed_data/activities_cleaned.csv
    features_csv : str, optional
        Path where the feature-enriched CSV will be saved, defaults to data/processed_data/activities_features.csv
    tabular_csv : str, optional
        Path where the tabular features CSV will be saved, defaults to data/processed_data/activities_tabular.csv
    sequence_npy : str, optional
        Path where the sequence data NPY will be saved, defaults to data/processed_data/activities_sequences.npy
    
    Returns:
    --------
    dict
        A dictionary with pipeline status information
    """
    start_time = datetime.now()
    
    # Set default file paths if not provided
    if cleaned_csv is None:
        cleaned_csv = "data/processed_data/activities_cleaned.csv"
    if features_csv is None:
        features_csv = "data/processed_data/activities_features.csv"
    if tabular_csv is None:
        tabular_csv = "data/processed_data/activities_tabular.csv"
    if sequence_npy is None:
        sequence_npy = "data/processed_data/activities_sequences.npy"
    
    try:
        # Step 1: Data Cleaning
        logger.info(f"Starting data cleaning for {input_csv}")
        clean_data(input_csv, cleaned_csv)
        logger.info(f"Data cleaning completed, saved to {cleaned_csv}")
        
        # Step 2: Feature Generation
        logger.info(f"Starting feature generation for {cleaned_csv}")
        generate_features(cleaned_csv, features_csv)
        logger.info(f"Feature generation completed, saved to {features_csv}")
        
        # Step 3: Data Transformations for ML/DL
        logger.info(f"Starting data transformations")
        
        # Load the feature-enriched data
        df = pd.read_csv(features_csv)
        logger.info(f"Loaded feature data with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert date column to datetime (needed for rolling windows)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # For XGBoost, create tabular features
        try:
            logger.info("Generating tabular features for XGBoost")
            df_tabular = get_tabular_features(df, lags=4)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(tabular_csv), exist_ok=True)
            df_tabular.to_csv(tabular_csv, index=False)
            logger.info(f"Tabular features created, saved to {tabular_csv}")
        except Exception as e:
            logger.error(f"Error creating tabular features: {str(e)}")
            raise
        
        # For NN, create sequence windows
        try:
            logger.info("Generating sequence data for neural networks")
            sequence_array = get_sequence_data(df, window_size=4)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(sequence_npy), exist_ok=True)
            np.save(sequence_npy, sequence_array)
            logger.info(f"Sequence data created, saved to {sequence_npy}")
        except Exception as e:
            logger.error(f"Error creating sequence data: {str(e)}")
            raise
        
        # Calculate runtime and return success
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        
        return {
            "status": "success",
            "input_file": input_csv,
            "cleaned_file": cleaned_csv,
            "features_file": features_csv,
            "tabular_file": tabular_csv,
            "sequence_file": sequence_npy,
            "runtime_seconds": runtime_seconds,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        # Calculate runtime even in case of error
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        
        logger.error(f"Pipeline failed: {str(e)}")
        
        return {
            "status": "error",
            "error_message": str(e),
            "runtime_seconds": runtime_seconds,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the data pipeline")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("--cleaned", help="Path for the cleaned CSV", default=None)
    parser.add_argument("--features", help="Path for the feature-enriched CSV", default=None)
    parser.add_argument("--tabular", help="Path for the tabular features CSV", default=None)
    parser.add_argument("--sequence", help="Path for the sequence data NPY", default=None)
    
    args = parser.parse_args()
    
    # Run the pipeline
    result = run_pipeline(
        args.input_csv, 
        args.cleaned, 
        args.features,
        args.tabular,
        args.sequence
    )
    
    # Print final status
    if result["status"] == "success":
        print(f"Pipeline completed successfully in {result['runtime_seconds']:.2f} seconds")
        print(f"Input file: {result['input_file']}")
        print(f"Cleaned file: {result['cleaned_file']}")
        print(f"Features file: {result['features_file']}")
        print(f"Tabular features file: {result['tabular_file']}")
        print(f"Sequence data file: {result['sequence_file']}")
    else:
        print(f"Pipeline failed after {result['runtime_seconds']:.2f} seconds")
        print(f"Error: {result['error_message']}")
