import os
import sys
import logging
import argparse
from datetime import datetime
from data_cleaning import clean_data
from feature_generation import generate_features

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

def run_pipeline(input_csv: str, cleaned_csv: str = None, features_csv: str = None):
    """
    Orchestrates the data pipeline: cleaning -> feature generation.
    
    Parameters:
    -----------
    input_csv : str
        Path to the raw CSV file
    cleaned_csv : str, optional
        Path where the cleaned CSV will be saved. If None, a default path will be used.
    features_csv : str, optional
        Path where the feature-enriched CSV will be saved. If None, a default path will be used.
    """
    start_time = datetime.now()
    logger.info(f"Starting pipeline for {input_csv}")
    
    try:
        # Generate default output paths if not provided
        if cleaned_csv is None:
            input_filename = os.path.basename(input_csv)
            input_name = os.path.splitext(input_filename)[0]
            cleaned_csv = os.path.join("data/processed_data", f"{input_name}_cleaned.csv")
        
        if features_csv is None:
            input_filename = os.path.basename(input_csv)
            input_name = os.path.splitext(input_filename)[0]
            features_csv = os.path.join("data/processed_data", f"{input_name}_features.csv")
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
        os.makedirs(os.path.dirname(features_csv), exist_ok=True)
        
        # Step 1: Clean the data
        logger.info("Starting data cleaning step")
        clean_data(input_csv, cleaned_csv)
        logger.info(f"Data cleaning completed. Output saved to {cleaned_csv}")
        
        # Step 2: Generate features
        logger.info("Starting feature generation step")
        generate_features(cleaned_csv, features_csv)
        logger.info(f"Feature generation completed. Output saved to {features_csv}")
        
        # Calculate and log total runtime
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        logger.info(f"Pipeline completed successfully in {runtime:.2f} seconds")
        
        return {
            "status": "success",
            "input_file": input_csv,
            "cleaned_file": cleaned_csv,
            "features_file": features_csv,
            "runtime_seconds": runtime
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        logger.info(f"Pipeline failed after {runtime:.2f} seconds")
        
        return {
            "status": "error",
            "input_file": input_csv,
            "error_message": str(e),
            "runtime_seconds": runtime
        }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the data pipeline for runner data')
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('--cleaned', help='Path where the cleaned CSV will be saved (optional)')
    parser.add_argument('--features', help='Path where the feature-enriched CSV will be saved (optional)')
    
    args = parser.parse_args()
    
    # Run the pipeline
    result = run_pipeline(args.input_csv, args.cleaned, args.features)
    
    # Print final status
    if result["status"] == "success":
        print(f"Pipeline completed successfully in {result['runtime_seconds']:.2f} seconds")
        print(f"Input file: {result['input_file']}")
        print(f"Cleaned file: {result['cleaned_file']}")
        print(f"Features file: {result['features_file']}")
    else:
        print(f"Pipeline failed after {result['runtime_seconds']:.2f} seconds")
        print(f"Error: {result['error_message']}")
