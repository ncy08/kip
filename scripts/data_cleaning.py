import pandas as pd
import numpy as np
import os
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_cleaning')

def clean_column_name(col_name):
    """
    Clean a column name by removing spaces, converting to lowercase, and replacing spaces with underscores.
    
    Parameters:
    -----------
    col_name : str
        The column name to clean
        
    Returns:
    --------
    str
        The cleaned column name
    """
    # Convert to string in case it's not already
    col_name = str(col_name)
    
    # Remove leading/trailing whitespace
    col_name = col_name.strip()
    
    # Convert to lowercase
    col_name = col_name.lower()
    
    # Replace spaces with underscores
    col_name = re.sub(r'\s+', '_', col_name)
    
    # Remove any special characters except underscores
    col_name = re.sub(r'[^\w_]', '', col_name)
    
    return col_name

def map_strava_columns(df):
    """
    Map Strava-like column names to standardized names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame with original column names
        
    Returns:
    --------
    pandas.DataFrame
        The DataFrame with standardized column names and a mapping dictionary
    """
    # Clean all column names first
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Define column mapping patterns
    column_mapping = {
        # Basic run details
        'activity_id': ['activity_id', 'id', 'activity_identifier'],
        'runner_id': ['athlete_id', 'user_id', 'runner_id', 'athlete_identifier'],
        'date': ['activity_date', 'start_time', 'start_date', 'date', 'activity_datetime'],
        'activity_type': ['activity_type', 'type', 'sport_type'],
        'distance': ['distance', 'distance_km', 'distance_miles'],
        'elapsed_time': ['elapsed_time', 'total_time', 'time'],
        'moving_time': ['moving_time', 'active_time'],
        
        # Performance metrics
        'max_heart_rate': ['max_heart_rate', 'max_hr', 'heart_rate_max'],
        'average_heart_rate': ['average_heart_rate', 'avg_heart_rate', 'avg_hr', 'heart_rate_avg', 'heart_rate_average'],
        'max_speed': ['max_speed', 'maximum_speed', 'speed_max'],
        'average_speed': ['average_speed', 'avg_speed', 'speed_avg', 'speed_average'],
        'pace': ['pace', 'average_pace', 'avg_pace'],
        'average_grade': ['average_grade', 'avg_grade', 'grade_avg', 'grade_average'],
        
        # Advanced metrics
        'average_cadence': ['average_cadence', 'avg_cadence', 'cadence_avg', 'cadence_average'],
        'relative_effort': ['relative_effort', 'effort_score', 'training_load'],
        'elevation_gain': ['elevation_gain', 'total_elevation_gain', 'ascent', 'total_ascent'],
        'elevation_loss': ['elevation_loss', 'total_elevation_loss', 'descent', 'total_descent'],
        
        # Environmental conditions
        'temperature': ['temperature', 'weather_temperature', 'temp'],
        'humidity': ['humidity', 'weather_humidity', 'relative_humidity']
    }
    
    # Create a mapping from original columns to standardized names
    actual_mapping = {}
    
    # For each standardized column name
    for standard_name, possible_names in column_mapping.items():
        # Check if any of the possible names exist in the DataFrame
        for possible_name in possible_names:
            if possible_name in df.columns:
                actual_mapping[possible_name] = standard_name
                break
    
    # Log the mapping
    logger.info(f"Column mapping: {actual_mapping}")
    
    # Rename the columns
    df = df.rename(columns=actual_mapping)
    
    # Return the DataFrame with standardized column names and the mapping
    return df, actual_mapping

def clean_data(input_csv: str, output_csv: str):
    """
    Reads raw data from `input_csv`, cleans and validates, saves to `output_csv`.
    
    Parameters:
    -----------
    input_csv : str
        Path to the raw CSV file
    output_csv : str
        Path where the cleaned CSV will be saved
    """
    logger.info(f"Starting data cleaning process for {input_csv}")
    
    # Check if file exists
    if not os.path.exists(input_csv):
        logger.error(f"Input file {input_csv} does not exist")
        raise FileNotFoundError(f"Input file {input_csv} does not exist")
    
    try:
        # 1. Load the data
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # 2. Map Strava-like columns to standardized names
        df, column_mapping = map_strava_columns(df)
        
        # 3. Check for essential columns
        essential_columns = ['date', 'distance', 'elapsed_time']
        missing_essential = [col for col in essential_columns if col not in df.columns]
        
        if missing_essential:
            logger.error(f"Missing essential columns: {missing_essential}")
            raise ValueError(f"Input CSV is missing essential columns: {missing_essential}")
        
        # 4. Keep only relevant columns
        relevant_columns = [
            # Basic run details
            'activity_id', 'runner_id', 'date', 'activity_type', 'distance', 
            'elapsed_time', 'moving_time',
            
            # Performance metrics
            'max_heart_rate', 'average_heart_rate', 'max_speed', 'average_speed', 
            'pace', 'average_grade',
            
            # Advanced metrics
            'average_cadence', 'relative_effort', 'elevation_gain', 'elevation_loss',
            
            # Environmental conditions
            'temperature', 'humidity'
        ]
        
        # Filter to only keep columns that exist in the DataFrame
        columns_to_keep = [col for col in relevant_columns if col in df.columns]
        
        # Log which columns we're keeping
        logger.info(f"Keeping {len(columns_to_keep)} columns: {columns_to_keep}")
        
        # Keep only the relevant columns
        df = df[columns_to_keep]
        
        # 5. Convert date to datetime
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                logger.info("Converted date column to datetime")
            except Exception as e:
                logger.warning(f"Could not convert date column to datetime: {e}")
        
        # 6. Ensure numeric columns are numeric
        numeric_columns = [
            'distance', 'elapsed_time', 'moving_time', 'max_heart_rate', 
            'average_heart_rate', 'max_speed', 'average_speed', 'pace', 
            'average_grade', 'average_cadence', 'relative_effort', 
            'elevation_gain', 'elevation_loss', 'temperature', 'humidity'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        # 7. Basic validation - remove rows with impossible values
        original_row_count = len(df)
        
        # Filter out impossible distances (e.g., negative or > 100 miles/km)
        if 'distance' in df.columns:
            df = df[df['distance'] > 0]
            df = df[df['distance'] < 100]  # Adjust threshold as needed
        
        # Filter out impossible times (e.g., negative or extremely long)
        if 'elapsed_time' in df.columns:
            df = df[df['elapsed_time'] > 0]
        
        # Filter out impossible heart rates if present
        if 'average_heart_rate' in df.columns:
            df = df[(df['average_heart_rate'] > 30) & (df['average_heart_rate'] < 250)]
        
        # Log how many rows were removed
        rows_removed = original_row_count - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with invalid values")
        
        # 8. Handle missing values
        # For heart rate, we can use forward fill within each runner's data
        if 'average_heart_rate' in df.columns and 'runner_id' in df.columns:
            # Count missing values before filling
            missing_hr = df['average_heart_rate'].isna().sum()
            if missing_hr > 0:
                logger.info(f"Found {missing_hr} missing heart rate values")
                
                # Group by runner_id and forward fill
                df = df.sort_values(['runner_id', 'date'])
                df['average_heart_rate'] = df.groupby('runner_id')['average_heart_rate'].fillna(method='ffill')
                
                # If still missing (e.g., first entry for a runner), use backward fill
                df['average_heart_rate'] = df.groupby('runner_id')['average_heart_rate'].fillna(method='bfill')
                
                # If still missing, fill with median
                remaining_missing = df['average_heart_rate'].isna().sum()
                if remaining_missing > 0:
                    median_hr = df['average_heart_rate'].median()
                    df['average_heart_rate'].fillna(median_hr, inplace=True)
                    logger.info(f"Filled {remaining_missing} remaining missing heart rate values with median ({median_hr})")
        
        # 9. Create a runner_id if it doesn't exist
        if 'runner_id' not in df.columns and 'activity_id' in df.columns:
            logger.info("Creating a default runner_id since none was found")
            df['runner_id'] = 'default_runner'
        
        # 10. Ensure runner_id is a string (for consistent grouping later)
        if 'runner_id' in df.columns:
            df['runner_id'] = df['runner_id'].astype(str)
        
        # 11. Save the cleaned data
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved cleaned data to {output_csv} with {len(df)} rows and {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    import sys
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        clean_data(input_file, output_file)
    else:
        print("Usage: python data_cleaning.py input_csv_path output_csv_path")