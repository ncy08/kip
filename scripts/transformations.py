import pandas as pd
import numpy as np
import logging

# Get the logger
logger = logging.getLogger('transformations')

def pace_to_minutes(pace_str):
    """Convert pace string (e.g., '7'03"' or '7:03') to numeric minutes (e.g., 7.05)"""
    if pd.isna(pace_str) or pace_str == '' or pace_str is None:
        return np.nan
    
    try:
        if isinstance(pace_str, str):
            if pace_str.startswith('-'):
                sign = -1
                pace_str = pace_str[1:]
            else:
                sign = 1
            
            # Handle MM'SS" format from feature_generation.py
            if "'" in pace_str and '"' in pace_str:
                parts = pace_str.split("'")
                minutes = int(parts[0])
                seconds = int(parts[1].replace('"', ''))
                return sign * (minutes + seconds/60)
            # Handle MM:SS format
            elif ':' in pace_str:
                parts = pace_str.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                return sign * (minutes + seconds/60)
            else:
                # Already a number
                return float(pace_str)
        else:
            # Already a number
            return float(pace_str)
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"Could not convert pace '{pace_str}' to minutes: {str(e)}")
        return np.nan

def minutes_to_pace_str(minutes):
    """Convert numeric minutes (e.g., 7.05) to pace string (e.g., '7:03')"""
    if pd.isna(minutes) or minutes is None:
        return ''
    
    try:
        # Handle negative values
        if minutes < 0:
            sign = '-'
            minutes = abs(minutes)
        else:
            sign = ''
            
        mins = int(minutes)
        secs = int(round((minutes - mins) * 60))
        
        # Handle case where seconds round to 60
        if secs == 60:
            mins += 1
            secs = 0
            
        return f"{sign}{mins}:{secs:02d}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert minutes '{minutes}' to pace string: {str(e)}")
        return ''

def get_tabular_features(df: pd.DataFrame, lags: int = 4) -> pd.DataFrame:
    """
    Returns a DataFrame with added lagged features and rolling averages for time-series data.
    For XGBoost, we flatten the sequence.
    
    Parameters:
      - df: Input DataFrame (must include at least columns: 'runner_id', 'date', 'pace', 'average_cadence', 'average_heart_rate', 'distance').
      - lags: Number of previous runs to include as lag features.
      
    Returns:
      A DataFrame augmented with new columns.
    """
    logger.info("Starting tabular feature generation")
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Verify data types of key columns
    logger.info(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Columns available: {df.columns.tolist()}")
    
    # Convert date column to datetime if it's not already
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.info("Converting date column to datetime")
        df['date'] = pd.to_datetime(df['date'])
    
    # Convert pace string to numeric for calculations
    if 'pace' in df.columns:
        logger.info(f"Pace column type: {df['pace'].dtype}")
        if df['pace'].dtype == 'object':
            logger.info("Converting pace string to numeric")
            # Store original pace strings
            df['pace_str'] = df['pace'].copy()
            # Convert to numeric for calculations
            df['pace'] = df['pace'].apply(pace_to_minutes)
            logger.info(f"After conversion, pace column has {df['pace'].isna().sum()} NaN values")
    else:
        logger.warning("Pace column not found in DataFrame")
    
    # Ensure the data is sorted by runner and date
    logger.info("Sorting data by runner_id and date")
    df = df.sort_values(by=['runner_id', 'date'])
    
    # Create lag features for each specified metric
    logger.info(f"Creating lag features with {lags} lags")
    for lag in range(1, lags + 1):
        if 'pace' in df.columns:
            df[f'pace_t-{lag}'] = df.groupby('runner_id')['pace'].shift(lag)
        if 'average_cadence' in df.columns:
            df[f'cadence_t-{lag}'] = df.groupby('runner_id')['average_cadence'].shift(lag)
        if 'average_heart_rate' in df.columns:
            df[f'hr_t-{lag}'] = df.groupby('runner_id')['average_heart_rate'].shift(lag)
    
    # Replace rolling calculations with this code:
    rolling_metrics = [
        # Heart rate metrics
        '7d_avg_heart_rate', '30d_avg_heart_rate',
        # Cadence metrics
        '7d_avg_cadence', '30d_avg_cadence',
        # Distance metrics
        '7d_avg_distance', '30d_avg_distance', '7d_total_distance', '30d_total_distance',
        # Pace metrics
        '7d_avg_pace', '30d_avg_pace',
        # Time metrics
        '7d_avg_elapsed_time', '30d_avg_elapsed_time',
        # Effort metrics
        '7d_avg_relative_effort', '30d_avg_relative_effort',
        # Grade metrics
        '7d_avg_grade', '30d_avg_grade'
    ]

    # Log which rolling metrics are already available
    existing_metrics = [col for col in rolling_metrics if col in df.columns]
    missing_metrics = [col for col in rolling_metrics if col not in df.columns]
    
    if existing_metrics:
        logger.info(f"Using existing rolling metrics: {existing_metrics}")
    if missing_metrics:
        logger.warning(f"Missing rolling metrics: {missing_metrics}")
    
    # Convert numeric pace columns back to string format
    if 'pace_str' in df.columns:
        logger.info("Converting numeric pace back to string format")
        # Convert pace back to string format
        if 'rolling_7d_avg_pace_num' in df.columns:
            df['rolling_7d_avg_pace'] = df['rolling_7d_avg_pace_num'].apply(minutes_to_pace_str)
            df = df.drop('rolling_7d_avg_pace_num', axis=1)
        
        # Convert lag columns back to string format
        for lag in range(1, lags + 1):
            col = f'pace_t-{lag}'
            if col in df.columns:
                df[f'{col}_str'] = df[col].apply(minutes_to_pace_str)
                df[col] = df[f'{col}_str']
                df = df.drop(f'{col}_str', axis=1)
        
        # Restore original pace column
        df['pace'] = df['pace_str']
        df = df.drop('pace_str', axis=1)
    
    # Reorganize columns into logical buckets before returning
    logger.info("Reorganizing tabular columns into logical groups")
    
    # Create dynamic lag column lists
    pace_lag_columns = [f'pace_t-{i}' for i in range(1, lags + 1) if f'pace_t-{i}' in df.columns]
    cadence_lag_columns = [f'average_cadence_t-{i}' for i in range(1, lags + 1) if f'average_cadence_t-{i}' in df.columns]
    hr_lag_columns = [f'average_heart_rate_t-{i}' for i in range(1, lags + 1) if f'average_heart_rate_t-{i}' in df.columns]
    distance_lag_columns = [f'distance_t-{i}' for i in range(1, lags + 1) if f'distance_t-{i}' in df.columns]
    
    # Define column groups (only include columns that exist in the DataFrame)
    column_groups = {
        "Identifiers": [
            'activity_id', 'runner_id', 'date', 'activity_type'
        ],
        
        "Core Metrics": [
            'distance', 'elapsed_time', 'moving_time', 'pace'
        ],
        
        "Physiological Metrics": [
            'max_heart_rate', 'average_heart_rate', 'average_cadence', 'relative_effort'
        ],
        
        "Route Metrics": [
            'average_grade', 'elevation_gain', 'elevation_loss', 
            'max_speed', 'average_speed'
        ],
        
        "Environmental": [
            'temperature', 'humidity'
        ],
        
        "Time Features": [
            'week_number', 'day_of_week', 'month', 'is_weekend'
        ],
        
        "Previous Activity Comparisons": [
            'pace_previous', 'pace_diff', 'distance_previous', 'distance_diff',
            'time_previous', 'time_diff', 'heart_rate_previous', 'heart_rate_diff',
            'days_since_last_run'
        ],
        
        "Lag Features": pace_lag_columns + cadence_lag_columns + hr_lag_columns + distance_lag_columns,
        
        "Rolling & Cumulative Metrics": [
            'total_distance', 'total_runs', 'weekly_distance', 'monthly_distance',
            '7d_avg_pace', '7d_avg_distance', '7d_avg_heart_rate', '7d_total_distance',
            '7d_avg_elapsed_time', '30d_avg_elapsed_time',
            '7d_avg_relative_effort', '30d_avg_relative_effort',
            '7d_avg_cadence', '30d_avg_cadence'
        ]
    }
    
    # Create a flat list of columns in the desired order
    # Only include columns that actually exist in the DataFrame
    ordered_columns = []
    for group, cols in column_groups.items():
        existing_cols = [col for col in cols if col in df.columns]
        ordered_columns.extend(existing_cols)
        
    # Add any remaining columns not explicitly placed in a group
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    if remaining_columns:
        logger.info(f"Adding {len(remaining_columns)} additional columns not in predefined groups")
        ordered_columns.extend(remaining_columns)
    
    # Reorder the DataFrame columns
    df = df[ordered_columns]
    logger.info(f"Reorganized columns into logical groups: {list(column_groups.keys())}")
    
    logger.info("Completed tabular feature generation")
    return df

def get_sequence_data(df: pd.DataFrame, window_size: int = 4) -> np.ndarray:
    """
    Returns a 3D NumPy array suitable for NN training.
    Each sequence (window) is of length 'window_size' and contains a subset of features.
    
    Parameters:
      - df: Input DataFrame (should be sorted by 'runner_id' and 'date').
      - window_size: Number of consecutive runs in each sequence.
      
    Returns:
      A 3D array of shape (num_sequences, window_size, num_features)
    """
    logger.info("Starting sequence data generation")
    
    # Create a copy of the dataframe
    df = df.copy()
    
    # Convert pace string to numeric for calculations if needed
    if 'pace' in df.columns:
        logger.info(f"Pace column type: {df['pace'].dtype}")
        if df['pace'].dtype == 'object':
            logger.info("Converting pace string to numeric for sequence data")
            df['pace'] = df['pace'].apply(pace_to_minutes)
    
    sequences = []
    # Use numeric columns for sequences
    feature_columns = []
    
    # Check for required columns and add them if they exist
    if 'pace' in df.columns:
        feature_columns.append('pace')
    if 'average_cadence' in df.columns:
        feature_columns.append('average_cadence')
    if 'average_heart_rate' in df.columns:
        feature_columns.append('average_heart_rate')
    if 'distance' in df.columns:
        feature_columns.append('distance')
    
    logger.info(f"Using features for sequences: {feature_columns}")
    
    if not feature_columns:
        logger.error("No required feature columns found in the dataframe")
        raise ValueError("No required feature columns found in the dataframe")
    
    # Check for NaN values in feature columns
    for col in feature_columns:
        na_count = df[col].isna().sum()
        if na_count > 0:
            logger.warning(f"Column {col} has {na_count} NaN values ({na_count/len(df):.1%} of data)")
    
    # Group data by runner_id and sort by date
    logger.info(f"Creating sequences with window size {window_size}")
    for runner, group in df.groupby('runner_id'):
        group = group.sort_values('date')
        # Fill NaN values with column mean to avoid issues in sequence creation
        for col in feature_columns:
            if group[col].isna().any():
                group[col] = group[col].fillna(group[col].mean())
        
        data = group[feature_columns].values
        
        # Create sliding windows
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            sequences.append(window)
    
    if not sequences:
        logger.error("No sequences could be created")
        raise ValueError("No sequences could be created. Check your data and window_size.")
    
    logger.info(f"Created {len(sequences)} sequences")
    return np.array(sequences)