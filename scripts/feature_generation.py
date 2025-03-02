import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('feature_generation')

def generate_features(cleaned_csv: str, output_csv: str):
    """
    Loads cleaned data, creates new features, and saves the feature-enriched dataset.
    
    Parameters:
    -----------
    cleaned_csv : str
        Path to the cleaned CSV file
    output_csv : str
        Path where the feature-enriched CSV will be saved
    """
    logger.info(f"Starting feature generation for {cleaned_csv}")
    
    # Check if file exists
    if not os.path.exists(cleaned_csv):
        logger.error(f"Input file {cleaned_csv} does not exist")
        raise FileNotFoundError(f"Input file {cleaned_csv} does not exist")
    
    try:
        # 1. Load the cleaned data
        df = pd.read_csv(cleaned_csv)
        logger.info(f"Loaded cleaned data with {len(df)} rows")
        
        # 2. Calculate pace from speed if pace is missing but speed exists
        if 'pace' not in df.columns and 'average_speed' in df.columns:
            # Pace is typically minutes per distance unit (e.g., min/mile or min/km)
            # Assuming speed is in distance units per hour (mph or kph)
            # Replace infinities or zeros with NaN first
            safe_speed = df['average_speed'].replace([np.inf, -np.inf, 0], np.nan)
            df['pace'] = 60 / safe_speed  # Convert to minutes per distance unit
            logger.info("Calculated pace from average_speed")
        
        # If we have elapsed_time and distance but no pace, calculate it
        if 'pace' not in df.columns and 'elapsed_time' in df.columns and 'distance' in df.columns:
            # Replace zeros in distance with NaN to avoid division by zero
            safe_distance = df['distance'].replace(0, np.nan)
            # Assuming elapsed_time is in seconds and distance is in miles or km
            df['pace'] = (df['elapsed_time'] / 60) / safe_distance  # Convert to minutes per distance unit
            logger.info("Calculated pace from elapsed_time and distance")
        
        # 3. Ensure date is in datetime format for time-based features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 4. Update runner_id to start at 01
        if 'runner_id' in df.columns:
            # Get unique runner IDs
            unique_runners = df['runner_id'].unique()
            # Create mapping from old IDs to new IDs starting at 01
            runner_mapping = {old_id: f"{i+1:02d}" for i, old_id in enumerate(unique_runners)}
            # Apply the mapping
            df['runner_id'] = df['runner_id'].map(runner_mapping)
            logger.info("Updated runner_id to start at 01")
        
        # 5. Sort by runner_id and date for time-series features
        df = df.sort_values(by=['runner_id', 'date'])
        
        # 6. Create week number feature
        if 'date' in df.columns:
            df['week_number'] = df['date'].dt.isocalendar().week
            logger.info("Created week number feature")
        
        # 7. Format pace columns in minute:second format (e.g., 7:03)
        pace_columns = [col for col in df.columns if 'pace' in col]
        for col in pace_columns:
            if col in df.columns:
                # First handle NaN and infinite values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Create a mask for valid values
                valid_mask = df[col].notna() & np.isfinite(df[col])
                
                # Initialize the formatted column with NaN
                formatted_pace = pd.Series(index=df.index, dtype='object')
                
                # Only process valid values
                if valid_mask.any():
                    # Extract minutes (integer part) and seconds (fractional part * 60)
                    minutes = df.loc[valid_mask, col].astype(float).astype(int)
                    seconds = ((df.loc[valid_mask, col].astype(float) - minutes) * 60).round().astype(int)
                    
                    # Handle case where seconds = 60
                    idx_60_seconds = seconds >= 60
                    minutes.loc[idx_60_seconds] += 1
                    seconds.loc[idx_60_seconds] -= 60
                    
                    # Format as MM:SS
                    formatted_pace.loc[valid_mask] = minutes.astype(str) + ':' + seconds.astype(str).str.zfill(2)
                
                # For invalid values, use a placeholder
                formatted_pace.loc[~valid_mask] = np.nan
                
                # Update the column in the dataframe
                df[col] = formatted_pace
                logger.info(f"Formatted {col} in minute:second format")
        
        # 8. Create lagged features (previous run metrics)
        if 'runner_id' in df.columns:
            # Lagged pace
            if 'pace' in df.columns:
                df['pace_previous'] = df.groupby('runner_id')['pace'].shift(1)
                logger.info("Created lagged pace feature")
            
            # Pace difference from previous run
            if 'pace' in df.columns and 'pace_previous' in df.columns:
                # Need to convert string pace back to numeric for calculations
                def pace_to_minutes(pace_str):
                    try:
                        if pd.isna(pace_str):
                            return np.nan
                        if ':' in str(pace_str):
                            minutes, seconds = str(pace_str).split(':')
                            return float(minutes) + float(seconds) / 60
                        else:
                            return float(pace_str)
                    except:
                        return np.nan
                
                numeric_pace = df['pace'].apply(pace_to_minutes)
                numeric_pace_prev = df['pace_previous'].apply(pace_to_minutes)
                
                # Calculate difference in numeric form
                pace_diff_numeric = numeric_pace - numeric_pace_prev
                
                # Initialize the formatted column with NaN
                formatted_diff = pd.Series(index=df.index, dtype='object')
                
                # Only process valid values
                valid_mask = pace_diff_numeric.notna() & np.isfinite(pace_diff_numeric)
                if valid_mask.any():
                    # Format the difference
                    minutes = pace_diff_numeric.loc[valid_mask].astype(float).astype(int)
                    seconds = ((pace_diff_numeric.loc[valid_mask].astype(float) - minutes) * 60).round().astype(int)
                    
                    # Handle case where seconds = 60 or seconds = -60
                    idx_60_seconds = seconds >= 60
                    minutes.loc[idx_60_seconds] += 1
                    seconds.loc[idx_60_seconds] -= 60
                    
                    idx_neg_60_seconds = seconds <= -60
                    minutes.loc[idx_neg_60_seconds] -= 1
                    seconds.loc[idx_neg_60_seconds] += 60
                    
                    # Handle negative seconds
                    idx_neg_seconds = (seconds < 0) & (minutes > 0)
                    minutes.loc[idx_neg_seconds] -= 1
                    seconds.loc[idx_neg_seconds] += 60
                    
                    # Format as MM:SS or -MM:SS
                    neg_mask = (minutes < 0) | ((minutes == 0) & (seconds < 0))
                    seconds = seconds.abs()  # Ensure seconds are positive for formatting
                    
                    formatted_diff.loc[valid_mask & ~neg_mask] = minutes.loc[~neg_mask].astype(str) + ':' + seconds.loc[~neg_mask].astype(str).str.zfill(2)
                    formatted_diff.loc[valid_mask & neg_mask] = '-' + minutes.loc[neg_mask].abs().astype(str) + ':' + seconds.loc[neg_mask].astype(str).str.zfill(2)
                
                # Update the column in the dataframe
                df['pace_diff'] = formatted_diff
                logger.info("Created pace difference feature")
            
            # Lagged distance
            if 'distance' in df.columns:
                df['distance_previous'] = df.groupby('runner_id')['distance'].shift(1)
                logger.info("Created lagged distance feature")
            
            # Distance difference from previous run
            if 'distance' in df.columns and 'distance_previous' in df.columns:
                df['distance_diff'] = df['distance'] - df['distance_previous']
                logger.info("Created distance difference feature")
            
            # 7-day rolling average pace
            if 'date' in df.columns and 'pace' in df.columns:
                # Create a numeric pace column for calculations
                df['numeric_pace'] = df['pace'].apply(pace_to_minutes)
                
                # Calculate rolling average on numeric pace
                df_indexed = df.set_index('date')
                rolling_pace = df_indexed.groupby('runner_id')['numeric_pace'].rolling('7D', min_periods=1).mean()
                rolling_pace = rolling_pace.reset_index()
                
                # Convert numeric rolling pace back to formatted string
                numeric_7d_pace = rolling_pace['numeric_pace'].values
                
                # Initialize the formatted column with NaN
                formatted_7d_pace = pd.Series(index=df.index, dtype='object')
                
                # Only process valid values
                valid_mask = pd.Series(np.isfinite(numeric_7d_pace), index=df.index)
                if valid_mask.any():
                    minutes = pd.Series(numeric_7d_pace, index=df.index).loc[valid_mask].astype(float).astype(int)
                    seconds = ((pd.Series(numeric_7d_pace, index=df.index).loc[valid_mask] - minutes) * 60).round().astype(int)
                    
                    # Handle case where seconds = 60
                    idx_60_seconds = seconds >= 60
                    minutes.loc[idx_60_seconds] += 1
                    seconds.loc[idx_60_seconds] -= 60
                    
                    formatted_7d_pace.loc[valid_mask] = minutes.astype(str) + ':' + seconds.astype(str).str.zfill(2)
                
                df['7d_avg_pace'] = formatted_7d_pace
                
                # Drop the temporary numeric pace column
                df = df.drop(columns=['numeric_pace'])
                logger.info("Created 7-day rolling average pace")
            
            # 30-day rolling average pace
            if 'date' in df.columns and 'pace' in df.columns:
                # Create a numeric pace column for calculations
                df['numeric_pace'] = df['pace'].apply(pace_to_minutes)
                
                # Calculate rolling average on numeric pace
                df_indexed = df.set_index('date')
                rolling_pace_30d = df_indexed.groupby('runner_id')['numeric_pace'].rolling('30D', min_periods=1).mean()
                rolling_pace_30d = rolling_pace_30d.reset_index()
                
                # Convert numeric rolling pace back to formatted string
                numeric_30d_pace = rolling_pace_30d['numeric_pace'].values
                
                # Initialize the formatted column with NaN
                formatted_30d_pace = pd.Series(index=df.index, dtype='object')
                
                # Only process valid values
                valid_mask = pd.Series(np.isfinite(numeric_30d_pace), index=df.index)
                if valid_mask.any():
                    minutes = pd.Series(numeric_30d_pace, index=df.index).loc[valid_mask].astype(float).astype(int)
                    seconds = ((pd.Series(numeric_30d_pace, index=df.index).loc[valid_mask] - minutes) * 60).round().astype(int)
                    
                    # Handle case where seconds = 60
                    idx_60_seconds = seconds >= 60
                    minutes.loc[idx_60_seconds] += 1
                    seconds.loc[idx_60_seconds] -= 60
                    
                    formatted_30d_pace.loc[valid_mask] = minutes.astype(str) + ':' + seconds.astype(str).str.zfill(2)
                
                df['30d_avg_pace'] = formatted_30d_pace
                
                # Drop the temporary numeric pace column
                df = df.drop(columns=['numeric_pace'])
                logger.info("Created 30-day rolling average pace")
            
            # 7-day total distance
            if 'date' in df.columns and 'distance' in df.columns:
                df_indexed = df.set_index('date')
                rolling_dist = df_indexed.groupby('runner_id')['distance'].rolling('7D', min_periods=1).sum()
                rolling_dist = rolling_dist.reset_index()
                df['7d_total_distance'] = rolling_dist['distance'].values
                logger.info("Created 7-day total distance feature")
            
            # 30-day total distance
            if 'date' in df.columns and 'distance' in df.columns:
                df_indexed = df.set_index('date')
                rolling_dist_30d = df_indexed.groupby('runner_id')['distance'].rolling('30D', min_periods=1).sum()
                rolling_dist_30d = rolling_dist_30d.reset_index()
                df['30d_total_distance'] = rolling_dist_30d['distance'].values
                logger.info("Created 30-day total distance feature")
            
            # 7-day run count
            if 'date' in df.columns:
                df_indexed = df.set_index('date')
                rolling_count = df_indexed.groupby('runner_id').rolling('7D', min_periods=1).count()
                rolling_count = rolling_count.reset_index()
                df['7d_run_count'] = rolling_count['activity_id'].values
                logger.info("Created 7-day run count feature")
            
            # 30-day run count
            if 'date' in df.columns:
                df_indexed = df.set_index('date')
                rolling_count_30d = df_indexed.groupby('runner_id').rolling('30D', min_periods=1).count()
                rolling_count_30d = rolling_count_30d.reset_index()
                df['30d_run_count'] = rolling_count_30d['activity_id'].values
                logger.info("Created 30-day run count feature")
            
            # Lagged heart rate
            if 'average_heart_rate' in df.columns:
                df['heart_rate_previous'] = df.groupby('runner_id')['average_heart_rate'].shift(1)
                logger.info("Created lagged heart rate feature")
            
            # Heart rate difference from previous run
            if 'average_heart_rate' in df.columns and 'heart_rate_previous' in df.columns:
                df['heart_rate_diff'] = df['average_heart_rate'] - df['heart_rate_previous']
                logger.info("Created heart rate difference feature")
            
            # 7-day rolling average heart rate
            if 'date' in df.columns and 'average_heart_rate' in df.columns:
                df_indexed = df.set_index('date')
                rolling_hr = df_indexed.groupby('runner_id')['average_heart_rate'].rolling('7D', min_periods=1).mean()
                rolling_hr = rolling_hr.reset_index()
                df['7d_avg_heart_rate'] = rolling_hr['average_heart_rate'].values
                logger.info("Created 7-day rolling average heart rate")
            
            # 7-day rolling average elapsed time
            if 'date' in df.columns and 'elapsed_time' in df.columns:
                df_indexed = df.set_index('date')
                rolling_time = df_indexed.groupby('runner_id')['elapsed_time'].rolling('7D', min_periods=1).mean()
                rolling_time = rolling_time.reset_index()
                df['7d_avg_elapsed_time'] = rolling_time['elapsed_time'].values
                logger.info("Created 7-day rolling average elapsed time")
            
            # 30-day rolling average elapsed time
            if 'date' in df.columns and 'elapsed_time' in df.columns:
                df_indexed = df.set_index('date')
                rolling_time_30d = df_indexed.groupby('runner_id')['elapsed_time'].rolling('30D', min_periods=1).mean()
                rolling_time_30d = rolling_time_30d.reset_index()
                df['30d_avg_elapsed_time'] = rolling_time_30d['elapsed_time'].values
                logger.info("Created 30-day rolling average elapsed time")
            
            # 7-day rolling average relative effort
            if 'date' in df.columns and 'relative_effort' in df.columns:
                df_indexed = df.set_index('date')
                rolling_effort = df_indexed.groupby('runner_id')['relative_effort'].rolling('7D', min_periods=1).mean()
                rolling_effort = rolling_effort.reset_index()
                df['7d_avg_relative_effort'] = rolling_effort['relative_effort'].values
                logger.info("Created 7-day rolling average relative effort")
            
            # 30-day rolling average relative effort
            if 'date' in df.columns and 'relative_effort' in df.columns:
                df_indexed = df.set_index('date')
                rolling_effort_30d = df_indexed.groupby('runner_id')['relative_effort'].rolling('30D', min_periods=1).mean()
                rolling_effort_30d = rolling_effort_30d.reset_index()
                df['30d_avg_relative_effort'] = rolling_effort_30d['relative_effort'].values
                logger.info("Created 30-day rolling average relative effort")
            
            # 7-day rolling average grade
            if 'date' in df.columns and 'average_grade' in df.columns:
                df_indexed = df.set_index('date')
                rolling_grade = df_indexed.groupby('runner_id')['average_grade'].rolling('7D', min_periods=1).mean()
                rolling_grade = rolling_grade.reset_index()
                df['7d_avg_grade'] = rolling_grade['average_grade'].values
                logger.info("Created 7-day rolling average grade")
            
            # 30-day rolling average grade
            if 'date' in df.columns and 'average_grade' in df.columns:
                df_indexed = df.set_index('date')
                rolling_grade_30d = df_indexed.groupby('runner_id')['average_grade'].rolling('30D', min_periods=1).mean()
                rolling_grade_30d = rolling_grade_30d.reset_index()
                df['30d_avg_grade'] = rolling_grade_30d['average_grade'].values
                logger.info("Created 30-day rolling average grade")
        
        # 9. Create time-based features
        if 'date' in df.columns:
            # Extract day of week (0=Monday, 6=Sunday)
            df['day_of_week'] = df['date'].dt.dayofweek
            logger.info("Created day of week feature")
            
            # Extract month
            df['month'] = df['date'].dt.month
            logger.info("Created month feature")
            
            # Extract if weekend (0=weekday, 1=weekend)
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            logger.info("Created weekend indicator feature")
        
        # 10. Calculate days since last run
        if 'date' in df.columns:
            df['days_since_last_run'] = df.groupby('runner_id')['date'].diff().dt.days
            logger.info("Created days since last run feature")
        
        # 11. Fill NaN values in the newly created features
        # For lagged and diff features, we can't meaningfully fill the first entry
        # But for rolling features, we've used min_periods=1 so they should be filled
        
        # 12. Save the feature-enriched data
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved feature-enriched data to {output_csv} with {len(df)} rows and {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during feature generation: {str(e)}")
        raise

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    import sys
    
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        generate_features(input_file, output_file)
    else:
        print("Usage: python feature_generation.py input_csv_path output_csv_path")
