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

def minutes_to_pace_format(minutes):
    """Convert numeric minutes to consistent MM'SS" format"""
    if pd.isna(minutes) or not np.isfinite(minutes):
        return ''
    
    try:
        mins = int(minutes)
        secs = int(round((minutes - mins) * 60))
        
        # Handle case where seconds round to 60
        if secs == 60:
            mins += 1
            secs = 0
            
        return f"{mins}'{secs:02d}\""
    except (ValueError, TypeError):
        return ''

def pace_to_minutes(pace_str):
    """Convert pace string to numeric minutes"""
    if pd.isna(pace_str) or pace_str == '' or pace_str is None:
        return np.nan
    
    try:
        if isinstance(pace_str, str):
            # Handle MM'SS" format
            if "'" in pace_str and '"' in pace_str:
                parts = pace_str.split("'")
                minutes = int(parts[0])
                seconds = int(parts[1].replace('"', ''))
                return minutes + seconds/60
            # Handle MM:SS format
            elif ':' in pace_str:
                parts = pace_str.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes + seconds/60
            else:
                # Already a number
                return float(pace_str)
        else:
            # Already a number
            return float(pace_str)
    except (ValueError, TypeError, IndexError):
        return np.nan

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
        
        # SIMPLIFIED PACE CALCULATION - ONE TIME ONLY

        # Calculate pace once, at the beginning
        if 'pace' not in df.columns:
            logger.info("Calculating pace")
            if 'elapsed_time' in df.columns and 'distance' in df.columns:
                # Replace zeros in distance with NaN to avoid division by zero
                safe_distance = df['distance'].replace(0, np.nan)
                
                # Calculate pace: elapsed_time (seconds) ÷ 60 = minutes, then divide by distance
                df['pace_numeric'] = (df['elapsed_time'] / 60) / safe_distance
                df['pace'] = df['pace_numeric'].apply(minutes_to_pace_format)
                logger.info("Calculated pace from elapsed_time and distance")
            elif 'average_speed' in df.columns:
                # Replace zeros in speed with NaN to avoid division by zero
                safe_speed = df['average_speed'].replace([np.inf, -np.inf, 0], np.nan)
                
                # Calculate pace: 60 minutes/hour ÷ speed
                df['pace_numeric'] = 60 / safe_speed
                df['pace'] = df['pace_numeric'].apply(minutes_to_pace_format)
                logger.info("Calculated pace from average_speed")
        # Convert existing pace to numeric if needed
        elif 'pace' in df.columns and 'pace_numeric' not in df.columns:
            df['pace_numeric'] = df['pace'].apply(pace_to_minutes)
            logger.info("Converted existing pace to numeric values")
        
        # 2. Ensure date is in datetime format for time-based features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 3. Update runner_id to start at 01
        if 'runner_id' in df.columns:
            # Get unique runner IDs
            unique_runners = df['runner_id'].unique()
            # Create mapping from old IDs to new IDs starting at 01
            runner_mapping = {old_id: f"{i+1:02d}" for i, old_id in enumerate(unique_runners)}
            # Apply the mapping
            df['runner_id'] = df['runner_id'].map(runner_mapping)
            logger.info("Updated runner_id to start at 01")
        
        # 4. Sort by runner_id and date for time-series features
        df = df.sort_values(by=['runner_id', 'date'])
        
        # 5. Create week number feature
        if 'date' in df.columns:
            df['week_number'] = df['date'].dt.isocalendar().week
            logger.info("Created week number feature")
        
        # 6. Format all pace columns consistently
        pace_columns = [col for col in df.columns if 'pace' in col and 'numeric' not in col and not col.endswith('_str')]
        for col in pace_columns:
            if col in df.columns:
                # Create a numeric version if it doesn't exist
                numeric_col = f"{col}_numeric"
                if numeric_col not in df.columns:
                    # First try to convert from string format to numeric
                    df[numeric_col] = df[col].apply(pace_to_minutes)
                
                # Format all pace columns consistently
                valid_mask = df[numeric_col].notna() & np.isfinite(df[numeric_col])
                
                # Initialize with empty strings
                df[col] = ''
                
                # Apply formatting only to valid values
                if valid_mask.any():
                    df.loc[valid_mask, col] = df.loc[valid_mask, numeric_col].apply(minutes_to_pace_format)
                
                logger.info(f"Formatted {col} using consistent MM'SS\" format")
        
        # 7. Create lagged features (previous run metrics)
        if 'runner_id' in df.columns:
            # Lagged pace
            if 'pace' in df.columns:
                df['pace_previous_numeric'] = df.groupby('runner_id')['pace_numeric'].shift(1)
                df['pace_previous'] = df['pace_previous_numeric'].apply(minutes_to_pace_format)
                logger.info("Created lagged pace feature")
            
            # Pace difference from previous run
            if 'pace' in df.columns and 'pace_previous_numeric' in df.columns:
                df['pace_diff_numeric'] = df['pace_numeric'] - df['pace_previous_numeric']
                
                # Format as MM'SS" with sign
                df['pace_diff'] = df['pace_diff_numeric'].apply(lambda x: 
                    ('' if x >= 0 else '-') + 
                    minutes_to_pace_format(abs(x)) if pd.notnull(x) else '')
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
                # Calculate rolling average on numeric pace
                df_indexed = df.set_index('date')
                rolling_pace = df_indexed.groupby('runner_id')['pace_numeric'].rolling('7D', min_periods=1).mean()
                rolling_pace = rolling_pace.reset_index()
                
                # Convert numeric rolling pace back to formatted string
                numeric_7d_pace = rolling_pace['pace_numeric'].values
                
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
                    
                    formatted_7d_pace.loc[valid_mask] = minutes.astype(str) + "'" + seconds.astype(str).str.zfill(2) + '"'
                
                df['7d_avg_pace'] = formatted_7d_pace
                logger.info("Created 7-day rolling average pace")
            
            # 30-day rolling average pace
            if 'date' in df.columns and 'pace' in df.columns:
                # Calculate rolling average on numeric pace
                df_indexed = df.set_index('date')
                rolling_pace_30d = df_indexed.groupby('runner_id')['pace_numeric'].rolling('30D', min_periods=1).mean()
                rolling_pace_30d = rolling_pace_30d.reset_index()
                
                # Convert numeric rolling pace back to formatted string
                numeric_30d_pace = rolling_pace_30d['pace_numeric'].values
                
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
                # 7-day heart rate
                df_indexed = df.set_index('date')
                rolling_hr = df_indexed.groupby('runner_id')['average_heart_rate'].rolling('7D', min_periods=1).mean()
                rolling_hr = rolling_hr.reset_index()
                df['7d_avg_heart_rate'] = rolling_hr['average_heart_rate'].values
                logger.info("Created 7-day rolling average heart rate")
                
                # 30-day heart rate
                rolling_hr_30d = df_indexed.groupby('runner_id')['average_heart_rate'].rolling('30D', min_periods=1).mean()
                rolling_hr_30d = rolling_hr_30d.reset_index()
                df['30d_avg_heart_rate'] = rolling_hr_30d['average_heart_rate'].values
                logger.info("Created 30-day rolling average heart rate")
            
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
            
            # 7-day and 30-day rolling average cadence
            if 'date' in df.columns and 'average_cadence' in df.columns:
                # 7-day cadence
                df_indexed = df.set_index('date')
                rolling_cadence = df_indexed.groupby('runner_id')['average_cadence'].rolling('7D', min_periods=1).mean()
                rolling_cadence = rolling_cadence.reset_index()
                df['7d_avg_cadence'] = rolling_cadence['average_cadence'].values
                logger.info("Created 7-day rolling average cadence")
                
                # 30-day cadence
                rolling_cadence_30d = df_indexed.groupby('runner_id')['average_cadence'].rolling('30D', min_periods=1).mean()
                rolling_cadence_30d = rolling_cadence_30d.reset_index()
                df['30d_avg_cadence'] = rolling_cadence_30d['average_cadence'].values
                logger.info("Created 30-day rolling average cadence")
            
            # 7-day and 30-day rolling average distance
            if 'date' in df.columns and 'distance' in df.columns:
                # 7-day distance (average per run)
                df_indexed = df.set_index('date')
                rolling_distance = df_indexed.groupby('runner_id')['distance'].rolling('7D', min_periods=1).mean()
                rolling_distance = rolling_distance.reset_index()
                df['7d_avg_distance'] = rolling_distance['distance'].values
                logger.info("Created 7-day rolling average distance")
                
                # 30-day distance (average per run)
                rolling_distance_30d = df_indexed.groupby('runner_id')['distance'].rolling('30D', min_periods=1).mean()
                rolling_distance_30d = rolling_distance_30d.reset_index()
                df['30d_avg_distance'] = rolling_distance_30d['distance'].values
                logger.info("Created 30-day rolling average distance")
        
        # 8. Create time-based features
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
        
        # 9. Calculate days since last run
        if 'date' in df.columns:
            df['days_since_last_run'] = df.groupby('runner_id')['date'].diff().dt.days
            logger.info("Created days since last run feature")
        
        # 10. Fill NaN values in the newly created features
        # For lagged and diff features, we can't meaningfully fill the first entry
        # But for rolling features, we've used min_periods=1 so they should be filled
        
        # 11. Reorganize columns to group related metrics together
        logger.info("Reorganizing columns to group related metrics together")
        
        # First, get all column names
        all_columns = df.columns.tolist()

        # Identify key column groups that should appear first
        id_columns = [col for col in all_columns if any(x in col.lower() for x in ['id', 'date', 'runner'])]
        basic_metrics = ['distance', 'elapsed_time', 'moving_time', 'pace', 'average_speed', 'max_speed']
        heart_rate_metrics = [col for col in all_columns if 'heart_rate' in col.lower()]
        cadence_metrics = [col for col in all_columns if 'cadence' in col.lower()]
        elevation_metrics = [col for col in all_columns if any(x in col.lower() for x in ['elevation', 'grade'])]
        effort_metrics = [col for col in all_columns if 'effort' in col.lower()]
        time_metrics = [col for col in all_columns if any(x in col.lower() for x in ['week', 'day', 'month', 'weekend'])]

        # Create pairs of related 7d and 30d metrics
        def pair_metrics(metric_list):
            base_metrics = []
            for col in metric_list:
                if '7d_' in col or '30d_' in col:
                    # Extract the base metric name
                    base_name = col.split('7d_')[-1] if '7d_' in col else col.split('30d_')[-1]
                    if base_name not in base_metrics:
                        base_metrics.append(base_name)
        
            # Group related metrics
            paired_metrics = []
            for base in base_metrics:
                seven_day = [col for col in metric_list if f'7d_{base}' in col]
                thirty_day = [col for col in metric_list if f'30d_{base}' in col]
                paired_metrics.extend(seven_day + thirty_day)
        
            return paired_metrics

        # Get the remaining columns that aren't in any special group
        paired_columns = pair_metrics(all_columns)
        remaining_cols = [col for col in all_columns if col not in id_columns + basic_metrics + 
                         heart_rate_metrics + cadence_metrics + elevation_metrics + 
                         effort_metrics + time_metrics + paired_columns]

        # Create the final column order
        ordered_columns = (
            id_columns + 
            basic_metrics + 
            heart_rate_metrics + 
            cadence_metrics + 
            elevation_metrics +
            effort_metrics +
            time_metrics +
            paired_columns +
            remaining_cols
        )

        # Check if we have all columns accounted for
        missing_cols = [col for col in all_columns if col not in ordered_columns]
        if missing_cols:
            ordered_columns.extend(missing_cols)
            logger.warning(f"Found {len(missing_cols)} columns that weren't in any grouping: {missing_cols}")

        # Reorder the DataFrame
        df = df[ordered_columns]
        logger.info("Reorganized columns to group related metrics together")
        
        # FIXED CONSOLIDATED ROLLING METRICS CALCULATION
        # Replace the previous rolling metrics code with this

        # Define all the metrics we want to calculate rolling averages for
        rolling_metrics = {
            'average_heart_rate': 'heart_rate',
            'average_cadence': 'cadence',
            'distance': 'distance',
            'elapsed_time': 'elapsed_time',
            'relative_effort': 'relative_effort',
            'average_grade': 'grade',
            'pace_numeric': 'pace'  # Special handling for pace
        }

        # Calculate all rolling metrics in one consistent way
        if 'runner_id' in df.columns:
            logger.info("Calculating rolling metrics based on last N runs")
            
            # Check for and safely remove existing rolling metrics to avoid duplicates
            for col in df.columns.tolist():  # Create a copy of columns to avoid modification during iteration
                if '7d_avg_' in col or '30d_avg_' in col or '7d_total_' in col or '30d_total_' in col:
                    if col in df.columns:  # Extra check to make sure column still exists
                        try:
                            df = df.drop(columns=[col])
                            logger.info(f"Removed existing column: {col}")
                        except KeyError:
                            logger.warning(f"Tried to remove {col} but it was already gone")
            
            # Now calculate all metrics in a consistent way
            for source_col, metric_name in rolling_metrics.items():
                if source_col in df.columns:
                    # Last 7 runs average
                    df[f'7d_avg_{metric_name}'] = df.groupby('runner_id')[source_col] \
                                                .rolling(window=7, min_periods=1).mean() \
                                                .reset_index(level=0, drop=True)
                    
                    # Last 30 runs average
                    df[f'30d_avg_{metric_name}'] = df.groupby('runner_id')[source_col] \
                                                .rolling(window=30, min_periods=1).mean() \
                                                .reset_index(level=0, drop=True)
                    
                    logger.info(f"Created rolling averages for {metric_name}")
            
            # Special handling for total distance (sum instead of average)
            if 'distance' in df.columns:
                df['7d_total_distance'] = df.groupby('runner_id')['distance'] \
                                        .rolling(window=7, min_periods=1).sum() \
                                        .reset_index(level=0, drop=True)
                
                df['30d_total_distance'] = df.groupby('runner_id')['distance'] \
                                        .rolling(window=30, min_periods=1).sum() \
                                        .reset_index(level=0, drop=True)
                
                logger.info("Created total distance metrics")
            
            # Format pace metrics if they exist
            if 'pace_numeric' in df.columns and '7d_avg_pace' in df.columns:
                df['7d_avg_pace'] = df['7d_avg_pace'].apply(minutes_to_pace_format)
                
            if 'pace_numeric' in df.columns and '30d_avg_pace' in df.columns:
                df['30d_avg_pace'] = df['30d_avg_pace'].apply(minutes_to_pace_format)
                
            if 'pace_numeric' in df.columns:
                logger.info("Formatted pace rolling averages")
            
            # Special handling for pace (needs to be calculated from pace_numeric)
            logger.info("Calculating rolling average pace metrics")
            
            # Last 7 runs average pace
            df['7d_avg_pace_numeric'] = df.groupby('runner_id')['pace_numeric'] \
                                    .rolling(window=7, min_periods=1).mean() \
                                    .reset_index(level=0, drop=True)
            
            # Last 30 runs average pace
            df['30d_avg_pace_numeric'] = df.groupby('runner_id')['pace_numeric'] \
                                     .rolling(window=30, min_periods=1).mean() \
                                     .reset_index(level=0, drop=True)
            
            # Format pace into proper string format
            df['7d_avg_pace'] = df['7d_avg_pace_numeric'].apply(minutes_to_pace_format)
            df['30d_avg_pace'] = df['30d_avg_pace_numeric'].apply(minutes_to_pace_format)
            
            # Remove the numeric versions after formatting
            df = df.drop(columns=['7d_avg_pace_numeric', '30d_avg_pace_numeric'])
            
            logger.info("Created and formatted rolling average pace metrics")
        
        # 12. Save the feature-enriched data
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Force removal of ALL pace-related numeric columns
        logger.info("Final cleanup: explicitly removing all pace numeric columns")
        columns_to_keep = [col for col in df.columns if not ('pace' in col.lower() and 'numeric' in col.lower())]
        df = df[columns_to_keep]  # Keep only non-pace-numeric columns

        # Double-check to make sure they're gone
        remaining_pace_numeric = [col for col in df.columns if 'pace' in col.lower() and 'numeric' in col.lower()]
        if remaining_pace_numeric:
            logger.warning(f"WARNING: Still found pace numeric columns after cleanup: {remaining_pace_numeric}")
            # Force remove them again with a different method
            for col in remaining_pace_numeric:
                try:
                    del df[col]
                    logger.info(f"Forcibly deleted column: {col}")
                except:
                    logger.error(f"Failed to delete column: {col}")

        # Log final column list before saving
        logger.info(f"Final columns before saving: {df.columns.tolist()}")
        
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
