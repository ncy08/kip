import pandas as pd
import numpy as np

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
    # Ensure the data is sorted by runner and date
    df = df.sort_values(by=['runner_id', 'date']).copy()
    
    # Create lag features for each specified metric
    for lag in range(1, lags + 1):
        df[f'pace_t-{lag}'] = df.groupby('runner_id')['pace'].shift(lag)
        df[f'cadence_t-{lag}'] = df.groupby('runner_id')['average_cadence'].shift(lag)
        df[f'hr_t-{lag}'] = df.groupby('runner_id')['average_heart_rate'].shift(lag)
    
    # Create rolling average features (example: over 7 runs)
    window_size = 7
    df['rolling_7d_avg_pace'] = df.groupby('runner_id')['pace'] \
                                  .rolling(window=window_size, min_periods=1).mean() \
                                  .reset_index(level=0, drop=True)
    df['rolling_7d_avg_cadence'] = df.groupby('runner_id')['average_cadence'] \
                                     .rolling(window=window_size, min_periods=1).mean() \
                                     .reset_index(level=0, drop=True)
    df['rolling_7d_avg_hr'] = df.groupby('runner_id')['average_heart_rate'] \
                                .rolling(window=window_size, min_periods=1).mean() \
                                .reset_index(level=0, drop=True)
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
    sequences = []
    feature_columns = ['pace', 'average_cadence', 'average_heart_rate', 'distance']
    
    # Group data by runner_id and sort by date
    for runner, group in df.groupby('runner_id'):
        group = group.sort_values('date')
        data = group[feature_columns].values
        
        # Create sliding windows
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            sequences.append(window)
    
    return np.array(sequences)