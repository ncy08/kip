import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data/processed_data"
MODELS_DIR = "models"
RESULTS_FOLDER = "results"

# Fixed directories (no timestamps)
MODEL_FOLDER = os.path.join(MODELS_DIR, "xgboost_pace_no_direct_time")

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_data(filename):
    """Load the preprocessed data"""
    file_path = os.path.join(DATA_DIR, filename)
    logger.info(f"Attempting to load data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Cannot find processed data file: {file_path}.")
    
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded data with shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare data for training - removing pace history and specific time features"""
    logger.info("Preparing data for training")
    
    # Convert pace (minutes per mile) from string to float
    if 'pace' in df.columns and df['pace'].dtype == 'object':
        # Extract minutes and seconds from pace strings like "6'33""
        df['pace_minutes'] = df['pace'].apply(lambda x: float(x.split("'")[0]) + float(x.split("'")[1].replace('"', '')) / 60)
        logger.info("Converted pace strings to minutes")
    
    # Drop all pace-related historical features
    pace_pattern = re.compile(r'pace|.*_pace_.*')
    cols_to_drop = [col for col in df.columns if pace_pattern.search(col)]
    
    # Keep pace_minutes for target
    if 'pace_minutes' in cols_to_drop:
        cols_to_drop.remove('pace_minutes')
    
    # Add specific time features to drop list
    specific_time_features = ['elapsed_time', 'moving_time']
    for feature in specific_time_features:
        if feature in df.columns:
            cols_to_drop.append(feature)
    
    # Add 'average_speed' to drop list as it's directly derived from pace
    if 'average_speed' in df.columns:
        cols_to_drop.append('average_speed')
    
    logger.info(f"Dropping pace and specific time columns: {cols_to_drop}")
    df = df.drop(cols_to_drop, axis=1)
    
    # Drop non-feature columns
    non_feature_cols = ['activity_id', 'runner_id', 'date', 'pace', 'activity_type']
    cols_to_drop = [col for col in non_feature_cols if col in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Split features and target
    X = df.drop('pace_minutes', axis=1)
    y = df['pace_minutes']
    
    logger.info(f"Data prepared with {X.shape[1]} features")
    return X, y

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    logger.info("Training XGBoost model")
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    logger.info(f"Model trained with best iteration: {model.best_iteration}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model performance")
    
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    
    # Create example comparison
    results = {
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'examples': []
    }
    
    # Add some example predictions
    for i in range(min(10, len(y_test))):
        actual_minutes = y_test.iloc[i]
        predicted_minutes = y_pred[i]
        error = abs(predicted_minutes - actual_minutes)
        
        # Convert to pace format
        actual_pace = f"{int(actual_minutes)}:{int((actual_minutes % 1) * 60):02d}"
        predicted_pace = f"{int(predicted_minutes)}:{int((predicted_minutes % 1) * 60):02d}"
        
        results['examples'].append({
            'Actual (minutes)': actual_minutes,
            'Predicted (minutes)': predicted_minutes,
            'Actual (pace)': actual_pace,
            'Predicted (pace)': predicted_pace,
            'Error (minutes)': error
        })
    
    return results

def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance, save to file, and generate CSV table"""
    logger.info("Plotting feature importance and generating table")
    
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    
    # If there are no features with importance, log warning and return
    if not importance:
        logger.warning("No feature importance found.")
        return
    
    # Convert to feature names if needed
    if list(importance.keys())[0].startswith('f'):
        importance = {feature_names[int(k.replace('f', ''))]: v for k, v in importance.items()}
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Only plot up to 15 features, or all if fewer than 15
    num_features = min(15, len(importance))
    plt.barh(list(importance.keys())[:num_features], list(importance.values())[:num_features])
    
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature Importance (No Pace & No Direct Time)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    logger.info(f"Feature importance plot saved to {output_path}")
    
    # Save data as CSV
    csv_path = output_path.replace('.png', '.csv')
    
    # Convert to DataFrame for easier output
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Save to CSV
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Feature importance table saved to {csv_path}")
    
    # Also save as markdown table for easier viewing
    md_path = output_path.replace('.png', '.md')
    with open(md_path, 'w') as f:
        f.write("# Feature Importance Table\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, (feature, importance_value) in enumerate(importance.items(), 1):
            f.write(f"| {i} | {feature} | {importance_value:.2f} |\n")
    
    logger.info(f"Feature importance markdown table saved to {md_path}")

def save_model(model, feature_names, results):
    """Save model, metadata, and results to files, overwriting previous files"""
    logger.info("Saving model and results")
    
    # Create or clean model folder
    if os.path.exists(MODEL_FOLDER):
        logger.info(f"Removing previous model contents from: {MODEL_FOLDER}")
        for item in os.listdir(MODEL_FOLDER):
            item_path = os.path.join(MODEL_FOLDER, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
    else:
        logger.info(f"Creating model directory: {MODEL_FOLDER}")
        os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODEL_FOLDER, "model.json")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Convert numpy values to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy(obj.tolist())
        else:
            return obj
    
    # Convert results
    results_json = convert_numpy(results)
    
    # Save results
    results_path = os.path.join(MODEL_FOLDER, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save feature names
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        'feature_names': feature_names,
        'last_updated': timestamp,
        'description': 'XGBoost model to predict running pace without pace features and direct time features'
    }
    
    metadata_path = os.path.join(MODEL_FOLDER, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main function to run the training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model without pace features and direct time features")
    parser.add_argument("--file", type=str, default="activities_tabular.csv", 
                        help="Name of the CSV file to use for training")
    args = parser.parse_args()
    
    # Create necessary directories
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_FOLDER]:
        ensure_directory(directory)
        
    # Load data
    df = load_data(args.file)
    
    # Prepare data - this will remove pace and specific time features
    X, y = prepare_data(df)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Plot feature importance with fixed name (no timestamp)
    plot_path = os.path.join(RESULTS_FOLDER, 'feature_importance_no_pace_no_direct_time.png')
    # Remove previous plot if it exists
    if os.path.exists(plot_path):
        os.remove(plot_path)
    plot_feature_importance(model, list(X.columns), plot_path)
    
    # Save model and results
    save_model(model, list(X.columns), results)
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main() 