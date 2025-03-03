import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
import json
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xgboost_training')

# File paths
DATA_DIR = "data/processed_data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def ensure_dirs_exist(dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")

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

def prepare_data(df, target_col='pace', test_size=0.2, val_size=0.25):
    """Prepare data for XGBoost training"""
    logger.info("Preparing data for model training")
    
    # Drop non-feature columns
    drop_cols = ['activity_id', 'runner_id', 'date']
    feat_cols = [col for col in df.columns if col not in drop_cols and col != target_col]
    
    # Convert target column to numeric if it's a pace string
    if df[target_col].dtype == 'object':
        from scripts.transformations import pace_to_minutes
        y = df[target_col].apply(pace_to_minutes)
        logger.info(f"Converted target '{target_col}' from string to minutes")
    else:
        y = df[target_col]
    
    # Prepare feature matrix
    X = df[feat_cols]
    
    # Handle categorical features (if any)
    cat_cols = X.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        logger.info(f"One-hot encoding categorical features: {list(cat_cols)}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Check for and handle NaN values
    if X.isna().any().any():
        logger.warning(f"Found NaN values in features, filling with median")
        X = X.fillna(X.median())
    
    # Split data: training (60%), validation (20%), test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size*2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feat_cols

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train the XGBoost model with early stopping"""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'n_jobs': -1,
            'random_state': 42
        }
    
    logger.info(f"Training XGBoost with parameters: {params}")
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    logger.info(f"Model trained with best iteration: {model.best_iteration}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data"""
    logger.info("Evaluating model on test data")
    
    # Create DMatrix for test data
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    
    # Convert pace minutes back to pace string for some examples
    from scripts.transformations import minutes_to_pace_str
    examples = []
    for i in range(min(10, len(y_test))):
        examples.append({
            'Actual (minutes)': y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i],
            'Predicted (minutes)': y_pred[i],
            'Actual (pace)': minutes_to_pace_str(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]),
            'Predicted (pace)': minutes_to_pace_str(y_pred[i]),
            'Error (minutes)': abs((y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]) - y_pred[i])
        })
    
    results = {
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'examples': examples
    }
    
    return results, y_pred

def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance and save to file"""
    logger.info("Plotting feature importance")
    
    # Check if model is a Booster or an XGBRegressor
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
    else:
        booster = model  # It's already a Booster object
    
    # Get feature importance
    importance = booster.get_score(importance_type='gain')
    
    # If there are no features with importance, log warning and return
    if not importance:
        logger.warning("No feature importance found. This may happen with early stopping or if features weren't used.")
        return
    
    # No need to convert keys if they're already feature names
    if list(importance.keys())[0].startswith('f'):
        # The old way - convert f0, f1, etc. to feature names
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
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    logger.info(f"Feature importance plot saved to {output_path}")

def plot_predictions(y_test, y_pred, output_file):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Pace (minutes)')
    plt.ylabel('Predicted Pace (minutes)')
    plt.title('Actual vs Predicted Pace')
    plt.grid(True)
    plt.savefig(output_file)
    logger.info(f"Predictions plot saved to {output_file}")

def save_model(model, feature_names, results, version):
    """Save model, metadata, and results to files"""
    logger.info("Saving model and results")
    
    # Create model folder
    model_folder = os.path.join(MODELS_DIR, f"xgboost_pace_{version}")
    ensure_directory(model_folder)
    
    # Save model
    model_path = os.path.join(model_folder, "model.json")
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
    results_path = os.path.join(model_folder, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save feature names
    metadata = {
        'feature_names': feature_names,
        'timestamp': version,
        'description': 'XGBoost model to predict running pace based on various factors'
    }
    
    metadata_path = os.path.join(model_folder, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

def hyperparameter_tuning(X_train, y_train, X_val, y_val, cv=3):
    """Perform hyperparameter tuning using cross-validation"""
    logger.info("Starting hyperparameter tuning")
    
    # Create parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # For faster tuning, uncomment this smaller grid
    # param_grid = {
    #     'max_depth': [3, 6],
    #     'learning_rate': [0.05, 0.1],
    #     'n_estimators': [100, 500],
    #     'subsample': [0.8, 1.0]
    # }
    
    # Combine train and validation for cross-validation
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    # Initialize XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Perform grid search
    logger.info("Performing grid search (this may take a while)...")
    grid_search.fit(X_combined, y_combined)
    
    # Get best parameters
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    
    # Train model with best parameters
    best_model = train_xgboost(X_train, y_train, X_val, y_val, params=best_params)
    
    return best_model, best_params

def main():
    """Main function to run the training pipeline"""
    parser = argparse.ArgumentParser(description="Train XGBoost model on running data")
    parser.add_argument("--file", type=str, default="activities_tabular.csv", 
                        help="Name of the CSV file to use for training")
    args = parser.parse_args()
    
    # Create necessary directories
    for directory in [DATA_DIR, MODELS_DIR, 'results']:
        ensure_dirs_exist([directory])
        
    # Use the filename from command line arguments
    df = load_data(args.file)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(df, target_col='pace')
    
    # Train model
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results, y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot results
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(RESULTS_DIR, f"xgboost_pace_{version}")
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_feature_importance(model, feature_names, os.path.join(plot_dir, "feature_importance.png"))
    plot_predictions(y_test, y_pred, os.path.join(plot_dir, "predictions.png"))
    
    # Save model
    save_model(model, feature_names, results, version)
    
    logger.info("XGBoost training pipeline completed successfully")

if __name__ == "__main__":
    main() 