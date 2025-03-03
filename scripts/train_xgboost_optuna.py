import os
import sys
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import logging
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data/processed_data"
MODELS_DIR = "models"
RESULTS_FOLDER = "results"

# Fixed model directory (no timestamps)
MODEL_FOLDER = os.path.join(MODELS_DIR, "xgboost_optuna")

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

def prepare_data(df, include_pace_history=False, include_time_features=False):
    """
    Prepare data for training with options to include/exclude pace history and time features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data frame
    include_pace_history : bool
        Whether to include pace-related historical features
    include_time_features : bool
        Whether to include direct time features (elapsed_time, moving_time)
        
    Returns:
    --------
    X, y : features and target variables
    """
    logger.info(f"Preparing data with pace_history={include_pace_history}, time_features={include_time_features}")
    
    # Convert pace (minutes per mile) from string to float if needed
    if 'pace' in df.columns and df['pace'].dtype == 'object':
        # Extract minutes and seconds from pace strings like "6'33""
        df['pace_minutes'] = df['pace'].apply(lambda x: 
            float(x.split("'")[0]) + float(x.split("'")[1].replace('"', '')) / 60 
            if isinstance(x, str) and "'" in x else np.nan)
        logger.info("Converted pace strings to minutes")
    
    # Determine which columns to drop
    cols_to_drop = []
    
    # Handle pace history features
    if not include_pace_history:
        pace_cols = [col for col in df.columns if 'pace' in col.lower() and col != 'pace_minutes']
        cols_to_drop.extend(pace_cols)
        logger.info(f"Excluding {len(pace_cols)} pace history features")
    
    # Handle time features (excluding average_speed)
    if not include_time_features:
        time_cols = ['elapsed_time', 'moving_time']
        time_cols = [col for col in time_cols if col in df.columns]
        cols_to_drop.extend(time_cols)
        logger.info(f"Excluding {len(time_cols)} direct time features")
    
    # Always exclude average_speed as it's directly related to pace
    if 'average_speed' in df.columns:
        cols_to_drop.append('average_speed')
        logger.info("Excluding average_speed (directly related to pace)")
    
    # Remove duplicates from cols_to_drop
    cols_to_drop = list(set(cols_to_drop))
    
    # Drop the identified columns
    if cols_to_drop:
        logger.info(f"Dropping columns: {cols_to_drop}")
        df = df.drop(cols_to_drop, axis=1)
    
    # Drop non-feature columns
    non_feature_cols = ['activity_id', 'runner_id', 'date', 'pace', 'activity_type']
    cols_to_drop = [col for col in non_feature_cols if col in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    
    # Handle missing values - fixed to handle mixed data types
    logger.info("Handling missing values with proper type checking")
    
    # First, identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Fill numeric columns with mean
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill categorical columns with mode (most frequent value)
    for col in categorical_cols:
        if df[col].isna().any():
            # Get the most frequent value, defaulting to empty string if no mode
            most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else ""
            df[col] = df[col].fillna(most_frequent)
    
    # Ensure all categorical variables are properly encoded
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    logger.info(f"Converted {len(categorical_cols)} categorical columns to dummy variables")
    
    # Check for any remaining NaN values
    if df.isna().any().any():
        missing_cols = df.columns[df.isna().any()].tolist()
        logger.warning(f"Warning: Still found missing values in columns: {missing_cols}")
        # Last resort - drop rows with any remaining NaN values
        df = df.dropna()
        logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
    
    # Split features and target
    if 'pace_minutes' not in df.columns:
        logger.error("Target column 'pace_minutes' not found in dataframe")
        raise ValueError("Target column 'pace_minutes' not found. Cannot continue training.")
    
    X = df.drop('pace_minutes', axis=1)
    y = df['pace_minutes']
    
    logger.info(f"Data prepared with {X.shape[1]} features")
    return X, y

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function that defines the hyperparameter search space
    and returns validation MAE to be minimized.
    """
    # Define hyperparameter search space for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    
    # Set an early stopping value
    early_stopping_rounds = 50
    num_boost_round = 1000
    
    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train the model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False  # Set to True to see training progress
    )
    
    # Get validation predictions
    preds = model.predict(dval)
    
    # Calculate validation MAE (to be minimized)
    val_mae = mean_absolute_error(y_val, preds)
    
    # Return the validation MAE that Optuna will try to minimize
    return val_mae

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
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
    
    return results, y_pred

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
    plt.title('Top 15 Feature Importance (Optuna Optimized)')
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

def plot_optuna_optimization_history(study, output_path):
    """Plot Optuna optimization history"""
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optuna Optimization History')
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Optimization history saved to {output_path}")

def save_model(model, feature_names, results, study):
    """Save model, metadata, results, and hyperparameter optimization details"""
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
    
    # Save best hyperparameters
    hyperparams_path = os.path.join(MODEL_FOLDER, "best_params.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Best hyperparameters saved to {hyperparams_path}")
    
    # Save feature names and metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        'feature_names': feature_names,
        'last_updated': timestamp,
        'description': 'XGBoost model to predict running pace with Optuna-optimized hyperparameters',
        'optimization_trials': len(study.trials),
        'best_validation_mae': study.best_value
    }
    
    metadata_path = os.path.join(MODEL_FOLDER, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

def plot_predictions(y_test, y_pred, output_path):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Pace (minutes/unit)')
    plt.ylabel('Predicted Pace (minutes/unit)')
    plt.title('Actual vs Predicted Pace')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Predictions plot saved to {output_path}")

def main():
    """Main function to run the training pipeline with Optuna hyperparameter search"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model with Optuna hyperparameter optimization")
    parser.add_argument("--file", type=str, default="activities_tabular.csv", 
                        help="Name of the CSV file to use for training")
    parser.add_argument("--exclude-pace", action="store_true",
                        help="Exclude pace history features (included by default)")
    parser.add_argument("--exclude-time", action="store_true",
                        help="Exclude direct time features (included by default)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials to run")
    args = parser.parse_args()
    
    # Create necessary directories
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_FOLDER]:
        ensure_directory(directory)
    
    # Load data
    df = load_data(args.file)
    
    # Prepare data based on arguments (REVERSED logic - we include features by default)
    X, y = prepare_data(
        df, 
        include_pace_history=not args.exclude_pace,
        include_time_features=not args.exclude_time
    )
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    
    # Define lambda function to pass additional arguments to objective
    objective_func = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    
    # Run optimization
    logger.info(f"Starting Optuna optimization with {args.trials} trials")
    study.optimize(objective_func, n_trials=args.trials, n_jobs=-1)
    
    logger.info(f"Best validation MAE: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params['eval_metric'] = 'mae'
    best_params['objective'] = 'reg:squarederror'
    
    # Combine train and validation sets for final model
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    dtest = xgb.DMatrix(X_test)
    
    # Train final model
    logger.info("Training final model with best hyperparameters")
    final_model = xgb.train(
        best_params,
        dtrain_full,
        num_boost_round=1000,
    )
    
    # Evaluate final model
    results, y_pred = evaluate_model(final_model, X_test, y_test)
    
    # Generate plots
    model_type = []
    if not args.exclude_pace:
        model_type.append("with_pace")
    else:
        model_type.append("no_pace")
    
    if not args.exclude_time:
        model_type.append("with_time")
    else:
        model_type.append("no_time")
    
    model_name = "_".join(model_type)
    
    # Create fixed output paths (overwrites previous files)
    feature_importance_path = os.path.join(RESULTS_FOLDER, f"feature_importance_optuna_{model_name}.png")
    optimization_history_path = os.path.join(RESULTS_FOLDER, f"optuna_history_{model_name}.png")
    predictions_path = os.path.join(RESULTS_FOLDER, f"predictions_optuna_{model_name}.png")
    
    # Remove previous plots if they exist
    for path in [feature_importance_path, optimization_history_path, predictions_path]:
        if os.path.exists(path):
            os.remove(path)
    
    # Generate plots
    plot_feature_importance(final_model, list(X.columns), feature_importance_path)
    plot_optuna_optimization_history(study, optimization_history_path)
    plot_predictions(y_test, y_pred, predictions_path)
    
    # Save model and results
    save_model(final_model, list(X.columns), results, study)
    
    logger.info("Training pipeline with Optuna completed successfully")

if __name__ == "__main__":
    main() 