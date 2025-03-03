import mlflow
import mlflow.xgboost
from old.train_xgboost import (
    load_data, prepare_data, train_xgboost, evaluate_model,
    plot_feature_importance, plot_predictions, ensure_dirs_exist
)
import logging
import os
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xgboost_mlflow')

# File paths
DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Fixed model and results directories (no timestamps)
MODEL_DIR = os.path.join(MODELS_DIR, "xgboost_pace_model")
RESULT_DIR = os.path.join(RESULTS_DIR, "xgboost_pace_results")

def train_with_mlflow(experiment_name="running_pace_prediction"):
    """Run training pipeline with MLflow tracking"""
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Ensure directories exist
    ensure_dirs_exist([DATA_DIR, MODELS_DIR, RESULTS_DIR])
    
    # Clean previous results if they exist
    if os.path.exists(MODEL_DIR):
        logger.info(f"Removing previous model directory: {MODEL_DIR}")
        shutil.rmtree(MODEL_DIR)
    
    if os.path.exists(RESULT_DIR):
        logger.info(f"Removing previous results directory: {RESULT_DIR}")
        shutil.rmtree(RESULT_DIR)
    
    # Create fresh directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Load data
    df = load_data("processed_running_data.csv")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(df, target_col='pace')
    
    # Define parameters to try
    params_list = [
        {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 500,
            'n_jobs': -1,
            'random_state': 42
        },
        {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'n_estimators': 1000,
            'n_jobs': -1,
            'random_state': 42
        }
    ]
    
    best_model = None
    best_rmse = float('inf')
    best_params = None
    
    # Try different parameters
    for i, params in enumerate(params_list):
        logger.info(f"Training model {i+1}/{len(params_list)} with parameters: {params}")
        
        with mlflow.start_run(run_name=f"xgboost_model_{i+1}"):
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Train model
            model = train_xgboost(X_train, y_train, X_val, y_val, params)
            
            # Evaluate model
            results, y_pred = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("mae", results['metrics']['mae'])
            mlflow.log_metric("rmse", results['metrics']['rmse'])
            mlflow.log_metric("r2", results['metrics']['r2'])
            
            # Create and log plots with fixed names (no timestamp)
            importance_plot = os.path.join(RESULT_DIR, "feature_importance.png")
            predictions_plot = os.path.join(RESULT_DIR, "predictions.png")
            
            plot_feature_importance(model, feature_names, importance_plot)
            plot_predictions(y_test, y_pred, predictions_plot)
            
            mlflow.log_artifact(importance_plot)
            mlflow.log_artifact(predictions_plot)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            # Track best model
            if results['metrics']['rmse'] < best_rmse:
                best_rmse = results['metrics']['rmse']
                best_model = model
                best_params = params
    
    # Save the best model to disk (overwriting any previous model)
    best_model_path = os.path.join(MODEL_DIR, "best_model.json")
    best_model.save_model(best_model_path)
    
    # Save the best parameters
    import json
    with open(os.path.join(MODEL_DIR, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Best model RMSE: {best_rmse}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best model saved to: {best_model_path}")
    
    return best_model, feature_names

if __name__ == "__main__":
    train_with_mlflow() 