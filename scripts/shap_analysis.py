import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional


def extract_latest_run(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Extract the latest run from the dataset.
    
    Args:
        df: DataFrame containing run data
        date_column: Column name containing date information
        
    Returns:
        DataFrame containing only the latest run
    """
    # Ensure date column is datetime type
    if pd.api.types.is_string_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Get the latest date
    latest_date = df[date_column].max()
    
    # Filter for the latest run
    latest_run = df[df[date_column] == latest_date]
    
    return latest_run


def calculate_shap_values(model: xgb.Booster, X: pd.DataFrame) -> Tuple[shap.Explanation, shap.Explanation]:
    """
    Calculate SHAP values for the given data using TreeExplainer.
    
    Args:
        model: Trained XGBoost model
        X: Feature data for which to calculate SHAP values
        
    Returns:
        Tuple of (shap_values, expected_value)
    """
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    return shap_values, explainer.expected_value


def rank_feature_importance(shap_values: shap.Explanation, X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a ranked list of features and their impact.
    
    Args:
        shap_values: SHAP values from TreeExplainer
        X: Feature data used for SHAP calculation
        
    Returns:
        DataFrame with features ranked by importance
    """
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Calculate mean absolute SHAP value for each feature
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values.values).mean(0),
        'mean_shap_value': shap_values.values.mean(0)
    })
    
    # Sort by importance (descending)
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance


def generate_improvement_report(
    feature_importance: pd.DataFrame, 
    shap_values: shap.Explanation, 
    X: pd.DataFrame,
    baseline_pace: float,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Generate a human-readable improvement report.
    
    Args:
        feature_importance: Ranked feature importance DataFrame
        shap_values: SHAP values from TreeExplainer
        X: Feature data used for SHAP calculation
        baseline_pace: The baseline/expected pace value
        top_n: Number of top features to include in report
        
    Returns:
        Dictionary containing report data
    """
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    # Initialize report
    report = {
        'baseline_pace': baseline_pace,
        'features': [],
        'summary': "",
        'recommendations': []
    }
    
    # Process each top feature
    for _, row in top_features.iterrows():
        feature = row['feature']
        importance = row['importance']
        mean_shap = row['mean_shap_value']
        
        # Determine if this feature increases or decreases pace
        impact = "increasing" if mean_shap > 0 else "decreasing"
        
        # Get the feature value for this run
        feature_value = X[feature].iloc[0]
        
        # Create feature dictionary
        feature_dict = {
            'name': feature,
            'importance': float(importance),
            'impact': impact,
            'value': float(feature_value) if not isinstance(feature_value, str) else feature_value,
            'mean_shap': float(mean_shap)
        }
        
        report['features'].append(feature_dict)
        
        # Generate recommendation
        if mean_shap > 0:
            recommendation = f"Consider decreasing '{feature}' to improve pace"
        else:
            recommendation = f"Consider increasing '{feature}' to improve pace"
        
        report['recommendations'].append(recommendation)
    
    # Generate summary text
    positive_impacts = [f['name'] for f in report['features'] if f['impact'] == 'increasing']
    negative_impacts = [f['name'] for f in report['features'] if f['impact'] == 'decreasing']
    
    summary = "Run Analysis Summary:\n"
    if positive_impacts:
        summary += f"- The following factors are increasing your pace: {', '.join(positive_impacts)}\n"
    if negative_impacts:
        summary += f"- The following factors are decreasing your pace: {', '.join(negative_impacts)}\n"
    
    report['summary'] = summary
    
    return report


def visualize_shap_values(shap_values: shap.Explanation, X: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Create visualizations of SHAP values.
    
    Args:
        shap_values: SHAP values from TreeExplainer
        X: Feature data used for SHAP calculation
        top_n: Number of top features to show
        
    Returns:
        Figure with SHAP visualizations
    """
    # Create a new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Summary plot
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n, show=False)
    plt.tight_layout()
    
    # Plot 2: Beeswarm plot
    plt.subplot(1, 2, 2)
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=top_n, show=False)
    plt.tight_layout()
    
    return fig


def run_shap_analysis(
    model: xgb.Booster, 
    data: pd.DataFrame, 
    baseline_pace: Optional[float] = None,
    date_column: str = 'date',
    top_n: int = 5,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline to run SHAP analysis on the latest run.
    
    Args:
        model: Trained XGBoost model
        data: DataFrame containing run data
        baseline_pace: Baseline/expected pace value (if None, will use model's expected value)
        date_column: Column name containing date information
        top_n: Number of top features to include in report
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary containing analysis results and report
    """
    # Extract latest run
    latest_run = extract_latest_run(data, date_column)
    
    # Get features (assuming target is not in the data)
    X = latest_run.drop([date_column], axis=1, errors='ignore')
    
    # If there are other columns to drop (like target variable), do it here
    # For example: X = X.drop(['pace'], axis=1, errors='ignore')
    
    # Calculate SHAP values
    shap_values, expected_value = calculate_shap_values(model, X)
    
    # If baseline_pace wasn't provided, use the model's expected value
    if baseline_pace is None:
        baseline_pace = expected_value
    
    # Rank feature importance
    feature_importance = rank_feature_importance(shap_values, X)
    
    # Generate improvement report
    report = generate_improvement_report(
        feature_importance, 
        shap_values, 
        X, 
        baseline_pace, 
        top_n
    )
    
    # Add raw SHAP data to the results
    results = {
        'report': report,
        'feature_importance': feature_importance.to_dict(orient='records'),
        'shap_values': shap_values,
        'latest_run': latest_run
    }
    
    # Generate visualizations if requested
    if visualize:
        fig = visualize_shap_values(shap_values, X, top_n)
        results['visualization'] = fig
    
    return results


if __name__ == "__main__":
    # Example usage
    # Assuming you have a trained model and data loaded
    # model = xgb.Booster()
    # model.load_model('trained_xgb_model.json')
    # data = pd.read_csv('running_data.csv')
    # results = run_shap_analysis(model, data)
    # 
    # # Print report summary
    # print(results['report']['summary'])
    # for rec in results['report']['recommendations']:
    #     print(f"- {rec}")
    # 
    # # Show visualizations
    # plt.show()
    pass 