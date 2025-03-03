# Running Data Pipeline & Pace Prediction

A comprehensive toolkit for runner data that cleans, transforms, prepares, and models running data for analysis and pace prediction.

## Project Overview

This project features:

1. A data processing pipeline that cleans and validates running data
2. Feature engineering to create useful metrics from raw activity data
3. Multiple XGBoost models to predict running pace with different feature sets
4. Visualization tools to understand feature importance and model performance

## Project Structure

```
project-root/
├── data/
│   ├── raw_data/          # Place new CSV files here
│   └── processed_data/    # Cleaned and feature-engineered files
├── scripts/
│   ├── data_cleaning.py   # Data validation and cleaning
│   ├── feature_generation.py  # Feature engineering
│   ├── pipeline.py        # Main orchestration script
│   ├── train_xgboost.py   # Base pace prediction model
│   ├── train_xgboost_no_pace_history.py  # Model without pace history
│   └── train_xgboost_no_pace_no_time.py  # Model without pace or direct time
├── models/                # Trained models stored here
│   ├── xgboost_pace_TIMESTAMP/  # Base model
│   ├── xgboost_pace_no_history_TIMESTAMP/  # No pace history model
│   └── xgboost_pace_no_direct_time_TIMESTAMP/  # No pace/time model
├── results/               # Feature importance plots and metrics
├── docs/                  # Documentation
└── README.md              # Project description and instructions
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git

### Environment Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/running-data-pipeline.git
   cd running-data-pipeline
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Processing Pipeline

1. Place your raw CSV file in the `data/raw_data/` directory
2. Run the pipeline:
   ```
   python -m scripts.pipeline data/raw_data/your_file.csv
   ```
3. Find the processed output in `data/processed_data/`

### Pace Prediction Models

#### 1. Base Model (using all features)

Best performance for pure prediction accuracy:

```
python -m scripts.train_xgboost
```

#### 2. No Pace History Model

Predicts pace without using historical pace data:

```
python -m scripts.train_xgboost_no_pace_history
```

#### 3. No Pace/No Direct Time Model

Predicts pace without using pace history or direct time metrics:

```
python -m scripts.train_xgboost_no_pace_no_time
```

### Model Performance & Insights

Our experiments revealed:

1. **Best Predictors of Pace**:

   - With all features: Recent pace history (7-day avg pace)
   - Without pace history: Current elapsed time
   - Without pace & direct time: 7-day average elapsed time + heart rate

2. **Performance Metrics**:

   - Base model: MAE ~0.76 min, R² ~0.59
   - No pace history: MAE ~0.91 min, R² ~0.56
   - No pace/direct time: MAE higher, R² lower

3. **Key Insights**:
   - Heart rate becomes much more important when pace history is unavailable
   - 7-day average metrics strongly predict performance
   - Training consistency (30d_run_count) is a moderate predictor

## Pipeline Steps

1. **Data Cleaning**: Validates data, handles missing values, and removes anomalies
2. **Feature Generation**: Creates derived features like rolling averages, lagged values, and cumulative metrics
3. **Output**: Produces cleaned and feature-rich datasets ready for analysis or modeling
4. **Model Training**: Builds XGBoost regression models for pace prediction
5. **Evaluation**: Generates metrics, example predictions, and feature importance plots

## Data Format

The expected input CSV should contain the following columns:

- Runner ID
- Date
- Distance (miles/km)
- Time (duration)
- Heart Rate
- Pace
- Cadence (optional)
- Elevation data (optional)
- Weather conditions (optional)

## Implementation Guide

### Script Structure

#### data_cleaning.py

```python
# data_cleaning.py
import pandas as pd

def clean_data(input_csv: str, output_csv: str):
    """
    Reads raw data from `input_csv`, cleans and validates, saves to `output_csv`.
    """
    # 1. Load the data
    df = pd.read_csv(input_csv)

    # 2. Basic validation
    # e.g., remove rows with impossible distances
    df = df[df['distance'] < 100]  # example threshold

    # 3. Handle missing heart rate or pace
    # e.g., fill missing values with 'NaN' or forward fill
    df['heart_rate'].fillna(method='ffill', inplace=True)

    # 4. Save cleaned data
    df.to_csv(output_csv, index=False)
```

#### feature_generation.py

```python
# feature_generation.py
import pandas as pd

def generate_features(cleaned_csv: str, output_csv: str):
    """
    Loads cleaned data, creates new features, and saves the feature-enriched dataset.
    """
    df = pd.read_csv(cleaned_csv)
    df.sort_values(by=['runner_id', 'date'], inplace=True)

    # Create a lagged feature for pace
    df['pace_previous'] = df.groupby('runner_id')['pace'].shift(1)

    # Create rolling averages
    df['7d_avg_pace'] = (df.groupby('runner_id')['pace']
                            .rolling(7, min_periods=1).mean()
                            .reset_index(drop=True))

    # Create training volume metrics
    df['7d_run_count'] = (df.groupby('runner_id')['date']
                            .rolling(7, min_periods=1).count()
                            .reset_index(drop=True))

    df.to_csv(output_csv, index=False)
```

#### train_xgboost.py (excerpt)

```python
# train_xgboost.py
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'eta': 0.1
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

    return model
```

## Future Enhancements

- Web dashboard for visualizing pace predictions
- Integration with Strava API for automatic data updates
- Model hyperparameter tuning for improved accuracy
- Weather API integration for environmental factors
- Personalized pace prediction based on target race events

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

[Your chosen license]

# Running Performance Analysis with SHAP

This repository provides tools for analyzing running performance data using SHAP (SHapley Additive exPlanations) with XGBoost models.

## Overview

The core functionality is implemented in `shap_analysis.py`, which:

1. Extracts the latest run from your dataset
2. Uses SHAP TreeExplainer to calculate SHAP values for the run
3. Generates a ranked list of top contributors and their impact on pace
4. Formats a human-readable improvement report with actionable insights

## Requirements

```
pandas
numpy
xgboost
shap
matplotlib
```

## Usage

Assuming you already have a trained XGBoost model and a dataset of run information:

```python
from shap_analysis import run_shap_analysis
import xgboost as xgb
import pandas as pd

# Load your trained model
model = xgb.Booster()
model.load_model('path/to/your/model.json')

# Load your running data
data = pd.read_csv('your_running_data.csv')

# Run the analysis
results = run_shap_analysis(
    model=model,
    data=data,
    baseline_pace=5.0,  # Your expected pace (e.g., min/km)
    top_n=5  # Show top 5 features
)

# Print the summary report
print(results['report']['summary'])
for rec in results['report']['recommendations']:
    print(f"- {rec}")

# Display visualizations
import matplotlib.pyplot as plt
plt.show()
```

See `run_analysis_example.py` for a complete working example.

## Features

- **SHAP Analysis**: Uses TreeExplainer to provide accurate feature importance values
- **Run Extraction**: Automatically identifies and analyzes your most recent run
- **Prioritized Insights**: Ranks features by their impact on your running pace
- **Actionable Recommendations**: Generates specific suggestions to improve performance
- **Visualizations**: Creates SHAP summary and beeswarm plots to visualize feature impacts

## Example Output

The analysis provides:

1. A summary of factors increasing or decreasing your pace
2. Ranked list of top contributors and their impact
3. Specific recommendations for improvement
4. Visualizations showing feature importance and impact direction

## Adding to Your Pipeline

To integrate this analysis into your existing pipeline:

1. Ensure your model is trained and saved in a format loadable by XGBoost
2. Prepare your run data with appropriate features
3. Call `run_shap_analysis()` with your model and data
4. Use the results to provide insights to the runner

## Customization

You can customize the analysis by:

- Adjusting the `top_n` parameter to show more or fewer features
- Providing a specific `baseline_pace` or letting it use the model's expected value
- Modifying the visualization settings
- Extending the report generation to include additional metrics

## How It Works

The SHAP analysis works by:

1. Using TreeExplainer to decompose the model's prediction into contributions from each feature
2. Calculating the magnitude and direction of each feature's impact on pace
3. Ranking features by their absolute contribution
4. Generating insights based on whether features increase or decrease pace

This provides a data-driven approach to understanding what specific factors are most affecting your running performance.
