# Running Data Pipeline

A data processing pipeline for runner data that cleans, transforms, and prepares running data for analysis and modeling.

## Project Overview

This pipeline takes raw CSV files containing running data, cleans and validates the data, generates useful features, and prepares it for machine learning models like XGBoost. The system is designed to be repeatable, automated, and extensible.

## Project Structure

```
project-root/
├── data/
│   ├── raw_data/         # Place new CSV files here
│   └── processed_data/   # Cleaned and feature-engineered files
├── scripts/
│   ├── data_cleaning.py  # Data validation and cleaning
│   ├── feature_generation.py  # Feature engineering
│   └── pipeline.py       # Main orchestration script
├── models/               # Trained models will be stored here
├── docs/                 # Documentation
└── README.md             # Project description and instructions
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

### Basic Usage

1. Place your raw CSV file in the `data/raw_data/` directory
2. Run the pipeline:
   ```
   python scripts/pipeline.py data/raw_data/your_file.csv
   ```
3. Find the processed output in `data/processed_data/`

### Pipeline Steps

1. **Data Cleaning**: Validates data, handles missing values, and removes anomalies
2. **Feature Generation**: Creates derived features like rolling averages, lagged values, and cumulative metrics
3. **Output**: Produces cleaned and feature-rich datasets ready for analysis or modeling

## Data Format

The expected input CSV should contain the following columns:

- Runner ID
- Date
- Distance (miles/km)
- Time (duration)
- Heart Rate (optional)
- Pace
- [Other relevant metrics]

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

    # Example: Create a lagged feature for pace
    df['pace_t-1'] = df.groupby('runner_id')['pace'].shift(1)

    # Example: 7-day rolling average if your data is daily
    # df['7d_avg_pace'] = (df.groupby('runner_id')['pace']
    #                         .rolling(7, min_periods=1).mean()
    #                         .reset_index(drop=True))

    df.to_csv(output_csv, index=False)
```

#### pipeline.py

```python
# pipeline.py
import os
from data_cleaning import clean_data
from feature_generation import generate_features

def run_pipeline(input_csv: str, cleaned_csv: str, features_csv: str):
    clean_data(input_csv, cleaned_csv)
    generate_features(cleaned_csv, features_csv)
    print("Pipeline completed.")

if __name__ == "__main__":
    # Example usage:
    input_path = "data/raw_data/runs.csv"
    cleaned_path = "data/processed_data/runs_cleaned.csv"
    features_path = "data/processed_data/runs_features.csv"

    # Make sure these directories exist
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    run_pipeline(input_path, cleaned_path, features_path)
```

### Testing the Pipeline

1. Place a sample CSV in `data/raw_data/runs.csv`
2. Run the pipeline:
   ```
   cd project-root
   source venv/bin/activate
   python scripts/pipeline.py
   ```
3. Check `data/processed_data/` for the output files

## Automation Options

### Command-Line / Cron Job

Schedule the pipeline to run at specific intervals using cron:

```
cd /path/to/project-root && source venv/bin/activate && python scripts/pipeline.py
```

### Simple API Integration

Create a minimal Flask or FastAPI server with an upload endpoint:

```python
from flask import Flask, request
from scripts.pipeline import run_pipeline
import os

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["csv_file"]
    raw_path = os.path.join("data/raw_data", file.filename)
    file.save(raw_path)

    cleaned_path = "data/processed_data/runs_cleaned.csv"
    features_path = "data/processed_data/runs_features.csv"

    run_pipeline(raw_path, cleaned_path, features_path)
    return "File uploaded and pipeline run!"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

### Cloud Functions

Deploy as a serverless function triggered by file uploads to cloud storage.

## Best Practices for Repeatability

1. **Documentation**: Keep this README updated with any changes to the pipeline
2. **Environment Management**: Use `pip freeze > requirements.txt` to capture dependencies
3. **Logging & Error Handling**: Add robust error handling to all scripts
4. **Data Validation**: Ensure input CSVs meet expected format requirements
5. **Version Control**: Use Git for code and consider DVC for data versioning

## Future Enhancements

- Model training integration
- Database storage instead of CSV files
- Web interface for file uploads and results visualization
- Containerization with Docker

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

[Your chosen license]
