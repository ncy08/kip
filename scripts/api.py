from flask import Flask, request, jsonify
import os
import uuid
from pipeline import run_pipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('api')

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    """
    API endpoint to upload a CSV file and run the pipeline on it.
    
    Expects a multipart/form-data POST request with a 'csv_file' field.
    """
    try:
        # Check if the post request has the file part
        if 'csv_file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['csv_file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Check if the file is a CSV
        if not file.filename.endswith('.csv'):
            logger.warning(f"File {file.filename} is not a CSV")
            return jsonify({"status": "error", "message": "File must be a CSV"}), 400
        
        # Generate a unique filename to avoid collisions
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{file.filename}"
        raw_path = os.path.join("data/raw_data", filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        
        # Save the file
        file.save(raw_path)
        logger.info(f"File saved to {raw_path}")
        
        # Generate output paths
        base_name = os.path.splitext(filename)[0]
        cleaned_path = os.path.join("data/processed_data", f"{base_name}_cleaned.csv")
        features_path = os.path.join("data/processed_data", f"{base_name}_features.csv")
        
        # Run the pipeline
        logger.info(f"Starting pipeline for {raw_path}")
        result = run_pipeline(raw_path, cleaned_path, features_path)
        
        # Return the result
        if result["status"] == "success":
            return jsonify({
                "status": "success",
                "message": "File uploaded and pipeline run successfully",
                "input_file": raw_path,
                "cleaned_file": cleaned_path,
                "features_file": features_path,
                "runtime_seconds": result["runtime_seconds"]
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": f"Pipeline failed: {result['error_message']}",
                "input_file": raw_path,
                "runtime_seconds": result["runtime_seconds"]
            }), 500
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    # Make sure the data directories exist
    os.makedirs("data/raw_data", exist_ok=True)
    os.makedirs("data/processed_data", exist_ok=True)
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True) 