from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import sys
import logging
import joblib
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the diabetes_model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diabetes_model import DiabetesModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define field constraints for validation
FIELD_CONSTRAINTS = {
    'pregnancies': {'min': 0, 'max': 20},
    'glucose': {'min': 0, 'max': 300},
    'bloodPressure': {'min': 0, 'max': 200},
    'skinThickness': {'min': 0, 'max': 100},
    'insulin': {'min': 0, 'max': 1000},
    'bmi': {'min': 10, 'max': 50},
    'dpf': {'min': 0, 'max': 3},
    'age': {'min': 0, 'max': 120}
}

# Request timing decorator
def timing_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Request to {f.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Initialize the model
model = DiabetesModel()
model_path = 'diabetes_model.joblib'

try:
    # Check if a saved model exists, otherwise train a new one
    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        logger.info("Training new model...")
        # Train and save a new model
        model.load_data()
        model.preprocess_data()
        model.split_data()
        
        # Use a smaller parameter grid for faster training
        param_grid = {
            'n_estimators': [100],
            'max_depth': [None, 20],
            'min_samples_split': [2]
        }
        
        model.train_model(param_grid=param_grid)
        model.save_model(model_path)
        logger.info(f"New model saved to {model_path}")
        
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    sys.exit(1)

def validate_prediction_input(data):
    """
    Validate the input data for a prediction request.
    
    Args:
        data (dict): Input data for prediction
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if all required fields are present
    required_fields = list(FIELD_CONSTRAINTS.keys())
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Check if values are numeric and within constraints
    for field, constraints in FIELD_CONSTRAINTS.items():
        try:
            value = float(data[field])
            if value < constraints['min'] or value > constraints['max']:
                return False, f"{field} should be between {constraints['min']} and {constraints['max']}"
        except (ValueError, TypeError):
            return False, f"{field} must be a number"
    
    return True, None

@app.route('/api/predict', methods=['POST'])
@timing_decorator
def predict():
    """
    API endpoint for making diabetes predictions.
    
    Expected JSON format:
    {
        "pregnancies": 6,
        "glucose": 148,
        "bloodPressure": 72,
        "skinThickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "dpf": 0.627,
        "age": 50
    }
    
    Returns:
    {
        "prediction": "Diabetic" or "Not Diabetic",
        "probability": 0.75 (if available)
    }
    """
    try:
        # Get the data from the request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate input data
        is_valid, error_message = validate_prediction_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Extract the features
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('bloodPressure', 0)),
            float(data.get('skinThickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('dpf', 0)),
            float(data.get('age', 0))
        ]
        
        # Make prediction
        result = model.predict(features)
        logger.info(f"Prediction result: {result['prediction']} with probability: {result['probability']}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return jsonify({
        'status': 'ok', 
        'message': 'Diabetes prediction API is running',
        'model_loaded': model.model is not None
    })


# Add an error handler for 404 errors
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


# Add an error handler for 405 errors
@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


# Add an error handler for 500 errors
@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Diabetes Prediction API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 