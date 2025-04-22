from flask import Flask, request, jsonify, Response
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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up the path for the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../diabetes_model.joblib')

# Import the DiabetesModel class from a separate file
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/models'))
from src.models.diabetes_model import DiabetesModel

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

# Load the model only if it exists
if os.path.exists(MODEL_PATH):
    logger.info(f"Loading existing model from {MODEL_PATH}")
    try:
        model.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
else:
    logger.warning(f"Model file not found at {MODEL_PATH}. Prediction will not work.")

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
        
        # Check if model is loaded
        if model.model is None:
            return jsonify({'error': 'Model not loaded. Please try again later.'}), 500
        
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


# This is the required function for Vercel serverless functions
# It converts Flask responses to Vercel-compatible serverless responses
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({"error": "Invalid endpoint"}), 404


# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Diabetes Prediction API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 