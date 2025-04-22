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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'diabetes_model.joblib')
sys.path.append(BASE_DIR)

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

# Initialize model
model = None
model_loaded = False

# Try to load the DiabetesModel class
try:
    # First try to import from local directory
    try:
        from diabetes_model import DiabetesModel
        logger.info("Imported DiabetesModel from local API directory")
    except ImportError:
        # Fall back to the src/models directory
        sys.path.append(os.path.join(BASE_DIR, 'src/models'))
        from src.models.diabetes_model import DiabetesModel
        logger.info("Imported DiabetesModel from src/models directory")
    
    # Initialize the model
    model = DiabetesModel()
    
    # Load the model if it exists
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}")
        try:
            model.load_model(MODEL_PATH)
            model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}")
except ImportError as e:
    logger.error(f"Error importing DiabetesModel: {str(e)}")


def validate_prediction_input(data):
    """Validate the input data for a prediction request."""
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
def predict():
    """API endpoint for making diabetes predictions."""
    try:
        # Check if model is loaded
        if not model_loaded or model is None:
            return jsonify({
                'error': 'Model not loaded. Please contact the administrator.'
            }), 503
        
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
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return jsonify({
        'status': 'ok', 
        'message': 'Diabetes prediction API is running',
        'model_loaded': model_loaded
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Handle all other routes"""
    return jsonify({
        "status": "error",
        "message": "Invalid endpoint", 
        "path": path
    }), 404


# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 