from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def simple_predict(features):
    """
    A lightweight prediction function that mimics a diabetes model
    without requiring scikit-learn or other heavy ML libraries.
    
    This is based on simple risk factors for Type 2 Diabetes:
    - High blood glucose
    - Higher BMI
    - Age (older adults have higher risk)
    - Family history (represented by diabetes pedigree function)
    """
    # Unpack features
    pregnancies = features[0]
    glucose = features[1]
    blood_pressure = features[2]
    skin_thickness = features[3]
    insulin = features[4]
    bmi = features[5]
    dpf = features[6]
    age = features[7]
    
    # Normalize values
    norm_glucose = max(0, min(1, glucose / 200))
    norm_bmi = max(0, min(1, (bmi - 18.5) / 15))
    norm_age = max(0, min(1, age / 80))
    norm_dpf = max(0, min(1, dpf / 1.5))
    
    # Weight the factors (these weights approximate general risk factors)
    # Glucose is heavily weighted as the primary indicator
    risk_score = (
        norm_glucose * 0.45 +  # Glucose is the most important factor
        norm_bmi * 0.25 +      # BMI is another significant factor
        norm_age * 0.20 +      # Age is a risk factor
        norm_dpf * 0.10        # Family history affects risk
    )
    
    # Calibrate the prediction threshold
    is_diabetic = risk_score > 0.5
    
    return {
        'prediction': 'Diabetic' if is_diabetic else 'Not Diabetic',
        'probability': float(max(0.1, min(0.9, risk_score)))  # Limit between 0.1-0.9
    }

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making diabetes predictions."""
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
        
        # Make prediction using our simple model
        result = simple_predict(features)
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return jsonify({
        'status': 'ok', 
        'message': 'Diabetes prediction API is running (lightweight version)'
    })

# For Vercel, we need a "catch-all" route
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Root catchall handler"""
    if not path:
        return "Diabetes Prediction API is running. Use the React frontend to access the application."
    
    # For API endpoints, handle them all through the main API route
    if path.startswith('api/'):
        return jsonify({"error": "Invalid API endpoint"}), 404
        
    return jsonify({"error": f"Invalid path: /{path}"}), 404 