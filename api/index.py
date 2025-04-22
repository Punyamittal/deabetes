from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import os
import sys
import logging
import json

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
        
        # Simplified prediction (mock response for demo)
        # In a real setup, this would call a ML model
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
        
        # For demo: higher glucose, age and BMI increase the risk
        glucose = features[1]
        bmi = features[5]
        age = features[7]
        
        risk_score = (glucose / 300) * 0.5 + (bmi / 50) * 0.3 + (age / 120) * 0.2
        is_diabetic = risk_score > 0.5
        
        result = {
            'prediction': 'Diabetic' if is_diabetic else 'Not Diabetic',
            'probability': min(max(risk_score, 0.1), 0.9)  # Keep between 0.1 and 0.9
        }
        
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
        'message': 'Diabetes prediction API is running'
    })

@app.route('/api', defaults={'path': ''})
@app.route('/api/<path:path>')
def api_catchall(path):
    """Handle all other API routes"""
    return jsonify({
        "status": "error",
        "message": "Invalid API endpoint", 
        "path": f"/api/{path}"
    }), 404

# For Vercel, we need a "catch-all" route
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Root catchall handler"""
    if not path or path.startswith('static'):
        return "Welcome to the Diabetes Prediction API. Use the React frontend to access the application."
    return jsonify({"error": f"Invalid path: /{path}"}), 404 