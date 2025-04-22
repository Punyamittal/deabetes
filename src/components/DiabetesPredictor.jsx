import React, { useState, useCallback, useMemo } from 'react';
import axios from 'axios';
import '../styles/DiabetesPredictor.css';

const API_URL = 'http://localhost:5000/api/predict';

// Field validation constraints
const FIELD_CONSTRAINTS = {
  pregnancies: { min: 0, max: 20 },
  glucose: { min: 0, max: 300 },
  bloodPressure: { min: 0, max: 200 },
  skinThickness: { min: 0, max: 100 },
  insulin: { min: 0, max: 1000 },
  bmi: { min: 10, max: 50 },
  dpf: { min: 0, max: 3 },
  age: { min: 0, max: 120 }
};

const DiabetesPredictor = () => {
  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    bloodPressure: '',
    skinThickness: '',
    insulin: '',
    bmi: '',
    dpf: '',
    age: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Memoize the form labels and fields to prevent unnecessary re-renders
  const formFields = useMemo(() => [
    { id: 'pregnancies', label: 'Pregnancies', placeholder: 'Number of pregnancies', step: '1' },
    { id: 'glucose', label: 'Glucose (mg/dL)', placeholder: 'Plasma glucose concentration' },
    { id: 'bloodPressure', label: 'Blood Pressure (mm Hg)', placeholder: 'Diastolic blood pressure' },
    { id: 'skinThickness', label: 'Skin Thickness (mm)', placeholder: 'Triceps skin fold thickness' },
    { id: 'insulin', label: 'Insulin (μU/ml)', placeholder: '2-Hour serum insulin' },
    { id: 'bmi', label: 'BMI (kg/m²)', placeholder: 'Body mass index', step: '0.1' },
    { id: 'dpf', label: 'Diabetes Pedigree Function', placeholder: 'Diabetes pedigree function', step: '0.001' },
    { id: 'age', label: 'Age (years)', placeholder: 'Age in years' }
  ], []);

  const handleInputChange = useCallback((e) => {
    const { id, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [id]: value
    }));
    
    // Clear error when user starts typing again
    if (error) setError(null);
  }, [error]);

  const validateField = useCallback((field, value) => {
    const numValue = Number(value);
    const constraints = FIELD_CONSTRAINTS[field];
    
    if (value === '' || isNaN(numValue)) {
      return `Please enter a valid value for ${field}`;
    }
    
    if (numValue < constraints.min || numValue > constraints.max) {
      return `${field} should be between ${constraints.min} and ${constraints.max}`;
    }
    
    return null;
  }, []);

  const validateForm = useCallback(() => {
    // Check each field against its constraints
    for (const field in formData) {
      const errorMessage = validateField(field, formData[field]);
      if (errorMessage) {
        setError(errorMessage);
        return false;
      }
    }
    
    setError(null);
    return true;
  }, [formData, validateField]);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Convert form data to numbers
      const requestData = Object.keys(formData).reduce((acc, key) => {
        acc[key] = Number(formData[key]);
        return acc;
      }, {});
      
      // API call to the Flask backend with timeout
      const response = await axios.post(API_URL, requestData, {
        timeout: 10000, // 10 second timeout
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      // Update prediction state with the response
      setPrediction({
        outcome: response.data.prediction,
        probability: response.data.probability || 0.5
      });
    } catch (error) {
      console.error('Error making prediction:', error);
      
      // Handle different types of errors
      if (error.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again later.');
      } else if (!navigator.onLine) {
        setError('You appear to be offline. Please check your internet connection.');
      } else if (error.response) {
        // The server responded with a status code outside the 2xx range
        setError(error.response.data?.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        // The request was made but no response was received
        setError('No response from server. Please check if the API is running.');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  }, [formData, validateForm]);

  // Reset the form
  const handleReset = useCallback(() => {
    setFormData({
      pregnancies: '',
      glucose: '',
      bloodPressure: '',
      skinThickness: '',
      insulin: '',
      bmi: '',
      dpf: '',
      age: ''
    });
    setPrediction(null);
    setError(null);
  }, []);

  return (
    <div className="diabetes-predictor-container">
      <div className="predictor-card">
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="text-3xl font-bold text-primary-800 mb-2 flex items-center justify-center">
            <span className="material-symbols-outlined mr-2 text-primary-600 animate-pulse-subtle">
              medical_services
            </span>
            Diabetes Risk Predictor
            <span className="material-symbols-outlined ml-2 text-primary-600 animate-pulse-subtle">
              monitoring
            </span>
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto animate-fade-in-slow">Enter your health information below to assess your diabetes risk using our predictive model.</p>
        </div>
      
        <div className="form-container">
          <h2 className="section-title">
            Health Parameters
          </h2>
      
          <form onSubmit={handleSubmit} noValidate>
            <div className="form-grid">
              {formFields.map(field => (
                <div className="form-group" key={field.id}>
                  <label htmlFor={field.id} className="form-label">{field.label}</label>
                  <input 
                    type="number" 
                    id={field.id} 
                    className="form-input"
                    placeholder={field.placeholder}
                    value={formData[field.id]}
                    onChange={handleInputChange}
                    min={FIELD_CONSTRAINTS[field.id].min}
                    max={FIELD_CONSTRAINTS[field.id].max}
                    step={field.step || '1'}
                    required
                  />
                </div>
              ))}
            </div>
            
            {error && (
              <div className="error-message">
                <span className="material-symbols-outlined error-icon">error</span>
                {error}
              </div>
            )}
      
            <div className="mt-8 text-center animate-fade-in-up button-group">
              <button 
                className="predict-button"
                type="submit"
                disabled={isLoading}
              >
                {isLoading ? 'Processing...' : 'Predict Diabetes Risk'}
                <span className="material-symbols-outlined ml-2 align-text-bottom animate-pulse">analytics</span>
              </button>
              
              <button 
                className="reset-button"
                type="button"
                onClick={handleReset}
                disabled={isLoading}
              >
                Reset
                <span className="material-symbols-outlined ml-2 align-text-bottom">refresh</span>
              </button>
            </div>
          </form>
        </div>
      
        <div className="results-container">
          <h2 className="section-title">
            Prediction Results
          </h2>
          
          {prediction ? (
            <div className="prediction-result">
              <div className={`result-circle ${prediction.outcome === 'Diabetic' ? 'diabetic' : 'not-diabetic'}`}>
                <div className="result-text">
                  <span className="result-value">{prediction.outcome}</span>
                  <p className="result-probability">Confidence: {(prediction.probability * 100).toFixed(1)}%</p>
                </div>
              </div>
              
              <p className="result-explanation">
                {prediction.outcome === 'Diabetic' 
                  ? 'Based on the provided information, you may be at risk for diabetes. Please consult with a healthcare professional for a proper diagnosis.'
                  : 'Based on the provided information, you appear to have a lower risk for diabetes. Continue maintaining a healthy lifestyle.'}
              </p>
            </div>
          ) : (
            <div className="awaiting-prediction">
              <div className="loading-circle">
                <div className="awaiting-text">
                  <span className="awaiting-label">Awaiting</span>
                  <p className="awaiting-sublabel">Input Data</p>
                </div>
                {isLoading && <div className="loading-spinner"></div>}
              </div>
              
              <p className="awaiting-explanation">
                Complete the form above and click "Predict" to receive your diabetes risk assessment.
              </p>
            </div>
          )}
        </div>
        
        <div className="disclaimer">
          <p>This tool is for informational purposes only and should not replace professional medical advice.</p>
          <p className="learn-more">
            <span className="material-symbols-outlined align-text-bottom text-xs mr-1 animate-bounce-subtle">info</span>
            Learn more about diabetes risk factors
          </p>
        </div>
      </div>
    </div>
  );
};

export default DiabetesPredictor; 