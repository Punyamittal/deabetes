# Diabetes Predictor

A machine learning web application for predicting diabetes risk based on health parameters. This project uses the Pima Indians Diabetes Dataset to train a Random Forest classifier and presents a modern, user-friendly interface for performing predictions.

## Features

- Interactive web interface for entering health parameters
- Machine learning model with high accuracy for diabetes prediction
- Data preprocessing including handling of missing values and outliers
- Visual results with confidence score
- REST API for making predictions

## Technology Stack

### Frontend
- React.js
- CSS3 with modern animations
- Material Symbols for icons

### Backend
- Python
- Flask REST API
- scikit-learn for machine learning
- Pandas and NumPy for data processing

## Setup Instructions

### Prerequisites
- Node.js (v14+)
- Python (v3.8+)
- npm or yarn

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/diabetes-predictor.git
cd diabetes-predictor
```

2. Install frontend dependencies:
```
npm install
```

3. Install backend dependencies:
```
pip install -r requirements.txt
```

### Running the Application

1. Start the backend API server:
```
npm run api
```

2. Start the React frontend (in a new terminal):
```
npm start
```

3. Open your browser and navigate to http://localhost:3000

## Model Details

The diabetes prediction model uses a Random Forest classifier with the following features:
- Pregnancies
- Glucose concentration
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age

The model is trained with hyperparameter tuning using GridSearchCV and handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

## API Documentation

### Predict Endpoint

**URL**: `/api/predict`
**Method**: `POST`
**Content-Type**: `application/json`

**Request Body**:
```json
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
```

**Response**:
```json
{
  "prediction": "Diabetic",
  "probability": 0.75
}
```

## License

MIT

## Acknowledgments

- Original dataset: Pima Indians Diabetes Database
- Dataset source: National Institute of Diabetes and Digestive and Kidney Diseases
- Dataset on Kaggle: [Diabetes Dataset for Beginners](https://www.kaggle.com/code/melikedilekci/diabetes-dataset-for-beginners) 