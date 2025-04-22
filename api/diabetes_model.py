import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


class DiabetesModel:
    """
    A machine learning model for diabetes prediction using the Pima Indians Diabetes Dataset.
    This class handles data preprocessing, model training, evaluation, and prediction.
    """
    
    def __init__(self):
        """Initialize the DiabetesModel with default attributes."""
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.pipeline = None
        
    def load_data(self, url=None, file_path=None):
        """
        Load the diabetes dataset from a URL or local file.
        
        Parameters:
        -----------
        url : str, optional
            URL to the dataset
        file_path : str, optional
            Path to local dataset file
            
        Returns:
        --------
        self : DiabetesModel
            Returns self for method chaining
        """
        # Default URL if none provided
        if url is None and file_path is None:
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        # Column names for the dataset
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        # Load data from URL or file
        if url:
            self.df = pd.read_csv(url, names=columns)
        elif file_path:
            self.df = pd.read_csv(file_path, names=columns)
            
        return self
    
    def explore_data(self, show_plots=False):
        """
        Perform exploratory data analysis on the dataset.
        
        Parameters:
        -----------
        show_plots : bool, default=False
            Whether to display EDA plots
            
        Returns:
        --------
        dict
            Dictionary containing summary statistics and data insights
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Basic statistics
        summary = {
            'shape': self.df.shape,
            'info': self.df.info(),
            'describe': self.df.describe(),
            'missing_values': self.df.isnull().sum(),
            'class_distribution': self.df['Outcome'].value_counts()
        }
        
        # Zero values that should be missing values
        zeros = {
            'Glucose': (self.df['Glucose'] == 0).sum(),
            'BloodPressure': (self.df['BloodPressure'] == 0).sum(),
            'SkinThickness': (self.df['SkinThickness'] == 0).sum(),
            'Insulin': (self.df['Insulin'] == 0).sum(),
            'BMI': (self.df['BMI'] == 0).sum()
        }
        summary['zeros'] = zeros
        
        if show_plots:
            # Distribution of target variable
            plt.figure(figsize=(8, 6))
            sns.countplot(x='Outcome', data=self.df)
            plt.title('Class Distribution (0: No Diabetes, 1: Diabetes)')
            plt.show()
            
            # Distribution of features
            plt.figure(figsize=(15, 10))
            self.df.hist(figsize=(15, 10))
            plt.tight_layout()
            plt.show()
            
            # Correlation matrix
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.show()
            
            # Box plots to identify outliers
            plt.figure(figsize=(15, 10))
            self.df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(15, 10))
            plt.tight_layout()
            plt.show()
        
        return summary
    
    def preprocess_data(self, handle_zeros=True, remove_outliers=True):
        """
        Preprocess the dataset for model training.
        
        Parameters:
        -----------
        handle_zeros : bool, default=True
            Whether to handle zero values that should be missing
        remove_outliers : bool, default=True
            Whether to remove outliers using IQR method
            
        Returns:
        --------
        self : DiabetesModel
            Returns self for method chaining
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a copy of the dataframe to avoid modifying the original
        df_processed = self.df.copy()
        
        # Handle zero values in columns where zeros are not valid
        if handle_zeros:
            # Replace zeros with NaN for columns where zero is not a valid value
            columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for column in columns_to_process:
                df_processed[column] = df_processed[column].replace(0, np.nan)
            
            # Impute missing values with median for each column
            for column in columns_to_process:
                median_value = df_processed[column].median(skipna=True)
                df_processed[column].fillna(median_value, inplace=True)
        
        # Remove outliers using IQR method
        if remove_outliers:
            for column in df_processed.columns[:-1]:  # Exclude outcome column
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_processed = df_processed[(df_processed[column] >= lower_bound) & 
                                           (df_processed[column] <= upper_bound)]
        
        # Split into features and target
        self.X = df_processed.drop('Outcome', axis=1)
        self.y = df_processed['Outcome']
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42, apply_smote=True):
        """
        Split the data into training and testing sets.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random seed for reproducibility
        apply_smote : bool, default=True
            Whether to apply SMOTE to handle class imbalance
            
        Returns:
        --------
        self : DiabetesModel
            Returns self for method chaining
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Apply SMOTE to handle class imbalance in the training set
        if apply_smote:
            smote = SMOTE(random_state=random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
        return self
    
    def train_model(self, param_grid=None, cv=5):
        """
        Train a Random Forest model with hyperparameter tuning.
        
        Parameters:
        -----------
        param_grid : dict, optional
            Grid of hyperparameters to search
        cv : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        self : DiabetesModel
            Returns self for method chaining
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        
        # Create a pipeline with scaling and the model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid={'classifier__' + key: value for key, value in param_grid.items()},
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search to the data
        grid_search.fit(self.X_train, self.y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self
    
    def evaluate_model(self, show_plots=False):
        """
        Evaluate the trained model on the test set.
        
        Parameters:
        -----------
        show_plots : bool, default=False
            Whether to display evaluation plots
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Calculate ROC curve and AUC
        try:
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc = roc_auc_score(self.y_test, y_prob)
        except:
            fpr, tpr, auc = None, None, None
        
        # Compile results
        evaluation = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'roc': {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        }
        
        # Print results
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        if show_plots:
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()
            
            # ROC Curve
            if fpr is not None and tpr is not None:
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.show()
                
            # Feature Importance
            if hasattr(self.model['classifier'], 'feature_importances_'):
                importances = self.model['classifier'].feature_importances_
                feature_names = self.X.columns
                
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title('Feature Importances')
                plt.bar(range(len(importances)), importances[indices], align='center')
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.show()
        
        return evaluation
    
    def predict(self, input_data):
        """
        Make a prediction for new input data.
        
        Parameters:
        -----------
        input_data : array-like
            Array of input features in the same order as the training data
            
        Returns:
        --------
        dict
            Dictionary containing prediction result
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert input to numpy array if it's a list
        if isinstance(input_data, list):
            input_data = np.array(input_data).reshape(1, -1)
            
        # Ensure input has the right shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        # Get probability if the model supports it
        probability = None
        try:
            probability = self.model.predict_proba(input_data)[0][1]
        except:
            pass
        
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'probability': probability
        }
        
        return result
    
    def save_model(self, file_path='diabetes_model.joblib'):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        file_path : str, default='diabetes_model.joblib'
            Path where the model will be saved
            
        Returns:
        --------
        str
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")
        return file_path
    
    def load_model(self, file_path='diabetes_model.joblib'):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        file_path : str, default='diabetes_model.joblib'
            Path to the saved model
            
        Returns:
        --------
        self : DiabetesModel
            Returns self for method chaining
        """
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return self


def train_and_save_model():
    """
    Train a model and save it to disk.
    
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create model instance
    diabetes_model = DiabetesModel()
    
    # Load, preprocess, and split data
    (diabetes_model
        .load_data()
        .preprocess_data()
        .split_data())
    
    # Define a smaller parameter grid for quicker training
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }
    
    # Train and evaluate model
    (diabetes_model
        .train_model(param_grid=param_grid)
        .evaluate_model())
    
    # Save the model
    return diabetes_model.save_model('diabetes_model.joblib')


def example_usage():
    """
    Example of how to use the DiabetesModel class.
    """
    # Create model instance
    model = DiabetesModel()
    
    # Full pipeline example
    (model
        .load_data()
        .preprocess_data()
        .split_data()
        .train_model()
        .evaluate_model(show_plots=True))
    
    # Save the model
    model.save_model()
    
    # Make a prediction for a new person
    sample_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Example input
    prediction = model.predict(sample_input)
    print(f"Prediction: {prediction['prediction']}")
    if prediction['probability'] is not None:
        print(f"Probability: {prediction['probability']:.2f}")


if __name__ == "__main__":
    # Only run example usage when the script is run directly
    example_usage() 