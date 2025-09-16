import joblib
import pandas as pd
from .feature_engineering import create_features

def load_model(filepath):
    """Loads a pre-trained model from a file."""
    return joblib.load(filepath)

def make_prediction(model, input_data):
    """
    Makes a churn prediction on new customer data.
    input_data: A dictionary or DataFrame with a single row of customer data.
    """
    # Convert input dict to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
        
    # Ensure correct data types (especially for numerical features)
    input_df['tenure'] = pd.to_numeric(input_df['tenure'])
    input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'])
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'])

    # Apply the same feature engineering
    input_df_featured = create_features(input_df.copy())
    
    # Predict probability
    churn_probability = model.predict_proba(input_df_featured)[:, 1][0]
    
    return churn_probability