import pandas as pd

def create_features(df):
    """Creates new features to improve model performance."""
    
    # Tenure in years
    df['tenure_years'] = df['tenure'] / 12
    
    # Monthly-to-total charges ratio
    # Add a small epsilon to avoid division by zero, though NaNs were dropped
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)
    
    # Interaction term: tenure and monthly charges
    df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']
    
    return df