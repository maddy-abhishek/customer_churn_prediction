import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Performs initial data cleaning."""
    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing 'TotalCharges' (they are few)
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Drop customerID as it's not a feature
    df.drop('customerID', axis=1, inplace=True)
    
    return df

def get_preprocessor(df):
    """Creates a preprocessing pipeline for categorical and numerical features."""

    #Separate features (X) and target (y) FIRST
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Identify categorical and numerical features (excluding the target 'Churn')
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns



    # Create a column transformer for preprocessing
    # One-hot encode categorical features, and scale numerical features later in model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def encode_target(df, target_column='Churn'):
    """Encodes the target variable."""
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    return df, le