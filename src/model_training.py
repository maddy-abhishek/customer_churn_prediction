from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

def split_data(df, target_column='Churn'):
    """Splits data into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor):
    """Trains and evaluates Logistic Regression, Random Forest, and XGBoost models."""
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=sum(y_train==0)/sum(y_train==1))
    }
    
    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        # Create a full pipeline with preprocessing, scaling, and the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)), # with_mean=False for sparse matrices from OneHotEncoder
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC-ROC': auc}
        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}\n")

        # Keep track of the best model based on AUC
        if auc > best_auc:
            best_auc = auc
            best_model = pipeline
            
    return best_model, results

def save_model(model, filepath):
    """Saves the trained model to a file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

if __name__ == '__main__':
    # This block allows running this script directly to train and save the model
    from data_preprocessing import load_data, clean_data, get_preprocessor, encode_target
    from feature_engineering import create_features
    
    # 1. Load and Clean Data
    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = clean_data(df)
    
    # 2. Feature Engineering
    df = create_features(df)

    # 3. Encode Target Variable
    df, target_encoder = encode_target(df) # We might need encoder later
    
    # 4. Get Preprocessor
    preprocessor = get_preprocessor(df)
    
    # 5. Split Data
    X_train, X_test, y_train, y_test = split_data(df)

    # 6. Train and Evaluate Models
    best_model, evaluation_results = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor)
    
    # 7. Save the Best Model
    save_model(best_model, 'saved_models/best_churn_model.pkl')