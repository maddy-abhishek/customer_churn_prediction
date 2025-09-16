# Customer Churn Prediction Project 🔮

This project demonstrates a complete end-to-end machine learning workflow for predicting customer churn in a telecom company. It includes data analysis, feature engineering, model training, evaluation, and a simple web interface for real-time predictions.

# ✨ Features

  - Exploratory Data Analysis (EDA): In-depth analysis with visualizations (seaborn, matplotlib) to uncover key insights about churn drivers.

  - Feature Engineering: Creation of new features to improve model performance.

  - Model Training: Implementation and comparison of three powerful classification models: Logistic Regression, Random Forest, and XGBoost.

  - Model Evaluation: Rigorous evaluation using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC, which are suitable for imbalanced datasets.

  - Interactive Web App: A user-friendly interface built with Streamlit that allows users to input customer data and receive an instant churn probability prediction.

  - Modular Code: The entire project is structured into modular, reusable Python scripts for better maintainability and clarity.

# 📂 Project Structure

The project is organized in a clean and logical manner to separate concerns.

customer_churn_prediction/

|

├── data/

│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv

|

├── saved_models/

│   └── best_churn_model.pkl

|

├── src/

│   ├── __init__.py

│   ├── data_preprocessing.py

│   ├── feature_engineering.py

│   ├── model_training.py

│   └── predict.py

|

├── app.py                     # The main Streamlit web application

├── requirements.txt           # Project dependencies

└── README.md                  # This file

# 🚀 Getting Started

1. Create and activate a virtual environment (recommended):
   
    - python -m venv venv
      
    - venv\Scripts\activate

3. Install the required dependencies:
   
    - pip install -r requirements.txt

# 🏃‍♂️ How to Run

The project runs in two main steps: first, training the model, and second, launching the web application.

Step 1: Train the Machine Learning Model

Run the training script from the src directory. This script will load the data, perform preprocessing and feature engineering, train multiple models, evaluate them, and save the best-performing model to the saved_models/ directory.
  - python src/model_training.py

you will see the evaluation metrics for each model printed in the console.

Step 2: Launch the Streamlit Web App

Once the model is trained and saved, launch the web application from the project's root directory.
  - streamlit run app.py

This command will open a new tab in your web browser with the interactive prediction interface. You can now input different customer attributes to see the model's churn prediction in real-time.

# 🛠️ Skills Demonstrated
This project showcases a range of skills essential for a data science and machine learning role:

 - Data Analysis & Preprocessing: Handling missing values, data type conversion, and encoding categorical features using pandas and scikit-learn.

 - Feature Engineering: Creating insightful features like tenure_years and monthly_to_total_ratio.

 - Machine Learning Modeling: Implementing, training, and comparing Logistic Regression, Random Forest, and XGBoost. Special attention is given to handling class imbalance.

 - Model Evaluation: Using appropriate metrics (Precision, Recall, F1-Score, AUC-ROC) to assess model performance on an imbalanced classification task.

 - Web Application Development: Building a simple, interactive UI with Streamlit to serve the ML model.

 - Software Engineering Best Practices: Writing clean, modular, and well-documented code.
