import streamlit as st
import pandas as pd
from src.predict import load_model, make_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction 🔮",
    page_icon="🤖",
    layout="centered"
)

# --- Load Model ---
# Cache the model loading to improve performance
@st.cache_resource
def get_model():
    return load_model('saved_models/best_churn_model.pkl')

model = get_model()

# --- App Title and Description ---
st.title("Customer Churn Prediction")
st.markdown("Enter customer details below to predict the probability of churn. This app uses a machine learning model to provide a prediction.")

# --- Input Fields for User ---
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

with col3:
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])

st.subheader("Numerical Details")
num_col1, num_col2, num_col3 = st.columns(3)

with num_col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
with num_col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, format="%.2f")
with num_col3:
    # Estimate TotalCharges for simplicity in the UI
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), format="%.2f")

# --- Prediction Logic ---
if st.button("Predict Churn", type="primary"):
    # Create a dictionary from the inputs
    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': 'DSL', # Hardcoded for simplicity, can be expanded
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': 'No', # Hardcoded
        'StreamingMovies': 'No', # Hardcoded
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'SeniorCitizen': 0, # Hardcoded
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Get prediction
    churn_probability = make_prediction(model, input_data)
    churn_prediction = "Yes" if churn_probability > 0.5 else "No"
    
    # Display the result
    st.subheader("Prediction Result")
    if churn_prediction == "Yes":
        st.error(f"**This customer is likely to CHURN.**")
        st.info(f"**Probability of Churn:** {churn_probability:.2%}")
    else:
        st.success(f"**This customer is likely to STAY.**")
        st.info(f"**Probability of Churn:** {churn_probability:.2%}")

    # Display progress bar for probability
    st.progress(churn_probability)