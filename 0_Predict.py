import streamlit as st
import joblib
import numpy as np
import pandas as pd

#Terminal command to view browser
# streamlit run 0_Predict.py

# Load the trained model and encoder
rf_model = joblib.load('models/trained_rf_model.joblib')
locality_encoder = joblib.load('encoder/locality_encoder.joblib')

st.set_page_config(
    page_title="Belgian House Price Predictor",
    page_icon="🏠",
    initial_sidebar_state="expanded",
    layout="centered",
)

st.title("Belgian Real Estate Price Predictor")
st.markdown(
    """
    Welcome to the Belgian Real Estate Price Predictor! This app is designed to predict house prices in Belgium
    using data collected from immoweb.be. By employing a **Random Forest model**, our goal is to provide a
     price forecast for the Belgian Real Estate market.
    """
)

# Input fields for user parameters
st.header("📊 Input Parameters")

property_type = st.selectbox(
    "Select Property Type", ["Apartment", "House"]
)
property_type_encoded = 1 if property_type == "House" else 0

number_of_bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=16, value=2)

living_area = st.slider("Living Area (m²)", min_value=1, max_value=1500, value=200)

equipped_kitchen = st.selectbox("Equipped Kitchen?", ["Yes", "No"]) == "Yes"

furnished = st.selectbox("Furnished?", ["Yes", "No"]) == "Yes"

swimming_pool = st.selectbox("Swimming Pool?", ["Yes", "No"]) == "Yes"

building_condition = st.selectbox(
    "Building Condition",
    ["To restore", "To renovate", "To be done up", "Good", "Just renovated", "As new"]
)
condition_mapping = {
    "To restore": 0,
    "To renovate": 1,
    "To be done up": 2,
    "Good": 3,
    "Just renovated": 4,
    "As new": 5,
}
building_condition_encoded = condition_mapping[building_condition]

locality_data = st.text_input(
    "Enter the Locality (e.g., postal code at least 2 characters)", "1000"
)

if st.button("Predict", key="predict-button"):
    # Encode locality data using the pre-trained encoder
    locality_prefix = locality_data[:2]
    locality_encoded = locality_encoder.transform([[locality_prefix]])

    # Combine inputs into a single array
    input_data = np.array([
        number_of_bedrooms,
        living_area,
        int(equipped_kitchen),
        int(furnished),
        int(swimming_pool),
        building_condition_encoded,
        property_type_encoded
    ])

    # Append the locality encoding
    input_data = np.concatenate((input_data, locality_encoded[0]))

    # Get feature names from the model and sort accordingly
    model_features = rf_model.feature_names_in_
    input_df = pd.DataFrame([input_data], columns=model_features)
    input_df = input_df[model_features]  # Ensure feature order matches the training

    # Predict and display the result
    prediction_log = rf_model.predict(input_df)
    prediction = np.exp(prediction_log) - 1
    st.success(f"Predicted Price: €{prediction[0]:,.2f}")
