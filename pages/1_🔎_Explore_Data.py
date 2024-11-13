import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model using the correct path
try:
    rf_model = joblib.load('./models/trained_rf_model.joblib')
except FileNotFoundError:
    st.error("The trained Random Forest model could not be found. Please ensure the model file is in the correct location.")

# Streamlit settings for the Explore Data page
st.set_page_config(page_title="Explore Data", page_icon="📊")

st.title("Explore Data")
st.markdown("### Feature Importance")
st.write("Below are some visualizations to help understand the model and the dataset used.")

# Function to visualize feature importance
def plot_feature_importance(model, feature_names):
    filtered_indices = [i for i, feature in enumerate(feature_names) if not feature.startswith("Locality_")]
    filtered_importances = model.feature_importances_[filtered_indices]
    filtered_features = [feature_names[i] for i in filtered_indices]
    sorted_indices = np.argsort(filtered_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh([filtered_features[i] for i in sorted_indices], filtered_importances[sorted_indices], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance (Excluding Locality)")
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Ensure that the model is loaded successfully before proceeding
if 'rf_model' in locals():
    feature_names = ['Number of Bedrooms', 'Living Area', 'Equipped Kitchen', 'Furnished', 'Swimming Pool',
                     'Encoded Building Condition', 'Property Type']  # Adjusted list of features used in the training

    plot_feature_importance(rf_model, feature_names)
else:
    st.warning("Model is not available. Visualizations cannot be generated.")
