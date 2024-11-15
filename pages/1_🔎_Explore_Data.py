import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Make sure to import pandas as it's used for type checking

# Load the trained model using the correct path
rf_model = None
try:
    rf_model = joblib.load('./models/trained_rf_model.joblib')
except FileNotFoundError:
    st.error("The trained Random Forest model could not be found. Please ensure the model file is in the correct location.")

# Load y_test and y_pred values
y_test, y_pred = None, None
try:
    y_test = joblib.load('./models/y_test.joblib')
    y_pred = joblib.load('./models/y_pred.joblib')

    # Convert y_test and y_pred to numpy arrays if they are pandas Series
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

except FileNotFoundError:
    st.error("The y_test or y_pred files could not be found. Please ensure these files are in the correct location.")

# Streamlit settings for the Explore Data page
st.set_page_config(page_title="Explore Data", page_icon="📊")

st.title("Explore Data")
st.markdown("### Feature Importance")
st.write("Below is a visualizations to help understand how each feature contributes to the accuracy of the model.")

# Function to visualize feature importance
def plot_feature_importance(model, feature_names):
    filtered_indices = [i for i, feature in enumerate(feature_names) if not feature.startswith("Locality_")]
    filtered_importances = model.feature_importances_[filtered_indices]
    filtered_features = [feature_names[i] for i in filtered_indices]
    sorted_indices = np.argsort(filtered_importances)[::-1]

    # Increase figure size and adjust font sizes proportionally
    plt.figure(figsize=(50, 50))  # Larger figure size

    # Adjust the fontsize to be proportionally larger
    plt.barh([filtered_features[i] for i in sorted_indices], filtered_importances[sorted_indices], color="skyblue")
    plt.xlabel("Feature Importance", fontsize=100)
    plt.ylabel("Feature", fontsize=100)
    plt.title("Random Forest Feature Importance (Excluding Locality)", fontsize=120)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=80)
    plt.yticks(fontsize=80)
    st.pyplot(plt)



# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(5, 5))
    # Plot the perfect prediction line (y=x line) for reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")

    # Calculate the regression line (best fit line)
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    regression_line = slope * y_test + intercept

    # Plot the regression line
    plt.plot(y_test, regression_line, color='blue', lw=2, label="Regression Line")

    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Values", fontsize = 12)
    plt.ylabel("Predicted Values", fontsize = 12)
    plt.title("Actual vs Predicted Values", fontsize = 15)
    st.pyplot(plt)

# Ensure that the model and data are loaded successfully before proceeding
if rf_model is not None:
    feature_names = ['Number of Bedrooms', 'Living Area', 'Equipped Kitchen', 'Furnished', 'Swimming Pool',
                     'Encoded Building Condition', 'Property Type']  # Adjusted list of features used in the training

    plot_feature_importance(rf_model, feature_names)
else:
    st.warning("Model is not available. Visualizations cannot be generated.")

# Ensure y_test and y_pred are available before plotting
if y_test is not None and y_pred is not None:
    st.markdown("### Actual vs Predicted Values")
    st.write("Below is a visualization to help understand the correlation between the actual data and the predicted data.")
    plot_actual_vs_predicted(y_test, y_pred)
else:
    st.warning("Actual vs Predicted values cannot be plotted as y_test or y_pred is not available.")
