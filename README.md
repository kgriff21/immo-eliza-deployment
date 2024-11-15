# Belgian Real Estate Price Predictor

Welcome to the **Belgian Real Estate Price Predictor**! This project aims to provide predictions for house prices in Belgium, using a **Random Forest model** trained on data collected from immoweb.be.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Features](#features)
- [Model Description](#model-description)
- [Future Improvements](#future-improvements)

## Overview
The **Belgian Real Estate Price Predictor** is a web application built with **Streamlit** to predict the price of real estate in Belgium. Users input various property details, and the application estimates a predicted price using a pre-trained Random Forest model. This project is intended to assist users in gauging property values in an intuitive and user-friendly way.  

## Installation
To run this project locally, you need to set up the required Python environment and install the necessary dependencies.

1. **Clone the repository**:
   ```pip
   git clone <repository-url>
   ```

2. **Navigate to the project directory**:
   ```pip
   cd immo-eliza-deployment
   ```

3. **Create a virtual environment** (optional but recommended):
   ```pip
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

4. **Install the required dependencies**:
   ```pip
   pip install -r requirements.txt
   ```

## How to Run
To start the **Belgian Real Estate Price Predictor** app locally, run the following command in the terminal:

```pip
streamlit run 0_Predict.py
```
This command will launch the web application, which you can access in your browser by following the link provided in the terminal output.

## Usage
Once the app is running, you will be able to enter property details like:
- Property Type (Apartment or House)
- Number of Bedrooms
- Living Area (in square meters)
- Whether the kitchen is equipped
- Whether the property is furnished
- Whether there is a swimming pool
- Building Condition
- Locality (using postal code or locality information with at least 2 numbers input)

After filling in the necessary fields, click on the **"Predict"** button to see the estimated property price.

## Features
- **Streamlit UI**: A clean, easy-to-use interface for entering property details.
- **Model Prediction**: The model provides price predictions based on user inputs.
- **Input Flexibility**: Users can specify a variety of features such as locality, property type, and more.

## Model Description
The model is a pre-trained **Random Forest** regression model built using **scikit-learn**. It uses features like the number of bedrooms, living area, property type, and building condition to predict the logarithmic price of a property. The output is then converted back to its original value to display the predicted price.
Note the models accuracy is limited. The Root Mean Squared Error is €118159.49, meaning on average, the predicted prices are off by around €118,159.49.

### Model Files
- **`trained_rf_model.joblib`**: The trained Random Forest model.
- **`locality_encoder.joblib`**: An encoder used to transform locality information into numerical values that the model can understand.

## Future Improvements
- **More Property Features**: Add additional features like parking availability, garden area, energy efficiency ratings, and distance to school or bus stop.
- **Enhanced Model**: Experiment with different models such as XGBoost or CatBoost to improve accuracy.
- **User Feedback Loop**: Add functionality for users to provide feedback on prediction accuracy to further improve the model.

## Acknowledgments
- **Immoweb.be** for the data used to train the model.
- **Streamlit** for making it easy to develop interactive web applications.


