import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
import pandas as pd

#Terminal command to view browser
# streamlit run streamlit_app.py

# # Load the trained RFR model
rf_model : RandomForestRegressor = joblib.load('models/trained_rf_model.joblib')

st.set_page_config(
    page_title="Belgian Real Estate Price Predictor",
    page_icon="🏡",
    initial_sidebar_state="expanded",
    layout="centered",
    menu_items={
        "Get Help": "https://github.com/kgriff21",
    },
)

def main():
    st.title("Belgian Real Estate Price Predictor")

    st.markdown("Welcome to the Belgian Real Estate Price Predictor! This web application is designed to predict real estate prices based on certain input criteria. "
                "The modeled data has been collected from immoweb.be, a widely used real estate platform in Belgium. This will deploy a Random Forest Regression Model"
                "to predict the real estate price from user parameters.")

    st.header("📈 Make a prediction")
    property_type_options = {
        "Apartment": 0,
        "House": 1
    }
    # Display the selectbox and get the selected option
    selected_property_type_option = st.selectbox("Select Property Type", list(property_type_options.keys()))
    property_type = property_type_options[selected_property_type_option]

    number_of_bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=16, value=2)

    living_area = st.slider("Living area m²", min_value=1, max_value=1500, value=200)

    # Create mappings for options and their corresponding values
    kitchen_options = {"Yes": 1, "No": 0}
    furnished_options = {"Yes": 1, "No": 0}
    pool_options = {"Yes": 1, "No": 0}

    # Display the selectboxes and get the selected options
    selected_kitchen_option = st.selectbox("Equipped Kitchen?", list(kitchen_options.keys()))
    equipped_kitchen = kitchen_options[selected_kitchen_option]

    selected_furnished_option = st.selectbox("Furnished?", list(furnished_options.keys()))
    furnished = furnished_options[selected_furnished_option]

    selected_pool_option = st.selectbox("Swimming Pool?", list(pool_options.keys()))
    swimming_pool = pool_options[selected_pool_option]

    # Create a mapping for building condition options
    condition_options = {
        'To restore': 0,
        'To renovate': 1,
        'To be done up': 2,
        'Good': 3,
        'Just renovated': 4,
        'As new': 5
    }

    # Display the selectbox and get the selected option
    selected_condition_option = st.selectbox("What is the condition of the building?", list(condition_options.keys()))
    building_condition = condition_options[selected_condition_option]

    province = st.selectbox(
        "Which province is the residence located?",
        ['Brussels', 'East Flanders', 'Flemish Brabant', 'Hainaut', 'Limburg',
        'Liège', 'Luxembourg', 'Namur', 'Walloon Brabant', 'West Flanders']
    )

    if st.button("Predict", key="start-button"):
        data = [
            number_of_bedrooms,
            living_area,
            equipped_kitchen,
            furnished,
            swimming_pool,
            building_condition,
            property_type
        ]

        columns = ['Number of bedrooms', 'Living area m²', 'Equipped kitchen',
       'Furnished', 'Swimming pool', 'Encoded Building Condition', 'house',
       'Brussels', 'East Flanders', 'Flemish Brabant', 'Hainaut', 'Limburg',
       'Liège', 'Luxembourg', 'Namur', 'Walloon Brabant', 'West Flanders']

        province_columns = ['Brussels', 'East Flanders', 'Flemish Brabant', 'Hainaut', 'Limburg',
                   'Liège', 'Luxembourg', 'Namur', 'Walloon Brabant', 'West Flanders']

        province_array = np.zeros(10)
        province_index = province_columns.index(province)
        province_array[province_index] = 1

        np_data = np.array(data)
        predict_input = np.concatenate((np_data, province_array))
        # st.info(predict_input)

        predict_row = pd.DataFrame(
            [np.array(predict_input)],
            columns=columns
        )

        test=int(rf_model.predict(predict_row).round(2)[0])
        st.info(f"Predicted price: €{test}")

# This prevents the main logic in this script from executing when it's imported as a module in another script. This is
# important because the main logic(like print statements, data processing, other code outside this function) will not execute automatically.
if __name__ == "__main__":
    main()