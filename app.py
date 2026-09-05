import subprocess
import sys

# Force install dependencies (this runs ONCE when the app starts)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn', 'streamlit', 'numpy', 'pandas', 'joblib', '--quiet'])

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sklearn

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="")

st.title("Airbnb NYC Price Predictor")
st.markdown("**Predict the optimal nightly price for your Airbnb listing**")

# Check if model files exist
if not os.path.exists('model.pkl'):
    st.error("Model file model.pkl not found.")
    st.stop()

if not os.path.exists('scaler.pkl'):
    st.error("Scaler file scaler.pkl not found.")
    st.stop()

# Load model and scaler with pickle
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# -------------------------------------------------------------------
# Feature columns (must match training order)
feature_columns = [
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
    'days_since_review', 'popularity_score', 'is_superhost',
    'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
    'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island',
    'room_type_Private room', 'room_type_Shared room'
]

numeric_cols = [
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count',
    'availability_365', 'days_since_review', 'popularity_score'
]

# -------------------------------------------------------------------
# UI Inputs
st.subheader("Enter Your Listing Details")

col1, col2 = st.columns(2)

with col1:
    neighbourhood = st.selectbox(
        "Neighborhood Group",
        ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    )
    room_type = st.selectbox(
        "Room Type",
        ['Entire home/apt', 'Private room', 'Shared room']
    )
    minimum_nights = st.slider(
        "Minimum Nights",
        min_value=1, max_value=30, value=3
    )
    availability = st.slider(
        "Availability (days per year)",
        min_value=0, max_value=365, value=200
    )

with col2:
    reviews = st.number_input(
        "Number of Reviews",
        min_value=0, max_value=500, value=50
    )
    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1
    )
    listings_count = st.number_input(
        "Host Listings Count",
        min_value=1, max_value=100, value=5
    )
    latitude = st.slider(
        "Latitude",
        min_value=40.5, max_value=40.9, value=40.7, step=0.01
    )
    longitude = st.slider(
        "Longitude",
        min_value=-74.3, max_value=-73.7, value=-74.0, step=0.01
    )

# -------------------------------------------------------------------
# Predict Button
if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        # Create feature vector
        input_data = {
            'latitude': latitude,
            'longitude': longitude,
            'minimum_nights': minimum_nights,
            'number_of_reviews': reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': listings_count,
            'availability_365': availability,
            'days_since_review': 30,
            'popularity_score': reviews / (availability + 1),
            'is_superhost': 1 if reviews > 50 else 0,
            'neighbourhood_group_Brooklyn': 1 if neighbourhood == 'Brooklyn' else 0,
            'neighbourhood_group_Manhattan': 1 if neighbourhood == 'Manhattan' else 0,
            'neighbourhood_group_Queens': 1 if neighbourhood == 'Queens' else 0,
            'neighbourhood_group_Staten Island': 1 if neighbourhood == 'Staten Island' else 0,
            'room_type_Private room': 1 if room_type == 'Private room' else 0,
            'room_type_Shared room': 1 if room_type == 'Shared room' else 0
        }

        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])

        # Ensure all columns exist
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        # Scale numerical features
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        prediction = model.predict(input_df)[0]

        # Display result
        st.markdown("---")
        st.subheader("Predicted Nightly Price")
        st.markdown(f"## ${prediction:.2f} per night")
        st.caption(f"Location: {neighbourhood} | Room Type: {room_type} | Reviews: {reviews}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check all input values and try again.")

st.markdown("---")
st.caption("Built with Streamlit | Model: Gradient Boosting | R-Squared: 0.68")