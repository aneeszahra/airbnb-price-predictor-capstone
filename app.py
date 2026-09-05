import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# HARDCODE sklearn import - will work if streamlit cloud pre-installs it
try:
    import sklearn
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn', '--quiet'])
    import sklearn

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="")
st.title("Airbnb NYC Price Predictor")
st.markdown("Predict nightly price for your Airbnb listing")

# Check for model file
if not os.path.exists('model.pkl'):
    st.warning("Model file not found. Using fallback pricing estimate.")
    
    # FALLBACK: Simple price estimator (NO MODEL NEEDED)
    neighbourhood = st.selectbox("Neighbourhood", ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
    room_type = st.selectbox("Room Type", ['Entire home/apt', 'Private room', 'Shared room'])
    minimum_nights = st.slider("Minimum Nights", 1, 30, 3)
    
    if st.button("Estimate Price"):
        # Simple logic (no sklearn)
        base_price = 150
        if neighbourhood == 'Manhattan':
            base_price += 100
        elif neighbourhood == 'Brooklyn':
            base_price += 30
        elif neighbourhood == 'Queens':
            base_price -= 20
            
        if room_type == 'Entire home/apt':
            base_price += 80
        elif room_type == 'Private room':
            base_price -= 30
            
        base_price += minimum_nights * 2
        
        st.success(f"Estimated Price: ${base_price:.2f} per night")
        st.info("This is a fallback estimate (model file not loaded)")
    
    st.stop()

# Load model if exists
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Model loaded")
except:
    st.error("Error loading model. Please check model files.")
    st.stop()

# Rest of your app...
neighbourhood = st.selectbox("Neighbourhood", ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])
room_type = st.selectbox("Room Type", ['Entire home/apt', 'Private room', 'Shared room'])
minimum_nights = st.slider("Minimum Nights", 1, 30, 3)

if st.button("Predict Price"):
    # Feature vector
    data = {
        'latitude': 40.7,
        'longitude': -74.0,
        'minimum_nights': minimum_nights,
        'number_of_reviews': 50,
        'reviews_per_month': 1.0,
        'calculated_host_listings_count': 5,
        'availability_365': 200,
        'days_since_review': 30,
        'popularity_score': 0.25,
        'is_superhost': 0,
        'neighbourhood_group_Brooklyn': 1 if neighbourhood == 'Brooklyn' else 0,
        'neighbourhood_group_Manhattan': 1 if neighbourhood == 'Manhattan' else 0,
        'neighbourhood_group_Queens': 1 if neighbourhood == 'Queens' else 0,
        'neighbourhood_group_Staten Island': 1 if neighbourhood == 'Staten Island' else 0,
        'room_type_Private room': 1 if room_type == 'Private room' else 0,
        'room_type_Shared room': 1 if room_type == 'Shared room' else 0
    }
    
    input_df = pd.DataFrame([data])
    
    # Scale
    numeric_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                    'reviews_per_month', 'calculated_host_listings_count', 
                    'availability_365', 'days_since_review', 'popularity_score']
    
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${prediction:.2f} per night")