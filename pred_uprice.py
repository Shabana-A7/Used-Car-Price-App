import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Load the trained model and preprocessing objects
model = joblib.load('xgb_model.pkl')
preprocessor = joblib.load('preprocessor2.pkl') 

# Define the app layout
st.title("Used-Car Price Prediction")

# Create input fields for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    name = st.sidebar.text_input('Name', 'Maruti Wagon')
    location = st.sidebar.selectbox('Location', ['Mumbai', 'Pune', 'Chennai', 'Coimbatore', 'Delhi', 'Jaipur', 'Kolkata', 'Hyderabad'])
    year = st.sidebar.number_input('Year', 2000, 2024, 2022)
    kilometers_driven = st.sidebar.number_input('Kilometers Driven', -1.0, 1.0, 0.0)
    fuel_type = st.sidebar.selectbox('Fuel Type', ['CNG', 'Diesel', 'Petrol'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    owner_type = st.sidebar.selectbox('Owner Type', ['First', 'Second'])
    mileage = st.sidebar.number_input('Mileage', -1.0, 1.0, 0.0)
    engine = st.sidebar.number_input('Engine', -1.0, 1.0, 0.0)
    power = st.sidebar.number_input('Power', -1.0, 1.0, 0.0)
    seats = st.sidebar.number_input('Seats', 2, 7, 5)
    type_ = st.sidebar.selectbox('Type', ['Maruti', 'Hyundai', 'Honda', 'Audi', 'Mahindra', 'Chevrolet'])

    data = pd.DataFrame({
        'Name': [name],
        'Location': [location],
        'Year': [year],
        'Kilometers_Driven': [kilometers_driven],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner_Type': [owner_type],
        'Mileage': [mileage],
        'Engine': [engine],
        'Power': [power],
        'Seats': [seats],
        'Type': [type_]
    })
    return data

# Get user input
user_input = user_input_features()

# Preprocess user input
def preprocess_input(data):
    # Apply preprocessing pipeline to user input
    processed_data = preprocessor2.transform(data)
    return processed_data

preprocessed_input = preprocess_input(user_input)

# Make prediction
prediction = model.predict(preprocessed_input)

# Display results
st.subheader('Prediction')
st.write(f'The predicted price of the used car is: {prediction[0]:,.2f}')

# Optionally display the raw data
st.subheader('User Input Data')
st.write(user_input)
