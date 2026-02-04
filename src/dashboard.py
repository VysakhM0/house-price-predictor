import streamlit as st
import pandas as pd
import joblib
import os

# 1. LOAD MODEL DIRECTLY
# We use @st.cache_resource so it only loads once and stays in memory (faster)
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "house_price_model.pkl")
    return joblib.load(model_path)

model = load_model()

# 2. TITLE
st.title("🏡 House Price Predictor")
st.write("Enter the details of the house to estimate its market value.")

# 3. INPUT FORM
col1, col2 = st.columns(2)
with col1:
    square_feet = st.number_input("Square Feet", 500, 10000, 2500)
    num_rooms = st.slider("Number of Rooms", 1, 10, 4)
with col2:
    age = st.number_input("Age of House (Years)", 0, 200, 10)
    distance_km = st.number_input("Distance to City (km)", 0.0, 15.0)

# 4. PREDICT BUTTON
if st.button("Estimate Price"):
    # Create a DataFrame directly (Same format as training!)
    input_data = pd.DataFrame([{
        "square_feet": square_feet,
        "num_rooms": num_rooms,
        "age": age,
        "distance_to_city(km)": distance_km  # Note: Use the exact column name from training
    }])
    
    # Predict directly using the loaded model
    prediction = model.predict(input_data)[0]
    
    st.success(f"Estimated Price: ${prediction:,.2f}")