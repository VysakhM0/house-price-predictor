import streamlit as st
import requests

# 1. TITLE AND DESCRIPTION
st.title("🏡 House Price Predictor")
st.write("Enter the details of the house to estimate its market value.")

# 2. CREATE INPUT FORM
# We use columns to make it look professional
col1, col2 = st.columns(2)

with col1:
    square_feet = st.number_input("Square Feet", min_value=500, max_value=10000, value=2500)
    num_rooms = st.slider("Number of Rooms", 1, 10, 4)

with col2:
    age = st.number_input("Age of House (Years)", min_value=0, max_value=200, value=10)
    distance_km = st.number_input("Distance to City (km)", min_value=0.0, value=15.0)

# 3. PREDICT BUTTON
if st.button("Estimate Price"):
    # Prepare the data dictionary (matches the API expectation)
    payload = {
        "square_feet": square_feet,
        "num_rooms": num_rooms,
        "age": age,
        "distance_km": distance_km
    }
    
    # 4. SEND REQUEST TO API
    # We are talking to the FastAPI server running on localhost:8000
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            price = result['predicted_price']
            
            # Display the result in big bold text
            st.success(f"Estimated Price: ${price:,.2f}")
        else:
            st.error("Error: Could not get prediction from server.")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Connection Error: Is the API server running?")
        st.info("Make sure you have 'uvicorn src.app:app --reload' running in a separate terminal!")