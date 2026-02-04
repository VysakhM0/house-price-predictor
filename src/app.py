from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. INITIALIZE APP
app = FastAPI(title="House Price Prediction API")

# 2. LOAD MODEL
# We load it once when the server starts, not for every request
MODEL_PATH = os.path.join("models", "house_price_model.pkl")
model = joblib.load(MODEL_PATH)

# 3. DEFINE INPUT STRUCTURE
# This ensures users send numbers, not text like "five hundred"
class HouseData(BaseModel):
    square_feet: float
    num_rooms: int
    age: int
    distance_km: float  # We use a clean name here

# 4. DEFINE PREDICTION ENDPOINT
@app.post("/predict")
def predict_price(data: HouseData):
    # Convert incoming JSON to a dictionary
    features = {
        "square_feet": data.square_feet,
        "num_rooms": data.num_rooms,
        "age": data.age,
        # MAP the clean input name to the EXACT column name the model expects
        "distance_to_city(km)": data.distance_km 
    }
    
    # Create a DataFrame (models prefer DataFrames over raw lists)
    df = pd.DataFrame([features])
    
    # Predict
    prediction = model.predict(df)
    
    # Return result
    return {
        "predicted_price": round(prediction[0], 2),
        "currency": "USD"
    }

# 5. HEALTH CHECK (Optional but good practice)
@app.get("/")
def home():
    return {"message": "House Price API is running!"}