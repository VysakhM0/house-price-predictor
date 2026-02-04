import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# 1. CONSTANTS
DATA_PATH = os.path.join("data", "housing_cleaned.csv")
MODEL_PATH = os.path.join("models", "house_price_model.pkl")

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # 2. SEPARATE FEATURES (X) AND TARGET (y)
    # The target is what we want to predict: 'price'
    X = df.drop(columns=["price"])
    y = df["price"]

    # 3. SPLIT DATA
    # 80% for training, 20% for testing
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. TRAIN MODEL
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. EVALUATE
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print(f"Model Performance:")
    print(f"MAE (Average Error): ${mae:,.2f}")
    print(f"RMSE (Root Mean Sq Error): ${rmse:,.2f}")

    # 6. SAVE MODEL
    # We save the model file so the API can load it later
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()