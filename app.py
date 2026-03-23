import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

model = load_model("model/heart_failure_model.keras")

def predict(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)

    if prediction > 0.5:
        return "High Risk"
    return "Low Risk"

print("Sample Prediction:", predict([60,1,200,1,20,1,250000,1.9,130,1,1,4]))
