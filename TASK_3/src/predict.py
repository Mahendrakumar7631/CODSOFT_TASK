import os
import joblib
import pandas as pd

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/customer_churn_model.pkl')
    return joblib.load(model_path)

def predict_churn(input_data):
    """
    input_data: dictionary containing features
    Returns: (prediction, probability)
    """
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability
