import os
import joblib

try:
    from src.preprocessing import clean_text
except ImportError:
    from preprocessing import clean_text



def load_model():
    """Load the saved spam SMS detection pipeline."""
    model_path = os.path.join(os.path.dirname(__file__), '../models/spam_sms_model.pkl')
    return joblib.load(model_path)


def predict_spam(message):
    """
    Predict whether an SMS message is spam or ham.

    Args:
        message: SMS text string

    Returns:
        tuple: (prediction_label, probability)
            prediction_label: 'spam' or 'ham'
            probability: spam probability (float) if supported, else None
    """
    model = load_model()

    # Apply the same text cleaning used during training
    cleaned = clean_text(message)

    # Predict
    prediction = model.predict([cleaned])[0]
    label = 'spam' if prediction == 1 else 'ham'

    # Get probability if supported
    probability = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba([cleaned])[0]
            probability = proba[1]  # probability of spam
        except Exception:
            probability = None

    return label, probability


if __name__ == "__main__":
    # Quick test
    test_messages = [
        "Hey, are you coming to the party tonight?",
        "WINNER! You have been selected for a free prize. Call now!",
        "Reminder: Your appointment is at 3pm tomorrow.",
        "FREE entry to win £1000 cash! Text WIN to 80085 now!",
    ]

    for msg in test_messages:
        label, prob = predict_spam(msg)
        prob_str = f" (spam probability: {prob:.4f})" if prob is not None else ""
        print(f"[{label.upper()}]{prob_str} — {msg[:60]}...")
