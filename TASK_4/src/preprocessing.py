import pandas as pd
import re


def load_data(filepath):
    """
    Loads the SMS Spam Collection dataset.
    - Uses latin-1 encoding (required for this dataset).
    - Drops unnecessary unnamed columns.
    - Renames v1 -> label, v2 -> message.
    """
    df = pd.read_csv(filepath, encoding='latin-1')

    # Drop unnamed/empty columns (Unnamed: 2, 3, 4)
    cols_to_drop = [col for col in df.columns if col.startswith('Unnamed')]
    df = df.drop(columns=cols_to_drop)

    # Rename columns to meaningful names
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})

    return df


def clean_text(text):
    """
    Lightweight text cleaning for SMS messages.
    - Converts to string
    - Lowercases
    - Removes extra whitespace
    - Removes non-printable characters
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Remove non-printable / control characters but keep basic punctuation
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset(df):
    """
    Full preprocessing pipeline for the SMS dataset.
    - Handles missing message values
    - Removes duplicate rows
    - Applies text cleaning
    - Encodes labels (ham=0, spam=1)
    Returns cleaned DataFrame and number of duplicates removed.
    """
    # Handle missing messages
    df['message'] = df['message'].fillna('')

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=['label', 'message'], keep='first').reset_index(drop=True)
    n_duplicates_removed = n_before - len(df)

    # Clean text
    df['message_clean'] = df['message'].apply(clean_text)

    # Encode labels: ham=0, spam=1
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

    return df, n_duplicates_removed
