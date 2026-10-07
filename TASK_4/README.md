# SMS Spam Detection — CodSoft ML Internship Task 4

## Overview

This project builds a traditional Machine Learning classification system that predicts whether an SMS message is **Spam** or **Ham** (not spam). It uses TF-IDF text vectorization combined with traditional ML classifiers, and includes a Streamlit web application for interactive predictions.

## Problem Statement

SMS spam is a widespread issue where unsolicited messages are sent to users, often containing fraudulent offers, phishing links, or promotional scams. Automatically detecting and filtering spam messages helps protect users and improve their messaging experience.

## Objective

Build an ML pipeline that:
1. Preprocesses SMS text data
2. Extracts features using TF-IDF
3. Trains and compares multiple classifiers
4. Deploys the best model via a Streamlit web application

## Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** Kaggle / UCI Machine Learning Repository
- **File:** `data/spam.csv`
- **Encoding:** latin-1

### Dataset Statistics

| Metric | Value |
|---|---|
| Original rows | 5,572 |
| Columns (original) | 5 (`v1`, `v2`, `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) |
| Useful columns | 2 (`v1` → label, `v2` → message) |
| Duplicate rows removed | 403 |
| Final dataset size | 5,169 |
| Ham messages | 4,516 (87.4%) |
| Spam messages | 653 (12.6%) |
| Missing values in label/message | 0 |

### Features / Input

| Feature | Description |
|---|---|
| `message` (v2) | The raw SMS text message |
| `label` (v1) | Target class: `ham` (not spam) or `spam` |

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Workflow

1. Load and inspect dataset
2. Drop unnecessary columns (`Unnamed: 2/3/4`)
3. Rename columns (`v1` → `label`, `v2` → `message`)
4. Remove 403 duplicate rows
5. Lightweight text preprocessing (lowercase, whitespace normalization)
6. Stratified train-test split (80/20)
7. TF-IDF feature extraction (fitted on training data only)
8. Train 3 baseline models
9. Compare models using multiple metrics
10. Hyperparameter tuning on best model
11. Save complete pipeline (TF-IDF + Classifier)
12. Deploy via Streamlit

## Data Preprocessing

- **Dropped columns:** `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4` (mostly empty — 50, 12, 6 non-null values respectively)
- **Duplicates:** 403 duplicate rows removed (based on label + message)
- **Text cleaning:** Lowercase, remove non-printable characters, normalize whitespace
- **Label encoding:** ham → 0, spam → 1

## TF-IDF Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 3,000 | Reasonable vocabulary for SMS-length text |
| `ngram_range` | (1, 2) | Captures unigrams and bigrams (e.g., "free call") |
| `stop_words` | english | Removes common English stop words |
| `min_df` | 2 | Ignores terms appearing in fewer than 2 documents |

TF-IDF is fitted **only on training data** to prevent data leakage.

## Models Used

1. **Multinomial Naive Bayes** — Fast, effective for text classification
2. **Logistic Regression** — Linear classifier with probability support
3. **Linear SVM** (with CalibratedClassifierCV) — Strong text classifier with calibrated probabilities

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.9681 | 0.9900 | 0.7557 | 0.8571 |
| Logistic Regression | 0.9555 | 0.9775 | 0.6641 | 0.7909 |
| **Linear SVM** | **0.9758** | **0.9344** | **0.8702** | **0.9012** |

## Evaluation Metrics

The final model was selected based on **F1-Score**, which balances Precision and Recall — critical for imbalanced spam detection.

### Final Model: Linear SVM (Tuned)

| Metric | Value |
|---|---|
| Accuracy | 0.9758 |
| Precision | 0.9344 |
| Recall | 0.8702 |
| F1 Score | 0.9012 |

### Final Classification Report

```
              precision    recall  f1-score   support

         Ham       0.98      0.99      0.99       903
        Spam       0.93      0.87      0.90       131

    accuracy                           0.98      1034
   macro avg       0.96      0.93      0.94      1034
weighted avg       0.98      0.98      0.98      1034
```

### Hyperparameter Tuning

- **Method:** GridSearchCV (cv=3, scoring='f1')
- **Best parameters:** `C=1.0`, `max_features=3000`
- The tuned model matched the baseline Linear SVM performance.

## Streamlit Application

The web application provides:
- **Left panel:** Project information, ML approach, dataset stats
- **Right panel:** SMS input area, prediction button, result display
- **Predictions:** Shows Spam/Ham classification with genuine probability scores

## Project Structure

```
TASK_4/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── model_development.ipynb
│
├── models/
│   └── spam_sms_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Mahendrakumar7631/CODSOFT_TASKS.git
cd CODSOFT_TASKS/TASK_4

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
python -m pip install -r requirements.txt
```

## How to Run

### Train the Model
```bash
cd src
python train_model.py
```

### Run the Streamlit Application
```bash
streamlit run app.py
```

### Run Predictions from Terminal
```bash
cd src
python predict.py
```

## Results

- **Linear SVM** achieved the best F1-Score of **0.9012** among the three classifiers
- The model correctly classifies **97.6%** of all messages
- **93.4%** precision ensures very few ham messages are incorrectly flagged as spam
- **87.0%** recall means the model catches most spam messages
- Probability scores are genuinely produced by the calibrated SVM model

## Future Improvements

- Experiment with character-level n-grams for catching obfuscated spam
- Try ensemble methods combining multiple classifiers
- Add more text features (message length, special character count)
- Collect more diverse spam samples to improve recall
- Implement cross-validation for more robust evaluation
- Add support for multilingual SMS messages
