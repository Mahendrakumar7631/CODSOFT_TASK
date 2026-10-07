import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
try:
    from src.preprocessing import load_data, preprocess_dataset
except ImportError:
    from preprocessing import load_data, preprocess_dataset


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }
    return metrics, y_pred


def train_and_evaluate():
    # ---- Load and preprocess ----
    data_path = os.path.join(os.path.dirname(__file__), '../data/spam.csv')
    df = load_data(data_path)
    df, n_duplicates = preprocess_dataset(df)

    print(f"Dataset loaded: {len(df)} rows (after removing {n_duplicates} duplicates)")
    print(f"Class distribution: ham={len(df[df['label']=='ham'])}, spam={len(df[df['label']=='spam'])}")
    print()

    X = df['message_clean']
    y = df['label_encoded']

    # ---- Stratified train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()

    # ---- Define pipelines ----
    # TF-IDF parameters: reasonable defaults for SMS-sized text
    tfidf_params = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'stop_words': 'english',
        'min_df': 2
    }

    pipelines = {
        "Multinomial Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=42, dual='auto'), cv=3
            ))
        ]),
    }

    # ---- Train and evaluate baseline models ----
    results = {}
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None

    print("=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1 Score:  {metrics['F1 Score']:.4f}")
        print(f"  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"    {cm}")

        if metrics["F1 Score"] > best_f1:
            best_f1 = metrics["F1 Score"]
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n{'=' * 60}")
    print(f"Best baseline model: {best_model_name} (F1: {best_f1:.4f})")
    print(f"{'=' * 60}")

    # ---- Hyperparameter tuning on best model ----
    print(f"\nTuning {best_model_name}...")

    if best_model_name == "Multinomial Naive Bayes":
        param_grid = {
            'classifier__alpha': [0.1, 0.5, 1.0, 2.0],
            'tfidf__max_features': [3000, 5000, 7000],
        }
        tuning_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', MultinomialNB())
        ])
    elif best_model_name == "Logistic Regression":
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'tfidf__max_features': [3000, 5000, 7000],
        }
        tuning_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
    else:  # Linear SVM
        param_grid = {
            'classifier__estimator__C': [0.1, 1.0, 10.0],
            'tfidf__max_features': [3000, 5000, 7000],
        }
        tuning_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=42, dual='auto'), cv=3
            ))
        ])

    search = GridSearchCV(
        tuning_pipeline, param_grid,
        cv=3, scoring='f1', n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    tuned_metrics, tuned_pred = evaluate_model(tuned_model, X_test, y_test)

    print(f"\nTuned {best_model_name}:")
    print(f"  Best params: {search.best_params_}")
    print(f"  Accuracy:  {tuned_metrics['Accuracy']:.4f}")
    print(f"  Precision: {tuned_metrics['Precision']:.4f}")
    print(f"  Recall:    {tuned_metrics['Recall']:.4f}")
    print(f"  F1 Score:  {tuned_metrics['F1 Score']:.4f}")

    # ---- Select final model ----
    if tuned_metrics["F1 Score"] >= best_f1:
        print(f"\nTuned model is better or equal. Saving tuned {best_model_name}...")
        final_model = tuned_model
        final_metrics = tuned_metrics
    else:
        print(f"\nBaseline {best_model_name} is better. Saving baseline model...")
        final_model = best_pipeline
        final_metrics = results[best_model_name]

    # ---- Final Classification Report ----
    y_final_pred = final_model.predict(X_test)
    print(f"\n{'=' * 60}")
    print("FINAL MODEL — CLASSIFICATION REPORT")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_final_pred, target_names=['Ham', 'Spam']))

    # ---- Save model ----
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '../models/spam_sms_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Final model saved to: {model_path}")

    return final_model, final_metrics


if __name__ == "__main__":
    train_and_evaluate()
