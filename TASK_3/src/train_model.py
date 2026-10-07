import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from preprocessing import get_preprocessor, load_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
    else:
        metrics["ROC-AUC"] = None
    return metrics

def train_and_evaluate():
    data_path = os.path.join(os.path.dirname(__file__), '../data/Churn_Modelling.csv')
    df = load_data(data_path)
    
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = get_preprocessor()
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None
    
    print("Training baseline models...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        print(f"{name}: {metrics}")
        
        if metrics["F1 Score"] > best_f1:
            best_f1 = metrics["F1 Score"]
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest baseline model: {best_model_name} with F1 Score: {best_f1:.4f}")
    
    # Tuning Random Forest (often provides good balance without being as slow to tune as GB)
    print("\nTuning Random Forest Classifier...")
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(class_weight="balanced", random_state=42))])
    param_distributions = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    search = RandomizedSearchCV(rf_pipeline, param_distributions, n_iter=5, cv=3, scoring='f1', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    tuned_model = search.best_estimator_
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)
    print(f"Tuned Random Forest F1: {tuned_metrics['F1 Score']:.4f}")
    
    # Compare with best baseline
    if tuned_metrics["F1 Score"] > best_f1:
        print("Tuned model is better. Saving Final Selected Model...")
        final_model = tuned_model
    else:
        print(f"Baseline {best_model_name} is better. Saving Final Selected Model...")
        final_model = best_pipeline
        
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '../models/customer_churn_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
