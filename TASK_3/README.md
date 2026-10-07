# Customer Churn Prediction

## Overview
This is a Machine Learning project developed for the CodSoft Machine Learning Internship (Task 3). The objective is to build a predictive model that identifies whether a customer is likely to churn from a subscription-based service or business. 

## Problem Statement
Customer churn (when a customer stops using a company's product or service) is a major problem for subscription-based businesses. Identifying customers who are likely to churn before they leave allows businesses to proactively engage and offer retention incentives.

## Objective
Develop a traditional machine learning system to accurately predict customer churn based on historical data, demographics, and usage behavior.

## Dataset
The dataset `Churn_Modelling.csv` contains 10,000 records of customer information. 
- **Predictive Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
- **Target Variable:** `Exited` (1 = Churned, 0 = Retained)

## Features
- All relevant features from the dataset were retained after exploratory data analysis (EDA).
- Identifier columns (`RowNumber`, `CustomerId`, `Surname`) were removed as they provide no predictive power.

## Technologies Used
- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Application:** Streamlit

## Machine Learning Workflow
1. **Data Inspection & Cleaning:** Checked for duplicates, missing values, and dropped non-predictive identifiers.
2. **Exploratory Data Analysis (EDA):** Visualized distribution of features and correlation with churn.
3. **Preprocessing:** Split data (80/20 stratify), utilized `StandardScaler` for numerical features and `OneHotEncoder` for categorical features within a `ColumnTransformer`.
4. **Model Training:** Trained Logistic Regression, Random Forest, and Gradient Boosting Classifiers. Handled class imbalance using the `class_weight='balanced'` parameter where appropriate.
5. **Model Evaluation:** Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
6. **Hyperparameter Tuning:** Tuned the Random Forest Classifier using `RandomizedSearchCV` based on the F1-Score.

## Data Preprocessing
- **Numerical Features:** Numerical features were standardized using StandardScaler to approximately zero mean and unit variance.
- **Categorical Features:** One-hot encoded to convert text categories into numerical format.

## EDA
- Churners tend to be older customers compared to retained ones.
- Geography (e.g., Germany) shows differences in churn rates.
- Overall churn rate is relatively imbalanced (around ~20% churn vs ~80% retained).

## Models Used
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

## Model Evaluation
Results on test data:
- **Logistic Regression:** Accuracy: 0.71, F1-Score: 0.50, ROC-AUC: 0.78
- **Random Forest:** Accuracy: 0.86, F1-Score: 0.56, ROC-AUC: 0.85
- **Gradient Boosting:** Accuracy: 0.87, F1-Score: 0.60, ROC-AUC: 0.87

## Final Model
The Tuned Random Forest Classifier was selected as the **Final Selected Model** as it provides robust performance, balances precision and recall well, and adequately handles class imbalance.

## Streamlit Application
A professional, simple, and clean Streamlit UI was created to allow users to input customer attributes and get an instant churn prediction and probability score directly from the trained model.

## Project Structure
```text
TASK_3/
│
├── data/
│   └── Churn_Modelling.csv
│
├── notebooks/
│   └── model_development.ipynb
│
├── models/
│   └── customer_churn_model.pkl
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
cd CODSOFT_TASKS/TASK_3

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
python -m pip install -r requirements.txt
```
## How to Run
To run the Streamlit application, execute the following command:
```bash
streamlit run app.py
```

## Results
The trained model can predict churn effectively, achieving a solid ROC-AUC score and highlighting features like Age, Number of Products, and Account Balance as key churn indicators.

## Future Improvements
- Test more complex models like XGBoost or LightGBM.
- Deploy the Streamlit app on a cloud platform (e.g., Streamlit Community Cloud or Heroku).
- Gather more historical interaction data for feature engineering.
