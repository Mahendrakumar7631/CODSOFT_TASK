import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    """
    Returns the ColumnTransformer for preprocessing the features.
    """
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )
    return preprocessor

def load_data(filepath):
    """
    Loads data and drops identifier columns.
    """
    df = pd.read_csv(filepath)
    # Drop identifiers
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df
