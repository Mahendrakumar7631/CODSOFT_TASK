import os
import joblib
import pandas as pd

def print_feature_importance():
    model_path = os.path.join(os.path.dirname(__file__), '../models/customer_churn_model.pkl')
    pipeline = joblib.load(model_path)
    
    # Get feature names from the column transformer
    preprocessor = pipeline.named_steps['preprocessor']
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    
    all_features = list(num_features) + list(cat_features)
    
    model = pipeline.named_steps['classifier']
    importances = model.feature_importances_
    
    fi = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    fi = fi.sort_values(by='Importance', ascending=False)
    print(fi.to_markdown(index=False))

if __name__ == "__main__":
    print_feature_importance()
