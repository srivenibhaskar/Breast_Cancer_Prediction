from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from Dataset_Preparation import preprocess_data

# Preprocess data
X, y = preprocess_data("breast_cancer_dataset.csv")

# Feature selection using SelectKBest
def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

# Apply feature selection
X_selected, selected_features = select_features(X, y, k=10)
print(f"Selected Features: {list(selected_features)}")