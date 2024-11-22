import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save the dataset as CSV
df.to_csv("breast_cancer_dataset.csv", index=False)
print("Dataset saved as breast_cancer_dataset.csv")

# Data Preprocessing
def preprocess_data(filepath, scale=True, balance_data=True):
    df = pd.read_csv(filepath)

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Impute missing values with column means
        df.fillna(df.mean(), inplace=True)
        print("Missing values imputed.")

    # Check for duplicate rows and remove them
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print(f"Removed {df.duplicated().sum()} duplicate rows.")

    # Feature and target separation
    X = df.drop(columns=["target"])
    y = df["target"]

    # Scale features (standardization)
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        print("Features have been standardized.")

    # Handle class imbalance (if any)
    if balance_data:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("Class imbalance handled using SMOTE.")

    return X, y

# Load and preprocess the data
X, y = preprocess_data("breast_cancer_dataset.csv")
print(f"Data Preprocessed. Final dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")