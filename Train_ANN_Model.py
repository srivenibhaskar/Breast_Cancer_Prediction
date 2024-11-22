from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from Grid_Search_cv import best_model
from Feature_Selection import X_selected, y

# Train and evaluate ANN
def train_ann_model(X, y, test_size=0.3, random_state=42):
    """
    Function to train and evaluate the ANN model using the best parameters found by Grid Search.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target labels.
        test_size (float): Fraction of data to use for testing (default 30%).
        random_state (int): Random seed for reproducibility.

    Returns:
        best_model: The trained ANN model.
    """
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    print("Training the model with the best parameters from Grid Search...")
    best_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return best_model

# Train and evaluate the model
if __name__ == "__main__":
    print("Starting training and evaluation...")
    ann_model = train_ann_model(X_selected, y)

import joblib

# Save the trained model
joblib.dump(ann_model, "best_ann_model.pkl")
print("Model saved as best_ann_model.pkl")

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Display the confusion matrix
ConfusionMatrixDisplay.from_estimator(ann_model, X_selected, y, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()