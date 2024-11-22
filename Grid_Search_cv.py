from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from Feature_Selection import X_selected, y

# Define the ANN model and hyperparameter grid
def tune_ann_model(X, y):
    # Create the base model
    model = MLPClassifier(max_iter=500, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive']
    }

    # Perform Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_

# Run Grid Search
best_model, best_params = tune_ann_model(X_selected, y)
print(f"Best Parameters: {best_params}")