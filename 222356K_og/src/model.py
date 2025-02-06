import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains multiple models and evaluates them on the test data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target.

    Returns:
        dict: A dictionary with model names as keys and their evaluation metrics as values.
    """
    # Initialize models, including Gradient Boosting
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    # Loop through the models, train and evaluate them
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }

        # Store the results
        results[model_name] = metrics

    return results


def tune_random_forest(X_train, y_train):
    """
    Tunes the hyperparameters of a Random Forest Classifier using GridSearchCV.

    Args:
        X_train (pd.DataFrame): The feature data for training the model.
        y_train (pd.Series): The target variable data for training the model.

    Returns:
        best_model (RandomForestClassifier): The RandomForest model with the best hyperparameters.
        best_params (dict): The best hyperparameters found.
    """
    # Define the parameter grid for Random Forest tuning
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
        'max_features': ['sqrt', 'log2'],  # Number of features to consider for best split
        'bootstrap': [True, False]  # Whether to use bootstrap samples
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Use GridSearchCV to search the parameter grid
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='f1'
    )

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and best hyperparameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Display results
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Model: {best_model}")

    return best_model, best_params


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Trains a model and evaluates it using the test dataset.

    Args:
        model: The machine learning model to be trained.
        X_train (pd.DataFrame): The feature data for training the model.
        y_train (pd.Series): The target variable data for training the model.
        X_test (pd.DataFrame): The feature data for testing the model.
        y_test (pd.Series): The target variable data for testing the model.

    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, precision, recall, F1 score).
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the classification report
    print(f"{model.__class__.__name__} Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print individual metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Return metrics as a dictionary
    return {
        'classification_report': classification_report(y_test, y_pred),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_model(model, filename: str):
    """Saves the trained model to a specified file path using joblib."""
    # Define the directory path where the model should be saved
    model_path = os.path.join('saved_model', filename)

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained model to the specified path using joblib
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def make_predictions_and_save_to_csv(test_data: pd.DataFrame, model_path: str, output_filename: str):
    """
    Loads a saved model, makes predictions on the test data, and saves the results to a CSV file.

    Args:
        test_data (pd.DataFrame): The feature data for testing the model.
        model_path (str): Path to the saved model file.
        output_filename (str): Name of the output CSV file to save predictions.

    Returns:
        None
    """
    # Load the saved model
    model = joblib.load(model_path)

    # Make predictions on the test data
    predictions = model.predict(test_data)

    # Add the predictions to the test data (you can use index or passenger_id if needed)
    test_data['Predicted_Survived'] = predictions

    # Define the path to save the output CSV file
    output_path = os.path.join('data', '03_predicted', output_filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the test data with predictions to the CSV file
    test_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
