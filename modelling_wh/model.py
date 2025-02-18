import io
from flask import Flask, request, jsonify
import requests
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

app = Flask(__name__)

def split_data(df: pd.DataFrame):
    """Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Preprocessed Titanic dataset.
    
    Returns:
        X_train (pd.DataFrame), X_test (pd.DataFrame), y_train (pd.Series),
        y_test (pd.Series): Train-test split data.
    """
    X = df.drop('Survived', axis=1)  # Features
    y = df['Survived']  # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains multiple models and evaluates them on the test data.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}
    best_model_name = None
    best_model = None
    best_f1 = 0

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        # Store metrics
        results[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1
        }

        # Update best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model = model

    print(f"\nBest model based on F1 Score: {best_model_name} ({best_f1:.4f})\n")

    return best_model_name, best_model, results

def tune_best_model(model_name, model, X_train, y_train):
    """
    Performs hyperparameter tuning on the best model.
    """
    # Parameter grid
    param_grids = {
        'Decision Tree': {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]},
        'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
        'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 11]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]},
        'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    }

    if model_name not in param_grids:
        print(f"No hyperparameter tuning available for {model_name}. Using default model.")
        return model

    print(f"Tuning {model_name} using GridSearchCV...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='f1'
    )

    grid_search.fit(X_train, y_train)

    print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates the selected model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred)
    }

    print(f"\nFinal Model: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    return metrics

def save_model(model, model_name):
    """Saves the trained model using joblib with its name."""
    model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    model_save_path = "saved_model/" + model_filename

    # Save model
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved to {model_save_path}")
    return model_save_path


@app.route('/model', methods=['POST'])
def model_train():
    # Get files from post request
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    train_data = pd.read_csv(io.StringIO(file1.stream.read().decode('utf-8')))

    # Split data into training and testing
    X_train, X_test, y_train, y_test = split_data(train_data)

    # Train and evaluate models
    best_model_name, best_model, results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Tune best model
    best_model_tuned = tune_best_model(best_model_name, best_model, X_train, y_train)

    # Final evaluation with tuned model
    final_metrics = train_and_evaluate_model(best_model_tuned, X_train, y_train, X_test, y_test)

    # Save the best model
    model_path = save_model(best_model_tuned, best_model_name)

    # Send files to next container
    url = "http://localhost:5002/predict"
    with open(model_path, 'rb') as f1:
        files = {
            'file1': (model_path, f1, 'application/octet-stream'),
            'file2': (file2.filename, file2.stream, 'text/csv') 
        }
        response = requests.post(url, files=files)

    return f"Final Model Metrics: {final_metrics} <br>Best Model: {best_model_name} <br><br>Response from prediction: {response.text}"


if __name__ == '__main__':
    app.run(port=5001)
