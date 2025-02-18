import os
import io
import logging
import jsonify
from flask import Flask, request
import requests
import joblib
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
    model_save_path = os.path.join('saved_model', model_filename)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


@app.route('/model', methods=['POST'])
def model_train():
    try:
        # Get files from post request
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1 or not file2:
            logging.error("No files provided in the request.")
            return jsonify({"error": "No files provided"}), 400

        logging.info("Loading processed datasets from uploaded files...")
        try:
            train_data = pd.read_csv(file1)
            predict_data = pd.read_csv(file2)
            logging.info("Datasets loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading datasets: {str(e)}")
            return jsonify({"error": f"Error loading datasets: {str(e)}"}), 400

        # Split data into training and testing
        logging.info("Splitting data into training and testing sets...")
        try:
            X_train, X_test, y_train, y_test = split_data(train_data)
            logging.info("Data split completed successfully.")
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            return jsonify({"error": f"Error splitting data: {str(e)}"}), 500

        # Train and evaluate models
        logging.info("Training and evaluating models...")
        try:
            best_model_name, best_model, results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
            logging.info(f"Best model based on F1 Score: {best_model_name}")
        except Exception as e:
            logging.error(f"Error during model training and evaluation: {str(e)}")
            return jsonify({"error": f"Error during model training and evaluation: {str(e)}"}), 500

        # Tune best model
        logging.info(f"Tuning best model: {best_model_name}...")
        try:
            best_model_tuned = tune_best_model(best_model_name, best_model, X_train, y_train)
            logging.info(f"Best model tuned successfully.")
        except Exception as e:
            logging.error(f"Error tuning best model: {str(e)}")
            return jsonify({"error": f"Error tuning best model: {str(e)}"}), 500

        # Final evaluation with tuned model
        logging.info("Performing final evaluation with the tuned model...")
        try:
            final_metrics = train_and_evaluate_model(best_model_tuned, X_train, y_train, X_test, y_test)
            logging.info(f"Final model metrics: {final_metrics}")
        except Exception as e:
            logging.error(f"Error during final model evaluation: {str(e)}")
            return jsonify({"error": f"Error during final model evaluation: {str(e)}"}), 500

        # Save the best model
        logging.info("Saving the best model...")
        try:
            save_model(best_model_tuned, best_model_name)
            logging.info(f"Best model saved successfully.")
        except Exception as e:
            logging.error(f"Error saving the best model: {str(e)}")
            return jsonify({"error": f"Error saving the best model: {str(e)}"}), 500

        # Automatically trigger the prediction Flask app
        try:
            # Define the URL for the prediction Flask app
            prediction_url = "http://prediction-service:80/predict"
            logging.info(f"Attempting to send processed predict data to {prediction_url}")

            # Save the processed predict data to a file
            predict_processed_path = 'data/02_processed/test_processed.csv'
            os.makedirs(os.path.dirname(predict_processed_path), exist_ok=True)
            predict_data.to_csv(predict_processed_path, index=False)

            # Send the processed predict data to the prediction Flask app
            with open(predict_processed_path, 'rb') as f:
                files = {
                    'file': (predict_processed_path, f, 'text/csv')
                }
                response = requests.post(prediction_url, files=files)
                response.raise_for_status()  # Raise an error for bad status codes
                logging.info(f"Prediction triggered successfully. Response: {response.text}")
                return f"Final Model Metrics: {final_metrics} \nBest Model: {best_model_name} \nPrediction Triggered: {response.text}"
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Failed to connect to the prediction service at {prediction_url}. Ensure the service is running. Error: {str(e)}")
            return f"Failed to connect to the prediction service at {prediction_url}. Ensure the service is running. Error: {str(e)}"
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending files to prediction service. Error: {str(e)}")
            return f"Error sending files to prediction service. Error: {str(e)}"

    except Exception as e:
        logging.error(f"Unexpected error in /model endpoint: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)