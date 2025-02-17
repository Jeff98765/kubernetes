import os
import joblib
import pandas as pd

def make_predictions(test_data: pd.DataFrame, model_path: str, output_filename: str):
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

if __name__ == '__main__':
    test_data_path = 'data/02_processed/test_processed.csv'
    test_data = pd.read_csv(test_data_path)

    model_path = 'saved_model/random_forest_model.pkl'
    output_filename = 'predictions.csv'

    make_predictions(test_data, model_path, output_filename)    