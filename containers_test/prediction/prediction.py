import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

def make_predictions_and_save_to_csv(test_data: pd.DataFrame, model_path: str, output_filename: str):
    """Make predictions and save them to a CSV file."""
    model = joblib.load(model_path)
    predictions = model.predict(test_data)
    test_data['Predicted_Survived'] = predictions

    output_path = os.path.join('data', '03_predicted', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    return output_path

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions."""
    try:
        # Load the processed test dataset from the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        test_data = pd.read_csv(file)

        # Path to the trained model
        model_path = 'saved_model/random_forest_model.pkl'
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 400

        # Make predictions and save to CSV
        output_filename = 'predictions.csv'
        output_path = make_predictions_and_save_to_csv(test_data, model_path, output_filename)

        return jsonify({
            "message": "Predictions made successfully",
            "output_file": output_path
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)