from flask import Flask, request
import pickle
import pandas as pd
import io
from sklearn.base import BaseEstimator

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_on_data():
    # Get files from post request
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    model = pickle.load(file1.stream)
    predict_data = pd.read_csv(io.StringIO(file2.stream.read().decode('utf-8')))

    # Predict on prediction dataset
    predictions = model.predict(predict_data)
    predict_data['Predicted_Survived'] = predictions

    # Save predictions
    predicted_data_path = 'predictions/predicted_data.csv'
    predict_data.to_csv(predicted_data_path, index=False)
    
    return_text = f"Prediction complete, file saved at {predicted_data_path}"

    return return_text, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    app.run(port=5002) 