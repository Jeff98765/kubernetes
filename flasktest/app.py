from flask import Flask
import pandas as pd
import numpy as np
import requests
import os

app = Flask(__name__)


@app.route('/generate_csv')
def generate_csv():
    # Create a DataFrame with two columns of random numbers
    data = {
        'Column1': np.random.randint(1, 100, 10),
        'Column2': np.random.randint(1, 100, 10)
    }
    df = pd.DataFrame(data)
    
    # Save it to a CSV file
    file_path = 'random_data.csv'
    df.to_csv(file_path, index=False)

    # Send the file to app2
    url = "http://localhost:5001/receive_csv"
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Clean up the file after sending
    os.remove(file_path)
    
    return f"CSV file sent to app2, response: {response.text}"

if __name__ == '__main__':
    app.run(port=5000)