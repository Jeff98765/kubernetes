from flask import Flask, request, jsonify
import requests
import pandas as pd
import io

app = Flask(__name__)

@app.route('/receive_csv', methods=['POST'])
def receive_csv():

    # Get the file from the POST request
    file = request.files['file']
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
    
    # Add a new column that is the mean of the first two columns
    df['Mean'] = df[['Column1', 'Column2']].mean(axis=1)

    print(df['Mean'].tolist(), flush=True)
    
    # Save the result to a new CSV file or return the updated data
    df.to_csv('updated_data.csv', index=False)
    
    return "File received and processed successfully!"

if __name__ == '__main__':
    app.run(port=5001)
