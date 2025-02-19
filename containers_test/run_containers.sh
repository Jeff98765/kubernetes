# #!/bin/bash

# # # Build Docker Images
# # echo "Building Docker Images..."
# # docker build -t kubelete/dataprep-flask:v1 ./dataprep
# # docker build -t kubelete/model-flask:v1 ./model
# # docker build -t kubelete/prediction-flask:v1 ./prediction

# # Run Containers
# echo "Running Containers..."
# docker run -d -p 5001:5001 -v $(pwd)/data/01_raw:/app/data/01_raw -v /tmp/processed-data:/app/data/02_processed --name dataprep-container kubelete/dataprep-flask:v1
# docker run -d -p 5002:5002 -v /tmp/processed-data:/app/data/02_processed -v /tmp/saved-model:/app/saved_model --name model-container kubelete/model-flask:v1
# docker run -d -p 5003:5003 -v /tmp/processed-data:/app/data/02_processed -v /tmp/saved-model:/app/saved_model -v /tmp/predicted-data:/app/data/03_predicted --name prediction-container kubelete/prediction-flask:v1

# # Test APIs
# echo "Testing Dataprep API..."
# curl -X POST http://localhost:5001/preprocess -F "file=@data/01_raw/train.csv"

# echo "Testing Model API..."
# curl -X POST http://localhost:5002/train

# echo "Testing Prediction API..."
# curl -X POST http://localhost:5003/predict

# # Verify Outputs
# echo "Verifying Outputs..."
# ls -lh /tmp/processed-data
# ls -lh /tmp/saved-model
# ls -lh /tmp/predicted-data

# # Stop and Remove Containers
# echo "Cleaning Up..."
# docker stop dataprep-container model-container prediction-container
# docker rm dataprep-container model-container prediction-container