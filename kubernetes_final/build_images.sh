#!/bin/bash

# Build data preprocessing image
cd dataprep
docker build -t kubelete/dataprep-flask:v4 .

# Build model training image
cd ..
cd model
docker build -t kubelete/model-flask:v4 .

# Build prediction image
cd ..
cd prediction
docker build -t kubelete/prediction-flask:v4 .

cd ..
cd streamlit
docker build -t kubelete/streamlit-app:v1 .

cd ..