#!/bin/bash

# Build data preprocessing image
cd dataprep
docker build -t kubelete/dataprep-flask:v3 .

# Build model training image
cd ..
cd model
docker build -t kubelete/model-flask:v3 .

# Build prediction image
cd ..
cd prediction
docker build -t kubelete/prediction-flask:v3 .

cd ..