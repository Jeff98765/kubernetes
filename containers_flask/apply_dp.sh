#!/bin/bash

kubectl apply -f k8s/dataprep-deployment.yml
kubectl apply -f k8s/model-deployment.yml
kubectl apply -f k8s/prediction-deployment.yml
kubectl apply -f k8s/streamlit-deployment.yml