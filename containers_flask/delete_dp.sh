#!/bin/bash

kubectl delete -f k8s/dataprep-deployment.yml
kubectl delete -f k8s/model-deployment.yml
kubectl delete -f k8s/prediction-deployment.yml
kubectl delete -f k8s/streamlit-deployment.yml