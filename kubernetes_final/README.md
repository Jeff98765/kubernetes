# EGT309 Project: End-to-End AI Solution in Kubernetes

## Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation and Setup](#4-installation-and-setup)
5. [Usage Instructions](#5-usage-instructions)
6. [API Endpoints](#6-api-endpoints)
7. [Deployment in Kubernetes](#7-deployment-in-kubernetes)
8. [Version Control](#8-version-control)
9. [Additional Features](#9-additional-features)

## 1. Project Overview
This project demonstrates the deployment of an end-to-end AI solution in Kubernetes. The goal is to predict survival from the Titanic dataset.

### Solution Modules:
1. **Dataprep**: Preprocesses raw data into a format suitable for machine learning.
2. **Model**: Trains a machine learning model using the preprocessed data.
3. **Prediction**: Uses the trained model to generate predictions on new data.
4. **UI**: Provides a user-friendly interface to interact with the system.

The solution is fully containerized, scalable, and supports self-healing, rollouts, and rollbacks.

## 2. System Architecture
The system architecture is described in this section.

**Container Connections**
Each container interacts with Kubernetes resources (PVs, PVCs, and Services) to ensure modularity and data persistence. Below are the connections for each container, including the service ports and how they are exposed:

- **Dataprep Container**
  - **PV/PVC**: 
    - Reads raw data from `raw-data-pvc`.
    - Writes preprocessed data to `processed-data-pvc`.
  - **Service**: Exposed via `dataprep-service`.
    - **Internal Port**: 5000 (container listens on this port).
    - **Service Port**: 80 (exposed within the cluster).


- **Model Container**
  - **PV/PVC**: 
    - Reads preprocessed data from `processed-data-pvc`.
    - Writes trained models to `saved-model-pvc`.
  - **Service**: Exposed via `model-service`.
    - **Internal Port**: 5001
    - **Service Port**: 80


- **Prediction Container**
  - **PV/PVC**: 
    - Reads trained models from `saved-model-pvc`.
    - Reads test data from `processed-data-pvc`.
    - Writes predictions to `predicted-data-pvc`.
  - **Service**: Exposed via `prediction-service`.
    - **Internal Port**: 5002
    - **Service Port**: 80


- **UI (Streamlit) Container**
  - **PV/PVC**: 
    - Reads raw data from `raw-data-pvc`.
    - Reads preprocessed data from `processed-data-pvc`.
    - Reads predictions from `predicted-data-pvc`.
  - **Service**: Exposed via `streamlit-service`.
    - **Internal Port**: 8501
    - **Service Port**: 8501
    - **External Port**: 30001 (accessible externally via NodePort)


## 3. Prerequisites
To run this project, the following tools are required:
- Ubuntu/WSL
- Kubernetes with `kubectl` (Minikube or cloud-based cluster)

## 4. Installation and Setup

### Steps:
1. **Unzip the file**
2. **Start the WSL terminal** and navigate to the project directory
3. **Start Minikube**
   ```sh
   minikube start
   ```

4. **Navigate to the [`k8s`](./k8s/) folder and deploy in Kubernetes**
   ```sh
   kubectl apply -f k8s/pv.yml
   kubectl apply -f k8s/pvc.yml
   kubectl apply -f k8s/dataprep-deployment.yml
   kubectl apply -f k8s/model-deployment.yml
   kubectl apply -f k8s/prediction-deployment.yml
   kubectl apply -f k8s/streamlit-deployment.yml
   ```


## 5. Usage Instructions

### Test Streamlit UI:
```sh
minikube service streamlit-service --url
```
Copy and paste the second IP address into a browser.

### Check Service URLs:
```sh
minikube service dataprep-service --url
minikube service model-service --url
minikube service prediction-service --url
minikube service streamlit-service --url
```

## 6. API Endpoints
The UI interacts with the following backend services via HTTP requests:

### Data Preprocessing
- **Endpoint:** `POST /preprocess`
- **Service URL:** `http://dataprep-service:80/preprocess`
- **Description:** Cleans and transforms raw data.
- **Input:** Raw dataset files (`train.csv`, `predict.csv`) in `data/01_raw`
- **Output:** Preprocessed files (`train_processed.csv`, `predict_processed.csv`) in `data/02_processed`

### Model Training
- **Endpoint:** `POST /model`
- **Service URL:** `http://model-service:80/model`
- **Description:** Trains machine learning models.
- **Input:** Preprocessed dataset files from `data/02_processed`
- **Output:** Model metrics as JSON and saved model (`random_forest_model.pkl`)

### Prediction
- **Endpoint:** `POST /predict`
- **Service URL:** `http://prediction-service:80/predict`
- **Description:** Generates predictions using the trained model.
- **Input:** Preprocessed test dataset file (`predict_processed.csv` in `data/02_processed`)
- **Output:** Predictions saved in `data/03_predicted` (`predictions.csv`)

## 7. Deployment in Kubernetes

### Scaling
Adjust the number of replicas:
```sh
kubectl scale deployment <deployment-name> --replicas=3
```

### Self-Healing
Kubernetes automatically restarts failed pods:
```sh
kubectl delete pod <pod-name>
```

### Rollout/Rollback
Update the image version and perform rollbacks if needed:
```sh
kubectl set image deployment/<deployment-name> <container-name>=<image-name>:<version>
kubectl rollout undo deployment/<deployment-name>
```

## 8. Version Control
GitHub is used to track code commits. Access the [project repository](https://github.com/Jeff98765/kubernetes.git).

## 9. Additional Features
- Added progress bar in the UI to track progress
- Added a display in the UI to show predictions


