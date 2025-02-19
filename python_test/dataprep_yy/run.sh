#!/bin/bash

# Deployment name
DEPLOYMENT_NAME="ml-deployment"

# Namespace (change it if your deployment is in a different namespace)
NAMESPACE="default"

# Get the name of the pod associated with the deployment
POD_NAME=$(kubectl get pods -o jsonpath='{.items[0].metadata.name}')

# Check if POD_NAME is found
if [ -z "$POD_NAME" ]; then
  echo "No pod found for deployment $DEPLOYMENT_NAME"
  exit 1
fi

# Execute command on the pod (replace '/bin/bash' with your desired command)
kubectl exec $POD_NAME -n $NAMESPACE -- curl -sS localhost:5000/preprocess
kubectl exec $POD_NAME -n $NAMESPACE -- tar -cf - ./predictions/predicted_data.csv | tar -xvf -
