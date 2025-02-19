# Use a lightweight Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy app code
COPY model.py requirements.txt ./
COPY saved_model/ ./saved_model/

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 5001

# Run the app
CMD ["python", "model.py"]