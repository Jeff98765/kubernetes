# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first to take advantage of Docker’s caching
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Run the dataprep.py script when the container starts
CMD ["python", "dataprep.py"]