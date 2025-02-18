# Use a lightweight Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy app code
COPY dataprep.py requirements.txt ./
COPY data/ ./data/

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "dataprep.py"]