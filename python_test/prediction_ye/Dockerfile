# Use a lightweight Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy app code
COPY prediction.py requirements.txt ./
COPY predictions/ ./predictions/

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 5002

# Run the app
CMD ["python", "prediction.py"]