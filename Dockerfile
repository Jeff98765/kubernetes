# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first to take advantage of Docker’s caching
COPY 222356K_og/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Ensure the run.sh script has executable permissions
RUN chmod +x 222356K_og/run.sh

# Set the default command to run the pipeline using the run.sh script
CMD ["./222356K_og/run.sh"]