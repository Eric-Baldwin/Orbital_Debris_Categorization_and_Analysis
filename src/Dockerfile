# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install build-essential (includes gcc)
RUN apt-get update && apt-get install -y build-essential

# Copy the current directory contents into the container at /app
COPY . /app

# Ensure the model file is copied into the container
COPY best_logreg_model.pkl /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
