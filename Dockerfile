# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your local project directory to the container
COPY . .

# Install required Python dependencies
RUN pip install numpy pandas scikit-learn wandb

# Set the default command to run your script
CMD ["python", "distance_classification.py"]
