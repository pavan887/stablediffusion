#Python image as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirement.txt requirements.txt

# Install Flask and other required packages
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 9093

# Define the command to start your Flask application
CMD ["python", "app.py"]
