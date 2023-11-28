#Python image as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /workspace

# Create the necessary directory
RUN mkdir -p ~/.huggingface

# Set the HUGGINGFACE_TOKEN environment variable
ENV HUGGINGFACE_TOKEN="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"
# Create and write the token to the file
RUN echo -n "${HUGGINGFACE_TOKEN}" > ~/.huggingface/token

# Copy the requirements file into the container
COPY requirements.txt requirements.txt
COPY diffusers diffusers

# Install Flask and other required packages
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
RUN pip install diffusers

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 9093

# Define the command to start your Flask application
CMD ["python", "app.py"]
