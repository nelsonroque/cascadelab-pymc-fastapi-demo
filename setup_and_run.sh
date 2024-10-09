#!/bin/bash

# Name of the virtual environment
ENV_NAME="fastapi_pymc_env"

# Create a new virtual environment
echo "Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run FastAPI app
echo "Starting FastAPI app..."
uvicorn app:app --reload
