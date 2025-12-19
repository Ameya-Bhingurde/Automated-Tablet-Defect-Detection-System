#!/bin/bash

# Render.com deployment script
# This script sets up the environment and starts the Streamlit app

# Print Python version
echo "Python version:"
python --version

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p models results

# Note: Model will be trained automatically on first run by app.py
echo "Setup complete! Starting Streamlit app..."
