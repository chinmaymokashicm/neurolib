#!/bin/bash

# ==============================================================================
# Set up the Python environment
python_version=3.12
venv_name=venv

# Create the virtual environment using venv
if [ -d "$venv_name" ]; then
    echo "Virtual environment $venv_name already exists. Skipping creation."
else
    echo "Creating virtual environment: $venv_name..."
    python -m venv "$venv_name"
    echo "Virtual environment $venv_name created."
fi

# Activate the virtual environment
source "$venv_name/bin/activate"
echo "Virtual environment $venv_name activated."

# Install required packages
echo "Installing required packages..."
bash install_requirements.sh "$venv_name"
echo "Required packages installed."

poetry add $(cat requirements.txt) # Only for this project

# Set environment variables
source set_envs.sh