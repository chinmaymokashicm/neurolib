#!/bin/bash

# ==============================================================================
# Set up the Python environment
python_version=3.12
venv_name=venv

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "pyenv is not installed. Please install it first."
    exit 1
fi

# Set the Python version for the current shell session
if ! pyenv versions | grep -q "$python_version"; then
    echo "Python $python_version is not installed. Installing now..."
    pyenv install "$python_version" -s
fi
pyenv shell "$python_version"
echo "Using Python version: $(python --version)"

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