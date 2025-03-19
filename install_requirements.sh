#!/bin/bash

# ==============================================================================
# Install Python packages into the virtual environment
venv_name=$1

if [ -z "$venv_name" ]; then
    echo "Error: Virtual environment name is required."
    exit 1
fi

# Check if the virtual environment exists
if [ ! -d "$venv_name" ]; then
    echo "Error: Virtual environment $venv_name does not exist."
    exit 1
fi

# Activate the virtual environment
source "$venv_name/bin/activate"

# Install required packages
echo "Installing required packages in virtual environment $venv_name..."
pip install pydantic rich pybids nibabel pydicom nilearn matplotlib pandas
pip install lapy antspyx antspynet

pip install ipython ipykernel ipywidgets


# Install the latest release of pycortex from pip
# pip install setuptools wheel numpy cython
# pip install pycortex

pip freeze > requirements.txt