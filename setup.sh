#!/bin/bash

# Update package list and install pip if not installed
sudo apt-get update
sudo apt-get install -y python3-pip

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required Python packages within the virtual environment
pip install scikit-learn matplotlib stable-baselines3 numpy tensorflow scipy gym-minigrid torch minigrid
pip install stable-baselines3[extra]

# Install additional packages from requirements.txt if it exists
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "All packages installed successfully."

# Any other setup commands
# e.g., setting environment variables, downloading datasets, etc.
