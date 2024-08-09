@echo off

REM Update package list and install pip if not installed
REM Note: Windows doesn't need to update package lists like Linux.
REM Instead, make sure Python is installed and pip is available.

REM Install required Python packages globally (not recommended)
REM Instead, we'll use a virtual environment.

REM Create a virtual environment (recommended)
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Install required Python packages within the virtual environment
pip install scikit-learn matplotlib stable-baselines3 numpy tensorflow scipy gym-minigrid torch minigrid
pip install stable-baselines3[extra]

REM Install additional packages from requirements.txt if it exists
if exist requirements.txt (
    pip install -r requirements.txt
)

echo "All packages installed successfully."

REM Any other setup commands
REM e.g., setting environment variables, downloading datasets, etc.
