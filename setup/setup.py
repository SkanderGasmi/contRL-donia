import subprocess
import sys
import os

def clear_screen():
    """Clear the terminal screen."""
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix-like systems
        os.system('clear')

def upgrade_pip():
    """Upgrade pip to the latest version."""
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

def install_requirements(requirements_file):
    """Install packages listed in the requirements file."""
    requirements_path = os.path.join('setup', requirements_file)
    if os.path.isfile(requirements_path):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
    else:
        raise FileNotFoundError(f"{requirements_file} file not found in the setup directory.")


def create_virtualenv(env_dir):
    """Create a virtual environment if it doesn't already exist."""
    if not os.path.exists(env_dir):
        print("Creating a virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', env_dir])
        print(f"Virtual environment created in {env_dir}.")
    else:
        print("Virtual environment already exists.")

def print_in_red(text):
    """Print text in red."""
    RED = '\033[91m'
    RESET = '\033[0m'
    print(f"{RED}{text}{RESET}")

def print_in_green(text):
    """Print text in green."""
    GREEN = '\033[92m'
    RESET = '\033[0m'
    print(f"{GREEN}{text}{RESET}")

def main():
    # Clear the terminal screen
    clear_screen()

    # Define virtual environment directory
    env_dir = 'venv'

    # Upgrade pip
    upgrade_pip()

    # Create a virtual environment
    create_virtualenv(env_dir)

    # Check for requirements.txt and install packages
    if os.path.isfile('requirements.txt'):
        print("Installing packages from requirements.txt...")
        try:
            install_requirements('requirements.txt')
            print_in_green("\n\nAll packages installed successfully.")
        except subprocess.CalledProcessError:
            print_in_red("Failed to install packages from requirements.txt.")
    else:
        print_in_red("requirements.txt file not found. Please create one with the required packages.")
    
    # Provide activation instructions
    if os.name == 'nt':  # Windows
        activation_command = f"\nTo activate the virtual environment, run:\nvenv\\Scripts\\Activate.ps1\n"
    else:  # Unix/Linux/MacOS
        activation_command = f"\nTo activate the virtual environment, run:\nsource venv/bin/activate\n"
    
    print_in_green(f"Please activate the virtual environment before running your project.{activation_command}")

if __name__ == "__main__":
    main()
