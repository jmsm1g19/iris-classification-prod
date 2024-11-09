import os
import sys

"""
This script sets the project root directory by creating or updating a '.env' file with the PROJECT_ROOT path.
The root directory is identified as the parent directory of the 'env' folder containing this script.
"""

# Indicator file used to confirm we are in the project root
gitignore_indicator = '.gitignore'

# Get the directory containing this script (which is the 'env' folder)
env_dir = os.path.dirname(os.path.abspath(__file__))

# Set the project root to be the parent directory of the 'env' folder
root_dir = os.path.abspath(os.path.join(env_dir, os.pardir))

# Check if the '.gitignore' file exists in the identified root directory
if not os.path.exists(os.path.join(root_dir, gitignore_indicator)):
    print("This script should be run from the 'env' folder, and the project root should contain a .gitignore file.")
    sys.exit(1)

# Define the path to the '.env' file in the project root
env_path = os.path.join(root_dir, ".env")

# Create or update the '.env' file with the PROJECT_ROOT path
if not os.path.isfile(env_path):
    with open(env_path, 'w') as f:
        f.write(f'PROJECT_ROOT={root_dir}\n')
    print(f'Created .env file with PROJECT_ROOT={root_dir}')
else:
    with open(env_path, 'r+') as f:
        lines = f.readlines()
        found = False
        # Update the PROJECT_ROOT line if it already exists
        for i, line in enumerate(lines):
            if line.startswith('PROJECT_ROOT='):
                lines[i] = f'PROJECT_ROOT={root_dir}\n'
                found = True
                break
        # Append the PROJECT_ROOT line if it wasn't found
        if not found:
            lines.append(f'PROJECT_ROOT={root_dir}\n')
        f.seek(0)
        f.writelines(lines)
        f.truncate()
    print(f'Updated .env file with PROJECT_ROOT={root_dir}')
