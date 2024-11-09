import subprocess

def create_conda_environment(environment_name, requirements_file):
    """
    Create a Conda environment with the specified name and requirements file.

    Parameters:
    environment_name (str): The name of the Conda environment to create.
    requirements_file (str): The path to the requirements file for installing packages.

    Raises:
    subprocess.CalledProcessError: If the Conda environment creation fails.
    """
    try:
        command = f"conda create --name {environment_name} --file {requirements_file} -y"
        subprocess.run(command, check=True, shell=True)
        print(f"Environment '{environment_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating the environment: {e}")

def setup_environment():
    """
    Set up the Conda environment by specifying the environment name and requirements file.
    """
    requirements_file = "requirements.txt"
    environment_name = "iris"
    
    create_conda_environment(environment_name, requirements_file)

if __name__ == "__main__":
    setup_environment()
