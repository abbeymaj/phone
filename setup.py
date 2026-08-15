# Importing packages
from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ."

# Creating a function to fetch requirements from the requirements.txt file
def fetch_requirements(file_path) -> list:
    '''
    This function will return the list of packages that are specified in the 
    requirements file.
    ==========================================================================
    ---------------
    Parameters:
    ---------------
    file_path : str : This is the path to the requirements.txt file.
    
    ---------------
    Returns:
    ---------------
    List - List[str] - This is the list of packages that need to be installed
    for the project, which is specified in the requirements.txt file.
    ==========================================================================
    '''
    # Reading the file
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Removing any new line characters and any empty spaces
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Removing the hyphen e dot if it is present in the requirements list
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements

# Creating the setup function to install the packages
setup(
    name='phone_addiction_detection',
    author='Abhijit Majumdar',
    version='0.0.1',
    packages=find_packages(),
    install_requires=fetch_requirements('requirements.txt')
)
    