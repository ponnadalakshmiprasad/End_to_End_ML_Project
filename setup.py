from typing import List
from setuptools import setup, find_packages

hypen_dot_e="-e ."
def get_requirements(file_path):
    with open(file_path) as file:
        requirements=file.readlines()
        requirements =[req.replace("\n", "") for req in requirements]
        if hypen_dot_e in requirements:
            requirements.remove(hypen_dot_e)
        return requirements
setup(
    name="End-End_project_ML",
    version="0.0.1",
    author="Prasad",
    author_email="ponnadalakshmiprasad@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)