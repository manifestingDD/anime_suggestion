from setuptools import setup, find_packages 

with open("requirements.txt") as f:
    requirements = f.read().splitlines()  # Read the file line by line, we'll get a list


setup(
    name = "MLOPS-HOTEL-FORECAST", 
    version = "0.1",
    author = "DamienD",
    packages = find_packages(), #automatically find all packages, including {src, config, utils}
    install_requires = requirements,
)