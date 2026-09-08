import os
from setuptools import setup, find_packages

requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        requirements = [
            req.strip() for req in f.read().splitlines()
            if req.strip() and not req.strip().startswith("#") and not req.strip().startswith("-e")
        ]

setup(
    name="object-detection-fasterrcnn",
    version='0.1.0',
    author='mdzaheerjk',
    packages=find_packages(),
    install_requires=requirements
)