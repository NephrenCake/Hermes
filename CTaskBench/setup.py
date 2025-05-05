from setuptools import setup
import setuptools
from typing import List
import os

ROOT_DIR = os.path.dirname(__file__)
with open("README.md", "r") as fh:
    long_description = fh.read()

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements

setup(
    name='CTaskBench',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/gz944367214/CTaskBench',
    license='MIT',
    author='Zuo Gan',
    author_email='gz944367214@sjtu.edu.cn',
    description='Compound LLM Task Inference Benchmark',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)