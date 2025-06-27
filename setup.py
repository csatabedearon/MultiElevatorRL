# setup.py
"""
Setup configuration for the Multi-Elevator RL package.

This script is used to configure the installation of the 'multi_elevator' package,
including its dependencies, metadata, and other properties.
"""
from setuptools import setup, find_packages

setup(
    # Package metadata
    name="multi_elevator",
    version="0.1.0",
    author="Csata Bede Aron",
    author_email="csatabedearonka@gmail.com",
    description="A multi-elevator reinforcement learning environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/csatabedearon/multi_elevator",

    # Package structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Dependencies
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "stable-baselines3>=2.1.0",
        "flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "python-socketio>=5.9.0",
        "gevent>=23.9.1",
        "gevent-websocket>=0.10.1",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "rich>=13.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.1.0",
        ],
    },

    # Python version requirement
    python_requires=">=3.8",

    # PyPI classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)