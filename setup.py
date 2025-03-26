# setup.py
"""
Setup configuration for the Multi-Elevator RL package.
"""
from setuptools import setup, find_packages

setup(
    name="multi_elevator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "stable-baselines3>=2.1.0",
        "flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "python-socketio>=5.9.0",
        # Use either eventlet OR gevent, not both listed usually.
        # Gevent seems preferred by your web_app.py patches.
        # "eventlet>=0.33.0", # Keep if needed, but web_app uses gevent
        "gevent>=23.9.1",     # From requirements.txt
        "gevent-websocket>=0.10.1", # From requirements.txt
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
    python_requires=">=3.8",
    # ---- REMOVE OR COMMENT OUT THIS SECTION ----
    # entry_points={
    #     "gymnasium.envs": [
    #         "MultiElevator-v0=multi_elevator.environment:MultiElevatorEnv",
    #     ],
    # },
    # --------------------------------------------
    author="Csata Bede Aron",
    author_email="csatabedearonka@gmail.com",
    description="A multi-elevator reinforcement learning environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/csatabedearon/multi_elevator",
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