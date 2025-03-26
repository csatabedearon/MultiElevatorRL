from setuptools import setup, find_packages

setup(
    name="multi_elevator",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium",
        "numpy",
        "torch",
        "stable-baselines3",
        "flask",
    ],
    python_requires=">=3.8",
    author="Csata Bede Aron",
    author_email="csatabedearonka@gmail.com",
    description="A multi-elevator system with reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/csatabedearon/multi_elevator",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 