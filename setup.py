# ===== FILE: setup.py =====
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trading-simulator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular stock trading simulation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git remote add origin https://github.com/tekuro-io/mjolnir-prime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "jupyter": [
            "jupyter",
            "matplotlib",
            "seaborn",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-simulator=trading_simulator.examples.basic_demo:comprehensive_demo",
        ],
    },
)