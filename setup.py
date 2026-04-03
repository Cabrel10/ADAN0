#!/usr/bin/env python3
"""Setup script for ADAN Trading Bot"""

from setuptools import setup, find_packages

setup(
    name="adan_trading_bot",
    version="0.1.0",
    description="ADAN Trading Bot - Advanced Deep Adaptive Networks for Algorithmic Trading",
    author="ADAN Team",
    author_email="info@adan.trading",
    url="https://github.com/Cabrel10/ADAN0",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10,<3.12",
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "gymnasium==1.2.1",
        "stable-baselines3==2.7.0",
        "ccxt==4.5.6",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "ta>=0.11.0",
        "finta>=1.3",
        "plotly>=5.17.0",
        "matplotlib>=3.8.0",
        "pyyaml",
        "python-dotenv",
        "tqdm",
        "psutil",
        "rich",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    include_package_data=True,
    zip_safe=False,
)
