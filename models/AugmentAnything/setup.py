from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="augment-anything",
    version="0.1.0",
    author="Simon Kotchou",
    author_email="simonkotchou@mines.edu",
    description="Intelligent data augmentation using SAM2 for semantic-aware image synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simon-kotchou/augment-anything",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "augment-anything=augment_anything.core:main",
            "augment-build-db=augment_anything.faiss_ds:main",
        ],
    },
    include_package_data=True,
    keywords="data-augmentation computer-vision sam2 semantic-segmentation faiss",
    project_urls={
        "Bug Reports": "https://github.com/simon-kotchou/augment-anything/issues",
        "Source": "https://github.com/simon-kotchou/augment-anything",
        "Documentation": "https://github.com/simon-kotchou/augment-anything#readme",
    },
)
