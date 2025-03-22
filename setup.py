from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="essa",
    version="0.1",
    author="Eugene Turov",
    author_email="linker0broadcast@gmail.com",
    description="Easy Singular Spectrum Analysis (SSA) implementation for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ProtonEvgeny/essa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.5.0",
    ],
)
