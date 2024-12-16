from setuptools import setup, find_packages

setup(
    name="recsys_lakehouse",
    version="0.1.0",
    author="Dewastator",
    description="Package for lakehouse operations in recommendation systems",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "pyspark>=3.5.3"
    ],
)
