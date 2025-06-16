import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foresight",
    version="0.4",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Temporal modeling of patients and diseases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/foresight",
    packages=['foresight', 'foresight.datasets', 'foresight.metrics', 'foresight.utils',
              'foresight.tokenizers'],
    install_requires=[
        'datasets',
        'dill',
        'medcat',
        'numpy',
        'pandas',
        'torch',
        'transformers',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
