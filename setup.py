#!/usr/bin/env python3
"""
Setup script for the ProtFlex package.
"""

import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get version from __init__.py
with open(os.path.join('protflex', '__init__.py'), encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'\"")
            break
    else:
        version = '0.1.0'

setup(
    name='protflex',
    version=version,
    description='Deep Learning for Protein Flexibility Prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Felix',
    author_email='s.felix@example.com',
    url='https://github.com/example/protflex',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'protflex=protflex:run',
        ],
    },
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'torch>=1.9.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'h5py>=3.3.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.61.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-cov>=2.12.1',
            'mypy>=0.910',
            'black>=21.6b0',
            'isort>=5.9.1',
            'flake8>=3.9.2',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)
