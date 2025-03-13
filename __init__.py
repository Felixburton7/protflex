
"""
ProtFlex: Deep Learning for Protein Flexibility Prediction.

A comprehensive framework for predicting protein flexibility (RMSF) from voxelized
protein structures using deep convolutional neural networks.
"""

import os
import sys

__version__ = '0.1.0'
__author__ = 'Felix'
__email__ = 's.felix@example.com'

# Make package components available for import
from . import config
from . import data
from . import models
from . import training
from . import utils
from . import cli

# Version information dictionary
__info__ = {
    'name': 'ProtFlex',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'description': 'Deep Learning for Protein Flexibility Prediction',
    'url': 'https://github.com/example/protflex',
}

def run():
    """
    Run ProtFlex CLI from Python.
    """
    from .cli.cli import main
    sys.exit(main())
