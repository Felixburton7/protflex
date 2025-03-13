"""
Model module for ProtFlex.

This module provides neural network models for protein flexibility prediction.
"""

from .cnn_models import create_model, ProtFlexCNN, DilatedResNet3D, MultipathRMSFNet
from .loss import create_loss_function, RMSFLoss, WeightedRMSFLoss, ElasticRMSFLoss
