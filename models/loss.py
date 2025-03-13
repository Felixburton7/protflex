"""
Custom loss functions for protein flexibility prediction.

This module provides specialized loss functions for training RMSF prediction models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSFLoss(nn.Module):
    """
    Custom loss function for RMSF prediction that combines MSE and RMSE.

    This loss function is designed specifically for RMSF prediction, using a weighted
    combination of MSE (mean squared error) and RMSE (root mean squared error) to
    better handle the natural distribution of RMSF values.
    """

    def __init__(self, mse_weight: float = 0.7, rmse_weight: float = 0.3, eps: float = 1e-8):
        """
        Initialize the RMSF loss function.

        Args:
            mse_weight: Weight for the MSE component
            rmse_weight: Weight for the RMSE component
            eps: Small constant to avoid numerical instability
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.rmse_weight = rmse_weight
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Weighted loss value
        """
        # Mean squared error
        mse = F.mse_loss(y_pred, y_true)

        # Root mean squared error
        rmse = torch.sqrt(mse + self.eps)

        # Combined loss
        loss = self.mse_weight * mse + self.rmse_weight * rmse

        return loss

class WeightedRMSFLoss(nn.Module):
    """
    Weighted loss function that gives higher weight to high RMSF values.

    This loss function places more emphasis on accurately predicting residues with
    high flexibility (high RMSF), which are often functionally important but fewer
    in number than low-RMSF residues.
    """

    def __init__(self, threshold: float = 0.5, high_weight: float = 2.0, low_weight: float = 1.0):
        """
        Initialize the weighted RMSF loss function.

        Args:
            threshold: RMSF threshold to separate high and low values
            high_weight: Weight for high RMSF values
            low_weight: Weight for low RMSF values
        """
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weighted loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Weighted loss value
        """
        # Create weight mask based on true values
        weights = torch.ones_like(y_true)
        weights[y_true > self.threshold] = self.high_weight
        weights[y_true <= self.threshold] = self.low_weight

        # Calculate squared differences
        squared_diff = (y_pred - y_true) ** 2

        # Weighted MSE
        loss = torch.mean(weights * squared_diff)

        return loss

class ElasticRMSFLoss(nn.Module):
    """
    Elastic net style loss combining L1 and L2 penalties for RMSF prediction.

    This loss function combines MSE (L2) and MAE (L1) losses, similar to an elastic
    net regularization, to benefit from both the smooth gradients of MSE and the
    robustness to outliers of MAE.
    """

    def __init__(self, l1_weight: float = 0.5, l2_weight: float = 0.5):
        """
        Initialize the elastic RMSF loss function.

        Args:
            l1_weight: Weight for the L1 (MAE) component
            l2_weight: Weight for the L2 (MSE) component
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the elastic loss.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Elastic loss value
        """
        # L1 loss (MAE)
        l1_loss = F.l1_loss(y_pred, y_true)

        # L2 loss (MSE)
        l2_loss = F.mse_loss(y_pred, y_true)

        # Combined loss
        loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss

        return loss

def create_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create a loss function based on name and parameters.

    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function

    Returns:
        Loss function instance
    """
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "rmsf":
        return RMSFLoss(**kwargs)
    elif loss_name == "weighted_rmsf":
        return WeightedRMSFLoss(**kwargs)
    elif loss_name == "elastic_rmsf":
        return ElasticRMSFLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
