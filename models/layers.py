"""
Custom neural network layers for protein flexibility prediction.

This module provides specialized neural network layers for processing 3D protein structure data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention layer for highlighting important regions in voxel data.
    """
    
    def __init__(self, in_channels: int, kernel_size: int = 7):
        """
        Initialize the spatial attention layer.
        
        Args:
            in_channels: Number of input channels
            kernel_size: Size of the convolutional kernel
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, depth, height, width]
            
        Returns:
            Tensor with same shape as input, with spatial attention applied
        """
        # Calculate max and average along channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate the pooled features
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        
        # Apply convolution and sigmoid activation
        attention = self.sigmoid(self.conv(pooled))
        
        # Apply attention to input
        return x * attention

class ChannelAttention3D(nn.Module):
    """
    Channel Attention layer for highlighting important atom channels.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        """
        Initialize the channel attention layer.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP for channel attention
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, depth, height, width]
            
        Returns:
            Tensor with same shape as input, with channel attention applied
        """
        # Apply average pooling
        avg_out = self.mlp(self.avg_pool(x))
        
        # Apply max pooling
        max_out = self.mlp(self.max_pool(x))
        
        # Combine pooled features and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention to input
        return x * attention