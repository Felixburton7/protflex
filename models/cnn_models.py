"""
Advanced 3D CNN architectures for protein flexibility prediction.

This module contains implementations of various 3D CNN architectures designed
to predict RMSF (Root Mean Square Fluctuation) values from voxelized protein
structures. The networks are specifically designed to preserve spatial relationships
among backbone atoms (C, N, O, CA, CB) which are crucial for understanding
protein dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

class ResidualBlock3D(nn.Module):
    """A 3D residual block with optional dilation and bottle-necking.
    
    This block implements the core residual connection pattern where the input
    is added to the output of a series of convolutional operations. It supports
    dilated convolutions to increase the receptive field without increasing
    parameter count.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        dilation: int = 1, 
        bottleneck: bool = False, 
        bottleneck_factor: int = 4
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            dilation: Dilation rate of the convolution
            bottleneck: Whether to use bottleneck architecture
            bottleneck_factor: Bottleneck reduction factor
        """
        super().__init__()
        
        self.bottleneck = bottleneck
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Calculate padding to maintain spatial dimensions
        padding = dilation * (kernel_size - 1) // 2
        
        if bottleneck:
            # Bottleneck architecture - reduce channels, then 3x3, then expand channels
            bottleneck_channels = out_channels // bottleneck_factor
            self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1)
            self.bn1 = nn.BatchNorm3d(bottleneck_channels)
            self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, 
                                kernel_size=kernel_size, stride=stride, 
                                padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm3d(bottleneck_channels)
            self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1)
            self.bn3 = nn.BatchNorm3d(out_channels)
        else:
            # Standard ResNet block - two 3x3 convolutions
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                                 padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        if self.bottleneck:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class MultiscaleLayer3D(nn.Module):
    """A multiscale feature extraction layer using parallel convolutions.
    
    This layer performs parallel convolutions with different kernel sizes and 
    dilations to capture features at multiple scales, then concatenates 
    the results.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_sizes: List[int] = [3, 5],
        dilations: List[int] = [1, 2, 3]
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per parallel path
            kernel_sizes: List of kernel sizes for parallel paths
            dilations: List of dilation rates for parallel paths
        """
        super().__init__()
        
        self.paths = nn.ModuleList()
        total_paths = len(kernel_sizes) * len(dilations)
        path_channels = out_channels // total_paths
        
        # Ensure we use all output channels by assigning the remainder to the first path
        remainder = out_channels - (path_channels * total_paths)
        
        path_count = 0
        for k in kernel_sizes:
            for d in dilations:
                # Determine output channels for this path
                if path_count == 0:
                    path_out_channels = path_channels + remainder
                else:
                    path_out_channels = path_channels
                
                # Calculate padding to maintain spatial dimensions
                padding = d * (k - 1) // 2
                
                # Create the convolutional path
                path = nn.Sequential(
                    nn.Conv3d(in_channels, path_out_channels, kernel_size=k, 
                              padding=padding, dilation=d),
                    nn.BatchNorm3d(path_out_channels),
                    nn.ReLU(inplace=True)
                )
                self.paths.append(path)
                path_count += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [path(x) for path in self.paths]
        return torch.cat(outputs, dim=1)

class ProtFlexCNN(nn.Module):
    """3D CNN model for protein flexibility prediction.
    
    This model architecture is specifically designed for predicting residue flexibility
    from voxelized protein structures. It employs 3D convolutions with residual
    connections and multiscale feature extraction to capture both local atomic
    arrangements and global structural contexts.
    """
    def __init__(
        self, 
        input_channels: int = 5,  # C, N, O, CA, CB channels
        channel_growth_rate: float = 1.5,
        num_residual_blocks: int = 4,
        use_multiscale: bool = True,
        use_bottleneck: bool = True,
        dropout_rate: float = 0.2,
        include_metadata: bool = False,
        metadata_features: int = 0
    ):
        """
        Args:
            input_channels: Number of input channels (default 5 for CNOCBCA encoding)
            channel_growth_rate: Factor by which channel count increases in deeper layers
            num_residual_blocks: Number of residual blocks in the network
            use_multiscale: Whether to use multiscale feature extraction
            use_bottleneck: Whether to use bottleneck architecture in residual blocks
            dropout_rate: Dropout probability in fully connected layers
            include_metadata: Whether to include additional metadata features
            metadata_features: Number of metadata features to include
        """
        super().__init__()
        
        self.include_metadata = include_metadata
        self.metadata_features = metadata_features
        
        # Initial 3D convolution for feature extraction
        base_channels = 32
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction using residual blocks and multiscale convolutions
        self.feature_extractor = nn.ModuleList()
        in_channels = base_channels
        
        for i in range(num_residual_blocks):
            # Calculate output channels with growth rate
            out_channels = int(in_channels * channel_growth_rate)
            
            # Decide on stride for downsampling
            # Downsample after the 1st and 3rd residual block
            stride = 2 if i in [1, 3] else 1
            
            # Add a multiscale feature extraction layer
            if use_multiscale and i < num_residual_blocks - 1:
                self.feature_extractor.append(
                    MultiscaleLayer3D(in_channels, out_channels, 
                                     kernel_sizes=[3, 5], 
                                     dilations=[1, 2, 3])
                )
                in_channels = out_channels
            
            # Add residual block
            self.feature_extractor.append(
                ResidualBlock3D(in_channels, out_channels, 
                               kernel_size=3, stride=stride,
                               dilation=1, bottleneck=use_bottleneck)
            )
            in_channels = out_channels
        
        # Global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers for regression
        fc_input_size = in_channels
        if include_metadata:
            fc_input_size += metadata_features
            
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Output a single RMSF value
        )
        
    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape [batch_size, channels, depth, height, width]
            metadata: Optional tensor of additional features
            
        Returns:
            Tensor of RMSF predictions with shape [batch_size, 1]
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Feature extraction blocks
        for layer in self.feature_extractor:
            x = layer(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Combine with metadata if provided
        if self.include_metadata and metadata is not None:
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.cat([x, metadata], dim=1)
        
        # Fully connected layers for regression
        x = self.fc_layers(x)
        
        return x

class DilatedResNet3D(nn.Module):
    """A more advanced 3D ResNet with dilated convolutions specialized for protein structures.
    
    This model focuses heavily on capturing long-range interactions through extensive
    use of dilated convolutions while maintaining high-resolution feature maps.
    """
    def __init__(
        self,
        input_channels: int = 5,
        init_features: int = 32,
        block_config: List[int] = [2, 2, 2, 2],
        dilations: List[int] = [1, 2, 4, 8],
        include_metadata: bool = False,
        metadata_features: int = 0,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            input_channels: Number of input channels
            init_features: Initial number of feature maps
            block_config: Number of residual blocks in each stage
            dilations: Dilation rates for each stage
            include_metadata: Whether to include additional metadata features
            metadata_features: Number of metadata features
            dropout_rate: Dropout probability in fully connected layers
        """
        super().__init__()
        
        self.include_metadata = include_metadata
        self.metadata_features = metadata_features
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, init_features, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True)
        )
        
        # Create residual stages
        self.stages = nn.ModuleList()
        in_channels = init_features
        
        for i, (num_blocks, dilation) in enumerate(zip(block_config, dilations)):
            # Double the number of features at each stage
            out_channels = in_channels * 2 if i > 0 else in_channels
            
            # First block of each stage may include strided convolution for downsampling
            # We only downsample in the first two stages to preserve spatial information
            stride = 2 if i < 2 else 1
            
            # Create a sequential container for this stage
            stage = nn.Sequential()
            
            # Add the first block with potential stride
            stage.add_module(
                f"block_{i}_0",
                ResidualBlock3D(in_channels, out_channels, stride=stride, 
                               dilation=dilation, bottleneck=True)
            )
            
            # Add the remaining blocks
            for j in range(1, num_blocks):
                stage.add_module(
                    f"block_{i}_{j}",
                    ResidualBlock3D(out_channels, out_channels, 
                                   dilation=dilation, bottleneck=True)
                )
            
            self.stages.append(stage)
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers for regression
        fc_input_size = in_channels
        if include_metadata:
            fc_input_size += metadata_features
            
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Output a single RMSF value
        )
    
    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the network."""
        # Initial convolution
        x = self.conv1(x)
        
        # Process through all stages
        for stage in self.stages:
            x = stage(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Combine with metadata if provided
        if self.include_metadata and metadata is not None:
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.cat([x, metadata], dim=1)
        
        # Regression head
        x = self.fc_layers(x)
        
        return x

class MultipathRMSFNet(nn.Module):
    """A CNN with multiple parallel paths to capture different aspects of protein structure.
    
    This network uses separate pathways to process different atom types and structural scales,
    then combines them for the final prediction. This design is inspired by how different
    structural elements (backbone geometry, side chain orientation, etc.) contribute to
    protein flexibility.
    """
    def __init__(
        self,
        input_channels: int = 5,
        base_filters: int = 32,
        include_metadata: bool = False,
        metadata_features: int = 0,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            input_channels: Number of input channels
            base_filters: Base number of filters for convolutional layers
            include_metadata: Whether to include additional metadata features
            metadata_features: Number of metadata features
            dropout_rate: Dropout probability in fully connected layers
        """
        super().__init__()
        
        self.include_metadata = include_metadata
        self.metadata_features = metadata_features
        
        # We'll create three parallel paths with different kernel and dilation configurations
        # Path 1: Focus on local features with small kernels
        self.local_path = nn.Sequential(
            nn.Conv3d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_filters, base_filters * 2, kernel_size=3, stride=2),
            ResidualBlock3D(base_filters * 2, base_filters * 2),
            ResidualBlock3D(base_filters * 2, base_filters * 4, stride=2),
            ResidualBlock3D(base_filters * 4, base_filters * 4)
        )
        
        # Path 2: Focus on medium-range interactions with medium kernels and dilations
        self.medium_path = nn.Sequential(
            nn.Conv3d(input_channels, base_filters, kernel_size=5, padding=2),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_filters, base_filters * 2, kernel_size=3, stride=2, dilation=2),
            ResidualBlock3D(base_filters * 2, base_filters * 2, dilation=2),
            ResidualBlock3D(base_filters * 2, base_filters * 4, stride=2),
            ResidualBlock3D(base_filters * 4, base_filters * 4, dilation=2)
        )
        
        # Path 3: Focus on long-range interactions with large dilations
        self.global_path = nn.Sequential(
            nn.Conv3d(input_channels, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_filters, base_filters * 2, kernel_size=3, stride=2, dilation=4),
            ResidualBlock3D(base_filters * 2, base_filters * 2, dilation=4),
            ResidualBlock3D(base_filters * 2, base_filters * 4, stride=2),
            ResidualBlock3D(base_filters * 4, base_filters * 4, dilation=4)
        )
        
        # Global pooling for each path
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # Combine features from all paths
        combined_features = base_filters * 4 * 3  # 3 paths with base_filters * 4 features each
        if include_metadata:
            combined_features += metadata_features
        
        # Fully connected layers for regression
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Output a single RMSF value
        )
    
    def forward(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the network."""
        # Process through each path
        local_features = self.pool(self.local_path(x))
        medium_features = self.pool(self.medium_path(x))
        global_features = self.pool(self.global_path(x))
        
        # Concatenate features from all paths
        combined = torch.cat([
            local_features.view(local_features.size(0), -1),
            medium_features.view(medium_features.size(0), -1),
            global_features.view(global_features.size(0), -1)
        ], dim=1)
        
        # Combine with metadata if provided
        if self.include_metadata and metadata is not None:
            combined = torch.cat([combined, metadata], dim=1)
        
        # Final prediction
        return self.fc_layers(combined)

# Factory function to create models based on configuration
def create_model(model_name: str, model_params: Dict) -> nn.Module:
    """
    Factory function to create a model instance based on name and parameters.
    
    Args:
        model_name: Name of the model architecture to use
        model_params: Dictionary of model parameters
        
    Returns:
        Instance of the requested model
    """
    if model_name == "protflex_cnn":
        return ProtFlexCNN(**model_params)
    elif model_name == "dilated_resnet3d":
        return DilatedResNet3D(**model_params)
    elif model_name == "multipath_rmsf_net":
        return MultipathRMSFNet(**model_params)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")