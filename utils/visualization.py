"""
Visualization utilities for ProtFlex.

This module provides functions for creating visualizations to analyze model performance
and protein flexibility predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    lr: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 6),
    dpi: int = 300
) -> None:
    """
    Plot training and validation loss curves with optional learning rate.
    
    Args:
        train_loss: List of training loss values
        val_loss: List of validation loss values
        lr: List of learning rates
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    epochs = list(range(1, len(train_loss) + 1))
    
    fig, ax1 = plt.subplots(figsize=fig_size)
    
    # Plot loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Add learning rate on secondary y-axis if provided
    if lr is not None:
        ax2 = ax1.twinx()
        ax2.plot(epochs, lr, 'g--', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
    # Add legend and title
    lines1, labels1 = ax1.get_legend_handles_labels()
    if lr is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    plt.title('Loss and Learning Rate vs. Epoch')
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 10),
    dpi: int = 300,
    max_points: int = 1000
) -> None:
    """
    Create a scatter plot of predicted vs. actual RMSF values.
    
    Args:
        predictions: Array of predicted RMSF values
        targets: Array of actual RMSF values
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
        max_points: Maximum number of points to plot
    """
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # Limit the number of points for clearer visualization
    if len(predictions) > max_points:
        indices = np.random.choice(len(predictions), max_points, replace=False)
        predictions = predictions[indices]
        targets = targets[indices]
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Create scatter plot
    scatter = plt.scatter(targets, predictions, alpha=0.6, s=20, c='blue')
    
    # Add identity line
    min_val = min(min(predictions), min(targets))
    max_val = max(max(predictions), max(targets))
    buffer = (max_val - min_val) * 0.05  # 5% buffer
    plt.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'r--')
    
    # Add metrics as text
    plt.text(
        0.05, 0.95, 
        f'RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Add labels and title
    plt.xlabel('Actual RMSF')
    plt.ylabel('Predicted RMSF')
    plt.title('Predicted vs. Actual RMSF Values')
    
    # Set equal aspect ratio
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> None:
    """
    Plot the distribution of prediction errors.
    
    Args:
        predictions: Array of predicted RMSF values
        targets: Array of actual RMSF values
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    # Calculate errors
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    # Plot histograms
    sns.histplot(errors, bins=30, kde=True, ax=ax1)
    ax1.set_xlabel('Prediction Error (Predicted - Actual)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Errors')
    
    # Add error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    
    ax1.text(
        0.05, 0.95,
        f'Mean: {mean_error:.4f}\nStd Dev: {std_error:.4f}\nMedian: {median_error:.4f}',
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Plot error vs. actual value
    scatter = ax2.scatter(targets, abs_errors, alpha=0.6, s=10, c='blue')
    ax2.set_xlabel('Actual RMSF')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Absolute Error vs. Actual RMSF')
    
    # Add a trend line using polynomial regression
    z = np.polyfit(targets, abs_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(targets), max(targets), 100)
    ax2.plot(x_line, p(x_line), 'r--', linewidth=2)
    
    # Add correlation coefficient
    corr = np.corrcoef(targets, abs_errors)[0, 1]
    ax2.text(
        0.05, 0.95,
        f'Correlation: {corr:.4f}',
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_residue_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    residue_types: List[str],
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (14, 10),
    dpi: int = 300
) -> None:
    """
    Analyze prediction performance by residue type.
    
    Args:
        predictions: Array of predicted RMSF values
        targets: Array of actual RMSF values
        residue_types: List of residue types (3-letter codes)
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    # Calculate errors
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    # Create a DataFrame for analysis
    df = pd.DataFrame({
        'Residue': residue_types,
        'Actual': targets,
        'Predicted': predictions,
        'Error': errors,
        'AbsError': abs_errors
    })
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=fig_size)
    
    # 1. Mean Absolute Error by Residue Type
    residue_mae = df.groupby('Residue')['AbsError'].mean().sort_values(ascending=False)
    residue_mae.plot(kind='bar', ax=ax1)
    ax1.set_xlabel('Residue Type')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Mean Absolute Error by Residue Type')
    
    # 2. Mean RMSF by Residue Type (Actual vs. Predicted)
    residue_means = df.groupby('Residue').agg({
        'Actual': 'mean',
        'Predicted': 'mean'
    }).sort_values(by='Actual', ascending=False)
    
    residue_means.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Residue Type')
    ax2.set_ylabel('Mean RMSF')
    ax2.set_title('Mean RMSF by Residue Type (Actual vs. Predicted)')
    
    # 3. Boxplot of Absolute Errors by Residue Type
    # Get the top 10 residues with highest mean absolute errors
    top_residues = residue_mae.head(10).index.tolist()
    sns.boxplot(x='Residue', y='AbsError', data=df[df['Residue'].isin(top_residues)], ax=ax3)
    ax3.set_xlabel('Residue Type')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Distribution of Absolute Errors for Top 10 Residues')
    
    # 4. Correlation between Actual RMSF and Error
    for residue in df['Residue'].unique():
        residue_df = df[df['Residue'] == residue]
        ax4.scatter(residue_df['Actual'], residue_df['AbsError'], alpha=0.5, label=residue)
    
    # Add a trend line for all data
    z = np.polyfit(df['Actual'], df['AbsError'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(df['Actual']), max(df['Actual']), 100)
    ax4.plot(x_line, p(x_line), 'r--', linewidth=2)
    
    ax4.set_xlabel('Actual RMSF')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Absolute Error vs. Actual RMSF by Residue Type')
    
    # Handle the legend - only show a subset if there are many residue types
    if len(df['Residue'].unique()) > 6:
        ax4.legend().set_visible(False)
    else:
        ax4.legend(loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_temperature_comparison(
    temperatures: List[int],
    metrics: Dict[int, Dict[str, float]],
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 8),
    dpi: int = 300
) -> None:
    """
    Compare model performance across different temperatures.
    
    Args:
        temperatures: List of temperature values
        metrics: Dictionary mapping temperatures to metrics
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    # Extract metrics
    mse_values = [metrics[temp].get('mse', 0) for temp in temperatures]
    mae_values = [metrics[temp].get('mae', 0) for temp in temperatures]
    r2_values = [metrics[temp].get('r2', 0) for temp in temperatures]
    
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size)
    
    # Plot MSE by temperature
    ax1.plot(temperatures, mse_values, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE vs. Temperature')
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE by temperature
    ax2.plot(temperatures, mae_values, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('MAE vs. Temperature')
    ax2.grid(True, alpha=0.3)
    
    # Plot R² by temperature
    ax3.plot(temperatures, r2_values, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('R² Score')
    ax3.set_title('R² vs. Temperature')
    ax3.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> None:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Create bar plot
    sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_atom_channel_examples(
    voxel_data: np.ndarray,
    channel_names: List[str] = ['C', 'N', 'O', 'CA', 'CB'],
    slice_idx: int = None,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (15, 3),
    dpi: int = 300
) -> None:
    """
    Visualize atom channel slices from a voxel grid.
    
    Args:
        voxel_data: Voxel grid data with shape [channels, depth, height, width]
        channel_names: Names of the channels
        slice_idx: Index of the slice to visualize (default: middle slice)
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    num_channels = voxel_data.shape[0]
    depth = voxel_data.shape[1]
    
    # If no slice index is provided, use the middle slice
    if slice_idx is None:
        slice_idx = depth // 2
    
    # Create figure
    fig, axes = plt.subplots(1, num_channels, figsize=fig_size)
    
    # Plot each channel
    for i in range(num_channels):
        channel_slice = voxel_data[i, slice_idx, :, :]
        
        # Plot the slice
        im = axes[i].imshow(channel_slice, cmap='viridis')
        axes[i].set_title(f'{channel_names[i]} Atoms')
        axes[i].axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Add overall title
    plt.suptitle(f'Voxel Grid Slice (z={slice_idx})', fontsize=16)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_rmsf_profiles(
    domain_id: str,
    residue_ids: List[int],
    actual_rmsf: List[float],
    predicted_rmsf: List[float],
    residue_names: Optional[List[str]] = None,
    secondary_structure: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 6),
    dpi: int = 300
) -> None:
    """
    Plot actual and predicted RMSF profiles for a protein domain.
    
    Args:
        domain_id: Protein domain identifier
        residue_ids: List of residue IDs
        actual_rmsf: List of actual RMSF values
        predicted_rmsf: List of predicted RMSF values
        residue_names: List of residue names (optional)
        secondary_structure: List of secondary structure assignments (optional)
        output_path: Path to save the figure
        fig_size: Figure size
        dpi: Figure resolution
    """
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot RMSF profiles
    ax.plot(residue_ids, actual_rmsf, '-', label='Actual RMSF', color='blue')
    ax.plot(residue_ids, predicted_rmsf, '--', label='Predicted RMSF', color='red')
    
    # Add secondary structure bars if provided
    if secondary_structure is not None:
        # Initialize secondary structure regions
        ss_regions = []
        current_ss = secondary_structure[0]
        start_idx = residue_ids[0]
        
        for i, ss in enumerate(secondary_structure[1:], 1):
            if ss != current_ss:
                ss_regions.append((current_ss, start_idx, residue_ids[i-1]))
                current_ss = ss
                start_idx = residue_ids[i]
        
        # Add the last region
        ss_regions.append((current_ss, start_idx, residue_ids[-1]))
        
        # Plot secondary structure as horizontal bars
        y_pos = min(actual_rmsf + predicted_rmsf) - 0.1
        height = 0.05
        
        ss_colors = {
            'H': 'red',       # Alpha helix
            'G': 'orange',    # 3-10 helix
            'I': 'yellow',    # Pi helix
            'E': 'blue',      # Extended strand
            'B': 'purple',    # Beta bridge
            'T': 'green',     # Turn
            'S': 'pink',      # Bend
            'C': 'gray',      # Coil
            'L': 'gray'       # Loop
        }
        
        for ss, start, end in ss_regions:
            color = ss_colors.get(ss, 'gray')
            ax.axhspan(y_pos, y_pos + height, xmin=(start - residue_ids[0])/(residue_ids[-1] - residue_ids[0]),
                     xmax=(end - residue_ids[0])/(residue_ids[-1] - residue_ids[0]), color=color, alpha=0.3)
        
        # Add legend for secondary structure
        import matplotlib.patches as mpatches
        ss_patches = []
        for ss, color in ss_colors.items():
            if any(region[0] == ss for region in ss_regions):
                ss_patches.append(mpatches.Patch(color=color, alpha=0.3, label=f'{ss}'))
        
        # Add the secondary structure legend below the main legend
        ax.legend(handles=ss_patches, loc='upper right', bbox_to_anchor=(1, 0.5), title='Secondary Structure')
    
    # Add labels and title
    ax.set_xlabel('Residue ID')
    ax.set_ylabel('RMSF')
    ax.set_title(f'RMSF Profile for Domain {domain_id}')
    
    # Add the main legend
    ax.legend(loc='upper right')
    
    # Annotate residue names if provided and not too many
    if residue_names is not None and len(residue_ids) <= 50:
        for i, (res_id, res_name) in enumerate(zip(residue_ids, residue_names)):
            if i % 5 == 0:  # Annotate every 5th residue to avoid crowding
                ax.text(res_id, max(actual_rmsf[i], predicted_rmsf[i]) + 0.05, res_name, 
                       ha='center', fontsize=8, rotation=90)
    
    # Grid and tight layout
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()