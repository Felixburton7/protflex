"""
Evaluation command for ProtFlex CLI.

This module provides the command to evaluate a protein flexibility prediction model.
"""

import os
import argparse
import logging
import json
from typing import Dict, Any, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from protflex.models import cnn_models
from protflex.data.data_loader import create_data_loaders
from protflex.utils.visualization import (
    plot_predictions,
    plot_error_distribution,
    plot_residue_analysis
)

logger = logging.getLogger(__name__)

def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a subparser for the evaluate command.

    Args:
        subparsers: Subparsers object

    Returns:
        Parser for the evaluate command
    """
    parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained protein flexibility prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str,
                      help='Directory containing data (overrides config)')
    parser.add_argument('--temperature', type=str,
                      help='Temperature to use for evaluation (overrides config)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')

    # Output arguments
    parser.add_argument('--output-dir', type=str,
                      help='Directory to save evaluation results (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU even if GPU is available')

    return parser

def run(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Run the evaluate command.

    Args:
        args: Command line arguments
        config: Configuration dictionary

    Returns:
        Exit code
    """
    # Override config with command-line arguments
    if args.data_dir:
        config['input']['data_dir'] = args.data_dir
    if args.temperature:
        config['input']['temperature'] = args.temperature
    if args.output_dir:
        config['output']['base_dir'] = args.output_dir

    # Set up directories
    data_dir = config['input']['data_dir']
    voxel_dir = os.path.join(data_dir, config['input']['voxel_dir'])
    rmsf_dir = os.path.join(data_dir, config['input']['rmsf_dir'])
    output_dir = config['output']['base_dir']
    results_dir = os.path.join(output_dir, config['output']['results_dir'])
    eval_dir = os.path.join(results_dir, 'evaluation')

    # Create output directories
    os.makedirs(eval_dir, exist_ok=True)

    # Load model checkpoint
    logger.info(f"Loading model from {args.model_path}")

    # Determine device (CPU/GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model_name = checkpoint.get('model_name', config['model']['architecture'])
        model_params = config['model'].copy()
        model_params.pop('architecture', None)

        # Create model
        model = cnn_models.create_model(model_name, model_params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Get temperature from checkpoint or config
        temperature = checkpoint.get('temperature', config['input']['temperature'])

        logger.info(f"Model loaded successfully. Using temperature: {temperature}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return 1

    # Create data loader for evaluation
    logger.info("Creating evaluation data loader...")
    _, _, test_loader = create_data_loaders(
        voxel_dir=voxel_dir,
        rmsf_dir=rmsf_dir,
        temperature=temperature,
        domain_ids=config['input']['domain_ids'],
        use_metadata=config['input']['use_metadata'],
        metadata_fields=config['input']['metadata_fields'],
        batch_size=args.batch_size,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,  # Use all data for testing
        random_seed=config['training']['random_seed'],
        use_augmentation=False
    )

    # Get RMSF scaler from dataset
    rmsf_scaler = getattr(test_loader.dataset.dataset, 'rmsf_scaler', None)

    # Evaluate the model
    logger.info("Evaluating model...")
    all_targets = []
    all_preds = []
    all_residue_types = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data to device
            if isinstance(data, tuple):
                voxel_data, metadata = data
                voxel_data = voxel_data.to(device)
                metadata = metadata.to(device)
                data = (voxel_data, metadata)
            else:
                data = data.to(device)

            # Forward pass
            if isinstance(data, tuple):
                output = model(data[0], data[1])
            else:
                output = model(data)

            # Collect predictions and targets
            targets_np = target.numpy()
            preds_np = output.cpu().numpy()

            # Denormalize if needed
            if rmsf_scaler is not None:
                targets_np = rmsf_scaler.inverse_transform(targets_np)
                preds_np = rmsf_scaler.inverse_transform(preds_np)

            all_targets.append(targets_np)
            all_preds.append(preds_np)

            # Collect residue types if available
            # This is a simplification - adapt based on your dataset
            if hasattr(test_loader.dataset.dataset, 'samples'):
                batch_indices = list(range(
                    batch_idx * test_loader.batch_size,
                    min((batch_idx + 1) * test_loader.batch_size, len(test_loader.dataset))
                ))
                for idx in batch_indices:
                    sample = test_loader.dataset.dataset.samples[idx]
                    if 'metadata' in sample and 'resname' in sample['metadata']:
                        all_residue_types.append(sample['metadata']['resname'])

    # Combine results
    all_targets = np.vstack(all_targets).flatten()
    all_preds = np.vstack(all_preds).flatten()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    # Log results
    logger.info(f"Evaluation metrics:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R²: {r2:.4f}")

    # Save metrics to file
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'num_samples': int(len(all_targets)),
        'temperature': temperature
    }

    metrics_file = os.path.join(eval_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Create visualizations
    logger.info("Creating visualizations...")

    # Predictions vs. Actual
    pred_plot_file = os.path.join(eval_dir, 'predictions.png')
    plot_predictions(
        predictions=all_preds,
        targets=all_targets,
        output_path=pred_plot_file,
        max_points=config['visualization'].get('max_scatter_points', 1000)
    )
    logger.info(f"Saved predictions plot to {pred_plot_file}")

    # Error distribution
    error_plot_file = os.path.join(eval_dir, 'error_distribution.png')
    plot_error_distribution(
        predictions=all_preds,
        targets=all_targets,
        output_path=error_plot_file
    )
    logger.info(f"Saved error distribution plot to {error_plot_file}")

    # Residue type analysis
    if all_residue_types:
        residue_plot_file = os.path.join(eval_dir, 'residue_analysis.png')
        plot_residue_analysis(
            predictions=all_preds,
            targets=all_targets,
            residue_types=all_residue_types,
            output_path=residue_plot_file
        )
        logger.info(f"Saved residue type analysis plot to {residue_plot_file}")

    logger.info("Evaluation completed successfully")
    return 0
