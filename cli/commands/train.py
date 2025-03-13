"""
Training command for ProtFlex CLI.

This module provides the command to train protein flexibility prediction models.
"""

import os
import argparse
import logging
import json
from typing import Dict, Any, Optional

import torch
import numpy as np

from protflex.models import cnn_models
from protflex.models.loss import create_loss_function
from protflex.data.data_loader import create_data_loaders
from protflex.training.trainer import (
    RMSFTrainer,
    create_optimizer,
    create_scheduler
)
from protflex.utils.file_utils import ensure_dir, save_model_summary

logger = logging.getLogger(__name__)

def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a subparser for the train command.

    Args:
        subparsers: Subparsers object

    Returns:
        Parser for the train command
    """
    parser = subparsers.add_parser(
        'train',
        help='Train a protein flexibility prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                      help='Directory containing data (overrides config)')
    parser.add_argument('--temperature', type=str,
                      help='Temperature to use for training (overrides config)')
    parser.add_argument('--domain-ids', type=str, nargs='+',
                      help='List of domain IDs to train on (if not specified, all available domains will be used)')

    # Model arguments
    parser.add_argument('--model', type=str, choices=['protflex_cnn', 'dilated_resnet3d', 'multipath_rmsf_net'],
                      help='Model architecture to use (overrides config)')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint to resume training from')

    # Training arguments
    parser.add_argument('--batch-size', type=int,
                      help='Batch size for training (overrides config)')
    parser.add_argument('--epochs', type=int,
                      help='Number of epochs to train (overrides config)')
    parser.add_argument('--learning-rate', type=float,
                      help='Learning rate (overrides config)')
    parser.add_argument('--loss-function', type=str, 
                      choices=['mse', 'mae', 'rmsf', 'weighted_rmsf', 'elastic_rmsf'],
                      help='Loss function to use (overrides config)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str,
                      help='Directory to save results (overrides config)')
    parser.add_argument('--no-eval', action='store_true',
                      help='Skip evaluation on test set after training')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU even if GPU is available')

    return parser

def run(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Run the train command.

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
    if args.domain_ids:
        config['input']['domain_ids'] = args.domain_ids
    if args.model:
        config['model']['architecture'] = args.model
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.output_dir:
        config['output']['base_dir'] = args.output_dir
    if args.loss_function:
        config['training']['loss_function'] = args.loss_function

    # Set up directories
    data_dir = config['input']['data_dir']
    voxel_dir = os.path.join(data_dir, config['input']['voxel_dir'])
    rmsf_dir = os.path.join(data_dir, config['input']['rmsf_dir'])
    output_dir = config['output']['base_dir']
    model_dir = os.path.join(output_dir, config['output']['model_dir'])
    
    # Create output directories
    ensure_dir(output_dir)
    ensure_dir(model_dir)
    
    # Set up device (CPU/GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = config['training']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            voxel_dir=voxel_dir,
            rmsf_dir=rmsf_dir,
            temperature=config['input']['temperature'],
            domain_ids=config['input']['domain_ids'],
            use_metadata=config['input']['use_metadata'],
            metadata_fields=config['input']['metadata_fields'],
            batch_size=config['training']['batch_size'],
            train_split=config['training']['train_split'],
            val_split=config['training']['val_split'],
            test_split=config['training']['test_split'],
            random_seed=config['training']['random_seed'],
            use_augmentation=config['training'].get('use_augmentation', True),
            augmentation_params=config['training'].get('augmentation_params', None)
        )
        
        # Get RMSF scaler for later de-normalization
        rmsf_scaler = getattr(train_loader.dataset.dataset, 'rmsf_scaler', None)
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return 1
    
    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
    try:
        # Get model parameters from config
        model_params = config['model'].copy()
        model_params.pop('architecture', None)  # Remove architecture key
        
        # Create model instance
        model = cnn_models.create_model(config['model']['architecture'], model_params)
        model.to(device)
        
        # Load checkpoint if specified
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save model summary
        summary_file = os.path.join(model_dir, "model_summary.txt")
        save_model_summary(model, summary_file)
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return 1
    
    # Create optimizer
    logger.info("Creating optimizer and scheduler...")
    try:
        optimizer = create_optimizer(
            model=model,
            optimizer_name=config['training'].get('optimizer', 'adam'),
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name=config['training']['lr_scheduler'],
            scheduler_params=config['training'].get('lr_scheduler_params', {})
        )
        
        # Create loss function
        loss_fn = create_loss_function(
            config['training'].get('loss_function', 'mse')
        )
        
    except Exception as e:
        logger.error(f"Error creating optimizer, scheduler, or loss function: {e}")
        return 1
    
    # Create trainer
    logger.info("Setting up trainer...")
    try:
        trainer = RMSFTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None if args.no_eval else test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            output_dir=model_dir,
            rmsf_scaler=rmsf_scaler
        )
        
    except Exception as e:
        logger.error(f"Error creating trainer: {e}")
        return 1
    
    # Train the model
    logger.info("Starting training...")
    try:
        results = trainer.train(
            num_epochs=config['training']['num_epochs'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        # Save training metadata
        metadata = {
            "model_architecture": config['model']['architecture'],
            "temperature": config['input']['temperature'],
            "training_time": results['training_time'],
            "best_val_loss": results['best_val_loss'],
            "final_metrics": results['final_metrics'],
            "training_params": {
                "batch_size": config['training']['batch_size'],
                "learning_rate": config['training']['learning_rate'],
                "optimizer": config['training'].get('optimizer', 'adam'),
                "scheduler": config['training']['lr_scheduler'],
                "loss_function": config['training'].get('loss_function', 'mse'),
                "epochs": config['training']['num_epochs'],
                "early_stopping_patience": config['training']['early_stopping_patience']
            },
            "model_params": model_params,
            "rmsf_stats": train_loader.dataset.dataset.get_rmsf_stats() if hasattr(train_loader.dataset.dataset, 'get_rmsf_stats') else {}
        }
        
        metadata_file = os.path.join(model_dir, "training_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Training completed. Results saved to {model_dir}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1
    
    return 0