"""
Model training and evaluation for ProtFlex.

This module provides utilities for training and evaluating models that predict
protein flexibility (RMSF) from voxelized protein structures.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

from protflex.utils.visualization import (
    plot_loss_curves,
    plot_predictions,
    plot_residue_analysis,
    plot_error_distribution
)

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            current_score: Current validation score
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False

class MetricTracker:
    """Track and compute metrics during training and evaluation."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics or ["loss", "mse", "mae", "r2"]
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.values = {metric: [] for metric in self.metrics}
        self.counts = {metric: 0 for metric in self.metrics}
        self.running_total = {metric: 0.0 for metric in self.metrics}
    
    def update(self, metric: str, value: float, batch_size: int = 1) -> None:
        """
        Update a metric.
        
        Args:
            metric: Name of the metric
            value: Metric value
            batch_size: Batch size for weighted averaging
        """
        if metric not in self.metrics:
            self.metrics.append(metric)
            self.values[metric] = []
            self.counts[metric] = 0
            self.running_total[metric] = 0.0
        
        self.values[metric].append(value)
        self.running_total[metric] += value * batch_size
        self.counts[metric] += batch_size
    
    def avg(self, metric: str) -> float:
        """
        Get the average value for a metric.
        
        Args:
            metric: Name of the metric
            
        Returns:
            Average value of the metric
        """
        if self.counts[metric] == 0:
            return 0.0
        return self.running_total[metric] / self.counts[metric]
    
    def result(self) -> Dict[str, float]:
        """
        Get the averaged results for all metrics.
        
        Returns:
            Dictionary mapping metric names to average values
        """
        return {metric: self.avg(metric) for metric in self.metrics}
    
    def history(self, metric: str) -> List[float]:
        """
        Get the history of values for a metric.
        
        Args:
            metric: Name of the metric
            
        Returns:
            List of values for the metric
        """
        return self.values.get(metric, [])

class RMSFTrainer:
    """Trainer for RMSF prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "results",
        rmsf_scaler = None
    ):
        """
        Args:
            model: Neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Device for training
            output_dir: Directory to save results
            rmsf_scaler: Scaler used to normalize RMSF values
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set up device
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        
        # Set up training components
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Set up tracking
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        self.test_metrics = MetricTracker()
        
        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": []
        }
        
        # Save best model state
        self.best_val_loss = float("inf")
        self.best_model_path = os.path.join(self.output_dir, "best_model.pth")
        
        # Save predictions
        self.all_preds = []
        self.all_targets = []
        
        # Save RMSF scaler for denormalization
        self.rmsf_scaler = rmsf_scaler
        
        logger.info(f"Initialized trainer with model: {type(model).__name__}")
        logger.info(f"Training on device: {self.device}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Reset metrics
        self.train_metrics.reset()
        
        # Training loop
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for i, (data, target) in enumerate(pbar):
            # Move data to device
            if isinstance(data, tuple):
                voxel_data, metadata = data
                voxel_data = voxel_data.to(self.device)
                metadata = metadata.to(self.device)
                data = (voxel_data, metadata)
            else:
                data = data.to(self.device)
            
            target = target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(data, tuple):
                output = self.model(data[0], data[1])
            else:
                output = self.model(data)
            
            # Calculate loss
            loss = self.loss_fn(output, target)
            
            # Backward pass and update
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = target.size(0)
            self.train_metrics.update("loss", loss.item(), batch_size)
            
            # Convert to NumPy arrays for metrics calculation
            output_np = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            
            # Calculate additional metrics
            mse = mean_squared_error(target_np, output_np)
            mae = mean_absolute_error(target_np, output_np)
            
            # R² can be undefined if all targets are identical (zero variance)
            try:
                r2 = r2_score(target_np, output_np)
            except:
                r2 = 0.0
            
            self.train_metrics.update("mse", mse, batch_size)
            self.train_metrics.update("mae", mae, batch_size)
            self.train_metrics.update("r2", r2, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{self.train_metrics.avg('loss'):.4f}",
                "mse": f"{self.train_metrics.avg('mse'):.4f}",
                "mae": f"{self.train_metrics.avg('mae'):.4f}",
                "r2": f"{self.train_metrics.avg('r2'):.4f}"
            })
        
        # Log results
        metrics = self.train_metrics.result()
        logger.info(f"Epoch {epoch} - Train: loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}, "
                   f"mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}")
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Reset metrics
        self.val_metrics.reset()
        
        # Validation loop
        with torch.no_grad():
            for data, target in self.val_loader:
                # Move data to device
                if isinstance(data, tuple):
                    voxel_data, metadata = data
                    voxel_data = voxel_data.to(self.device)
                    metadata = metadata.to(self.device)
                    data = (voxel_data, metadata)
                else:
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Forward pass
                if isinstance(data, tuple):
                    output = self.model(data[0], data[1])
                else:
                    output = self.model(data)
                
                # Calculate loss
                loss = self.loss_fn(output, target)
                
                # Update metrics
                batch_size = target.size(0)
                self.val_metrics.update("loss", loss.item(), batch_size)
                
                # Convert to NumPy arrays for metrics calculation
                output_np = output.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                
                # Calculate additional metrics
                mse = mean_squared_error(target_np, output_np)
                mae = mean_absolute_error(target_np, output_np)
                
                try:
                    r2 = r2_score(target_np, output_np)
                except:
                    r2 = 0.0
                
                self.val_metrics.update("mse", mse, batch_size)
                self.val_metrics.update("mae", mae, batch_size)
                self.val_metrics.update("r2", r2, batch_size)
        
        # Log results
        metrics = self.val_metrics.result()
        logger.info(f"Epoch {epoch} - Validation: loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}, "
                   f"mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}")
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """
        Test the model on the test dataset.
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            logger.warning("No test loader provided. Skipping test.")
            return {}
        
        # Load best model if available
        if os.path.exists(self.best_model_path):
            logger.info(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(torch.load(self.best_model_path))
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Reset metrics
        self.test_metrics.reset()
        
        # Collect all predictions and targets for later analysis
        all_outputs = []
        all_targets = []
        
        # Test loop
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                if isinstance(data, tuple):
                    voxel_data, metadata = data
                    voxel_data = voxel_data.to(self.device)
                    metadata = metadata.to(self.device)
                    data = (voxel_data, metadata)
                else:
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Forward pass
                if isinstance(data, tuple):
                    output = self.model(data[0], data[1])
                else:
                    output = self.model(data)
                
                # Calculate loss
                loss = self.loss_fn(output, target)
                
                # Update metrics
                batch_size = target.size(0)
                self.test_metrics.update("loss", loss.item(), batch_size)
                
                # Convert to NumPy arrays for metrics calculation
                output_np = output.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                
                # Collect predictions and targets
                all_outputs.append(output_np)
                all_targets.append(target_np)
                
                # Calculate additional metrics
                mse = mean_squared_error(target_np, output_np)
                mae = mean_absolute_error(target_np, output_np)
                
                try:
                    r2 = r2_score(target_np, output_np)
                except:
                    r2 = 0.0
                
                self.test_metrics.update("mse", mse, batch_size)
                self.test_metrics.update("mae", mae, batch_size)
                self.test_metrics.update("r2", r2, batch_size)
        
        # Combine all predictions and targets
        self.all_preds = np.vstack(all_outputs)
        self.all_targets = np.vstack(all_targets)
        
        # Denormalize if scaler is provided
        if self.rmsf_scaler is not None:
            self.all_preds = self.rmsf_scaler.inverse_transform(self.all_preds)
            self.all_targets = self.rmsf_scaler.inverse_transform(self.all_targets)
        
        # Recalculate metrics on the entire test set
        mse = mean_squared_error(self.all_targets, self.all_preds)
        mae = mean_absolute_error(self.all_targets, self.all_preds)
        
        try:
            r2 = r2_score(self.all_targets, self.all_preds)
        except:
            r2 = 0.0
        
        # Set the final metrics
        metrics = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "loss": mse  # Use MSE as the loss
        }
        
        # Log results
        logger.info(f"Test Results: mse={metrics['mse']:.4f}, mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}")
        
        return metrics
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, Any]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary containing training history and results
        """
        # Set up early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Set up timing
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["lr"].append(current_lr)
            
            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                
                # Save the best model
                torch.save(self.model.state_dict(), self.best_model_path)
            
            # Early stopping
            if early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Calculate training time
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Test the model
        test_metrics = self.test()
        
        # Save training history
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "train_loss": self.history["train_loss"],
                "val_loss": self.history["val_loss"],
                "lr": self.history["lr"],
                "final_metrics": {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics
                },
                "training_time": train_time
            }, f, indent=4)
        
        # Create visualization plots
        self.create_visualization_plots()
        
        # Return training results
        return {
            "history": self.history,
            "final_metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics
            },
            "training_time": train_time,
            "best_val_loss": self.best_val_loss
        }
    
    def create_visualization_plots(self) -> None:
        """Create and save visualization plots."""
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot loss curves
        loss_curve_path = os.path.join(vis_dir, "loss_curves.png")
        plot_loss_curves(
            train_loss=self.history["train_loss"],
            val_loss=self.history["val_loss"],
            lr=self.history["lr"],
            output_path=loss_curve_path
        )
        
        # Plot predictions vs targets
        if len(self.all_preds) > 0:
            predictions_path = os.path.join(vis_dir, "predictions.png")
            plot_predictions(
                predictions=self.all_preds.flatten(),
                targets=self.all_targets.flatten(),
                output_path=predictions_path,
                max_points=1000  # Limit for better visualization
            )
            
            # Plot error distribution
            error_dist_path = os.path.join(vis_dir, "error_distribution.png")
            plot_error_distribution(
                predictions=self.all_preds.flatten(),
                targets=self.all_targets.flatten(),
                output_path=error_dist_path
            )
            
            # Note: residue-type analysis would need additional data about residue types
            # This could be added if the dataset provides this information


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: Neural network model
        optimizer_name: Name of the optimizer (adam, sgd, adagrad, etc.)
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_name}. Using Adam instead.")
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "reduce_on_plateau",
    scheduler_params: Optional[Dict[str, Any]] = None
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of the scheduler
        scheduler_params: Additional scheduler parameters
        
    Returns:
        Scheduler instance or None
    """
    params = scheduler_params or {}
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params.get("factor", 0.5),
            patience=params.get("patience", 5),
            verbose=True
        )
    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("t_max", 10),
            eta_min=params.get("eta_min", 0)
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 30),
            gamma=params.get("gamma", 0.1)
        )
    elif scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params.get("gamma", 0.95)
        )
    elif scheduler_name == "none" or not scheduler_name:
        return None
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}. Not using a scheduler.")
        return None