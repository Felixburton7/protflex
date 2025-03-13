"""
Dataset management for ProtFlex.

This module provides functions for loading, processing, and managing protein structure datasets.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_dataset(data_dir: str, domain_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load a dataset from the specified directory.
    
    Args:
        data_dir: Path to the data directory
        domain_ids: Optional list of domain IDs to load (if None, load all domains)
    
    Returns:
        Dictionary containing loaded dataset information
    """
    logger.info(f"Loading dataset from {data_dir}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all domain directories
    domain_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if domain_ids is not None:
        domain_dirs = [d for d in domain_dirs if d in domain_ids]
    
    if not domain_dirs:
        logger.warning(f"No domain directories found in {data_dir}")
        return {}
    
    # Load domain data
    dataset = {}
    
    for domain_id in tqdm(domain_dirs, desc="Loading domains"):
        try:
            domain_data = load_domain_data(os.path.join(data_dir, domain_id), domain_id)
            if domain_data:
                dataset[domain_id] = domain_data
        except Exception as e:
            logger.error(f"Error loading domain {domain_id}: {e}")
    
    logger.info(f"Loaded {len(dataset)} domains")
    return dataset

def load_domain_data(domain_dir: str, domain_id: str) -> Dict[str, Any]:
    """
    Load data for a single protein domain.
    
    Args:
        domain_dir: Path to the domain directory
        domain_id: Domain identifier
    
    Returns:
        Dictionary containing domain data
    """
    # Validate domain directory
    if not os.path.exists(domain_dir):
        logger.warning(f"Domain directory not found: {domain_dir}")
        return {}
    
    # Find voxel and RMSF directories/files
    voxel_files = glob.glob(os.path.join(domain_dir, "**", "*.hdf5"), recursive=True)
    rmsf_files = glob.glob(os.path.join(domain_dir, "**", "*.csv"), recursive=True)
    
    if not voxel_files:
        logger.warning(f"No voxel files found for domain {domain_id}")
        return {}
    
    if not rmsf_files:
        logger.warning(f"No RMSF files found for domain {domain_id}")
        return {}
    
    # Create domain data dictionary
    domain_data = {
        "domain_id": domain_id,
        "voxel_files": voxel_files,
        "rmsf_files": rmsf_files
    }
    
    # Try to load RMSF data
    try:
        rmsf_df = pd.read_csv(rmsf_files[0])
        
        # Validate RMSF data
        if "resid" not in rmsf_df.columns:
            logger.warning(f"RMSF file for domain {domain_id} is missing 'resid' column")
            return {}
            
        # Find RMSF column
        rmsf_cols = [col for col in rmsf_df.columns if "rmsf" in col.lower()]
        if not rmsf_cols:
            logger.warning(f"No RMSF column found in data for domain {domain_id}")
            return {}
            
        domain_data["rmsf_data"] = rmsf_df
        domain_data["rmsf_column"] = rmsf_cols[0]  # Use first RMSF column
    except Exception as e:
        logger.error(f"Error loading RMSF data for domain {domain_id}: {e}")
        return {}
    
    return domain_data

def filter_dataset(dataset: Dict[str, Any], min_residues: int = 10) -> Dict[str, Any]:
    """
    Filter dataset based on certain criteria.
    
    Args:
        dataset: Dataset dictionary
        min_residues: Minimum number of residues required for a domain
    
    Returns:
        Filtered dataset
    """
    filtered_dataset = {}
    
    for domain_id, domain_data in dataset.items():
        # Check for required data
        if "rmsf_data" not in domain_data:
            logger.warning(f"Domain {domain_id} missing RMSF data, skipping")
            continue
        
        # Check minimum residue count
        if len(domain_data["rmsf_data"]) < min_residues:
            logger.warning(f"Domain {domain_id} has fewer than {min_residues} residues, skipping")
            continue
        
        # Add to filtered dataset
        filtered_dataset[domain_id] = domain_data
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} domains")
    return filtered_dataset

def split_dataset(
    dataset: Dict[str, Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        dataset: Dataset dictionary
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    import random
    random.seed(random_seed)
    
    # Check that ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Split ratios do not sum to 1: {total_ratio}")
        # Normalize ratios
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # Get list of domain IDs
    domain_ids = list(dataset.keys())
    random.shuffle(domain_ids)
    
    # Calculate split indices
    n_domains = len(domain_ids)
    n_train = int(n_domains * train_ratio)
    n_val = int(n_domains * val_ratio)
    
    # Split domain IDs
    train_ids = domain_ids[:n_train]
    val_ids = domain_ids[n_train:n_train + n_val]
    test_ids = domain_ids[n_train + n_val:]
    
    # Create split datasets
    train_dataset = {domain_id: dataset[domain_id] for domain_id in train_ids}
    val_dataset = {domain_id: dataset[domain_id] for domain_id in val_ids}
    test_dataset = {domain_id: dataset[domain_id] for domain_id in test_ids}
    
    logger.info(f"Split dataset into {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test domains")
    
    return train_dataset, val_dataset, test_dataset