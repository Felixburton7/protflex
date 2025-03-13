"""
File handling utilities for ProtFlex.

This module provides functions for file operations and path manipulation.
"""

import os
import glob
import shutil
import tempfile
import h5py
import json
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        Absolute path to the directory
    """
    dir_path = os.path.abspath(os.path.expanduser(directory))
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def find_voxel_files(base_dir: str, domain_id: str, temperature: str) -> List[str]:
    """
    Find voxel files for a domain at a specific temperature.

    Args:
        base_dir: Base directory containing voxelized data
        domain_id: Domain identifier
        temperature: Temperature value

    Returns:
        List of voxel file paths
    """
    domain_dir = os.path.join(base_dir, domain_id, str(temperature))

    if not os.path.exists(domain_dir):
        logger.warning(f"Domain directory not found: {domain_dir}")
        return []

    voxel_files = glob.glob(os.path.join(domain_dir, "*.hdf5"))
    return voxel_files

def find_rmsf_file(rmsf_dir: str, domain_id: str, temperature: Union[int, str]) -> Optional[str]:
    """Find the RMSF file for a domain at a specific temperature."""
    # Try standard naming patterns
    if temperature == "average":
        # Try different naming conventions for average
        patterns = [
            f"{domain_id}_total_average_rmsf.csv",
            f"{domain_id}_average_rmsf.csv",
            f"{domain_id}_rmsf_average.csv"
        ]
        for pattern in patterns:
            rmsf_file = os.path.join(rmsf_dir, "average", pattern)
            if os.path.exists(rmsf_file):
                return rmsf_file
    else:
        # Try different naming conventions for specific temperature
        patterns = [
            f"{domain_id}_temperature_{temperature}_average_rmsf.csv",
            f"{domain_id}_temp{temperature}_average_rmsf.csv",
            f"{domain_id}_rmsf_{temperature}.csv"
        ]
        for pattern in patterns:
            rmsf_file = os.path.join(rmsf_dir, str(temperature), pattern)
            if os.path.exists(rmsf_file):
                return rmsf_file

    logger.warning(f"RMSF file not found for domain {domain_id} at temperature {temperature}")
    return None

def read_h5_metadata(h5_file: str) -> Dict[str, Any]:
    """
    Read metadata from an HDF5 file.

    Args:
        h5_file: Path to HDF5 file

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    try:
        with h5py.File(h5_file, 'r') as f:
            # Read root attributes
            for key in f.attrs:
                metadata[key] = f.attrs[key]

            # Read metadata group if it exists
            if 'metadata' in f:
                metadata_group = f['metadata']
                for key in metadata_group.attrs:
                    metadata[f"metadata_{key}"] = metadata_group.attrs[key]

    except Exception as e:
        logger.error(f"Error reading metadata from {h5_file}: {e}")

    return metadata

def save_model_summary(model, output_file: str) -> None:
    """
    Save a summary of the model architecture to a file.

    Args:
        model: PyTorch model
        output_file: Path to output file
    """
    try:
        # Get model summary as string
        from io import StringIO
        import sys

        # Redirect stdout to capture summary
        original_stdout = sys.stdout
        sys.stdout = summary_io = StringIO()

        # Print model architecture
        print(model)

        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        # Restore stdout
        sys.stdout = original_stdout

        # Write to file
        with open(output_file, 'w') as f:
            f.write(summary_io.getvalue())

        logger.info(f"Saved model summary to {output_file}")

    except Exception as e:
        logger.error(f"Error saving model summary: {e}")

def load_rmsf_data(rmsf_file: str) -> pd.DataFrame:
    """
    Load RMSF data from a CSV file.

    Args:
        rmsf_file: Path to RMSF CSV file

    Returns:
        DataFrame with RMSF data
    """
    try:
        df = pd.read_csv(rmsf_file)
        return df
    except Exception as e:
        logger.error(f"Error loading RMSF data from {rmsf_file}: {e}")
        return pd.DataFrame()

def save_to_json(data: Dict[str, Any], output_file: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        output_file: Path to output file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert numpy/torch types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        serializable_data = convert_to_serializable(data)

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        logger.info(f"Saved data to {output_file}")

    except Exception as e:
        logger.error(f"Error saving data to JSON: {e}")

def load_from_json(input_file: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        input_file: Path to input file

    Returns:
        Loaded data
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {input_file}: {e}")
        return {}

def get_domain_ids(data_dir: str, voxel_dir: str, rmsf_dir: str,
                  temperature: Union[int, str]) -> List[str]:
    """
    Get list of domain IDs that have both voxel and RMSF data.

    Args:
        data_dir: Base data directory
        voxel_dir: Relative path to voxelized data
        rmsf_dir: Relative path to RMSF data
        temperature: Temperature value

    Returns:
        List of domain IDs
    """
    voxel_base_dir = os.path.join(data_dir, voxel_dir)
    rmsf_base_dir = os.path.join(data_dir, rmsf_dir)

    # Find domains with voxel data
    voxel_domains = set()
    if os.path.exists(voxel_base_dir):
        for item in os.listdir(voxel_base_dir):
            domain_dir = os.path.join(voxel_base_dir, item)
            if os.path.isdir(domain_dir):
                # Check if temperature directory exists
                temp_dir = os.path.join(domain_dir, str(temperature))
                if os.path.exists(temp_dir) and os.listdir(temp_dir):
                    voxel_domains.add(item)

    # Find domains with RMSF data
    rmsf_domains = set()
    if temperature == "average":
        rmsf_temp_dir = os.path.join(rmsf_base_dir, "average")
    else:
        rmsf_temp_dir = os.path.join(rmsf_base_dir, str(temperature))

    if os.path.exists(rmsf_temp_dir):
        for item in os.listdir(rmsf_temp_dir):
            if item.endswith(".csv"):
                if temperature == "average":
                    domain_id = item.replace("_total_average_rmsf.csv", "")
                else:
                    domain_id = item.split(f"_temperature_{temperature}")[0]
                rmsf_domains.add(domain_id)

    # Find domains with both voxel and RMSF data
    common_domains = voxel_domains.intersection(rmsf_domains)

    logger.info(f"Found {len(common_domains)} domains with both voxel and RMSF data for temperature {temperature}")

    return sorted(list(common_domains))
