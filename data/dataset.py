"""
Datasets management for ProtFlex.

This module provides utilities for loading, filtering, and splitting datasets
for protein flexibility prediction.
"""

import os
import glob
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import random

logger = logging.getLogger(__name__)

def load_dataset(
    data_dir: str,
    voxel_dir: str,
    rmsf_dir: str,
    temperature: Union[int, str] = 320,
    domain_ids: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load dataset information for training.
    
    Args:
        data_dir: Base data directory
        voxel_dir: Subdirectory containing voxelized data
        rmsf_dir: Subdirectory containing RMSF data
        temperature: Temperature value or "average"
        domain_ids: List of domain IDs to include (None = all domains)
        
    Returns:
        Dictionary mapping domain IDs to their data
    """
    # Convert temperature to string for path construction
    temp_str = str(temperature)
    
    # Set up paths
    voxel_base_dir = os.path.join(data_dir, voxel_dir)
    rmsf_base_dir = os.path.join(data_dir, rmsf_dir)
    
    # Determine temperature directories
    if temp_str == "average":
        rmsf_temp_dir = os.path.join(rmsf_base_dir, "average")
    else:
        rmsf_temp_dir = os.path.join(rmsf_base_dir, temp_str)
    
    # Validate paths
    if not os.path.exists(rmsf_temp_dir):
        logger.error(f"RMSF directory does not exist: {rmsf_temp_dir}")
        return {}
    
    # Get domain IDs based on availability of RMSF files
    available_domains = []
    
    if domain_ids is None:
        # Find all domains with RMSF data for the specified temperature
        rmsf_files = glob.glob(os.path.join(rmsf_temp_dir, "*.csv"))
        
        for rmsf_file in rmsf_files:
            file_name = os.path.basename(rmsf_file)
            
            if temp_str == "average":
                if "_total_average_rmsf.csv" in file_name:
                    domain_id = file_name.replace("_total_average_rmsf.csv", "")
                    available_domains.append(domain_id)
            else:
                if f"_temperature_{temp_str}" in file_name:
                    domain_id = file_name.split(f"_temperature_{temp_str}")[0]
                    available_domains.append(domain_id)
                elif f"_temp{temp_str}" in file_name:
                    domain_id = file_name.split(f"_temp{temp_str}")[0]
                    available_domains.append(domain_id)
    else:
        # Use the provided domain IDs
        available_domains = domain_ids
    
    # Load data for each domain
    dataset = {}
    
    for domain_id in available_domains:
        # Find RMSF file
        if temp_str == "average":
            rmsf_file = os.path.join(rmsf_temp_dir, f"{domain_id}_total_average_rmsf.csv")
            if not os.path.exists(rmsf_file):
                # Try alternative naming
                rmsf_file = os.path.join(rmsf_temp_dir, f"{domain_id}_average_rmsf.csv")
        else:
            rmsf_file = os.path.join(rmsf_temp_dir, f"{domain_id}_temperature_{temp_str}_average_rmsf.csv")
            if not os.path.exists(rmsf_file):
                # Try alternative naming
                rmsf_file = os.path.join(rmsf_temp_dir, f"{domain_id}_temp{temp_str}_average_rmsf.csv")
                if not os.path.exists(rmsf_file):
                    # Try another alternative naming
                    rmsf_file = os.path.join(rmsf_temp_dir, f"{domain_id}_rmsf_{temp_str}.csv")
        
        # Skip if RMSF file not found
        if not os.path.exists(rmsf_file):
            logger.warning(f"RMSF file not found for domain {domain_id}")
            continue
        
        # Find voxel directory
        voxel_domain_dir = os.path.join(voxel_base_dir, domain_id)
        
        # Check if domain directory exists
        if not os.path.exists(voxel_domain_dir):
            logger.warning(f"Voxel directory not found for domain {domain_id}")
            continue
        
        # Find voxel files based on temperature
        voxel_files = []
        
        # Check if temperature-specific directory exists
        temp_dir = os.path.join(voxel_domain_dir, temp_str)
        if os.path.exists(temp_dir):
            # Temperature-specific directory exists
            voxel_files = glob.glob(os.path.join(temp_dir, "*.hdf5"))
        else:
            # Try alternative directories
            pdb_dir = os.path.join(voxel_domain_dir, "pdb")
            if os.path.exists(pdb_dir):
                voxel_files = glob.glob(os.path.join(pdb_dir, "*.hdf5"))
            else:
                # Look directly in domain directory
                voxel_files = glob.glob(os.path.join(voxel_domain_dir, "*.hdf5"))
        
        # Skip if no voxel files found
        if not voxel_files:
            logger.warning(f"No voxel files found for domain {domain_id}")
            continue
        
        # Load RMSF data
        try:
            rmsf_df = pd.read_csv(rmsf_file)
            
            # Determine RMSF column name
            rmsf_columns = [col for col in rmsf_df.columns if 'rmsf' in col.lower()]
            if not rmsf_columns:
                logger.warning(f"No RMSF column found in {rmsf_file}")
                continue
            
            # Use the first matching column as default
            rmsf_col = rmsf_columns[0]
            
            # Try to find a more specific column if multiple exist
            if temp_str == "average":
                for col in rmsf_columns:
                    if 'average' in col.lower():
                        rmsf_col = col
                        break
            else:
                for col in rmsf_columns:
                    if temp_str in col:
                        rmsf_col = col
                        break
            
            # Store data
            dataset[domain_id] = {
                "rmsf_file": rmsf_file,
                "rmsf_df": rmsf_df,
                "rmsf_col": rmsf_col,
                "voxel_dir": voxel_domain_dir,
                "voxel_files": voxel_files,
                "temperature": temp_str
            }
            
            logger.info(f"Loaded data for domain {domain_id} with {len(rmsf_df)} residues and {len(voxel_files)} voxel files")
            
        except Exception as e:
            logger.error(f"Error loading data for domain {domain_id}: {e}")
    
    logger.info(f"Loaded dataset with {len(dataset)} domains")
    return dataset

def filter_dataset(
    dataset: Dict[str, Dict[str, Any]],
    min_residues: int = 10,
    max_residues: Optional[int] = None,
    min_voxel_files: int = 5,
    max_voxel_files: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Filter dataset based on criteria.
    
    Args:
        dataset: Dictionary mapping domain IDs to their data
        min_residues: Minimum number of residues required
        max_residues: Maximum number of residues allowed (None = no limit)
        min_voxel_files: Minimum number of voxel files required
        max_voxel_files: Maximum number of voxel files allowed (None = no limit)
        
    Returns:
        Filtered dataset
    """
    filtered_dataset = {}
    
    for domain_id, data in dataset.items():
        # Check number of residues
        num_residues = len(data["rmsf_df"])
        if num_residues < min_residues:
            logger.debug(f"Filtering out domain {domain_id}: too few residues ({num_residues} < {min_residues})")
            continue
        
        if max_residues is not None and num_residues > max_residues:
            logger.debug(f"Filtering out domain {domain_id}: too many residues ({num_residues} > {max_residues})")
            continue
        
        # Check number of voxel files
        num_voxel_files = len(data["voxel_files"])
        if num_voxel_files < min_voxel_files:
            logger.debug(f"Filtering out domain {domain_id}: too few voxel files ({num_voxel_files} < {min_voxel_files})")
            continue
        
        if max_voxel_files is not None and num_voxel_files > max_voxel_files:
            logger.debug(f"Filtering out domain {domain_id}: too many voxel files ({num_voxel_files} > {max_voxel_files})")
            continue
        
        # Include domain in filtered dataset
        filtered_dataset[domain_id] = data
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} domains")
    return filtered_dataset

def match_residues_to_voxels(
    dataset: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Match residues in RMSF files to voxel files.
    
    Args:
        dataset: Dictionary mapping domain IDs to their data
        
    Returns:
        Dataset with matched residues and voxel files
    """
    matched_dataset = {}
    
    for domain_id, data in dataset.items():
        rmsf_df = data["rmsf_df"]
        voxel_files = data["voxel_files"]
        rmsf_col = data["rmsf_col"]
        
        # Create a mapping of residue IDs to voxel files
        residue_to_voxel = {}
        
        # Extract residue IDs from RMSF dataframe
        residue_ids = []
        if "resid" in rmsf_df.columns:
            residue_ids = rmsf_df["resid"].tolist()
        elif "residue_id" in rmsf_df.columns:
            residue_ids = rmsf_df["residue_id"].tolist()
        else:
            # Try to extract from index if it's numeric
            try:
                residue_ids = [int(idx) for idx in rmsf_df.index]
            except:
                logger.warning(f"Could not determine residue IDs for domain {domain_id}")
                continue
        
        # Match residues to voxel files
        for residue_id in residue_ids:
            # Look for voxel files that match this residue
            matching_files = []
            
            for voxel_file in voxel_files:
                file_name = os.path.basename(voxel_file)
                
                # Try to extract residue ID from filename using different patterns
                try:
                    # Option 1: Residue ID after "res" or "residue"
                    if f"res{residue_id}_" in file_name or f"residue{residue_id}_" in file_name:
                        matching_files.append(voxel_file)
                        continue
                    
                    # Option 2: Residue ID as part of split filename
                    parts = file_name.split('_')
                    for i, part in enumerate(parts):
                        if part.isdigit() and int(part) == residue_id:
                            # Check if previous part is "res" or "residue"
                            if i > 0 and (parts[i-1].lower() == "res" or parts[i-1].lower() == "residue"):
                                matching_files.append(voxel_file)
                                break
                    
                    # Option 3: Check HDF5 file metadata
                    try:
                        with h5py.File(voxel_file, 'r') as f:
                            if "metadata" in f and "resid" in f["metadata"].attrs:
                                if f["metadata"].attrs["resid"] == residue_id:
                                    matching_files.append(voxel_file)
                            elif "resid" in f.attrs:
                                if f.attrs["resid"] == residue_id:
                                    matching_files.append(voxel_file)
                    except:
                        pass  # Silently continue if HDF5 metadata check fails
                    
                except:
                    pass  # Silently continue if parsing fails
            
            # Store the matching files
            if matching_files:
                residue_to_voxel[residue_id] = matching_files
        
        # Create matched samples for this domain
        matched_samples = []
        
        # For each row in the RMSF dataframe
        for _, row in rmsf_df.iterrows():
            # Get residue ID
            residue_id = None
            if "resid" in row:
                residue_id = row["resid"]
            elif "residue_id" in row:
                residue_id = row["residue_id"]
            else:
                try:
                    residue_id = int(row.name)
                except:
                    continue
            
            # Get RMSF value
            if rmsf_col in row:
                rmsf_value = row[rmsf_col]
            else:
                logger.warning(f"RMSF column {rmsf_col} not found in row for domain {domain_id}, residue {residue_id}")
                continue
            
            # Get matching voxel files
            if residue_id in residue_to_voxel:
                voxel_files_for_residue = residue_to_voxel[residue_id]
                
                # Create a sample for each matching voxel file
                for voxel_file in voxel_files_for_residue:
                    sample = {
                        "domain_id": domain_id,
                        "residue_id": residue_id,
                        "rmsf_value": rmsf_value,
                        "voxel_file": voxel_file
                    }
                    
                    # Add metadata if available
                    metadata = {}
                    
                    if "resname" in row:
                        metadata["resname"] = row["resname"]
                    
                    if "secondary_structure" in row:
                        metadata["secondary_structure"] = row["secondary_structure"]
                    
                    if "accessibility" in row:
                        metadata["accessibility"] = row["accessibility"]
                    
                    if metadata:
                        sample["metadata"] = metadata
                    
                    matched_samples.append(sample)
        
        # Store matched samples for this domain
        if matched_samples:
            matched_dataset[domain_id] = {
                "samples": matched_samples,
                "rmsf_col": rmsf_col,
                "temperature": data["temperature"]
            }
            logger.info(f"Matched {len(matched_samples)} samples for domain {domain_id}")
    
    logger.info(f"Created matched dataset with {len(matched_dataset)} domains")
    return matched_dataset

def split_dataset(
    dataset: Dict[str, Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dictionary mapping domain IDs to their data
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate split ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        logger.warning(f"Split ratios do not sum to 1: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all domain IDs
    domain_ids = list(dataset.keys())
    random.shuffle(domain_ids)
    
    # Calculate split indices
    train_end = int(len(domain_ids) * train_ratio)
    val_end = train_end + int(len(domain_ids) * val_ratio)
    
    # Split domain IDs
    train_domains = domain_ids[:train_end]
    val_domains = domain_ids[train_end:val_end]
    test_domains = domain_ids[val_end:]
    
    # Create split datasets
    train_dataset = {domain_id: dataset[domain_id] for domain_id in train_domains}
    val_dataset = {domain_id: dataset[domain_id] for domain_id in val_domains}
    test_dataset = {domain_id: dataset[domain_id] for domain_id in test_domains}
    
    logger.info(f"Split dataset into {len(train_dataset)} train, {len(val_dataset)} validation, and {len(test_dataset)} test domains")
    return train_dataset, val_dataset, test_dataset

def create_residue_level_dataset(
    matched_dataset: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create a residue-level dataset from matched domain data.
    
    Args:
        matched_dataset: Matched domain data from match_residues_to_voxels
        
    Returns:
        List of samples for all residues
    """
    all_samples = []
    
    for domain_id, data in matched_dataset.items():
        all_samples.extend(data["samples"])
    
    logger.info(f"Created residue-level dataset with {len(all_samples)} samples")
    return all_samples