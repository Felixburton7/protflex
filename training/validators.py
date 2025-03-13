"""
Validation utilities for ProtFlex.

This module provides validators for dataset integrity and model performance.
"""

import os
import logging
import numpy as np
import torch
import h5py
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validator for checking dataset integrity.
    """
    
    def __init__(self, voxel_dir: str, rmsf_dir: str):
        """
        Initialize the validator.
        
        Args:
            voxel_dir: Directory containing voxelized data
            rmsf_dir: Directory containing RMSF data
        """
        self.voxel_dir = voxel_dir
        self.rmsf_dir = rmsf_dir
    
    def validate_domain(self, domain_id: str, temperature: str) -> Dict[str, Any]:
        """
        Validate data for a single domain.
        
        Args:
            domain_id: Domain identifier
            temperature: Temperature value
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating domain {domain_id} at temperature {temperature}")
        
        results = {
            "domain_id": domain_id,
            "temperature": temperature,
            "is_valid": False,
            "voxel_dir_exists": False,
            "rmsf_file_exists": False,
            "voxel_files_count": 0,
            "residues_with_voxel_data": 0,
            "issues": []
        }
        
        # Check voxel directory
        domain_voxel_dir = os.path.join(self.voxel_dir, domain_id, str(temperature))
        results["voxel_dir_exists"] = os.path.exists(domain_voxel_dir)
        
        if not results["voxel_dir_exists"]:
            results["issues"].append(f"Voxel directory not found: {domain_voxel_dir}")
            return results
        
        # Check RMSF file
        if temperature == "average":
            rmsf_file = os.path.join(self.rmsf_dir, "average", f"{domain_id}_total_average_rmsf.csv")
        else:
            rmsf_file = os.path.join(self.rmsf_dir, str(temperature),
                                   f"{domain_id}_temperature_{temperature}_average_rmsf.csv")
        
        results["rmsf_file_exists"] = os.path.exists(rmsf_file)
        
        if not results["rmsf_file_exists"]:
            results["issues"].append(f"RMSF file not found: {rmsf_file}")
            return results
        
        # Count voxel files
        voxel_files = [f for f in os.listdir(domain_voxel_dir) if f.endswith('.hdf5')]
        results["voxel_files_count"] = len(voxel_files)
        
        if results["voxel_files_count"] == 0:
            results["issues"].append(f"No voxel files found in {domain_voxel_dir}")
            return results
        
        # Check voxel files for expected structure
        valid_voxel_files = 0
        residue_ids = set()
        
        for voxel_file in voxel_files:
            try:
                voxel_path = os.path.join(domain_voxel_dir, voxel_file)
                with h5py.File(voxel_path, 'r') as f:
                    # Check for required datasets
                    if "inputs" in f:
                        inputs = f["inputs"]
                        shape = inputs.shape
                        
                        # Check shape
                        if len(shape) >= 3:
                            valid_voxel_files += 1
                        else:
                            results["issues"].append(f"Invalid shape in {voxel_file}: {shape}")
                    
                    # Extract residue ID
                    resid = None
                    if "metadata" in f and "resid" in f["metadata"].attrs:
                        resid = f["metadata"].attrs["resid"]
                    elif "resid" in f.attrs:
                        resid = f.attrs["resid"]
                    
                    if resid is not None:
                        residue_ids.add(resid)
                    else:
                        results["issues"].append(f"Missing residue ID in {voxel_file}")
            except Exception as e:
                results["issues"].append(f"Error reading {voxel_file}: {str(e)}")
        
        results["residues_with_voxel_data"] = len(residue_ids)
        
        # Set overall validity
        results["is_valid"] = (
            results["voxel_dir_exists"] and
            results["rmsf_file_exists"] and
            valid_voxel_files > 0 and
            len(results["issues"]) == 0
        )
        
        return results