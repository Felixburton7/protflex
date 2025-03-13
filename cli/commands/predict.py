"""
Prediction command for ProtFlex CLI.

This module provides the command to predict protein flexibility using a trained model.
"""

import os
import argparse
import logging
import json
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from protflex.models import cnn_models
from protflex.data.data_loader import RMSFDataset
from protflex.utils.visualization import plot_rmsf_profiles

logger = logging.getLogger(__name__)

def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Add a subparser for the predict command.

    Args:
        subparsers: Subparsers object

    Returns:
        Parser for the predict command
    """
    parser = subparsers.add_parser(
        'predict',
        help='Make RMSF predictions using a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str,
                      help='Directory containing data (overrides config)')
    parser.add_argument('--domain-ids', type=str, nargs='+',
                      help='List of domain IDs to predict (if not specified, all available domains will be used)')

    # Output arguments
    parser.add_argument('--output-dir', type=str,
                      help='Directory to save prediction results (overrides config)')
    parser.add_argument('--plot', action='store_true',
                      help='Generate RMSF profile plots')
    parser.add_argument('--compare-actual', action='store_true',
                      help='Compare predictions with actual RMSF values if available')
    parser.add_argument('--cpu', action='store_true',
                      help='Force using CPU even if GPU is available')

    return parser


def predict_domain(
    model: torch.nn.Module,
    domain_id: str,
    voxel_dir: str,
    temperature: str,
    device: torch.device,
    rmsf_scaler = None
) -> Tuple[List[int], List[float], List[str]]:
    """
    Predict RMSF values for a domain with robust error handling.

    Args:
        model: Trained model
        domain_id: Domain ID
        voxel_dir: Directory containing voxelized data
        temperature: Temperature value
        device: Device to run the model on
        rmsf_scaler: Scaler for normalizing RMSF values

    Returns:
        Tuple of (residue_ids, predicted_rmsf_values, residue_names)
    """
    # Find voxel files for the domain
    domain_voxel_dir = os.path.join(voxel_dir, domain_id)
    
    # Check for temperature-specific directory
    temp_dir = os.path.join(domain_voxel_dir, str(temperature))
    if os.path.exists(temp_dir):
        domain_voxel_dir = temp_dir
    else:
        # Check for PDB directory as a fallback
        pdb_dir = os.path.join(domain_voxel_dir, "pdb")
        if os.path.exists(pdb_dir):
            domain_voxel_dir = pdb_dir
    
    if not os.path.exists(domain_voxel_dir):
        logger.error(f"Voxel directory not found for domain {domain_id} at temperature {temperature}")
        return [], [], []
    
    voxel_files = [f for f in os.listdir(domain_voxel_dir) if f.endswith('.hdf5')]
    
    if not voxel_files:
        logger.error(f"No voxel files found for domain {domain_id} in {domain_voxel_dir}")
        return [], [], []
    
    residue_ids = []
    predicted_rmsf = []
    residue_names = []
    
    # Process each voxel file
    for voxel_file in tqdm(voxel_files, desc=f"Predicting {domain_id}"):
        try:
            voxel_path = os.path.join(domain_voxel_dir, voxel_file)
            
            # Extract residue ID from filename
            residue_id = extract_residue_id_from_filename(voxel_path)
            
            if residue_id is None:
                logger.warning(f"Could not extract residue ID from {voxel_path}, skipping")
                continue
            
            # Load voxel data
            voxel_data, metadata = load_voxel_data(voxel_path)
            
            if voxel_data is None:
                logger.warning(f"Failed to load voxel data from {voxel_path}, skipping")
                continue
            
            # Check data shape for compatibility with model
            expected_channels = 5  # C, N, O, CA, CB channels
            if voxel_data.shape[0] != expected_channels:
                logger.warning(f"Unexpected channel count in {voxel_path}: expected {expected_channels}, got {voxel_data.shape[0]}")
                continue
            
            # Convert to tensor
            voxel_tensor = torch.from_numpy(voxel_data).float().unsqueeze(0).to(device)
            
            # Get residue name from metadata
            residue_name = "UNK"
            if "resname" in metadata:
                residue_name = metadata["resname"]
            elif "residue_name" in metadata:
                residue_name = metadata["residue_name"]
            
            # Forward pass through model
            with torch.no_grad():
                # Handle models that expect metadata
                if getattr(model, 'include_metadata', False):
                    # Create metadata tensor
                    metadata_size = getattr(model, 'metadata_features', 0)
                    metadata_tensor = torch.zeros(1, metadata_size, device=device)
                    
                    # Fill metadata tensor if possible
                    current_idx = 0
                    
                    # One-hot encode residue name if available
                    if metadata_size >= 20 and "resname" in metadata:
                        # Get amino acid index (simple mapping for standard amino acids)
                        aa_dict = {"ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
                                  "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
                                  "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
                                  "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19}
                        
                        aa_idx = aa_dict.get(metadata["resname"], -1)
                        if aa_idx >= 0:
                            metadata_tensor[0, aa_idx] = 1.0
                    
                    # Pass both voxel data and metadata to the model
                    output = model(voxel_tensor, metadata_tensor)
                else:
                    output = model(voxel_tensor)
                
                # Get prediction
                pred = output.item()
                
                # Denormalize if scaler provided
                if rmsf_scaler is not None:
                    pred = rmsf_scaler.inverse_transform([[pred]])[0, 0]
                
                # Store results
                residue_ids.append(residue_id)
                predicted_rmsf.append(pred)
                residue_names.append(residue_name)
        
        except Exception as e:
            logger.error(f"Error processing voxel file {voxel_file}: {e}")
    
    # Check if we have any predictions
    if not residue_ids:
        logger.warning(f"No predictions generated for domain {domain_id}")
        return [], [], []
    
    # Sort by residue ID
    sorted_indices = np.argsort(residue_ids)
    residue_ids = [residue_ids[i] for i in sorted_indices]
    predicted_rmsf = [predicted_rmsf[i] for i in sorted_indices]
    residue_names = [residue_names[i] for i in sorted_indices]
    
    return residue_ids, predicted_rmsf, residue_names

def load_voxel_data(voxel_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load voxel data from an HDF5 file with robust error handling.
    
    Args:
        voxel_path: Path to the voxel HDF5 file
        
    Returns:
        Tuple of (voxel_data, metadata) or (None, None) if loading fails
    """
    try:
        with h5py.File(voxel_path, 'r') as f:
            voxel_data = None
            metadata = {}
            
            # Extract metadata
            for key in f.attrs:
                metadata[key] = f.attrs[key]
            
            # Try different possible structures for voxel data
            
            # Option 1: Modern aposteriori format with "inputs" dataset
            if "inputs" in f:
                voxel_data = f["inputs"][:]
                
                # Get additional metadata from inputs group
                if hasattr(f["inputs"], "attrs"):
                    for key in f["inputs"].attrs:
                        metadata[f"input_{key}"] = f["inputs"].attrs[key]
            
            # Option 2: Legacy format with "voxel_data" dataset
            elif "voxel_data" in f:
                voxel_data = f["voxel_data"][:]
            
            # Option 3: Direct data in root dataset
            elif len(f.keys()) == 0 and isinstance(f, h5py.Dataset):
                voxel_data = f[:]
            
            # Option 4: Look for any dataset with 3D or 4D shape
            else:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset = f[key]
                        shape = dataset.shape
                        
                        if len(shape) >= 3:
                            voxel_data = dataset[:]
                            break
            
            # Extract metadata from metadata group if it exists
            if "metadata" in f:
                metadata_group = f["metadata"]
                
                # Extract attributes
                if hasattr(metadata_group, "attrs"):
                    for key in metadata_group.attrs:
                        metadata[key] = metadata_group.attrs[key]
            
            if voxel_data is None:
                logger.warning(f"Could not find voxel data in {voxel_path}")
                return None, metadata
            
            # Reshape if needed
            if len(voxel_data.shape) == 4:  # [batch, channels, height, width]
                voxel_data = voxel_data[0]  # Take the first batch item
            
            # Validate data shape
            if len(voxel_data.shape) != 3 and len(voxel_data.shape) != 4:
                logger.warning(f"Unexpected data shape in {voxel_path}: {voxel_data.shape}")
                return None, metadata
                
            return voxel_data, metadata
    
    except Exception as e:
        logger.error(f"Error loading voxel file {voxel_path}: {e}")
        return None, None
    
def extract_residue_id_from_filename(voxel_file: str) -> Optional[int]:
    """
    Extract residue ID from voxel filename using various patterns.
    
    Args:
        voxel_file: Path to voxel file
        
    Returns:
        Extracted residue ID or None if not found
    """
    file_name = os.path.basename(voxel_file)
    
    # Try several patterns to extract residue ID
    
    # Pattern 1: Look for _res{ID}_ or _residue{ID}_
    import re
    res_pattern = re.compile(r'_res(\d+)_|_residue(\d+)_')
    match = res_pattern.search(file_name)
    if match:
        # Get the matched group that's not None
        for group in match.groups():
            if group is not None:
                return int(group)
    
    # Pattern 2: Extract from file parts
    parts = file_name.split('_')
    
    # Look for numeric parts
    for i, part in enumerate(parts):
        if part.isdigit():
            # Check if previous part indicates this is a residue ID
            if i > 0 and (parts[i-1].lower() == 'res' or parts[i-1].lower() == 'residue'):
                return int(part)
    
    # Pattern 3: Try to extract from directory structure
    dir_path = os.path.dirname(voxel_file)
    dir_parts = dir_path.split(os.sep)
    
    # Look for numeric directory names
    for part in dir_parts:
        if part.isdigit():
            return int(part)
    
    # Pattern 4: Try to read ID from HDF5 file
    try:
        with h5py.File(voxel_file, 'r') as f:
            # Check metadata attributes
            if "metadata" in f and "resid" in f["metadata"].attrs:
                resid = f["metadata"].attrs["resid"]
                # Validate the residue ID
                try:
                    return int(resid)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid residue ID in metadata.attrs.resid: {resid}")
            
            # Check top-level attributes
            if "resid" in f.attrs:
                resid = f.attrs["resid"]
                try:
                    return int(resid)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid residue ID in attrs.resid: {resid}")
            
            # Check inputs dataset attributes
            if "inputs" in f and "resid" in f["inputs"].attrs:
                resid = f["inputs"].attrs["resid"]
                try:
                    return int(resid)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid residue ID in inputs.attrs.resid: {resid}")
    except Exception as e:
        logger.error(f"Error reading residue ID from HDF5 file {voxel_file}: {e}")
    
    # If all extraction methods fail, return None
    logger.warning(f"Could not extract residue ID from filename: {voxel_file}")
    return None

def run(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Run the predict command.

    Args:
        args: Command line arguments
        config: Configuration dictionary

    Returns:
        Exit code
    """
    # Override config with command-line arguments
    if args.data_dir:
        config['input']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output']['base_dir'] = args.output_dir
    if args.domain_ids:
        config['input']['domain_ids'] = args.domain_ids

    # Set up directories
    data_dir = config['input']['data_dir']
    voxel_dir = os.path.join(data_dir, config['input']['voxel_dir'])
    rmsf_dir = os.path.join(data_dir, config['input']['rmsf_dir'])
    output_dir = config['output']['base_dir']
    results_dir = os.path.join(output_dir, config['output']['results_dir'])

    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    predictions_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    if args.plot:
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

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

    # Get domain IDs to process
    domain_ids = config['input']['domain_ids']
    if not domain_ids:
        # Find all domains with voxel data
        domain_ids = [d for d in os.listdir(voxel_dir)
                     if os.path.isdir(os.path.join(voxel_dir, d))]

    if not domain_ids:
        logger.error(f"No domains found in {voxel_dir}")
        return 1

    logger.info(f"Processing {len(domain_ids)} domains")

    # Process each domain
    all_results = []

    for domain_id in domain_ids:
        logger.info(f"Processing domain {domain_id}")

        # Predict RMSF values
        residue_ids, predicted_rmsf, residue_names = predict_domain(
            model, domain_id, voxel_dir, temperature, device)

        if not residue_ids:
            logger.warning(f"No predictions generated for domain {domain_id}")
            continue

        # Create DataFrame with predictions
        df = pd.DataFrame({
            'domain_id': domain_id,
            'resid': residue_ids,
            'resname': residue_names,
            f'predicted_rmsf_{temperature}': predicted_rmsf
        })

        # Compare with actual RMSF values if requested
        if args.compare_actual:
            if temperature == "average":
                actual_rmsf_file = os.path.join(rmsf_dir, "average", f"{domain_id}_total_average_rmsf.csv")
            else:
                actual_rmsf_file = os.path.join(rmsf_dir, str(temperature),
                                             f"{domain_id}_temperature_{temperature}_average_rmsf.csv")

            if os.path.exists(actual_rmsf_file):
                actual_df = pd.read_csv(actual_rmsf_file)

                # Determine RMSF column name
                if temperature == "average":
                    rmsf_col = "average_rmsf"
                else:
                    rmsf_col = f"rmsf_{temperature}"

                # Merge predictions with actual values
                df = df.merge(actual_df[['resid', rmsf_col]], on='resid', how='left')
                df.rename(columns={rmsf_col: f'actual_rmsf_{temperature}'}, inplace=True)

        # Save predictions to CSV
        output_file = os.path.join(predictions_dir, f"{domain_id}_predictions.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")

        # Create plot if requested
        if args.plot:
            if 'actual_rmsf' in df.columns.str.contains(f'actual_rmsf_{temperature}').any():
                actual_col = f'actual_rmsf_{temperature}'
                actual_values = df[actual_col].values
            else:
                actual_values = None

            plot_file = os.path.join(plots_dir, f"{domain_id}_rmsf_profile.png")
            plot_rmsf_profiles(
                domain_id=domain_id,
                residue_ids=df['resid'].values,
                actual_rmsf=actual_values if actual_values is not None else [],
                predicted_rmsf=df[f'predicted_rmsf_{temperature}'].values,
                residue_names=df['resname'].values,
                output_path=plot_file
            )
            logger.info(f"Saved RMSF profile plot to {plot_file}")

        all_results.append(df)

    # Combine all results
    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        all_output_file = os.path.join(predictions_dir, f"all_domains_predictions.csv")
        all_df.to_csv(all_output_file, index=False)
        logger.info(f"Saved combined predictions to {all_output_file}")

    logger.info("Prediction completed successfully")
    return 0
