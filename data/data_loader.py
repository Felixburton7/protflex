"""
Data loading and preprocessing utilities for ProtFlex.

This module contains functions for loading and preprocessing voxelized protein data
and corresponding RMSF values.
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import glob
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict

logger = logging.getLogger(__name__)

class RMSFDataset(Dataset):
    """Dataset for protein voxel grids and RMSF values."""
    
    def __init__(
        self, 
        voxel_dir: str,
        rmsf_dir: str,
        temperature: Union[int, str] = 320,
        domain_ids: Optional[List[str]] = None,
        use_metadata: bool = True,
        metadata_fields: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        normalize_rmsf: bool = True,
        cache_size: int = 1000  # Maximum number of samples to cache in memory
    ):
        """
        Args:
            voxel_dir: Directory containing voxelized protein data
            rmsf_dir: Directory containing RMSF CSV files
            temperature: Temperature value (320, 348, 379, 413, 450 or "average")
            domain_ids: List of domain IDs to include (None = all domains)
            use_metadata: Whether to include metadata features
            metadata_fields: List of metadata fields to include
            transform: Optional transform to apply to the voxel data
            normalize_rmsf: Whether to normalize RMSF values
            cache_size: Maximum number of samples to cache in memory
        """
        self.voxel_dir = voxel_dir
        self.rmsf_dir = rmsf_dir
        self.temperature = str(temperature)
        self.use_metadata = use_metadata
        self.metadata_fields = metadata_fields or []
        self.transform = transform
        self.normalize_rmsf = normalize_rmsf
        self.cache_size = cache_size
        
        # Read all available domains and RMSF data
        self.domain_data = self._load_domain_data(domain_ids)
        
        # Build an index of all samples
        self.samples = self._build_sample_index()
        
        # Initialize data cache
        self.data_cache = {}
        
        # Initialize RMSF normalization
        self.rmsf_scaler = None
        if normalize_rmsf:
            self._setup_rmsf_normalization()
        
        logger.info(f"Initialized dataset with {len(self.samples)} samples " 
                   f"from {len(self.domain_data)} domains")
    
    def _load_domain_data(self, domain_ids: Optional[List[str]]) -> Dict[str, Dict]:
        """
        Load RMSF data for all domains and match with available voxel files.
        
        Args:
            domain_ids: List of domain IDs to include (None = all domains)
            
        Returns:
            Dictionary mapping domain IDs to their data
        """
        # Determine the temperature directory
        if self.temperature == "average":
            temp_dir = "average"
        else:
            temp_dir = self.temperature
        
        # Find all RMSF CSV files for the specified temperature
        rmsf_pattern = os.path.join(self.rmsf_dir, temp_dir, "*.csv")
        rmsf_files = glob.glob(rmsf_pattern)
        
        if not rmsf_files:
            raise ValueError(f"No RMSF files found in {os.path.join(self.rmsf_dir, temp_dir)}")
        
        # Extract domain IDs from file names
        if domain_ids is None:
            if self.temperature == "average":
                all_domains = [os.path.basename(f).split("_total_average_rmsf.csv")[0] for f in rmsf_files]
            else:
                all_domains = [os.path.basename(f).split(f"_temperature_{self.temperature}")[0] for f in rmsf_files]
        else:
            all_domains = domain_ids
        
        # Filter out domains that don't have both RMSF data and voxel files
        domain_data = {}
        
        for domain_id in all_domains:
            # Check for RMSF file
            if self.temperature == "average":
                rmsf_file = os.path.join(self.rmsf_dir, temp_dir, f"{domain_id}_total_average_rmsf.csv")
            else:
                rmsf_file = os.path.join(self.rmsf_dir, temp_dir, f"{domain_id}_temperature_{self.temperature}_average_rmsf.csv")
            
            # Check for voxel directory
            voxel_domain_dir = os.path.join(self.voxel_dir, domain_id, temp_dir)
            
            if os.path.exists(rmsf_file) and os.path.exists(voxel_domain_dir):
                # Load RMSF data
                try:
                    rmsf_df = pd.read_csv(rmsf_file)
                    
                    # Determine the RMSF column name
                    if self.temperature == "average":
                        rmsf_col = "average_rmsf"
                    else:
                        rmsf_col = f"rmsf_{self.temperature}"
                    
                    # Check if the RMSF column exists
                    if rmsf_col not in rmsf_df.columns:
                        logger.warning(f"RMSF column '{rmsf_col}' not found in {rmsf_file}")
                        continue
                    
                    # Find voxel files for this domain
                    voxel_files = glob.glob(os.path.join(voxel_domain_dir, "*.hdf5"))
                    
                    # Only include domains with both RMSF data and voxel files
                    if voxel_files:
                        domain_data[domain_id] = {
                            "rmsf_file": rmsf_file,
                            "rmsf_df": rmsf_df,
                            "rmsf_col": rmsf_col,
                            "voxel_dir": voxel_domain_dir,
                            "voxel_files": voxel_files
                        }
                        logger.debug(f"Added domain {domain_id} with {len(rmsf_df)} residues")
                    else:
                        logger.warning(f"No voxel files found for domain {domain_id} in {voxel_domain_dir}")
                except Exception as e:
                    logger.error(f"Error loading data for domain {domain_id}: {e}")
            else:
                logger.debug(f"Skipping domain {domain_id}: RMSF file or voxel directory missing")
        
        return domain_data
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build an index of all samples in the dataset.
        
        Returns:
            List of sample dictionaries containing domain_id, resid, etc.
        """
        samples = []
        
        for domain_id, data in self.domain_data.items():
            rmsf_df = data["rmsf_df"]
            rmsf_col = data["rmsf_col"]
            
            for _, row in rmsf_df.iterrows():
                # Extract necessary information
                resid = row["resid"]
                
                # Find the corresponding voxel file
                voxel_file = self._find_voxel_file_for_residue(domain_id, resid)
                
                if voxel_file:
                    # Create a sample entry
                    sample = {
                        "domain_id": domain_id,
                        "resid": resid,
                        "rmsf_value": row[rmsf_col],
                        "voxel_file": voxel_file
                    }
                    
                    # Add metadata if requested
                    if self.use_metadata and self.metadata_fields:
                        sample["metadata"] = {
                            field: row[field] if field in row else None
                            for field in self.metadata_fields
                        }
                    
                    samples.append(sample)
        
        return samples
    def _find_voxel_file_for_residue(self, domain_id: str, resid: int) -> Optional[str]:
        """
        Find the voxel file corresponding to a residue.
        
        This method searches for voxel files that match the given residue ID,
        using multiple matching strategies.
        
        Args:
            domain_id: Domain identifier
            resid: Residue identifier
            
        Returns:
            Path to the voxel file or None if not found
        """
        voxel_domain_dir = self.domain_data[domain_id]["voxel_dir"]
        voxel_files = self.domain_data[domain_id]["voxel_files"]
        
        # First, check if we have a direct frame file for this residue
        # Try multiple patterns:
        # Pattern 1: domain_tempXXX_frameX_resid_clean_CNOCBCA.hdf5
        # Pattern 2: domain_pdb_clean_CNOCBCA.hdf5 (for PDB files without temperature)
        # Pattern 3: domain_resid_tempXXX_frameX_clean_CNOCBCA.hdf5
        
        # Strategy 1: Look for explicit residue IDs in filenames
        matching_files = []
        for voxel_file in voxel_files:
            file_name = os.path.basename(voxel_file)
            
            # Check for residue ID in filename
            if f"_res{resid}_" in file_name or f"_residue{resid}_" in file_name:
                matching_files.append(voxel_file)
                continue
            
            # Check in file name parts
            parts = file_name.split('_')
            for i, part in enumerate(parts):
                if part.isdigit() and int(part) == resid:
                    # Check if previous part indicates this is a residue ID
                    if i > 0 and (parts[i-1].lower() == "res" or parts[i-1].lower() == "residue"):
                        matching_files.append(voxel_file)
                        break
        
        # If we found matching files, return the first one
        if matching_files:
            return matching_files[0]
        
        # Strategy 2: Check HDF5 metadata for residue information
        for voxel_file in voxel_files:
            try:
                with h5py.File(voxel_file, 'r') as f:
                    # Check metadata attributes
                    if "metadata" in f and hasattr(f["metadata"], "attrs"):
                        metadata = f["metadata"]
                        if "resid" in metadata.attrs and metadata.attrs["resid"] == resid:
                            return voxel_file
                    
                    # Check top-level attributes
                    if "resid" in f.attrs and f.attrs["resid"] == resid:
                        return voxel_file
                    
                    # Check for inputs dataset with attributes
                    if "inputs" in f and hasattr(f["inputs"], "attrs"):
                        inputs = f["inputs"]
                        if "resid" in inputs.attrs and inputs.attrs["resid"] == resid:
                            return voxel_file
                    
                    # Check if this file contains multiple residues
                    if "residues" in f and str(resid) in f["residues"]:
                        return voxel_file
            except Exception as e:
                logger.warning(f"Error checking HDF5 file {voxel_file}: {e}")
        
        # Strategy 3: Check temperature subdirectory
        temp_dir = self.temperature
        temp_subdir = os.path.join(voxel_domain_dir, temp_dir)
        if os.path.exists(temp_subdir):
            # Look for files in the temperature subdirectory
            temp_files = [f for f in os.listdir(temp_subdir) if f.endswith('.hdf5')]
            for temp_file in temp_files:
                file_path = os.path.join(temp_subdir, temp_file)
                if f"_res{resid}_" in temp_file or f"_residue{resid}_" in temp_file:
                    return file_path
        
        # Log warning and return None if no match found
        logger.warning(f"Could not find voxel file for domain {domain_id}, residue {resid}")
        return None
    
    def _setup_rmsf_normalization(self) -> None:
        """Set up normalization for RMSF values."""
        # Collect all RMSF values
        all_rmsf = [sample["rmsf_value"] for sample in self.samples]
        
        if not all_rmsf:
            logger.warning("No RMSF values found for normalization, using identity scaling")
            # Create a dummy scaler that does nothing
            self.rmsf_scaler = StandardScaler()
            self.rmsf_scaler.mean_ = np.array([0.0])
            self.rmsf_scaler.var_ = np.array([1.0])
            self.rmsf_scaler.scale_ = np.array([1.0])
            return
        
        # Create and fit a scaler
        self.rmsf_scaler = StandardScaler()
        self.rmsf_scaler.fit(np.array(all_rmsf).reshape(-1, 1))
        
        logger.info(f"RMSF normalization: mean={self.rmsf_scaler.mean_[0]:.4f}, "
                f"std={np.sqrt(self.rmsf_scaler.var_[0]):.4f}")
    
    # Add this import at the top:
    # import collections

    def _load_voxel_data(self, voxel_file: str) -> torch.Tensor:
        """
        Load voxel data from an HDF5 file with efficient caching.
        
        Args:
            voxel_file: Path to the voxel HDF5 file
            
        Returns:
            Tensor containing voxel data with shape [channels, 21, 21, 21]
        """
        # First check if this sample is cached
        if voxel_file in self.data_cache:
            # Move to end (most recently used)
            value = self.data_cache.pop(voxel_file)
            self.data_cache[voxel_file] = value
            return value["voxel_data"]
        
        # If not cached, load from disk
        try:
            with h5py.File(voxel_file, 'r') as f:
                # Extract the data based on the Aposteriori HDF5 format
                if "inputs" in f:
                    # Modern aposteriori format with "inputs" dataset
                    voxel_data = f["inputs"][:]
                    
                    # Reshape if needed
                    if len(voxel_data.shape) == 4:  # [batch, channels, height, width]
                        voxel_data = voxel_data[0]  # Take the first batch item
                    
                    # Convert to torch tensor
                    voxel_tensor = torch.from_numpy(voxel_data).float()
                else:
                    # Try other locations or formats
                    if "voxel_data" in f:
                        voxel_data = f["voxel_data"][:]
                        voxel_tensor = torch.from_numpy(voxel_data).float()
                    else:
                        # Legacy format - fallback to zeros
                        logger.warning(f"No recognized data format in {voxel_file}, using zeros")
                        voxel_tensor = torch.zeros((5, 21, 21, 21), dtype=torch.float32)
            
            # Apply any transformations
            if self.transform:
                voxel_tensor = self.transform(voxel_tensor)
            
            # Cache the tensor with LRU behavior
            if len(self.data_cache) >= self.cache_size:
                # Remove oldest item (first key in OrderedDict)
                self.data_cache.popitem(last=False)
            
            # Add new item to cache
            self.data_cache[voxel_file] = {"voxel_data": voxel_tensor}
            
            return voxel_tensor
        
        except Exception as e:
            logger.error(f"Error loading voxel file {voxel_file}: {e}")
            # Return zeros as a fallback
            return torch.zeros((5, 21, 21, 21), dtype=torch.float32)
    
    def _process_metadata(self, sample: Dict) -> torch.Tensor:
        """
        Process metadata for a sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tensor containing metadata features
        """
        metadata_tensors = []
        
        if "metadata" in sample:
            metadata = sample["metadata"]
            
            # Process each metadata field
            for field in self.metadata_fields:
                if field in metadata and metadata[field] is not None:
                    # Handle different types of metadata
                    if field == "resname_encoded":
                        # One-hot encoded residue name (20 values)
                        if isinstance(metadata[field], int):
                            # Already encoded as an integer
                            one_hot = torch.zeros(20, dtype=torch.float32)
                            if 0 <= metadata[field] < 20:
                                one_hot[metadata[field]] = 1.0
                            metadata_tensors.append(one_hot)
                        else:
                            # Need to encode the residue name
                            # This is a placeholder - implement proper encoding
                            placeholder = torch.zeros(20, dtype=torch.float32)
                            metadata_tensors.append(placeholder)
                    
                    elif field == "secondary_structure_encoded":
                        # One-hot encoded secondary structure (3 values: helix, sheet, loop)
                        if isinstance(metadata[field], int):
                            one_hot = torch.zeros(3, dtype=torch.float32)
                            if 0 <= metadata[field] < 3:
                                one_hot[metadata[field]] = 1.0
                            metadata_tensors.append(one_hot)
                        else:
                            placeholder = torch.zeros(3, dtype=torch.float32)
                            metadata_tensors.append(placeholder)
                    
                    elif field == "core_exterior_encoded":
                        # One-hot encoded core/exterior (2 values)
                        if isinstance(metadata[field], int):
                            one_hot = torch.zeros(2, dtype=torch.float32)
                            if 0 <= metadata[field] < 2:
                                one_hot[metadata[field]] = 1.0
                            metadata_tensors.append(one_hot)
                        else:
                            placeholder = torch.zeros(2, dtype=torch.float32)
                            metadata_tensors.append(placeholder)
                    
                    else:
                        # Scalar values like normalized_resid, relative_accessibility
                        value = float(metadata[field])
                        metadata_tensors.append(torch.tensor([value], dtype=torch.float32))
        
        # Concatenate all metadata tensors
        if metadata_tensors:
            return torch.cat(metadata_tensors)
        else:
            # Return an empty tensor if no metadata was processed
            return torch.tensor([], dtype=torch.float32)
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing the input tensor (voxel data with optional metadata)
            and the target tensor (RMSF value)
        """
        sample = self.samples[idx]
        
        # Load voxel data
        voxel_data = self._load_voxel_data(sample["voxel_file"])
        
        # Process metadata if needed
        if self.use_metadata and self.metadata_fields:
            metadata = self._process_metadata(sample)
            input_data = (voxel_data, metadata)
        else:
            input_data = voxel_data
        
        # Process target RMSF value
        rmsf_value = sample["rmsf_value"]
        if self.normalize_rmsf and self.rmsf_scaler is not None:
            rmsf_value = self.rmsf_scaler.transform([[rmsf_value]])[0, 0]
        
        target = torch.tensor([rmsf_value], dtype=torch.float32)
        
        return input_data, target

    def get_rmsf_stats(self) -> Dict[str, float]:
        """
        Get statistics for RMSF values.
        
        Returns:
            Dictionary of RMSF statistics
        """
        if self.rmsf_scaler is not None:
            return {
                "mean": float(self.rmsf_scaler.mean_[0]),
                "std": float(np.sqrt(self.rmsf_scaler.var_[0])),
                "min": float(min(sample["rmsf_value"] for sample in self.samples)),
                "max": float(max(sample["rmsf_value"] for sample in self.samples))
            }
        else:
            rmsf_values = [sample["rmsf_value"] for sample in self.samples]
            return {
                "mean": float(np.mean(rmsf_values)),
                "std": float(np.std(rmsf_values)),
                "min": float(min(rmsf_values)),
                "max": float(max(rmsf_values))
            }

    def get_residue_type_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of residue types in the dataset.
        
        Returns:
            Dictionary mapping residue names to counts
        """
        residue_counts = defaultdict(int)
        
        for domain_id, data in self.domain_data.items():
            rmsf_df = data["rmsf_df"]
            if "resname" in rmsf_df.columns:
                for resname in rmsf_df["resname"]:
                    residue_counts[resname] += 1
        
        return dict(residue_counts)


# Augmentation functions
class RandomRotation3D:
    """Apply random 3D rotation to voxel grids."""
    
    def __init__(self, prob: float = 0.5, angles: List[int] = [90, 180, 270]):
        """
        Args:
            prob: Probability of applying the rotation
            angles: Possible rotation angles in degrees
        """
        self.prob = prob
        self.angles = angles
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            # Choose a random axis and angle
            axis = random.randint(0, 2)  # 0=x, 1=y, 2=z
            angle = random.choice(self.angles)
            
            # Convert to numpy for rotation
            x_np = x.numpy()
            
            # Rotate along the chosen axis
            k = angle // 90  # Number of 90-degree rotations
            
            if axis == 0:
                x_np = np.rot90(x_np, k=k, axes=(2, 3))
            elif axis == 1:
                x_np = np.rot90(x_np, k=k, axes=(1, 3))
            else:
                x_np = np.rot90(x_np, k=k, axes=(1, 2))
            
            return torch.from_numpy(x_np)
        
        return x

class RandomFlip3D:
    """Apply random 3D flipping to voxel grids."""
    
    def __init__(self, prob: float = 0.3):
        """
        Args:
            prob: Probability of applying the flip
        """
        self.prob = prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            # Choose a random axis
            axis = random.randint(1, 3)  # 1=x, 2=y, 3=z
            
            # Flip along the chosen axis
            if axis == 1:
                return x.flip(1)
            elif axis == 2:
                return x.flip(2)
            else:
                return x.flip(3)
        
        return x

class AddGaussianNoise:
    """Add Gaussian noise to voxel grids."""
    
    def __init__(self, prob: float = 0.2, scale: float = 0.05):
        """
        Args:
            prob: Probability of adding noise
            scale: Scale of the Gaussian noise
        """
        self.prob = prob
        self.scale = scale
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            noise = torch.randn_like(x) * self.scale
            return x + noise
        
        return x

class ComposeTransforms:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[callable]):
        """
        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

def create_data_loaders(
    voxel_dir: str,
    rmsf_dir: str,
    temperature: Union[int, str] = 320,
    domain_ids: Optional[List[str]] = None,
    use_metadata: bool = True,
    metadata_fields: Optional[List[str]] = None,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 4,
    use_augmentation: bool = True,
    augmentation_params: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        voxel_dir: Directory containing voxelized protein data
        rmsf_dir: Directory containing RMSF CSV files
        temperature: Temperature value or "average"
        domain_ids: List of domain IDs to include (None = all domains)
        use_metadata: Whether to include metadata features
        metadata_fields: List of metadata fields to include
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        num_workers: Number of worker threads for data loading
        use_augmentation: Whether to use data augmentation
        augmentation_params: Parameters for data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate split proportions
    assert 0.0 <= train_split <= 1.0, "Train split must be between 0 and 1"
    assert 0.0 <= val_split <= 1.0, "Validation split must be between 0 and 1"
    assert 0.0 <= test_split <= 1.0, "Test split must be between 0 and 1"
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Split proportions must sum to 1"
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # Create transforms
    train_transform = None
    if use_augmentation:
        params = augmentation_params or {}
        train_transform = ComposeTransforms([
            RandomRotation3D(
                prob=params.get("rotation_prob", 0.5),
                angles=params.get("rotation_angles", [90, 180, 270])
            ),
            RandomFlip3D(prob=params.get("flip_prob", 0.3)),
            AddGaussianNoise(
                prob=params.get("noise_prob", 0.2),
                scale=params.get("noise_scale", 0.05)
            )
        ])
    
    # Create the full dataset
    full_dataset = RMSFDataset(
        voxel_dir=voxel_dir,
        rmsf_dir=rmsf_dir,
        temperature=temperature,
        domain_ids=domain_ids,
        use_metadata=use_metadata,
        metadata_fields=metadata_fields,
        transform=None,  # We'll apply transforms in the subset datasets
        normalize_rmsf=True
    )
    
    # Get RMSF statistics
    rmsf_stats = full_dataset.get_rmsf_stats()
    logger.info(f"RMSF statistics: mean={rmsf_stats['mean']:.4f}, std={rmsf_stats['std']:.4f}, "
              f"min={rmsf_stats['min']:.4f}, max={rmsf_stats['max']:.4f}")
    
    # Split the dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(train_split * dataset_size)
    val_end = train_end + int(val_split * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    logger.info(f"Data split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Create subset datasets
    class TransformSubset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, transform=None):
            super().__init__(dataset, indices)
            self.transform = transform
        
        def __getitem__(self, idx):
            input_data, target = super().__getitem__(idx)
            
            if self.transform is not None:
                if isinstance(input_data, tuple):
                    voxel_data, metadata = input_data
                    voxel_data = self.transform(voxel_data)
                    input_data = (voxel_data, metadata)
                else:
                    input_data = self.transform(input_data)
            
            return input_data, target
    
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, None)
    test_dataset = TransformSubset(full_dataset, test_indices, None)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader