"""
Configuration handling for ProtFlex.

This module loads and validates YAML configuration files for the ProtFlex pipeline.
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Union

# Default configuration settings
DEFAULT_CONFIG = {
    "input": {
        "data_dir": os.path.expanduser("~/data"),
        "voxel_dir": "processed/voxelized_output",
        "rmsf_dir": "interim/per-residue-rmsf",
        "temperature": 320,
        "domain_ids": [],  # Empty means process all domains
        "use_metadata": True,
        "metadata_fields": ["resname_encoded", "normalized_resid", "secondary_structure_encoded"]
    },
    "output": {
        "base_dir": os.path.expanduser("~/protflex_results"),
        "model_dir": "models",
        "results_dir": "results",
        "visualizations_dir": "visualizations",
        "log_file": "protflex.log"
    },
    "model": {
        "architecture": "protflex_cnn",
        "input_channels": 5,  # C, N, O, CA, CB channels
        "channel_growth_rate": 1.5,
        "num_residual_blocks": 4,
        "use_multiscale": True,
        "use_bottleneck": True,
        "dropout_rate": 0.2,
        "include_metadata": True
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "early_stopping_patience": 10,
        "lr_scheduler": "reduce_on_plateau",
        "lr_scheduler_params": {
            "factor": 0.5,
            "patience": 5
        },
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "random_seed": 42
    },
    "logging": {
        "level": "INFO",
        "console_level": "INFO",
        "file_level": "DEBUG",
        "show_progress_bars": True
    },
    "visualization": {
        "plot_loss": True,
        "plot_predictions": True,
        "plot_residue_type_analysis": True,
        "plot_error_distribution": True,
        "save_format": "png",
        "dpi": 300
    }
}

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to default values.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing merged configuration
    """
    logger = logging.getLogger(__name__)
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load user configuration if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # Deep merge configs
                    deep_merge(config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
    
    # Process and validate configuration
    config = process_config(config)
    
    return config

def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and validate configuration:
    - Expand user paths
    - Set derived values
    - Validate settings
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Processed configuration dictionary
    """
    # Expand user paths
    for section in ["input", "output"]:
        for key in config[section]:
            if isinstance(config[section][key], str) and "~" in config[section][key]:
                config[section][key] = os.path.expanduser(config[section][key])
    
    # Ensure all required directories are present in output configuration
    for dir_key in ["model_dir", "results_dir", "visualizations_dir"]:
        if dir_key in config["output"]:
            # Make the path relative to base_dir if not absolute
            if not os.path.isabs(config["output"][dir_key]):
                config["output"][dir_key] = os.path.join(
                    config["output"]["base_dir"], 
                    config["output"][dir_key]
                )
    
    # Process log file path
    if "log_file" in config["output"] and not os.path.isabs(config["output"]["log_file"]):
        config["output"]["log_file"] = os.path.join(
            config["output"]["base_dir"],
            config["output"]["log_file"]
        )
    
    # Set model metadata parameters based on input configuration
    if config["model"]["include_metadata"] and config["input"]["use_metadata"]:
        # Calculate metadata_features dimension based on the metadata fields
        metadata_features = 0
        if "resname_encoded" in config["input"]["metadata_fields"]:
            metadata_features += 20  # 20 standard amino acids
        if "normalized_resid" in config["input"]["metadata_fields"]:
            metadata_features += 1
        if "secondary_structure_encoded" in config["input"]["metadata_fields"]:
            metadata_features += 3  # 3 classes: helix, sheet, loop
        if "core_exterior_encoded" in config["input"]["metadata_fields"]:
            metadata_features += 2  # core or exterior
        if "relative_accessibility" in config["input"]["metadata_fields"]:
            metadata_features += 1
        
        config["model"]["metadata_features"] = metadata_features
    else:
        config["model"]["include_metadata"] = False
        config["model"]["metadata_features"] = 0
    
    return config

def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge source dict into target dict.
    
    Args:
        target: Target dictionary to merge into
        source: Source dictionary to merge from
        
    Returns:
        Merged dictionary (target is modified in-place)
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target

def create_example_config() -> None:
    """
    Create an example configuration file.
    """
    with open('config.yaml.example', 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

def get_temperature_dir(temperature: Union[int, str]) -> str:
    """
    Get the directory name for a temperature value.
    
    Args:
        temperature: Temperature value or "average"
        
    Returns:
        Directory name for the temperature
    """
    if temperature == "average":
        return "average"
    else:
        return str(temperature)

if __name__ == "__main__":
    # Generate example configuration file when run directly
    create_example_config()
    print("Created example configuration file: config.yaml.example")