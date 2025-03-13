"""
Logging utilities for ProtFlex.

This module provides functions for setting up and using logging in the ProtFlex package.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime

def setup_logging(config: Dict[str, Any], override_level: Optional[int] = None) -> logging.Logger:
    """
    Set up logging based on configuration.

    Args:
        config: Configuration dictionary
        override_level: Override the log level from config

    Returns:
        Root logger instance
    """
    # Get log file path from config
    log_file = config["output"]["log_file"]
    log_dir = os.path.dirname(log_file)

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Determine log levels
    if override_level is not None:
        console_level = override_level
        file_level = override_level
    else:
        console_level = getattr(logging, config["logging"]["console_level"])
        file_level = getattr(logging, config["logging"]["file_level"])

    # Set root logger level to the most verbose of the two
    root_level = min(console_level, file_level)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(root_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initialization
    logger.info(f"Logging initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.debug(f"Console log level: {logging.getLevelName(console_level)}")
    logger.debug(f"File log level: {logging.getLevelName(file_level)}")

    return logger

def log_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a section header to make log files more readable.

    Args:
        title: Section title
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()

    separator = "=" * (len(title) + 4)
    logger.info(f"\n{separator}\n  {title}  \n{separator}")

def log_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Log configuration parameters.

    Args:
        config: Configuration dictionary
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()

    log_section("Configuration", logger)

    # Log each section of the config
    for section_name, section in config.items():
        logger.info(f"{section_name}:")

        if isinstance(section, dict):
            for key, value in section.items():
                # Format the value for better readability
                if isinstance(value, dict):
                    logger.info(f"  {key}: <dict with {len(value)} items>")
                elif isinstance(value, list) and len(value) > 5:
                    logger.info(f"  {key}: <list with {len(value)} items>")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {section}")

def log_runtime_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log runtime information about the Python environment.

    Args:
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()

    import platform
    import torch

    log_section("Runtime Information", logger)

    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")

    # PyTorch information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Log CPU information
    try:
        import psutil
        logger.info(f"CPU count: {psutil.cpu_count(logical=True)} logical, {psutil.cpu_count(logical=False)} physical")
        mem_info = psutil.virtual_memory()
        logger.info(f"System memory: {mem_info.total / 1e9:.2f} GB total, {mem_info.available / 1e9:.2f} GB available")
    except ImportError:
        logger.info("CPU information not available (psutil not installed)")

def enable_verbose_logging() -> None:
    """
    Enable verbose logging for all loggers.
    """
    # Set root logger to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

    # Set all handlers to DEBUG
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)

    logging.debug("Verbose logging enabled")

def disable_logging() -> None:
    """
    Disable all logging except critical messages.
    """
    # Set root logger to CRITICAL
    logging.getLogger().setLevel(logging.CRITICAL)

    # Set all handlers to CRITICAL
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.CRITICAL)
