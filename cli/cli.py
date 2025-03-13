
#!/usr/bin/env python3
"""
Command-line interface for ProtFlex.

This module provides the main entry point for the ProtFlex CLI application.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

from protflex.cli.commands import train, predict, evaluate
from protflex.config import config
from protflex.utils.logging_utils import setup_logging

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the ProtFlex CLI.

    Args:
        args: Command line arguments (if None, sys.argv[1:] is used)

    Returns:
        Exit code
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description='ProtFlex: Deep learning for protein flexibility prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add global arguments
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Suppress all output except errors')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Add command subparsers
    train.add_subparser(subparsers)
    predict.add_subparser(subparsers)
    evaluate.add_subparser(subparsers)

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # If no command specified, show help and exit
    if not parsed_args.command:
        parser.print_help()
        return 1

    # Load configuration
    try:
        cfg = config.load_config(parsed_args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Configure logging based on verbosity
    log_level = logging.INFO
    if parsed_args.verbose:
        log_level = logging.DEBUG
    elif parsed_args.quiet:
        log_level = logging.ERROR

    logger = setup_logging(cfg, log_level)

    # Execute the specified command
    try:
        if parsed_args.command == 'train':
            return train.run(parsed_args, cfg)
        elif parsed_args.command == 'predict':
            return predict.run(parsed_args, cfg)
        elif parsed_args.command == 'evaluate':
            return evaluate.run(parsed_args, cfg)
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    except Exception as e:
        logger.exception(f"Error executing command {parsed_args.command}: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
