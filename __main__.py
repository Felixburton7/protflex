"""
Main entry point for ProtFlex package.

When the package is run directly using `python -m protflex`, this script will be executed.
"""

import os
import sys
import argparse
import logging
from protflex.cli.cli import main
from protflex.utils.logging_utils import setup_logging
from protflex.config import load_config, process_config
from protflex.utils.file_utils import ensure_dir
from protflex.training.validators import DataValidator

def validate_data():
    """Run data validation to identify common issues."""
    parser = argparse.ArgumentParser(description='Validate ProtFlex data for common issues')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--domain', '-d', type=str, nargs='+', help='Specific domain(s) to validate')
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(cfg, logging.INFO)
    
    # Set up data directories
    data_dir = cfg['input']['data_dir']
    voxel_dir = os.path.join(data_dir, cfg['input']['voxel_dir'])
    rmsf_dir = os.path.join(data_dir, cfg['input']['rmsf_dir'])
    temperature = cfg['input']['temperature']
    
    logger.info(f"Validating data in {data_dir}")
    logger.info(f"Temperature: {temperature}")
    
    # Create validator
    validator = DataValidator(voxel_dir, rmsf_dir)
    
    # Get domain IDs
    if args.domain:
        domain_ids = args.domain
    else:
        domain_ids = cfg['input']['domain_ids']
        if not domain_ids:
            # Find all domains
            domain_ids = [d for d in os.listdir(voxel_dir) if os.path.isdir(os.path.join(voxel_dir, d))]
    
    # Run validation
    results = validator.validate_dataset(domain_ids, temperature)
    
    # Print summary
    logger.info(f"Validation results: {results['valid_domains']} of {results['total_domains']} domains valid")
    
    # Print issues for each domain
    for domain_id, domain_result in results['domain_results'].items():
        if not domain_result['is_valid']:
            logger.warning(f"Domain {domain_id} has issues:")
            for issue in domain_result['issues']:
                logger.warning(f"  - {issue}")

if __name__ == "__main__":
    # Check if running validation
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        # Remove the 'validate' argument
        sys.argv.pop(1)
        validate_data()
    else:
        # Execute the main CLI function
        sys.exit(main())