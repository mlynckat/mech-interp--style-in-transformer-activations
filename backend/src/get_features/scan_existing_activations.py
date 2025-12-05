"""
Script to scan existing activation directories and populate tracking files retroactively.

This script:
1. Scans data/raw_features and data/raw_dense_features directories
2. Extracts metadata from directory structure and file patterns
3. Registers runs in the tracking system
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from backend.src.get_features.activation_tracking import get_tracker
from backend.src.utils.shared_utilities import ActivationMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse activation filename to extract metadata.
    
    Expected format after removing extensions:
    {unimportant}__{model}__{layer_type}__{unimportant}__{author}__layer_{N}
    
    Returns:
        Dict with parsed info or None if pattern doesn't match
    """
    # Remove .sparse and .npz extensions
    parsed = filename.replace(".sparse", "").replace(".npz", "").split("__")
    
    # Need at least 6 parts: [0] unimportant, [1] model, [2] layer_type, 
    # [3] unimportant, [4] author, [5] layer_{N}
    if len(parsed) < 6:
        return None
    
    # Extract layer number from parsed[5] which should be "layer_{N}"
    layer_match = re.match(r'layer_(\d+)', parsed[5])
    if not layer_match:
        return None
    
    layer = int(layer_match.group(1))
    
    # Extract model and convert underscores to slashes
    model = parsed[1].replace('_', '/')
    
    # Extract layer type
    layer_type = parsed[2]
    
    # Extract author (may contain spaces)
    author = parsed[4]
    
    # Determine format from original filename
    format_type = 'sparse' if '.sparse.' in filename or filename.endswith('.sparse.npz') else 'dense'
    
    # Extract setting from filename if present
    setting = None
    if 'baseline' in filename:
        setting = 'baseline'
    elif 'prompted' in filename:
        setting = 'prompted'

    
    result = {
        'model': model,
        'layer_type': layer_type,
        'author': author,
        'layer': layer,
        'format': format_type,
        'setting': setting
    }
    
    return result


def extract_metadata_from_dir(dir_path: Path) -> Optional[Dict]:
    """
    Extract metadata from directory structure and files.
    
    Expected structure:
    - data/raw_features/{dataset}/{category}/{model}/{run_name}/
    - data/raw_features/{dataset}/{model}/{run_name}/
    - data/raw_dense_features/{dataset}/{category}/{model}/{run_name}/
    - data/raw_dense_features/{dataset}/{model}/{run_name}/
    
    Returns:
        Dict with metadata or None if structure doesn't match
    """
    parts = dir_path.parts
    
    # Try to match expected structure
    # Look for pattern: .../data/{base_dir}/{dataset}/{category?}/{model}/{run_name}
    base_dirs = ['raw_features', 'raw_dense_features']
    base_dir = None
    base_idx = None
    
    for i, part in enumerate(parts):
        if part in base_dirs:
            base_dir = part
            base_idx = i
            break
    
    if base_dir is None:
        return None
    
    # Extract components
    try:
        dataset_idx = base_idx + 1
        if dataset_idx >= len(parts):
            return None
        
        dataset = parts[dataset_idx]
        
        # Check if next part is category or model
        next_idx = dataset_idx + 1
        if next_idx >= len(parts):
            return None
        
        category = None
        model_idx = next_idx
        
        # Check if it's a category (common categories: politics, sports, etc.)
        # or if it looks like a model name (contains gemma or has underscores)
        potential_category = parts[next_idx]
        if not ('gemma' in potential_category.lower() or '_' in potential_category):
            category = potential_category
            model_idx = next_idx + 1
        
        if model_idx >= len(parts):
            return None
        
        model_raw = parts[model_idx]
        model = model_raw.replace('_', '/')
        
        run_name_idx = model_idx + 1
        if run_name_idx >= len(parts):
            return None
        
        run_name = parts[run_name_idx]
        
        # Determine storage format from base directory
        storage_format = "dense" if base_dir == "raw_dense_features" else "sparse"
        
        return {
            'dataset': dataset,
            'category': category,
            'model': model,
            'run_name': run_name,
            'storage_format': storage_format,
            'base_dir': base_dir
        }
    except Exception as e:
        logger.warning(f"Error parsing directory structure for {dir_path}: {e}")
        return None


def scan_directory_for_activations(dir_path: Path) -> Dict:
    """
    Scan a directory for activation files and extract metadata.
    
    Returns:
        Dict with:
        - layers: set of layer indices
        - authors: set of authors
        - formats: set of formats (sparse, dense, both)
        - is_synthetic: bool
        - layer_types: set of layer types
        - settings: set of settings (baseline, prompted)
    """
    result = {
        'layers': set(),
        'authors': set(),
        'formats': set(),
        'layer_types': set(),
        'settings': set()
    }
    
    # Scan for activation files
    for file_path in dir_path.glob('*.npz'):
        parsed = parse_filename(file_path.name)
        if parsed:
            result['layers'].add(parsed['layer'])
            result['authors'].add(parsed['author'])
            result['formats'].add(parsed['format'])
            result['layer_types'].add(parsed['layer_type'])
            result['settings'].add(parsed['setting'])
    
    # Determine format
    if len(result['formats']) > 1:
        result['format'] = 'both'
    elif len(result['formats']) == 1:
        result['format'] = list(result['formats'])[0]
    else:
        result['format'] = 'unknown'
    
    return result


def get_file_modification_time(dir_path: Path) -> Optional[str]:
    """Get the earliest modification time of activation files in the directory."""
    times = []
    for file_path in dir_path.glob('*.npz'):
        if file_path.exists():
            print(os.stat(file_path))
            times.append(os.stat(file_path).st_ctime)
    
    if times:
        earliest = min(times)
        return datetime.fromtimestamp(earliest).strftime("%Y-%m-%d %H:%M:%S")
    return None


def scan_and_register_existing_runs(
    base_paths: List[Path],
    tracker=None
):
    """
    Scan existing activation directories and register them in the tracking system.
    
    Args:
        base_paths: List of base directories to scan (e.g., [Path("data/raw_features"), Path("data/raw_dense_features")])
        tracker: Optional tracker instance (will create if None)
    """
    if tracker is None:
        tracker = get_tracker()
    
    # Get existing run IDs to avoid duplicates
    existing_runs = tracker.list_runs()
    existing_paths = {run.get('activation_path') for run in existing_runs}
    
    registered_count = 0
    skipped_count = 0
    
    for base_path in base_paths:
        if not base_path.exists():
            logger.info(f"Base path does not exist, skipping: {base_path}")
            continue
        
        logger.info(f"Scanning {base_path}...")
        
        # Walk through directory structure
        for dir_path in base_path.rglob('*'):
            if not dir_path.is_dir():
                continue
            
            # Check if this directory contains activation files
            has_activations = any(dir_path.glob('*.npz'))
            if not has_activations:
                continue
            
            # Skip if already registered
            if str(dir_path) in existing_paths:
                skipped_count += 1
                continue
            
            # Extract metadata from directory structure
            dir_metadata = extract_metadata_from_dir(dir_path)
            if not dir_metadata:
                logger.warning(f"Could not parse directory structure: {dir_path}")
                continue
            
            # Scan directory for activation details
            scan_result = scan_directory_for_activations(dir_path)
            
            if not scan_result['layers']:
                logger.warning(f"No layers found in {dir_path}, skipping")
                continue
            
            # Get file modification time as proxy for date
            date = get_file_modification_time(dir_path)
            
            # Extract setting from scan results
            setting = None
            if scan_result['settings']:
                # If all files have the same setting, use it
                if len(scan_result['settings']) == 1:
                    setting = list(scan_result['settings'])[0]
                # If mixed settings, we'll use None (could be 'both' in the future)
            
            
            # Register the run
            try:
                run_id = tracker.register_run(
                    model=dir_metadata['model'],
                    dataset=dir_metadata['dataset'],
                    run_name=dir_metadata['run_name'],
                    layers=sorted(list(scan_result['layers'])),
                    activation_path=str(dir_path),
                    category=dir_metadata['category'],
                    authors=sorted(list(scan_result['authors'])) if scan_result['authors'] else None,
                    storage_format=scan_result['format'],
                    n_docs_per_author=None,  # Can't determine from directory
                    min_length_doc=None,  # Can't determine from directory
                    setting=setting,
                    date=date
                )
                logger.info(f"Registered run {run_id}: {dir_metadata['model']} - {dir_metadata['dataset']} - {dir_metadata['run_name']}")
                registered_count += 1
            except Exception as e:
                logger.error(f"Failed to register {dir_path}: {e}")
    
    logger.info(f"\n=== Scan Complete ===")
    logger.info(f"Registered: {registered_count} new runs")
    logger.info(f"Skipped: {skipped_count} already registered runs")


def main():
    """Main entry point for scanning existing activations."""
    # Get the project root directory (go up from backend/src/get_features to project root)
    # __file__ is: backend/src/get_features/scan_existing_activations.py
    # We need to go up 4 levels: get_features -> src -> backend -> project_root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Paths relative to project root
    base_paths = [
        project_root / "data" / "raw_features",
        project_root / "data" / "raw_dense_features"
    ]
    
    logger.info("Starting scan of existing activation directories...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Scanning paths: {[str(p) for p in base_paths]}")
    
    scan_and_register_existing_runs(base_paths)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
