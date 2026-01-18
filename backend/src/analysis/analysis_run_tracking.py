"""
Analysis run tracking system for mechanistic interpretability project.

This module provides functions to:
- Look up activation runs by ID
- Generate output directories automatically
- Track analysis runs in .md and .json files
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from backend.src.get_features.activation_tracking import get_tracker

logger = logging.getLogger(__name__)


class AnalysisRunTracker:
    """Manages tracking of analysis runs with .md and .json files."""
    
    def __init__(self, tracking_dir: Path = None):
        """
        Initialize the analysis tracker.
        
        Args:
            tracking_dir: Directory where tracking files will be stored.
                         Defaults to analysis directory.
        """
        if tracking_dir is None:
            # Default to analysis directory
            tracking_dir = Path(__file__).parent
        
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.md_file = self.tracking_dir / "analysis_runs.md"
        self.json_file = self.tracking_dir / "analysis_runs.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize tracking files if they don't exist."""
        if not self.json_file.exists():
            with open(self.json_file, 'w') as f:
                json.dump({"runs": {}, "next_id": 1}, f, indent=2)
        
        if not self.md_file.exists():
            with open(self.md_file, 'w') as f:
                f.write("# Analysis Runs Log\n\n")
                f.write("This file tracks all analysis runs performed on activations.\n\n")
                f.write("| ID | Date | Activation Run ID | Analysis Type | Data Path | Output Path |\n")
                f.write("|----|------|-------------------|---------------|-----------|-------------|\n")
    
    def _load_json(self) -> Dict[str, Any]:
        """Load the JSON tracking file."""
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def _save_json(self, data: Dict[str, Any]):
        """Save the JSON tracking file."""
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_id(self) -> str:
        """Generate a short numerical ID."""
        data = self._load_json()
        next_id = data.get("next_id", 1)
        return str(next_id).zfill(4)
    
    def _get_next_id(self) -> int:
        """Get the next ID number and increment it."""
        data = self._load_json()
        next_id = data.get("next_id", 1)
        data["next_id"] = next_id + 1
        self._save_json(data)
        return next_id
    
    def register_analysis(
        self,
        activation_run_id: str,
        analysis_type: str,
        data_path: str,
        output_path: str,
        date: Optional[str] = None
    ) -> str:
        """
        Register a new analysis run.
        
        Args:
            activation_run_id: ID of the activation run being analyzed
            analysis_type: Type of analysis (e.g., "explore", "cluster", "classification", "entropy", "feature_importance")
            data_path: Path to the activation data
            output_path: Path to the analysis outputs
            date: Date string (defaults to current date)
        
        Returns:
            The generated ID for this analysis run
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate ID
        analysis_id = self._generate_id()
        self._get_next_id()  # Increment the counter
        
        # Prepare metadata
        metadata = {
            "id": analysis_id,
            "date": date,
            "activation_run_id": activation_run_id,
            "analysis_type": analysis_type,
            "data_path": data_path,
            "output_path": output_path
        }
        
        # Save to JSON
        data = self._load_json()
        data["runs"][analysis_id] = metadata
        self._save_json(data)
        
        # Add to markdown file
        md_row = (
            f"| {analysis_id} | {date} | {activation_run_id} | {analysis_type} | "
            f"{data_path} | {output_path} |\n"
        )
        
        with open(self.md_file, 'a') as f:
            f.write(md_row)
        
        logger.info(f"Registered analysis run with ID {analysis_id}")
        return analysis_id
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis metadata by ID."""
        data = self._load_json()
        return data["runs"].get(analysis_id)


def get_activation_run_info(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get activation run information by ID.
    
    Args:
        run_id: The activation run ID
    
    Returns:
        Run metadata dictionary or None if not found
    """
    tracker = get_tracker()
    return tracker.get_run(run_id)


def generate_output_path(data_path: str, analysis_type: str, run_name: Optional[str] = None) -> Path:
    """
    Generate output path based on data path structure.
    
    Expected input structure:
    - data/raw_features/{dataset}/{category?}/{model}/{run_name}/
    - data/raw_dense_features/{dataset}/{category?}/{model}/{run_name}/
    
    Output structure:
    - data/output_data/{dataset}/{category?}/{model}/{run_name}/{analysis_type}/
    
    Args:
        data_path: Path to activation data
        analysis_type: Type of analysis (e.g., "explore", "cluster")
        run_name: Optional run name override
    
    Returns:
        Path to output directory
    """
    # Convert to Path object
    data_path_str = str(data_path)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Handle absolute paths
    if Path(data_path_str).is_absolute():
        # Find 'data' in the path
        parts = Path(data_path_str).parts
        data_idx = None
        for i, part in enumerate(parts):
            if part == 'data':
                data_idx = i
                break
        
        if data_idx is not None:
            # Reconstruct relative path from 'data' onwards
            relative_path = Path(*parts[data_idx:])
            project_root = Path(*parts[:data_idx])
        else:
            relative_path = Path(data_path_str)
    else:
        relative_path = Path(data_path_str)
    
    # Convert to string and replace base directory
    path_str = str(relative_path)
    
    # Replace raw_features or raw_dense_features with output_data
    # Or if already in output_data (e.g., residual activations), just use it directly
    if 'raw_features' in path_str:
        output_path_str = path_str.replace('raw_features', 'output_data', 1)
    elif 'raw_dense_features' in path_str:
        output_path_str = path_str.replace('raw_dense_features', 'output_data', 1)
    elif 'output_data' in path_str:
        # Path already points to output_data (e.g., residual activations from previous analysis)
        output_path_str = path_str
    else:
        raise ValueError(f"Could not determine output path from data path: {data_path}. "
                        f"Expected path to contain 'raw_features', 'raw_dense_features', or 'output_data'")
    
    # If run_name override is provided, replace the last part before analysis_type
    if run_name:
        output_path_obj = Path(output_path_str)
        # Replace the last component (run_name) if it exists
        if len(output_path_obj.parts) > 0:
            output_path_str = str(output_path_obj.parent / run_name)
    
    # Append analysis_type
    output_path_obj = Path(output_path_str) / analysis_type
    
    # Make absolute path
    if project_root:
        output_path = project_root / output_path_obj
    else:
        output_path = output_path_obj
    
    # Create directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def get_data_and_output_paths(
    run_id: Optional[str] = None,
    data_path: Optional[str] = None,
    analysis_type: str = "analysis",
    run_name_override: Optional[str] = None
) -> tuple[Path, Path, Optional[Dict[str, Any]]]:
    """
    Get data path and generate output path from either run_id or data_path.
    
    Args:
        run_id: Activation run ID (takes precedence over data_path)
        data_path: Direct path to activation data
        analysis_type: Type of analysis for output directory
        run_name_override: Optional run name override
    
    Returns:
        Tuple of (data_path, output_path, activation_run_info)
    """
    activation_run_info = None
    
    if run_id:
        # Look up activation run
        activation_run_info = get_activation_run_info(run_id)
        if not activation_run_info:
            raise ValueError(f"Activation run ID {run_id} not found in tracking system")
        
        data_path = activation_run_info['activation_path']
        logger.info(f"Using activation run {run_id}: {data_path}")
    elif data_path:
        # Use provided data path directly
        data_path = str(data_path)
        logger.info(f"Using provided data path: {data_path}")
    else:
        raise ValueError("Either --run_id or --data_path must be provided")
    
    # Generate output path
    run_name = run_name_override or (activation_run_info.get('run_name') if activation_run_info else None)
    output_path = generate_output_path(data_path, analysis_type, run_name)
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    
    return Path(data_path), output_path, activation_run_info
