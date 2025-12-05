"""
Activation tracking system for mechanistic interpretability project.

This module provides functions to track and manage activation runs:
- Generate unique IDs for each activation run
- Maintain a markdown file for human-readable documentation
- Maintain a JSON file for programmatic access
- Support lookups by ID
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ActivationTracker:
    """Manages tracking of activation runs with .md and .json files."""
    
    def __init__(self, tracking_dir: Path = None):
        """
        Initialize the activation tracker.
        
        Args:
            tracking_dir: Directory where tracking files will be stored.
                         Defaults to get_features directory.
        """
        if tracking_dir is None:
            # Default to get_features directory
            tracking_dir = Path(__file__).parent
        
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.md_file = self.tracking_dir / "activation_runs.md"
        self.json_file = self.tracking_dir / "activation_runs.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize tracking files if they don't exist."""
        if not self.json_file.exists():
            with open(self.json_file, 'w') as f:
                json.dump({"runs": {}, "next_id": 1}, f, indent=2)
        
        if not self.md_file.exists():
            with open(self.md_file, 'w') as f:
                f.write("# Activation Runs Log\n\n")
                f.write("This file tracks all activation extraction runs.\n\n")
                f.write("| ID | Date | Model | Dataset | Category | Run Name | Layers | Authors | Format | Docs/Author | Min Doc Length | Setting | Path |\n")
                f.write("|----|------|-------|---------|----------|----------|--------|---------|--------|-------------|----------------|---------|------|\n")
    
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
        return str(next_id).zfill(4)  # Zero-pad to 4 digits (e.g., "0001")
    
    def _get_next_id(self) -> int:
        """Get the next ID number and increment it."""
        data = self._load_json()
        next_id = data.get("next_id", 1)
        data["next_id"] = next_id + 1
        self._save_json(data)
        return next_id
    
    def register_run(
        self,
        model: str,
        dataset: str,
        run_name: str,
        layers: List[int],
        activation_path: str,
        category: Optional[str] = None,
        authors: Optional[List[str]] = None,
        storage_format: str = "sparse",  # "sparse", "dense", or "both"
        n_docs_per_author: Optional[int] = None,
        min_length_doc: Optional[int] = None,
        setting: Optional[str] = None,
        date: Optional[str] = None
    ) -> str:
        """
        Register a new activation run.
        
        Args:
            model: Model name (e.g., "google/gemma-2-2b")
            dataset: Dataset name (e.g., "AuthorMix", "news", "synthetic")
            run_name: Name of the run
            layers: List of layer indices
            activation_path: Path to the activation directory
            category: Category name (for news dataset)
            authors: List of authors processed
            storage_format: "sparse", "dense", or "both"
            n_docs_per_author: Number of documents per author
            min_length_doc: Minimum document length
            setting: Setting used (e.g., "baseline", "prompted")
            date: Date string (defaults to current date)
        
        Returns:
            The generated ID for this run
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate ID
        run_id = self._generate_id()
        self._get_next_id()  # Increment the counter
        
        # Format layers
        layers_str = ", ".join(map(str, sorted(layers)))
        
        # Format authors
        if authors is None or len(authors) == 0:
            authors_str = "N/A"
        elif len(authors) <= 5:
            authors_str = ", ".join(authors)
        else:
            authors_str = f"{len(authors)} authors ({', '.join(authors[:3])}...)"
        
        # Prepare metadata
        metadata = {
            "id": run_id,
            "date": date,
            "model": model,
            "dataset": dataset,
            "category": category,
            "run_name": run_name,
            "layers": layers,
            "authors": authors,
            "storage_format": storage_format,
            "n_docs_per_author": n_docs_per_author,
            "min_length_doc": min_length_doc,
            "setting": setting,
            "activation_path": activation_path
        }
        
        # Save to JSON
        data = self._load_json()
        data["runs"][run_id] = metadata
        self._save_json(data)
        
        # Add to markdown file
        category_str = category if category else "N/A"
        format_str = storage_format
        docs_str = str(n_docs_per_author) if n_docs_per_author else "N/A"
        min_len_str = str(min_length_doc) if min_length_doc else "N/A"
        setting_str = setting if setting else "N/A"
        
        md_row = (
            f"| {run_id} | {date} | {model} | {dataset} | {category_str} | "
            f"{run_name} | {layers_str} | {authors_str} | {format_str} | "
            f"{docs_str} | {min_len_str} | {setting_str} | {activation_path} |\n"
        )
        
        with open(self.md_file, 'a') as f:
            f.write(md_row)
        
        logger.info(f"Registered activation run with ID {run_id}")
        return run_id
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run metadata by ID.
        
        Args:
            run_id: The run ID
        
        Returns:
            Run metadata dictionary or None if not found
        """
        data = self._load_json()
        return data["runs"].get(run_id)
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all runs.
        
        Returns:
            List of all run metadata dictionaries
        """
        data = self._load_json()
        runs = list(data["runs"].values())
        # Sort by date (most recent first)
        runs.sort(key=lambda x: x.get("date", ""), reverse=True)
        return runs
    
    def search_runs(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        category: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for runs matching criteria.
        
        Args:
            model: Filter by model name
            dataset: Filter by dataset name
            category: Filter by category
            run_name: Filter by run name
        
        Returns:
            List of matching run metadata dictionaries
        """
        all_runs = self.list_runs()
        results = []
        
        for run in all_runs:
            if model and run.get("model") != model:
                continue
            if dataset and run.get("dataset") != dataset:
                continue
            if category and run.get("category") != category:
                continue
            if run_name and run.get("run_name") != run_name:
                continue
            results.append(run)
        
        return results


def get_tracker(tracking_dir: Optional[Path] = None) -> ActivationTracker:
    """
    Get or create a tracker instance.
    
    Args:
        tracking_dir: Optional directory for tracking files
    
    Returns:
        ActivationTracker instance
    """
    return ActivationTracker(tracking_dir)
