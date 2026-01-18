"""
Aggregate classification results from steering experiments.

This script traverses the results repositories and creates summary matrices
for NSS and CII metrics across generation runs and classifier runs.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd


def find_generation_runs(base_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all generation runs that have classification_results subfolder.
    
    Returns:
        List of tuples (generation_run_name, classification_results_path)
    """
    generation_runs = []
    
    for item in base_dir.iterdir():
        if item.is_dir():
            classification_results_path = item / "classification_results"
            if classification_results_path.exists() and classification_results_path.is_dir():
                generation_runs.append((item.name, classification_results_path))
    
    return generation_runs


def find_classifier_runs(classification_results_path: Path) -> List[Tuple[str, Path]]:
    """
    Find all classifier run subfolders containing structured_results_*.json files.
    
    Returns:
        List of tuples (classifier_run_name, json_file_path)
    """
    classifier_runs = []
    
    for item in classification_results_path.iterdir():
        if item.is_dir():
            # Look for structured_results_*.json in this subfolder
            json_files = list(item.glob("structured_results_*.json"))
            if json_files:
                # Take the first one (there should typically be only one)
                classifier_runs.append((item.name, json_files[0]))
    
    return classifier_runs


def extract_metrics(json_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Extract NSS and CII metrics for each target classifier from a results JSON file.
    
    Returns:
        Dict mapping target_classifier -> {"NSS": float, "CII": float}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = {}
    per_target = data.get("results", {}).get("per_target_classifier", {})
    
    for target_classifier, target_data in per_target.items():
        target_metrics = target_data.get("metrics", {})
        metrics[target_classifier] = {
            "NSS": target_metrics.get("NSS"),
            "CII": target_metrics.get("CII")
        }
    
    return metrics


def aggregate_results(base_dir: str | Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Aggregate all classification results into matrices.
    
    Args:
        base_dir: Path to the base directory containing generation run folders
        
    Returns:
        Dict mapping target_classifier -> {"NSS": DataFrame, "CII": DataFrame}
        Each DataFrame has classifier runs as rows and generation runs as columns
    """
    base_dir = Path(base_dir)
    
    # Collect all data
    # Structure: {target_classifier: {metric: {classifier_run: {generation_run: value}}}}
    all_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    
    # Track all classifier runs and generation runs for consistent matrix structure
    all_classifier_runs: Set[str] = set()
    all_generation_runs: Set[str] = set()
    all_target_classifiers: Set[str] = set()
    
    # Find all generation runs
    generation_runs = find_generation_runs(base_dir)
    print(f"Found {len(generation_runs)} generation runs with classification_results")
    
    for gen_run_name, classification_results_path in generation_runs:
        all_generation_runs.add(gen_run_name)
        print(f"\n  Processing generation run: {gen_run_name}")
        
        # Find all classifier runs in this generation run
        classifier_runs = find_classifier_runs(classification_results_path)
        print(f"    Found {len(classifier_runs)} classifier runs")
        
        for classifier_run_name, json_path in classifier_runs:
            all_classifier_runs.add(classifier_run_name)
            print(f"      Processing classifier: {classifier_run_name}")
            
            try:
                # Extract metrics from JSON
                metrics = extract_metrics(json_path)
                
                for target_classifier, target_metrics in metrics.items():
                    all_target_classifiers.add(target_classifier)
                    
                    # Initialize nested dicts if needed
                    if target_classifier not in all_data:
                        all_data[target_classifier] = {"NSS": {}, "CII": {}}
                    
                    for metric_name in ["NSS", "CII"]:
                        if classifier_run_name not in all_data[target_classifier][metric_name]:
                            all_data[target_classifier][metric_name][classifier_run_name] = {}
                        
                        value = target_metrics.get(metric_name)
                        all_data[target_classifier][metric_name][classifier_run_name][gen_run_name] = value
                        
            except Exception as e:
                print(f"        Error processing {json_path}: {e}")
    
    # Convert to DataFrames
    # Rows: classifier runs (sorted), Columns: generation runs (sorted)
    sorted_classifier_runs = sorted(all_classifier_runs)
    sorted_generation_runs = sorted(all_generation_runs)
    
    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    for target_classifier in sorted(all_target_classifiers):
        result[target_classifier] = {}
        
        for metric_name in ["NSS", "CII"]:
            # Build the matrix
            matrix_data = []
            
            for classifier_run in sorted_classifier_runs:
                row = []
                for gen_run in sorted_generation_runs:
                    value = all_data.get(target_classifier, {}).get(metric_name, {}).get(classifier_run, {}).get(gen_run)
                    row.append(value)
                matrix_data.append(row)
            
            df = pd.DataFrame(
                matrix_data,
                index=sorted_classifier_runs,
                columns=sorted_generation_runs
            )
            df.index.name = "classifier_run"
            result[target_classifier][metric_name] = df
    
    return result


def save_matrices(matrices: Dict[str, Dict[str, pd.DataFrame]], output_dir: str | Path) -> None:
    """
    Save all matrices as CSV files.
    
    Args:
        matrices: Dict from aggregate_results()
        output_dir: Directory to save CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for target_classifier, metrics_dict in matrices.items():
        for metric_name, df in metrics_dict.items():
            # Create safe filename (replace spaces with underscores)
            safe_name = target_classifier.replace(" ", "_")
            filename = f"{safe_name}_{metric_name}.csv"
            filepath = output_dir / filename
            
            df.to_csv(filepath)
            print(f"Saved: {filepath}")


def main():
    """Main entry point for aggregation script."""
    # Base directory relative to workspace root
    workspace_root = Path(__file__).parent.parent.parent.parent  # Go up from steering/ to workspace root
    base_dir = workspace_root / "data" / "steering" / "tests"
    output_dir = base_dir / "aggregated_results"
    # create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results
    matrices = aggregate_results(base_dir)
    
    print("\n" + "=" * 60)
    print("Summary of aggregated matrices:")
    print("=" * 60)
    
    for target_classifier, metrics_dict in matrices.items():
        print(f"\n{target_classifier}:")
        for metric_name, df in metrics_dict.items():
            print(f"  {metric_name}: {df.shape[0]} classifier runs x {df.shape[1]} generation runs")
    
    # Save to CSV
    print("\n" + "=" * 60)
    print("Saving matrices to CSV...")
    print("=" * 60)
    save_matrices(matrices, output_dir)
    
    print("\n" + "=" * 60)
    print("Aggregation complete!")


if __name__ == "__main__":
    main()

