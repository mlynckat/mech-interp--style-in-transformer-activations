import argparse
import os
import json
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import logging

# Set up logging
logger = logging.getLogger(__name__)

from backend.src.utils.shared_utilities import EntropyFilenamesLoader, TokenandFullTextFilenamesLoader
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
)


def create_heatmap(entropy: np.ndarray, tokens: List[List[str]], author: str, entropy_type: str, path_to_save_heatmap: str, prompted: str):
    """Create a heatmap of the entropy where tokens exist and are not <bos> or <pad>"""
    if entropy_type == "entropy_loss":
        doc_lengths = [len(token) for token in tokens]
    elif entropy_type == "cross_entropy_loss":
        doc_lengths = [len(token)-1 for token in tokens]
    else:
        raise ValueError(f"Invalid entropy type: {entropy_type}")
    

    # mask values in every row that is greater than the document length
    for i, length in enumerate(doc_lengths):
        entropy[i, length:] = np.nan
    logger.debug(entropy[0])

    entropy = entropy[:, :max(doc_lengths)]

    # Create your colormap
    #cmap = mpl.cm.viridis  # or whatever base cmap you want
    # Clone if necessary, or use get_cmap
    #cmap = mpl.cm.get_cmap("viridis").copy()
    #cmap.set_bad(color="white")  # masked / NaN values will be white
    
    plt.figure(figsize=(12, 10))
    # Plot
    if prompted == "prompted-baseline":
        ax = sns.heatmap(entropy, cmap="RdBu", 
                        mask=np.isnan(entropy),  # mask so the nan cells aren’t coloured
                        center=0,
                        cbar=True)
    else:
        ax = sns.heatmap(entropy, cmap="viridis", 
                    mask=np.isnan(entropy),  # mask so the nan cells aren’t coloured
                    cbar=True)



    plt.title(f"{author} {entropy_type} heatmap {prompted}")
    plt.savefig(path_to_save_heatmap / f"{author}_{entropy_type}_heatmap_{prompted}.png")
    plt.close()

def retrieve_doc_tok_positions_of_improved_entropies(entropy_diffs: np.ndarray):
    """Retrieve document and token positions of improved entropies"""
    doc_inds, token_inds = np.where(entropy_diffs < 0)
    return doc_inds, token_inds

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Activation run ID (takes precedence over --path_to_data)"
    )
    parser.add_argument(
        "--path_to_data",
        type=str,
        default=None,
        help="Path to the data (required if --run_id not provided)"
    )
    parser.add_argument(
        "--include_authors",
        type=str,
        nargs="+",
        default=None,
        help="The authors to include in the analysis"
    )
    
    args = parser.parse_args()
    
    # Validate that either run_id or path_to_data is provided
    if not args.run_id and not args.path_to_data:
        parser.error("Either --run_id or --path_to_data must be provided")
    
    return args

def main():
    args = parse_arguments()
    
    # Get data and output paths
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type="entropy_exploration",
        run_name_override=None
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type="entropy_exploration",
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")
    
    path_to_data = str(data_path)
    path_to_outputs = str(output_path)

    entropies_filenames_structured = EntropyFilenamesLoader(data_dir=Path(path_to_data), include_authors=args.include_authors).get_structured_filenames()
    tokens_and_full_text_filenames_structured = TokenandFullTextFilenamesLoader(data_dir=Path(path_to_data), include_authors=args.include_authors).get_structured_filenames()


    for entropy_type, authors_items in entropies_filenames_structured.items():
        for author, prompted_items in authors_items.items():
            filename_prompted = prompted_items["prompted"]
            filename_baseline = prompted_items["baseline"]

            entropy_prompted = np.load(Path(path_to_data) / filename_prompted)
            entropy_baseline = np.load(Path(path_to_data) / filename_baseline)

            with open(Path(path_to_data) / tokens_and_full_text_filenames_structured["tokens"][author]["prompted"], "r", encoding="utf-8") as f:
                tokens_prompted = json.load(f)
                tokens_prompted_clean = []
                for doc in tokens_prompted:
                    tokens_prompted_clean.append([token for token in doc if token != "<bos>" and token != "<pad>"])
            with open(Path(path_to_data) / tokens_and_full_text_filenames_structured["tokens"][author]["baseline"], "r", encoding="utf-8") as f:
                tokens_baseline = json.load(f)
                tokens_baseline_clean = []
                for doc in tokens_baseline:
                    tokens_baseline_clean.append([token for token in doc if token != "<bos>" and token != "<pad>"])

            logger.debug(tokens_prompted_clean[0:5])
            logger.debug("--------------------------------")
            logger.debug(tokens_baseline_clean[0:5])
            assert tokens_prompted_clean == tokens_baseline_clean, "Tokens prompted and baseline should be the same"
            tokens = tokens_prompted_clean
            

            logger.info(f"{entropy_type} for {author} prompted: shape {entropy_prompted.shape}")
            logger.info(f"{entropy_type} for {author} baseline: shape {entropy_baseline.shape}")

            for prompted, entropy in zip(["prompted", "baseline"], [entropy_prompted, entropy_baseline]):
                create_heatmap(entropy, tokens, author, entropy_type, Path(path_to_outputs), prompted)
            create_heatmap(entropy_prompted-entropy_baseline, tokens, author, entropy_type, Path(path_to_outputs), "prompted-baseline")


if __name__ == "__main__":
    main()