import argparse
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)
from sklearn.feature_extraction.text import TfidfTransformer
import json
import altair as alt
from scipy.stats import pointbiserialr

# Import shared utilities
from backend.src.utils.shared_utilities import (
    AuthorColorManager,
    BaseAnalyzer
)
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
)


class FeatureImportanceVisualizer:
    """Handles visualization of feature importance results"""
    
    def __init__(self, save_dir: Path, color_manager: AuthorColorManager):
        """
        Initialize the visualizer with a save directory and color manager.
        
        Args:
            save_dir: Directory to save visualizations
            color_manager: AuthorColorManager instance for consistent color mapping
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.color_manager = color_manager
    
    def plot_feature_importance(self, df, layer_ind, layer_type):
        """
        Create and save feature importance visualization.
        
        Args:
            df: DataFrame containing feature importance data
            layer_ind: Layer index
            layer_type: Type of layer (res, mlp, att)
        """
        
        # Filter out points where tfidf value is 0
        df_filtered = df[df['tfidf'] > 0.0001].copy()
        
        # Get unique authors and their colors for Altair
        unique_authors = df_filtered['author'].unique().tolist()
        domain, range_colors = self.color_manager.get_altair_domain_range(unique_authors)
        
        # Create the chart
        chart = alt.Chart(df_filtered).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X('feature_ind:Q', title='Feature Index'),
            y=alt.Y('tfidf:Q', title='TF-IDF-like Score'),
            color=alt.Color('author:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
            tooltip=[
                alt.Tooltip('feature_ind:Q', title='Feature Index'),
                alt.Tooltip('tfidf:Q', title='TF-IDF Score', format='.4f'),
                alt.Tooltip('author:N', title='Author'),
                alt.Tooltip('input_tokens:N', title='Input Tokens'),
                alt.Tooltip('predicted_tokens:N', title='Predicted Tokens')
            ]
        ).properties(
            title=f'Feature Importance - {layer_type} {layer_ind}',
            width=1400,
            height=1200
        ).interactive()

        # Save to the correct directory with the correct filename
        save_path = self.save_dir / f'tfidf_per_feature__{layer_type}_{layer_ind}.html'
        chart.save(str(save_path), inline=True)
        
        logger.info(f"Altair chart saved to: {save_path}")


class FeatureImportanceAnalyzer(BaseAnalyzer):
    """Main analyzer class for feature importance analysis"""
    
    def __init__(self, data_dir: Path, output_dir: Path, run_prefix: str = ""):
        """
        Initialize the feature importance analyzer.
        
        Args:
            data_dir: Directory containing SAE activation files
            output_dir: Directory to save results
            run_prefix: Prefix for the output directory
        """
        super().__init__(data_dir, output_dir, run_prefix)
        
        # Initialize visualizer with color manager
        self.visualizer = FeatureImportanceVisualizer(save_dir=self.output_dir, color_manager=self.color_manager)

    def get_most_important_tokens(self, feature_ind_to_tokens_counts: Dict[str, Dict[int, Counter]]) -> Dict[str, Dict[int, List[str]]]:
        """
        Extract the most important tokens for each feature using TF-IDF analysis.
        
        Args:
            feature_ind_to_tokens_counts: Dictionary mapping authors to feature indices to token counters
            
        Returns:
            Dictionary mapping authors to feature indices to lists of most important tokens
        """
        feature_ind_to_predicted_tokens_most_important = defaultdict(lambda: defaultdict(list))
        
        for author, feature_ind_to_predicted_tokens_dict in feature_ind_to_tokens_counts.items():
            # Build vocabulary from all tokens across all features for this author
            all_tokens = set()
            for predicted_tokens in feature_ind_to_predicted_tokens_dict.values():
                all_tokens.update(predicted_tokens.keys())
            
            # Create vocabulary mapping
            token_to_idx = {token: idx for idx, token in enumerate(sorted(all_tokens))}
            idx_to_token = {idx: token for token, idx in token_to_idx.items()}
            
            # Create document-term matrix directly from token counts
            n_features = len(feature_ind_to_predicted_tokens_dict)
            n_tokens = len(all_tokens)
            dtm = np.zeros((n_features, n_tokens))
            
            # Fill the document-term matrix
            for feature_idx, (feature_ind, predicted_tokens) in enumerate(feature_ind_to_predicted_tokens_dict.items()):
                for token, count in predicted_tokens.items():
                    if token in token_to_idx:
                        token_idx = token_to_idx[token]
                        dtm[feature_idx, token_idx] = count
            
            # Apply TF-IDF transformation
            tfidf_transformer = TfidfTransformer(use_idf=True, norm='l2')
            tfidf_matrix = tfidf_transformer.fit_transform(dtm)
            
            # Process each feature to get top 40 tokens
            for feature_idx, (feature_ind, _) in enumerate(feature_ind_to_predicted_tokens_dict.items()):
                # Get the row as a sparse vector
                row = tfidf_matrix[feature_idx]
                
                # Get non-zero indices and values
                indices = row.indices
                values = row.data
                
                if len(indices) > 0:
                    # Sort by TF-IDF scores and get top 40
                    sorted_indices = indices[np.argsort(values)[-40:]]
                    top_40_tokens = [idx_to_token[i] for i in sorted_indices]
                else:
                    # Handle case where no tokens are found
                    top_40_tokens = []
                
                feature_ind_to_predicted_tokens_most_important[author][feature_ind] = top_40_tokens
        
        return feature_ind_to_predicted_tokens_most_important

    def analyze_feature_importance(self, filenames: List[str]) -> Dict[str, Any]:
        """
        Analyze feature importance across all provided files.
        
        Args:
            filenames: List of SAE activation filenames to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Parse filenames and organize in hierarchy: layer_type -> layer_ind -> author 
        filenames_structured = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        authors = set()
        for filename in filenames:
             filename_parsed = self.filename_parser.parse_filename(filename)
             layer_type, layer_ind, author = filename_parsed["transformer_part"], filename_parsed["layer_ind"], filename_parsed["author"]
             filenames_structured[layer_type][layer_ind][author] = filename
             authors.add(author)

        # Get tokens
        all_input_tokens, all_full_texts = self.token_loader.load_tokens(list(authors))
        
        # Load in data
        for layer_type, layer_ind_dict in filenames_structured.items():
            for layer_ind, author_filename_dict in layer_ind_dict.items():
                csv_path = self.output_dir / f'feature_importance_data__{layer_type}_{layer_ind}.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    logger.info(f"Loaded existing CSV file for {layer_type} layer {layer_ind}: {csv_path}")
                    logger.info("Skipping NPZ file loading for this layer...")

                else:
                    data = {}
                    feature_ind_to_input_tokens = defaultdict(lambda: defaultdict(list))
                    feature_ind_to_predicted_tokens = defaultdict(lambda: defaultdict(list))
                    for author, filename in author_filename_dict.items():
                        activations = self.data_loader.load_sae_activations_simple(filepath=self.data_dir / filename)
                        data[author] = np.sum(activations, axis=(0, 1))
                        for doc_ind, (doc_activations, doc_tokens) in enumerate(zip(activations, all_input_tokens[author])):
                            for feature_ind, feature_vector in enumerate(doc_activations.T):
                                # Convert boolean mask to indices for list indexing
                                active_indices = np.where(feature_vector > 0)[0]
                                feature_ind_to_input_tokens[author][feature_ind].extend([doc_tokens[i] for i in active_indices])
                                feature_ind_to_predicted_tokens[author][feature_ind].extend([doc_tokens[i + 1] for i in active_indices if i + 1 < len(doc_tokens)]) 
                    
                    feature_ind_to_input_tokens_counts = defaultdict(lambda: defaultdict(Counter))
                    feature_ind_to_predicted_tokens_counts = defaultdict(lambda: defaultdict(Counter))
                    for author, feature_ind_to_input_tokens_dict in feature_ind_to_input_tokens.items():
                        for feature_ind, input_tokens in feature_ind_to_input_tokens_dict.items():
                            feature_ind_to_input_tokens_counts[author][feature_ind] = Counter(input_tokens)
                            # sort by frequency
                            feature_ind_to_input_tokens_counts[author][feature_ind] = dict(sorted(feature_ind_to_input_tokens_counts[author][feature_ind].items(), key=lambda x: x[1], reverse=True))
                    for author, feature_ind_to_predicted_tokens_dict in feature_ind_to_predicted_tokens.items():
                        for feature_ind, predicted_tokens in feature_ind_to_predicted_tokens_dict.items():
                            feature_ind_to_predicted_tokens_counts[author][feature_ind] = Counter(predicted_tokens)
                            feature_ind_to_predicted_tokens_counts[author][feature_ind] = dict(sorted(feature_ind_to_predicted_tokens_counts[author][feature_ind].items(), key=lambda x: x[1], reverse=True))

                    # get the most important tokens for each feature
                    feature_ind_to_predicted_tokens_most_important = self.get_most_important_tokens(feature_ind_to_predicted_tokens_counts)
                    feature_ind_to_input_tokens_most_important = self.get_most_important_tokens(feature_ind_to_input_tokens_counts)

                    # Transform data into numpy array
                    np_data = np.array(list(data.values()))
                    authors = list(data.keys())
                    logger.info(f"Shape of data: {np_data.shape}")

                    # compute feature importance similarly to TF-IDF
                    tfidf = TfidfTransformer(use_idf=True, norm='l2')
                    tfidf.fit(np_data)
                    tfidf_output_matrix = tfidf.transform(np_data).toarray()
                    logger.debug(f"Shape of tfidf_output_matrix: {tfidf_output_matrix.shape}")

                    df = pd.DataFrame(columns=["feature_ind", "input_tokens", "predicted_tokens", "tfidf", "author"])
            
                    for author_ind, author in enumerate(authors):
                        tfidfs = tfidf_output_matrix[author_ind]
                        inputs_tokens = feature_ind_to_input_tokens_most_important[author]
                        predicted_tokens = feature_ind_to_predicted_tokens_most_important[author]

                        temp_dict = {
                            "feature_ind": range(len(inputs_tokens)),
                            "input_tokens": [temp_input_tokens for temp_input_tokens in inputs_tokens.values()],
                            "predicted_tokens": [temp_predicted_tokens for temp_predicted_tokens in predicted_tokens.values()],
                            "tfidf": tfidfs,
                            "author": [author] * len(inputs_tokens)
                        }
                        df = pd.concat([df, pd.DataFrame(temp_dict)])

                    # Export DataFrame to CSV
                    csv_save_path = self.output_dir / f'feature_importance_data__{layer_type}_{layer_ind}.csv'
                    df.to_csv(csv_save_path, index=False)
                    logger.info(f"DataFrame exported to CSV: {csv_save_path}")

                # visualize
                self.visualizer.plot_feature_importance(df, layer_ind, layer_type)

    def get_entropy_feature_correlations(self, filenames: List[str], entropy_files: List[str], cross_entropy_files: List[str]):
        """
        Analyze correlations between features and entropy/cross-entropy.
        
        Args:
            filenames: List of SAE activation filenames
            entropy_files: List of entropy file names
            cross_entropy_files: List of cross-entropy file names
        """


        # Load entropies        # Load entropies
        entropies = defaultdict()
        logger.debug(f"Entropy files: {entropy_files}")
        for entropy_filename in entropy_files:
            stem_filename = entropy_filename.split(".")[0]
            author = stem_filename.split("_")[-1]
            logger.info(f"Loading entropy file for author {author}")
            entropies[author] = np.load(self.data_dir / entropy_filename)
        
        logger.debug(f"all authors in entropies: {entropies.keys()}")
        
        cross_entropies = defaultdict()
        logger.debug(f"Cross entropy files: {cross_entropy_files}")
        for cross_entropy_filename in cross_entropy_files:
            stem_filename = cross_entropy_filename.split(".")[0]
            author = stem_filename.split("_")[-1]
            cross_entropies[author] = np.load(self.data_dir / cross_entropy_filename)

        # Parse filenames and organize in hierarchy: layer_type -> layer_ind -> author 
        filenames_structured = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        authors = set()
        for filename in filenames:
             filename_parsed = self.filename_parser.parse_filename(filename)
             layer_type, layer_ind, author = filename_parsed["transformer_part"], filename_parsed["layer_ind"], filename_parsed["author"]
             filenames_structured[layer_type][layer_ind][author] = filename
             authors.add(author)

        authors = list(authors)

        point_biserial_correlations = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        point_biserial_p_values = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        point_biserial_correlations_cross = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        point_biserial_p_values_cross = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        # Load in data
        for layer_type, layer_ind_dict in filenames_structured.items():
            for layer_ind, author_filename_dict in layer_ind_dict.items():
                pbc_json_save_path = self.output_dir / f'point_biserial_correlations__{layer_type}_{layer_ind}.json'
                if pbc_json_save_path.exists():
                    logger.info(f"Skipping {layer_type} {layer_ind} because .json file already exists...")
                
                for author, filename in author_filename_dict.items():
                    activations = self.data_loader.load_sae_activations_simple(filepath=self.data_dir / filename)
                    activations_flat = activations.reshape(-1, activations.shape[2])
                    entropies_flat = entropies[author].reshape(-1)
                    cross_entropies_flat = cross_entropies[author].reshape(-1)

                    for feature_ind_in_activations in range(activations_flat.shape[1]):
                        activations_per_feature = activations_flat[:, feature_ind_in_activations]

                        # compute correlation between activations and entropies
                        point_biserial_correlation, point_biserial_p_value = pointbiserialr(activations_per_feature, entropies_flat)
                        point_biserial_correlation_cross, point_biserial_p_value_cross = pointbiserialr(activations_per_feature, cross_entropies_flat)

                        #logger.info(f"Feature {feature_ind_in_activations} - Point biserial correlation: {point_biserial_correlation}, Point biserial correlation cross: {point_biserial_correlation_cross}")
                        #logger.info(f"Feature {feature_ind_in_activations} - Point biserial p-value: {point_biserial_p_value}, Point biserial p-value cross: {point_biserial_p_value_cross}")

                        if point_biserial_correlation == point_biserial_correlation:

                            point_biserial_correlations[layer_type][layer_ind][author][feature_ind_in_activations] = float(point_biserial_correlation)
                            point_biserial_p_values[layer_type][layer_ind][author][feature_ind_in_activations] = float(point_biserial_p_value)

                        if point_biserial_correlation_cross == point_biserial_correlation_cross:

                            point_biserial_correlations_cross[layer_type][layer_ind][author][feature_ind_in_activations] = float(point_biserial_correlation_cross)
                            point_biserial_p_values_cross[layer_type][layer_ind][author][feature_ind_in_activations] = float(point_biserial_p_value_cross)

            
                    
                # save to json
                logger.debug(point_biserial_correlations)
                
                json_save_path = self.output_dir / f'point_biserial_correlations__{layer_type}_{layer_ind}.json'
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(point_biserial_correlations, f)
                logger.info(f"Point biserial correlations exported to JSON: {json_save_path}")
                
                json_save_path = self.output_dir / f'point_biserial_p_values__{layer_type}_{layer_ind}.json'
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(point_biserial_p_values, f)
                logger.info(f"Point biserial p-values exported to JSON: {json_save_path}")

                
                json_save_path = self.output_dir / f'point_biserial_correlations_cross__{layer_type}_{layer_ind}.json'
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(point_biserial_correlations_cross, f)
                logger.info(f"Point biserial correlations cross exported to JSON: {json_save_path}")
                
                json_save_path = self.output_dir / f'point_biserial_p_values_cross__{layer_type}_{layer_ind}.json'
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(point_biserial_p_values_cross, f)
                logger.info(f"Point biserial p-values cross exported to JSON: {json_save_path}")



        

    def get_unique_features_per_author(self, filenames: List[str], start_token_ind: int = 0) -> Dict[str, Any]:
        """
        Find features that are unique to each author.
        
        Args:
            filenames: List of SAE activation filenames
            start_token_ind: Starting token index for analysis
            
        Returns:
            Dictionary containing unique features per author
        """
        filenames_structured, authors = self.filename_parser.get_structured_filenames_and_authors(filenames)

        # Get tokens
        all_input_tokens, all_full_texts = self.token_loader.load_tokens(self.data_dir)

        author_inds = {author: i for i, author in enumerate(authors) if author not in ["h", "pp", "qq"]}
        
        # Load in data
        for layer_type, layer_ind_dict in filenames_structured.items():
            for layer_ind, author_filename_dict in layer_ind_dict.items():
                json_save_path = self.output_dir / f'raw_data_aggregated_over_docs__{layer_type}_{layer_ind}.json'
                if json_save_path.exists():
                    logger.info(f"Skipping {layer_type} {layer_ind} because .json file already exists...")
                    continue

                data = None
                output = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                active_tokens_and_docs = defaultdict(lambda: defaultdict(list))
                for author, filename in author_filename_dict.items():
                    if author in ["h", "pp", "qq"]:
                        continue
                    author_ind = author_inds[author]
                    activations = self.data_loader.load_sae_activations_simple(filepath=self.data_dir / filename)

                    docs_length_tokens = np.array([len([tok for tok in raw_tokens if tok != "<pad>"])-1 for raw_tokens in all_input_tokens[author]])
                    docs_lengths_activations = np.count_nonzero(np.sum(activations, axis=2), axis=1)

                    if docs_length_tokens.shape != docs_lengths_activations.shape:
                        logger.info(f"Docs length tokens and activations shapes do not match; {docs_length_tokens.shape} != {docs_lengths_activations.shape}")
                        logger.info(f"Docs length tokens: {docs_length_tokens}")
                        logger.info(f"Docs length activations: {docs_lengths_activations}")
                        logger.info(f"Activations: {activations}")


                    #assert np.all(docs_length_tokens == docs_lengths_activations[:len(docs_length_tokens)]), f"Docs length tokens and activations do not match, {docs_length_tokens} != {docs_lengths_activations}"

                    if data is None:
                        data = np.zeros((len(authors), activations.shape[2]))
                    data[author_ind] = self.activation_processor.aggregate_data_over_docs_and_tokens(activations, start_token_ind)
                    active_tokens_and_docs[author] = self.activation_processor.get_active_tokens_and_docs(activations, start_token_ind)

                # get unique features per author = features that are active only by one author
                for author, author_ind in author_inds.items():
                    logger.info(author)

                    # features indices which are active by this author
                    active_feature_inds = np.where(data[author_ind] > 0)[0]
                    # features indices which are active by other authors (excluding current author)
                    other_authors_data = np.delete(data, author_ind, axis=0)
                    active_feature_inds_others = np.where(np.sum(other_authors_data, axis=0) > 0)[0]
                    # features indices which are active by this author and not by any other author
                    unique_feature_inds = np.setdiff1d(active_feature_inds, active_feature_inds_others)
                    logger.info(f"Unique features for author {author}: {unique_feature_inds}")
                    # get tokens and docs for these features
                    for feature_ind in unique_feature_inds:
                        tokens_and_docs = active_tokens_and_docs[author][int(feature_ind)]
                        
                        # find input tokens, predicted tokens, and full text of documents
                        doc_inds, tokens_inds = zip(*tokens_and_docs)
                        input_tokens = Counter([all_input_tokens[author][doc_ind][token_ind] for doc_ind, token_ind in tokens_and_docs])
                        predicted_tokens = Counter([all_input_tokens[author][doc_ind][token_ind + 1] for doc_ind, token_ind in tokens_and_docs])

                        # sort by frequency
                        input_tokens = dict(sorted(input_tokens.items(), key=lambda x: x[1], reverse=True))
                        predicted_tokens = dict(sorted(predicted_tokens.items(), key=lambda x: x[1], reverse=True))
                        logger.info(tokens_and_docs)
                        full_text = [all_full_texts[author][doc_ind] for doc_ind in doc_inds]

                        output[author][int(feature_ind)] = {
                            "token_inds": tokens_inds,
                            "doc_inds": doc_inds,
                            "input_tokens": input_tokens,
                            "predicted_tokens": predicted_tokens,
                            "full_texts": full_text
                        }

                # save to json
                
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f)
                logger.info(f"Data exported to JSON: {json_save_path}")


def main():
    """Main entry point for feature importance analysis"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    parser = argparse.ArgumentParser(description="SAE Feature Importance Analysis Tool")
    
    # Run configuration
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
        help="Directory containing SAE activation files (required if --run_id not provided)"
    )
    parser.add_argument(
        "--include_authors",
        type=str,
        nargs="+",
        default=None,
        help="The authors to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_types",
        type=str,
        nargs="+",
        default=None,
        choices=["res", "mlp", "att"],
        help="The layer types to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_inds",
        type=int,
        nargs="+",
        default=None,
        help="The layers to include in the analysis"
    )
    
    args = parser.parse_args()
    
    # Validate that either run_id or path_to_data is provided
    if not args.run_id and not args.path_to_data:
        parser.error("Either --run_id or --path_to_data must be provided")
    
    # Get data and output paths
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type="feature_importance",
        run_name_override=None
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type="feature_importance",
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")
    
    dir_path = str(data_path)
    output_dir = str(output_path)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(dir_path, output_dir, "")
    
    # Get list of files to analyze
    filenames = [f for f in os.listdir(dir_path) if f.endswith(".npz")]

    entropies_files = [f for f in os.listdir(dir_path) if f.endswith(".npy") and "entropy" in f and "cross" not in f]
    cross_entropies_files = [f for f in os.listdir(dir_path) if f.endswith(".npy") and "cross_entropy" in f]
    
    # Apply filters using the base analyzer method
    filenames = analyzer.filter_filenames(
        filenames, 
        include_authors=args.include_authors,
        include_layer_types=args.include_layer_types,
        include_layer_inds=args.include_layer_inds
    )
    
    logger.info(f"Analyzing {len(filenames)} files for feature importance...")
    
    # Run feature importance analysis
    results = analyzer.get_unique_features_per_author(filenames)
    #results = analyzer.get_entropy_feature_correlations(filenames, entropies_files, cross_entropies_files)
    
    logger.info("Feature importance analysis completed!")
    logger.info(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
