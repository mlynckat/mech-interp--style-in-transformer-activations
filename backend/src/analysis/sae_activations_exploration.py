import argparse
import gc
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
import logging

# Set up logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import altair as alt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import shared utilities
from brain.my_scripts.shared_utilities import (
    AuthorColorManager,
    FilenameParser,
    DataLoader,
    ActivationProcessor,
    MetricsCalculator,
    TokenLoader,
    BaseAnalyzer,
    SAEMetadata,
    ClusterMetrics
)


class Visualizer:
    """Handles all visualization tasks"""

    def __init__(self, save_dir: Path, color_manager: AuthorColorManager):
        """
        Initialize the visualizer with save directory and color manager.
        
        Args:
            save_dir: Directory to save visualizations
            color_manager: AuthorColorManager instance for consistent color mapping
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.color_manager = color_manager

    def plot_activation_histogram(self, data: np.ndarray | pd.DataFrame, title_suffix: str, filename: str):
        """Create histogram of non-zero activations with density lines and boxplots"""
        # Create subplots: histogram on top, boxplot on bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        if isinstance(data, np.ndarray):
            # Single array case
            data_filtered = data[data > 0]
            ax1.hist(data_filtered, bins=50, alpha=0.5, density=True, label='All data')
            
            # Add density line
            kde = stats.gaussian_kde(data_filtered)
            x_range = np.linspace(data_filtered.min(), data_filtered.max(), 100)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Density')
            
            # Create boxplot
            ax2.boxplot(data_filtered, tick_labels=['All data'])
            
        elif isinstance(data, pd.DataFrame):
            # DataFrame case - assume it has author information
            if 'author' in data.columns and 'activations' in data.columns:
                # Group by author
                authors = data['author'].unique()
                author_colors = self.color_manager.get_author_colors(authors)
                
                # Prepare data for boxplot
                boxplot_data = []
                boxplot_labels = []
                
                for author in authors:
                    author_data = data[data['author'] == author]['activations']
                    author_data = author_data[author_data > 0]
                    logger.debug(f"Author data is of type {type(author_data)} and has shape {author_data.shape}")
                    
                    if len(author_data) > 0:
                        # Histogram
                        ax1.hist(author_data, bins=50, alpha=0.6, density=True, 
                                label=author, color=author_colors[author])
                        
                        # Add density line
                        kde = stats.gaussian_kde(author_data)
                        x_range = np.linspace(author_data.min(), author_data.max(), 100)
                        ax1.plot(x_range, kde(x_range), color=author_colors[author], linewidth=2, 
                                linestyle='--', alpha=0.8)
                        
                        # Prepare for boxplot
                        boxplot_data.append(author_data)
                        boxplot_labels.append(author)
                
                # Create boxplot
                if boxplot_data:
                    ax2.boxplot(boxplot_data, tick_labels=boxplot_labels)
            else:
                raise ValueError("Data does not have author and activations columns")
        
        # Set up histogram subplot
        ax1.set_title(f"Distribution of Active SAE Features per Token {title_suffix}")
        ax1.set_xlabel("Active SAE features")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set up boxplot subplot
        ax2.set_title(f"Boxplot of Active SAE Features per Token {title_suffix}")
        ax2.set_xlabel("Author")
        ax2.set_ylabel("Active SAE features")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_tsne_plots(self, features_dict: Dict[str, np.ndarray], labels: List[str],
                          sae_name: str, transformer_part: str):
        """Create t-SNE visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 20), constrained_layout=True)

        plot_configs = [
            ("Original Features", features_dict["original"]),
            ("Normalized Features", features_dict["normalized"]),
            ("Original Features, Lower Perprlexity", features_dict["original"]),
            ("Normalized Features, Lower Perprlexity", features_dict["normalized"]),
        ]

        for idx, (title, features) in enumerate(plot_configs):
            row, col = idx // 2, idx % 2
            if row == 0:
                embeddings = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(features)
            elif row == 1:
                embeddings = TSNE(n_components=2, perplexity=20, random_state=42).fit_transform(features)
            else:
                raise ValueError("Invalid row index.")

            # Get unique authors and their colors
            unique_authors = list(set(labels))
            author_palette = self.color_manager.get_seaborn_palette(unique_authors)
            
            sns.scatterplot(
                x=embeddings[:, 0], y=embeddings[:, 1],
                hue=labels, palette=author_palette, s=50, alpha=0.5, ax=axes[row, col]
            )
            axes[row, col].set_title(f't-SNE ({title})', fontsize=10)

        fig.suptitle(f't-SNE of Docs for {transformer_part} {sae_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        filename = f"clusters_google_gemma-2-2b_{transformer_part}_{sae_name.replace('/', '_')}.png"
        plt.savefig(self.save_dir / filename)
        plt.close()

    def create_heatmap(self, data: pd.DataFrame, title: str, filename: str):
        """Create and save heatmap"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, fmt=".1f", cmap="Greens")
        plt.title(title)
        plt.savefig(self.save_dir / filename)
        plt.close()

    def create_distribution_plot(self, data_dict: Dict[str, np.ndarray], title: str,
                                 xlabel: str, ylabel: str, filename: str):
        """Create overlaid histogram distribution plot"""
        plt.figure(figsize=(10, 6))
        
        # Get colors for authors
        authors = list(data_dict.keys())
        author_colors = self.color_manager.get_author_colors(authors)
        
        for author, data in data_dict.items():
            plt.hist(data, bins=50, alpha=0.5, label=author, color=author_colors[author])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(self.save_dir / filename)
        plt.close()

    def plot_activations_per_feature(self, activations_df: pd.DataFrame, title_suffix: str, plot_filename: str):
        """Plot activations per feature with altair"""
        
        # Ensure feature_ind is numeric for proper x-axis ordering
        activations_df = activations_df.copy()
        activations_df['feature_ind'] = pd.to_numeric(activations_df['feature_ind'], errors='coerce')
        
        # Remove the feature_counts column as it contains Counter objects that Altair can't handle
        # Keep only the string representation for tooltips
        chart_df = activations_df.drop(columns=['feature_counts'], errors='ignore')
        
        # Get unique authors and their colors for Altair
        unique_authors = chart_df['variable'].unique().tolist()
        domain, range_colors = self.color_manager.get_altair_domain_range(unique_authors)
        
        # Create the chart
        chart = alt.Chart(chart_df).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X('feature_ind:Q', title='Feature Index'),
            y=alt.Y('value:Q', title='Activation Count'),
            color=alt.Color('variable:N', title='Author', domain=domain, range=range_colors),
            tooltip=['feature_ind', 'value', 'variable', 'feature_counts_str']
        ).properties(
            title=f'Feature Activation Counts - {title_suffix}',
            width=800,
            height=400
        ).interactive()

        # Save to the correct directory with the correct filename
        save_path = self.save_dir / plot_filename.replace('.png', '.html')
        chart.save(str(save_path), inline=True)
        
        logger.info(f"Altair chart saved to: {save_path}")

    def plot_entropies(self, entropies: np.ndarray, cross_entropies: np.ndarray, layer_str: str, transformer_part: str, author: str, activations: np.ndarray):
        """Plot entropies and cross-entropies
        entropies: np.ndarray of shape (num_docs, num_tokens)
        cross_entropies: np.ndarray of shape (num_docs, num_tokens)
        layer_str: str
        transformer_part: str
        author: str
        activations: np.ndarray of shape (num_docs, num_tokens)
        """
        assert entropies.shape[0] == cross_entropies.shape[0] == activations.shape[0] and entropies.shape[1] == cross_entropies.shape[1] == activations.shape[1], "Entropies, cross-entropies, and activations must have the same shape"

        # flatten all arrays
        activations_flat = activations.flatten()
        activations_flat_non_zero = activations_flat[activations_flat > 0]
        logger.info(f"Removed {len(activations_flat) - len(activations_flat_non_zero)} tokens with no activations")
        entropies_flat = entropies.flatten()
        entropies_flat_non_zero = entropies_flat[activations_flat > 0]
        logger.info(f"Removed {len(entropies_flat) - len(entropies_flat_non_zero)} tokens with no entropies")
        cross_entropies_flat = cross_entropies.flatten()
        cross_entropies_flat_non_zero = cross_entropies_flat[activations_flat > 0]
        logger.info(f"Removed {len(cross_entropies_flat) - len(cross_entropies_flat_non_zero)} tokens with no cross-entropies")

        # create a dataframe
        df = pd.DataFrame({
            "entropies": entropies_flat,
            "cross_entropies": cross_entropies_flat,
            "activations": activations_flat
        })
        
        # plot the entropies vs activations
        plt.figure(figsize=(10, 6))
        plt.scatter(df["entropies"], df["activations"], alpha=0.2)
        plt.xlabel("Entropy")
        plt.ylabel("Activations")
        plt.title(f"Entropy vs Activations for {layer_str} {transformer_part} {author}")
        plt.savefig(self.save_dir / f"entropies_vs_activations_{layer_str}_{transformer_part}_{author}.png")
        plt.close()

        # plot the cross-entropies vs activations
        plt.figure(figsize=(10, 6))
        plt.scatter(df["cross_entropies"], df["activations"], alpha=0.2)
        plt.xlabel("Cross-Entropy")
        plt.ylabel("Activations")
        plt.title(f"Cross-Entropy vs Activations for {layer_str} {transformer_part} {author}")
        plt.savefig(self.save_dir / f"cross_entropies_vs_activations_{layer_str}_{transformer_part}_{author}.png")
        plt.close()

    def plot_token_positions(self, token_positions: Dict[int, List[int]], author: str, transformer_part: str, layer_str: str):  
        """Plot token positions"""
        plt.figure(figsize=(10, 6))
        for token_ind, number_activations in token_positions.items():
            plt.scatter(token_ind*np.ones(len(number_activations)), number_activations, alpha=0.2)
        plt.title(f"Token positions for {transformer_part} layer {layer_str} {author}")
        plt.xlabel("Token Index")
        plt.ylabel("Number of Activations")
        plt.savefig(self.save_dir / f"token_positions_{transformer_part}_{layer_str}_{author}.png")
        plt.close()


class SAEAnalyzer(BaseAnalyzer):
    """Main analyzer class that orchestrates the analysis"""

    def __init__(self, data_dir: Path, output_dir: Path, run_prefix: str = ""):
        """
        Initialize the SAE analyzer.
        
        Args:
            data_dir: Directory containing SAE activation files
            output_dir: Directory to save results
            run_prefix: Prefix for the output directory
        """
        super().__init__(data_dir, output_dir, run_prefix)
        
        # Initialize visualizer with color manager
        self.visualizer = Visualizer(self.save_dir, self.color_manager)

    def analyze_single_file(self, filename: str):
        """Analyze a single SAE activation file"""

        # Create activation histogram
        title_suffix = filename.split('.')[0][22:]
        hist_filename = f"non_zeroes_hist_boxplots_{Path(filename).stem}.png"

        # Check if .npy file already exists
        if (self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy").exists():
            logger.info(f"Skipping {filename} because .npy file already exists, loading from file...")
            non_zeroes_count = np.load(self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy")
            non_zeroes_count = non_zeroes_count.flatten()

        else:
            metadata, activations = self.data_loader.load_sae_activations(self.data_dir / filename)
            non_zeroes_count = np.count_nonzero(activations, axis=2)
            # save non_zeroes_count to npy
            np.save(self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy", non_zeroes_count)
            non_zeroes_count = non_zeroes_count.flatten()
            del activations

        self.visualizer.plot_activation_histogram(non_zeroes_count, title_suffix, hist_filename)
 
        gc.collect()

        return non_zeroes_count

    def analyze_all_files(self, filenames):
        """Analyze all files for activation statistics"""

        max_workers = min(mp.cpu_count(), 1)  # Limit to 8 workers to avoid memory issues

        logger.info(f"Using {max_workers} out of {mp.cpu_count()} available workers")

         # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.analyze_single_file, filename) for filename in filenames]

            aggregated_activations = defaultdict(lambda: defaultdict(dict))
            mean_activations_per_token = defaultdict(lambda: defaultdict(dict))

            for ind_filename, future in tqdm(enumerate(futures), desc="Processing files"):
                filename = filenames[ind_filename]
                try:
                    non_zeroes_count = future.result()

                    # Parse filename
                    parsed = self.filename_parser.parse_filename(filename)
                    self.filename_parser.validate_activation_type(parsed["transformer_part"])

                    layer_str = self.filename_parser.format_layer_string(parsed["layer_ind"])

                    averaged_activations = np.mean(non_zeroes_count[non_zeroes_count > 0])
                    mean_activations_per_token[parsed["author"]][layer_str][parsed["transformer_part"]] = averaged_activations
                    aggregated_activations[parsed["transformer_part"]][layer_str][parsed["author"]] = non_zeroes_count

                except Exception as e:
                    logger.info(f"Error processing file: {e}")    

        # Force garbage collection after processing all files
        gc.collect()
        
        logger.info("Memory cleanup completed. Generating reports...")
        # Generate reports
        self._generate_reports_parallel(aggregated_activations, mean_activations_per_token)

    def _generate_reports_parallel(self, aggregated_activations, mean_activations_per_token):
        """Generate reports with parallel processing"""
        # Process visualizations in parallel
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = []
            
            for transformer_part, data_items in aggregated_activations.items():
                for layer_str, author_data in data_items.items():
                    df = pd.DataFrame.from_dict(author_data)
                    logger.info(df.head())
                    df_long = df.melt(value_vars=author_data.keys(), var_name='author', value_name='activations') 
                    title_suffix = f"Layer {layer_str} {transformer_part}"
                    hist_filename = f"non_zeroes_hist_aggregated_{transformer_part}_{layer_str}.png"
                    
                    future = executor.submit(
                        self.visualizer.plot_activation_histogram, 
                        df_long, title_suffix, hist_filename
                    )
                    futures.append(future)
            
            # Wait for all visualizations to complete
            for future in tqdm(futures, desc="Generating visualizations"):
                future.result()

        # Generate author reports
        self._generate_author_reports(mean_activations_per_token)

    def _generate_author_reports(self, data: Dict):
        """Generate per-author analysis reports"""
        for author, layers_dict in data.items():
            df = pd.DataFrame.from_dict(layers_dict, orient='index')
            df.index.name = "layer"
            df.columns.name = "activation_type"
            df = df.sort_index()
            df = df.sort_index(axis=1)

            # Save CSV
            df.to_csv(self.output_dir / f"{author}_non_zero_count_activations.csv")

            # Create heatmap
            title = f"Average Number of active SAE features per layer and activation type for {author}"
            filename = f"{author}_non_zero_count_activations.png"
            self.visualizer.create_heatmap(df, title, filename)

    def _generate_distribution_plot(self, dataset_stats: Dict[str, np.ndarray]):
        """Generate token distribution plot"""
        self.visualizer.create_distribution_plot(
            dataset_stats,
            "Distribution of tokens per doc across authors",
            "Author",
            "Tokens per doc",
            f"tokens_per_doc_distribution.png"
        )

    def _save_cluster_metrics(self, cluster_metrics: Dict):
        """Save clustering metrics to CSV files"""
        for transformer_part, layers_dict in cluster_metrics.items():
            df_metrics = pd.DataFrame.from_dict(layers_dict, orient='index')
            df_metrics.index.name = "layer"
            df_metrics = df_metrics.sort_index()
            df_metrics.to_csv(self.output_dir / f"clusters_metrics_{transformer_part}.csv", index=False)

    def analyze_clusters(self, filenames):
        """Perform clustering analysis across all files"""

        # Group files by transformer part and layer
        grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

        for filename in filenames:
            parsed = self.filename_parser.parse_filename(filename)
            layer_str = self.filename_parser.format_layer_string(parsed["layer_ind"])
            grouped_files[parsed["transformer_part"]][layer_str][parsed["author"]] = filename

        # Analyze each group
        cluster_metrics = defaultdict(lambda: defaultdict(dict))

        jobs = []
        for transformer_part, layers_dict in grouped_files.items():
            for layer_str, authors_dict in layers_dict.items():
                jobs.append((transformer_part, layer_str, authors_dict))

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._analyze_layer_cluster, authors_dict, transformer_part): (transformer_part, layer_str)
                for transformer_part, layer_str, authors_dict in jobs
            }

        for future in as_completed(futures):
            transformer_part, layer_str = futures[future]
            metrics = future.result()
            cluster_metrics[transformer_part][layer_str] = metrics.to_dict()

        # Save metrics
        self._save_cluster_metrics(cluster_metrics)

    def _analyze_layer_cluster(self, authors_dict: Dict[str, str], transformer_part: str) -> ClusterMetrics:
        """Analyze clustering for a specific layer"""
        all_activations = []
        all_activations_short = []
        author_list = []
        author_to_docs = {}

        for author, filename in authors_dict.items():
            metadata, activations = self.data_loader.load_sae_activations(self.data_dir / filename)

            # Process activations
            agg_activations = self.activation_processor.aggregate_normalized(activations, metadata.tokens_per_doc)
            agg_activations_short = self.activation_processor.aggregate_normalized(activations, metadata.tokens_per_doc, 100)

            all_activations.append(agg_activations)
            all_activations_short.append(agg_activations_short)
            author_list.append(metadata.author)
            author_to_docs[metadata.author] = metadata.tokens_per_doc

        # Combine data
        combined_activations = np.concatenate(all_activations, axis=0)
        combined_activations_short = np.concatenate(all_activations_short, axis=0)

        # Create labels
        cluster_labels = []
        for i, author in enumerate(author_list):
            num_docs = len(author_to_docs[author])
            cluster_labels.extend([author] * num_docs)

        # Prepare features for analysis
        features_dict = {
            "original": combined_activations,
            "normalized": self.metrics_calculator.scaler.fit_transform(combined_activations),
        }

        # Create visualizations
        self.visualizer.create_tsne_plots(
            features_dict, cluster_labels, metadata.sae_name, transformer_part
        )

        # Compute metrics
        return self.metrics_calculator.compute_cluster_metrics(combined_activations, cluster_labels)

    def analyze_detailed_counts(self, filenames):
        """Analyze all files for activation statistics"""
        activations_aggregated = defaultdict(lambda: defaultdict(dict))
        text_lengths = defaultdict()
        tokens_counts = defaultdict(lambda: defaultdict(list))

        tokens = self.token_loader.load_tokens(list(set([self.filename_parser.parse_filename(f)['author'] for f in filenames])))

        for filename in filenames:
            metadata, activations = self.data_loader.load_sae_activations(self.data_dir / filename)

            # Compute statistics
            non_zeroes_per_feature = np.count_nonzero(activations, axis=(0, 1))
            logger.info(f"Shape of aggregated activations: {non_zeroes_per_feature.shape}. Should be equal to the number of features")

            # Parse filename
            parsed = self.filename_parser.parse_filename(filename)
            self.filename_parser.validate_activation_type(parsed["transformer_part"])

            layer_str = self.filename_parser.format_layer_string(parsed["layer_ind"])

            activations_aggregated[parsed["transformer_part"]][layer_str][parsed["author"]] = non_zeroes_per_feature
            text_lengths[parsed["author"]] = np.sum(metadata.tokens_per_doc)

            for doc_ind, doc_activations in enumerate(activations):
                for feature_ind, feature_vector in enumerate(doc_activations.T):
                    seq_ind = np.nonzero(feature_vector)[0]
                    if seq_ind.size == 0 and feature_ind not in tokens_counts[parsed["author"]]:
                        tokens_counts[parsed["author"]][feature_ind] = []
                    elif seq_ind.size > 0:
                        for seq_ind_i in seq_ind:
                            tokens_counts[parsed["author"]][feature_ind].append(tokens[parsed["author"]][doc_ind][seq_ind_i])
        
        tokens_counter = defaultdict(lambda: defaultdict(Counter))
        for author_in_tokens, tokens_per_author in tokens_counts.items():
            logger.info(tokens_per_author)
            for feature_ind, feature_tokens in tokens_per_author.items():
                tokens_counter[author_in_tokens][feature_ind] = Counter(tokens_counts[author_in_tokens][feature_ind])

        # Generate reports
        for transformer_part, transformer_part_items in activations_aggregated.items():
            for layer_str, layer_items in transformer_part_items.items():
                if not layer_items:  # Skip if no data for this layer
                    logger.info(f"No data found for {transformer_part} layer {layer_str}") 
                    continue
                    
                df = pd.DataFrame.from_dict(layer_items)  # Use layer_items instead of transformer_part_items
                if df.empty:
                    logger.info(f"Empty DataFrame for {transformer_part} layer {layer_str}")
                    continue
                    
                df["feature_ind"] = df.index
                df_long = df.melt(id_vars=['feature_ind'], value_vars=layer_items.keys())
                
                # Add feature_counts column with error handling
                feature_counts_list = []
                for ind, row in df_long.iterrows():
                    try:
                        author = row["variable"]
                        feature_ind = row["feature_ind"]
                        if author in tokens_counter and feature_ind in tokens_counter[author]:
                            feature_counts_list.append(tokens_counter[author][feature_ind])
                        else:
                            feature_counts_list.append(Counter())  # Empty counter if not found
                    except KeyError as e:
                        logger.info(f"Key error for row {ind}: {e}")
                        feature_counts_list.append(Counter())
                
                df_long["feature_counts"] = feature_counts_list

                # Convert Counter objects to strings for better visualization, sorted by count (descending)
                df_long["feature_counts_str"] = df_long["feature_counts"].apply(lambda x: str(dict(x.most_common())) if x else "{}")

                logger.info(df_long.head())

                # save df_long to csv
                df_long.to_csv(self.output_dir / f"frequencies_activated_per_feature_{transformer_part}_{layer_str}.csv", index=False)

                title_suffix = f"Layer {layer_str} {transformer_part}"
                plot_filename = f"frequencies_activated_per_feature_{transformer_part}_{layer_str}.png"
                self.visualizer.plot_activations_per_feature(df_long, title_suffix, plot_filename)
    
    def analyze_entropies(self, filenames, entropies_files, cross_entropies_files):
        """Analyze all files for entropy and cross-entropy statistics"""

        aggregated_activations = defaultdict(lambda: defaultdict(dict))
            
        for filename in filenames:
            # Check if .npy file already exists
            if (self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy").exists():
                logger.info(f"Skipping {filename} because .npy file already exists, loading from file...")
                non_zeroes_count = np.load(self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy")

            else:
                metadata, activations = self.data_loader.load_sae_activations(self.data_dir / filename)
                non_zeroes_count = np.count_nonzero(activations, axis=2)
                np.save(self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy", non_zeroes_count)
                del activations

            parsed = self.filename_parser.parse_filename(filename)
            self.filename_parser.validate_activation_type(parsed["transformer_part"])
            layer_str = self.filename_parser.format_layer_string(parsed["layer_ind"])

            aggregated_activations[parsed["transformer_part"]][layer_str][parsed["author"]] = non_zeroes_count

        entropies = defaultdict()
        logger.info(f"Entropy files: {entropies_files}")
        for entropy_filename in entropies_files:
            stem_filename = entropy_filename.split(".")[0]
            author = stem_filename.split("_")[-1]
            logger.info(f"Loading entropy file for author {author}")
            entropies[author] = np.load(self.data_dir / entropy_filename)
        
        logger.info(f"all authors in entropies: {entropies.keys()}")
        
        cross_entropies = defaultdict()
        logger.info(f"Cross entropy files: {cross_entropies_files}")
        for cross_entropy_filename in cross_entropies_files:
            stem_filename = cross_entropy_filename.split(".")[0]
            author = stem_filename.split("_")[-1]
            cross_entropies[author] = np.load(self.data_dir / cross_entropy_filename)

        for transformer_part, transformer_part_items in aggregated_activations.items():
            for layer_str, layer_items in transformer_part_items.items():
                for author, activations in layer_items.items():
                    if author in entropies and author in cross_entropies:
                        self.visualizer.plot_entropies(entropies[author], cross_entropies[author], layer_str, transformer_part, author, activations)
                    else:
                        logger.info(f"Author {author} not found in entropies or cross_entropies")

    def analyze_token_positions(self, filenames):
        """Analyze token positions"""

        token_positions = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for filename in filenames:
            # Check if .npy file already exists
            if (self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy").exists():
                logger.info(f"Skipping {filename} because .npy file already exists, loading from file...")
                non_zeroes_count = np.load(self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy")

            else:
                metadata, activations = self.data_loader.load_sae_activations(self.data_dir / filename)
                non_zeroes_count = np.count_nonzero(activations, axis=2)
                np.save(self.output_dir / f"non_zeroes_count_{Path(filename).stem}.npy", non_zeroes_count)
                del activations

            parsed = self.filename_parser.parse_filename(filename)
            self.filename_parser.validate_activation_type(parsed["transformer_part"])
            layer_str = self.filename_parser.format_layer_string(parsed["layer_ind"])

            for doc_ind, doc_activations in enumerate(non_zeroes_count):
                for token_ind, number_activations in enumerate(doc_activations):
                    if number_activations > 0:
                        token_positions[parsed["transformer_part"]][layer_str][parsed["author"]][token_ind].append(number_activations)
            
            for transformer_part, transformer_part_items in token_positions.items():
                for layer_str, layer_items in transformer_part_items.items():
                    for author, author_items in layer_items.items():
                        self.visualizer.plot_token_positions(author_items, author, transformer_part, layer_str)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SAE Activation Analysis Tool")
    parser.add_argument(
        "--dir_path",
        type=str,
        default="sae_features/reduced",
        help="Directory containing SAE activation files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="+",
        default="sae_features/outputs",
        help="The authors to include in the analysis"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specific file to analyze (if not provided, analyzes all files)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["explore", "cluster", "detailed", "entropies", "token_positions", "all"],
        default="explore",
        help="Analysis mode: explore (single file analysis), cluster (clustering analysis), or all"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="no_padding_all_reduced",
        help="The name of the run to create a folder in outputs and save visualizations and files there"
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

    analyzer = SAEAnalyzer(args.dir_path, args.output_dir, args.run_name)

    if args.filename:
        analyzer.analyze_single_file(args.filename)

    else:
        filenames = [f for f in os.listdir(args.dir_path) if f.endswith(".npz")]
        
        # Apply filters using the base analyzer method
        filenames = analyzer.filter_filenames(
            filenames, 
            include_authors=args.include_authors,
            include_layer_types=args.include_layer_types,
            include_layer_inds=args.include_layer_inds
        )

    if args.mode == "explore" or args.mode == "all":
        analyzer.analyze_all_files(filenames)

    if args.mode == "cluster" or args.mode == "all":
        analyzer.analyze_clusters(filenames)

    if args.mode == "detailed" or args.mode == "all":
        analyzer.analyze_detailed_counts(filenames)

    if args.mode == "entropies" or args.mode == "all":
        entropies_files = [f for f in os.listdir(args.dir_path) if f.endswith(".npy") and "entropy" in f and "cross" not in f]
        cross_entropies_files = [f for f in os.listdir(args.dir_path) if f.endswith(".npy") and "cross_entropy" in f]
        analyzer.analyze_entropies(filenames, entropies_files, cross_entropies_files)

    if args.mode == "token_positions" or args.mode == "all":
        analyzer.analyze_token_positions(filenames)

if __name__ == "__main__":
    main()
