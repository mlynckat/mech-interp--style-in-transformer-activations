import argparse
import gc
import json
from collections import defaultdict, Counter
from pathlib import Path
from re import L
from typing import Dict, List, Tuple, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp
from scipy import stats
from sklearn.manifold import TSNE
import umap
from tqdm.auto import tqdm
import altair as alt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import and apply plot styling
from backend.src.utils.plot_styling import PlotStyle, apply_style, create_figure

# Apply global matplotlib styling (Nordic Ocean Theme)
apply_style()

# Constants
ACTIVATION_THRESHOLD = 1.0
BINARY_ACTIVATION_TYPE = np.int8
HISTOGRAM_BINS = 50
TSNE_PERPLEXITY_HIGH = 40
TSNE_PERPLEXITY_LOW = 20
TSNE_RANDOM_STATE = 42
UMAP_N_NEIGHBORS_HIGH = 15
UMAP_N_NEIGHBORS_LOW = 5
UMAP_MIN_DIST_HIGH = 0.1
UMAP_MIN_DIST_LOW = 0.3
UMAP_RANDOM_STATE = 42
MAX_WORKERS_LIMIT = 1
CLUSTER_MAX_WORKERS = 4
FIGURE_DPI = 300
DEFAULT_FIGURE_SIZE = (10, 6)
LARGE_FIGURE_SIZE = (15, 12)
TSNE_FIGURE_SIZE = (20, 20)
UMAP_FIGURE_SIZE = (20, 20)
SCATTER_ALPHA = 0.5
HISTOGRAM_ALPHA = 0.6
DENSITY_LINE_WIDTH = 2
DENSITY_ALPHA = 0.8
SHORT_SEQUENCE_LENGTH = 100

# Import shared utilities
from backend.src.utils.shared_utilities import (
    AuthorColorManager,
    ActivationFilenamesLoader,
    EntropyFilenamesLoader,
    TokenandFullTextFilenamesLoader,
    BaseAnalyzer,
    ActivationMetadata,
    ClusterMetrics
)
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
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

    def plot_activation_histogram(self, data: Union[np.ndarray, pd.DataFrame], title_suffix: str, filename: str) -> None:
        """
        Create histogram of non-zero activations with density lines and boxplots.
        
        Args:
            data: Either a numpy array or pandas DataFrame with activation data
            title_suffix: Suffix to add to plot titles
            filename: Name of the file to save the plot
        """
        fig, (hist_ax, box_ax) = plt.subplots(2, 1, figsize=LARGE_FIGURE_SIZE)
        
        if isinstance(data, np.ndarray):
            self._plot_single_array_histogram(data, hist_ax, box_ax)
        elif isinstance(data, pd.DataFrame):
            self._plot_dataframe_histogram(data, hist_ax, box_ax)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        self._configure_histogram_axes(hist_ax, box_ax, title_suffix)
        self._save_plot(fig, filename)

    def _plot_single_array_histogram(self, data: np.ndarray, hist_ax: plt.Axes, box_ax: plt.Axes) -> None:
        """Plot histogram for a single numpy array."""
        filtered_data = data[data > 0]
        
        # Create histogram
        hist_ax.hist(filtered_data, bins=HISTOGRAM_BINS, alpha=SCATTER_ALPHA, 
                    density=True, label='All data')
            
            # Add density line
        self._add_density_line(hist_ax, filtered_data, 'r', 'Density')
            
            # Create boxplot
        box_ax.boxplot(filtered_data, tick_labels=['All data'])

    def _plot_dataframe_histogram(self, data: pd.DataFrame, hist_ax: plt.Axes, box_ax: plt.Axes) -> None:
        """Plot histogram for a pandas DataFrame with author information."""
        if not self._validate_dataframe_columns(data):
            raise ValueError("DataFrame must have 'author' and 'activations' columns")
        
        authors = data['author'].unique()
        author_colors = self.color_manager.get_author_colors(authors)
        boxplot_data, boxplot_labels = [], []
        
        for author in authors:
            author_data = self._get_author_activation_data(data, author)
            if len(author_data) > 0:
                self._plot_author_histogram(hist_ax, author_data, author, author_colors[author])
                self._add_density_line(hist_ax, author_data, color=author_colors[author], 
                                     linestyle='--', alpha=DENSITY_ALPHA)
                boxplot_data.append(author_data)
                boxplot_labels.append(author)
        
        if boxplot_data:
            box_ax.boxplot(boxplot_data, tick_labels=boxplot_labels)

    def _validate_dataframe_columns(self, data: pd.DataFrame) -> bool:
        """Validate that DataFrame has required columns."""
        return 'author' in data.columns and 'activations' in data.columns

    def _get_author_activation_data(self, data: pd.DataFrame, author: str) -> np.ndarray:
        """Get activation data for a specific author, filtered for positive values."""
        author_data = data[data['author'] == author]['activations']
        author_data = author_data[author_data > 0]
        logger.debug(f"Author {author} data type: {type(author_data)}, shape: {author_data.shape}")
        return author_data

    def _plot_author_histogram(self, ax: plt.Axes, data: np.ndarray, author: str, color: str) -> None:
        """Plot histogram for a specific author."""
        ax.hist(data, bins=HISTOGRAM_BINS, alpha=HISTOGRAM_ALPHA, 
               density=True, label=author, color=color)

    def _add_density_line(self, ax: plt.Axes, data: np.ndarray, color: str = 'r', 
                         label: str = 'Density', linestyle: str = '-', alpha: float = 1.0) -> None:
        """Add density line to histogram."""
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), color=color, linewidth=DENSITY_LINE_WIDTH, 
               linestyle=linestyle, alpha=alpha, label=label)

    def _configure_histogram_axes(self, hist_ax: plt.Axes, box_ax: plt.Axes, title_suffix: str) -> None:
        """Configure histogram and boxplot axes with Nordic Ocean styling."""
        # Configure histogram subplot using PlotStyle
        PlotStyle.style_axis(
            hist_ax,
            title=f"Distribution of Active SAE Features per Token {title_suffix}",
            xlabel="Active SAE features",
            ylabel="Density",
            grid_axis='y'
        )
        hist_ax.legend(frameon=False, fontsize=9)
        
        # Configure boxplot subplot using PlotStyle
        PlotStyle.style_axis(
            box_ax,
            title=f"Boxplot of Active SAE Features per Token {title_suffix}",
            xlabel="Author",
            ylabel="Active SAE features",
            grid_axis='y'
        )

    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save plot to file and close figure with Nordic Ocean styling."""
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=PlotStyle.COLORS['bg_white'])
        plt.close()

    def create_tsne_plots(self, features_dict: Dict[str, np.ndarray], labels: List[str],
                          layer_str: str, activation_type: str) -> None:
        """
        Create t-SNE visualization plots with different perplexity values.
        
        Args:
            features_dict: Dictionary containing original and normalized features
            labels: List of labels for each data point
            sae_name: Name of the SAE model
            transformer_part: Part of the transformer being analyzed
        """
        fig, axes = plt.subplots(2, 2, figsize=TSNE_FIGURE_SIZE, constrained_layout=True)

        plot_configs = self._create_tsne_plot_configs(features_dict)
        unique_authors = list(set(labels))
        author_palette = self.color_manager.get_seaborn_palette(unique_authors)
            
        for idx, (title, features, perplexity) in enumerate(plot_configs):
            row, col = idx // 2, idx % 2
            embeddings = self._compute_tsne_embeddings(features, perplexity)
            self._plot_tsne_scatter(axes[row, col], embeddings, labels, author_palette, title)

        self._finalize_tsne_plot(fig, layer_str, activation_type)

    def _create_tsne_plot_configs(self, features_dict: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray, int]]:
        """Create configuration for t-SNE plots."""
        return [
            ("Original Features", features_dict["original"], TSNE_PERPLEXITY_HIGH),
            ("Normalized Features", features_dict["normalized"], TSNE_PERPLEXITY_HIGH),
            ("Original Features, Lower Perplexity", features_dict["original"], TSNE_PERPLEXITY_LOW),
            ("Normalized Features, Lower Perplexity", features_dict["normalized"], TSNE_PERPLEXITY_LOW),
        ]

    def _compute_tsne_embeddings(self, features: np.ndarray, perplexity: int) -> np.ndarray:
        """Compute t-SNE embeddings for given features and perplexity."""
        return TSNE(n_components=2, perplexity=perplexity, random_state=TSNE_RANDOM_STATE).fit_transform(features)

    def _plot_tsne_scatter(self, ax: plt.Axes, embeddings: np.ndarray, labels: List[str], 
                          author_palette: Dict, title: str) -> None:
        """Plot t-SNE scatter plot on given axes with Nordic Ocean styling."""
        sns.scatterplot(
            x=embeddings[:, 0], y=embeddings[:, 1],
            hue=labels, palette=author_palette, s=50, alpha=SCATTER_ALPHA, ax=ax
        )
        PlotStyle.style_axis(ax, title=f't-SNE ({title})', grid_axis='both')

    def _finalize_tsne_plot(self, fig: plt.Figure, layer_str: str, activation_type: str) -> None:
        """Finalize t-SNE plot with title and save using Nordic Ocean styling."""
        fig.suptitle(f't-SNE of Docs for {activation_type} layer {layer_str}',
                     fontsize=14, color=PlotStyle.COLORS['text_dark'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

        filename = f"clusters_{activation_type}_layer_{layer_str}.png"
        plt.savefig(self.save_dir / filename, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=PlotStyle.COLORS['bg_white'])
        plt.close()

    def create_umap_plots(self, features_dict: Dict[str, np.ndarray], labels: List[str],
                          layer_str: str, activation_type: str) -> None:
        """
        Create UMAP visualization plots with different parameter values.
        
        Args:
            features_dict: Dictionary containing original and normalized features
            labels: List of labels for each data point
            layer_str: Name of the layer
            activation_type: Type of activation
        """
        fig, axes = plt.subplots(2, 2, figsize=UMAP_FIGURE_SIZE, constrained_layout=True)

        plot_configs = self._create_umap_plot_configs(features_dict)
        unique_authors = list(set(labels))
        author_palette = self.color_manager.get_seaborn_palette(unique_authors)
            
        for idx, (title, features, n_neighbors, min_dist) in enumerate(plot_configs):
            row, col = idx // 2, idx % 2
            embeddings = self._compute_umap_embeddings(features, n_neighbors, min_dist)
            self._plot_umap_scatter(axes[row, col], embeddings, labels, author_palette, title)

        self._finalize_umap_plot(fig, layer_str, activation_type)

    def _create_umap_plot_configs(self, features_dict: Dict[str, np.ndarray]) -> List[Tuple[str, np.ndarray, int, float]]:
        """Create configuration for UMAP plots."""
        return [
            ("Original Features", features_dict["original"], UMAP_N_NEIGHBORS_HIGH, UMAP_MIN_DIST_HIGH),
            ("Normalized Features", features_dict["normalized"], UMAP_N_NEIGHBORS_HIGH, UMAP_MIN_DIST_HIGH),
            ("Original Features, Lower Neighbors", features_dict["original"], UMAP_N_NEIGHBORS_LOW, UMAP_MIN_DIST_LOW),
            ("Normalized Features, Lower Neighbors", features_dict["normalized"], UMAP_N_NEIGHBORS_LOW, UMAP_MIN_DIST_LOW),
        ]

    def _compute_umap_embeddings(self, features: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
        """Compute UMAP embeddings for given features and parameters."""
        return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                        random_state=UMAP_RANDOM_STATE).fit_transform(features)

    def _plot_umap_scatter(self, ax: plt.Axes, embeddings: np.ndarray, labels: List[str], 
                          author_palette: Dict, title: str) -> None:
        """Plot UMAP scatter plot on given axes with Nordic Ocean styling."""
        sns.scatterplot(
            x=embeddings[:, 0], y=embeddings[:, 1],
            hue=labels, palette=author_palette, s=50, alpha=SCATTER_ALPHA, ax=ax
        )
        PlotStyle.style_axis(ax, title=f'UMAP ({title})', grid_axis='both')

    def _finalize_umap_plot(self, fig: plt.Figure, layer_str: str, activation_type: str) -> None:
        """Finalize UMAP plot with title and save using Nordic Ocean styling."""
        fig.suptitle(f'UMAP of Docs for {activation_type} layer {layer_str}',
                     fontsize=14, color=PlotStyle.COLORS['text_dark'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

        filename = f"umap_clusters_{activation_type}_layer_{layer_str}.png"
        plt.savefig(self.save_dir / filename, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor=PlotStyle.COLORS['bg_white'])
        plt.close()

    def create_heatmap(self, data: pd.DataFrame, title: str, filename: str) -> None:
        """
        Create and save heatmap visualization with Nordic Ocean styling.
        
        Args:
            data: DataFrame to visualize as heatmap
            title: Title for the plot
            filename: Name of the file to save the plot
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        # Use Nordic Ocean gradient colormap
        nordic_cmap = PlotStyle.create_full_gradient_cmap()
        sns.heatmap(data, annot=True, fmt=".1f", cmap=nordic_cmap, ax=ax,
                   cbar_kws={'shrink': 0.8})
        
        PlotStyle.style_axis(ax, title=title, grid_axis='')
        self._save_plot(fig, filename)

    def create_distribution_plot(self, data_dict: Dict[str, np.ndarray], title: str,
                                 xlabel: str, ylabel: str, filename: str) -> None:
        """
        Create overlaid histogram distribution plot with Nordic Ocean styling.
        
        Args:
            data_dict: Dictionary mapping author names to their data arrays
            title: Title for the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            filename: Name of the file to save the plot
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        authors = list(data_dict.keys())
        author_colors = self.color_manager.get_author_colors(authors)
        
        for author, data in data_dict.items():
            ax.hist(data, bins=HISTOGRAM_BINS, alpha=SCATTER_ALPHA, 
                   label=author, color=author_colors[author])

        PlotStyle.style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid_axis='y')
        ax.legend(frameon=False, fontsize=9)
        self._save_plot(fig, filename)

    def plot_activations_per_feature(self, activations_df: pd.DataFrame, title_suffix: str, plot_filename: str) -> None:
        """
        Plot activations per feature using Altair interactive visualization.
        
        Args:
            activations_df: DataFrame containing activation data with feature indices
            title_suffix: Suffix to add to the plot title
            plot_filename: Name of the file to save the plot
        """
        chart_df = self._prepare_activations_dataframe(activations_df)
        chart = self._create_altair_chart(chart_df, title_suffix)
        self._save_altair_chart(chart, plot_filename)

    def _prepare_activations_dataframe(self, activations_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Altair visualization."""
        chart_df = activations_df.copy()
        chart_df['feature_ind'] = pd.to_numeric(chart_df['feature_ind'], errors='coerce')
        # Remove Counter objects that Altair can't handle, keep string representation for tooltips
        return chart_df.drop(columns=['feature_counts'], errors='ignore')

    def _create_altair_chart(self, chart_df: pd.DataFrame, title_suffix: str) -> alt.Chart:
        """Create Altair chart for activation visualization with Nordic Ocean styling."""
        unique_authors = chart_df['variable'].unique().tolist()
        domain, range_colors = self.color_manager.get_altair_domain_range(unique_authors)
        
        return alt.Chart(chart_df).mark_circle(size=60, opacity=SCATTER_ALPHA).encode(
            x=alt.X('feature_ind:Q', title='Feature Index',
                   axis=alt.Axis(labelColor=PlotStyle.COLORS['text_medium'],
                                titleColor=PlotStyle.COLORS['text_dark'])),
            y=alt.Y('value:Q', title='Activation Count',
                   axis=alt.Axis(labelColor=PlotStyle.COLORS['text_medium'],
                                titleColor=PlotStyle.COLORS['text_dark'])),
            color=alt.Color('variable:N', title='Author', 
                          scale=alt.Scale(domain=domain, range=range_colors)),
            tooltip=['feature_ind', 'value', 'variable', 'feature_counts_str']
        ).properties(
            title=alt.TitleParams(
                text=f'Feature Activation Counts - {title_suffix}',
                color=PlotStyle.COLORS['text_dark'],
                fontSize=14
            ),
            width=800,
            height=400
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            gridColor=PlotStyle.COLORS['grid']
        ).interactive()

    def _save_altair_chart(self, chart: alt.Chart, plot_filename: str) -> None:
        """Save Altair chart to HTML file."""
        save_path = self.save_dir / plot_filename.replace('.png', '.html')
        chart.save(str(save_path), inline=True)
        logger.info(f"Altair chart saved to: {save_path}")

    def plot_entropies(self, entropies: np.ndarray, cross_entropies: np.ndarray, 
                      layer_str: str, transformer_part: str, author: str, 
                      activations: np.ndarray) -> None:
        """
        Plot entropies and cross-entropies against activations.
        
        Args:
            entropies: Array of shape (num_docs, num_tokens) containing entropy values
            cross_entropies: Array of shape (num_docs, num_tokens) containing cross-entropy values
            layer_str: String representation of the layer
            transformer_part: Part of the transformer being analyzed
            author: Author name for the plot title
            activations: Array of shape (num_docs, num_tokens) containing activation values
        """
        self._validate_entropy_arrays(entropies, cross_entropies, activations)
        
        df = self._prepare_entropy_dataframe(entropies, cross_entropies, activations)
        
        self._plot_entropy_scatter(df, layer_str, transformer_part, author, "entropies", "Entropy")
        self._plot_entropy_scatter(df, layer_str, transformer_part, author, "cross_entropies", "Cross-Entropy")

    def _validate_entropy_arrays(self, entropies: np.ndarray, cross_entropies: np.ndarray, 
                                activations: np.ndarray) -> None:
        """Validate that entropy arrays have matching shapes."""
        if not (entropies.shape == cross_entropies.shape == activations.shape):
            raise ValueError("Entropies, cross-entropies, and activations must have the same shape")

    def _prepare_entropy_dataframe(self, entropies: np.ndarray, cross_entropies: np.ndarray, 
                                  activations: np.ndarray) -> pd.DataFrame:
        """Prepare DataFrame for entropy plotting by flattening arrays and filtering non-zero activations."""
        activations_flat = activations.flatten()
        non_zero_mask = activations_flat > 0
        
        logger.info(f"Removed {len(activations_flat) - np.sum(non_zero_mask)} tokens with no activations")
        
        return pd.DataFrame({
            "entropies": entropies.flatten(),
            "cross_entropies": cross_entropies.flatten(),
            "activations": activations_flat
        })
        
    def _plot_entropy_scatter(self, df: pd.DataFrame, layer_str: str, transformer_part: str, 
                             author: str, entropy_type: str, entropy_label: str) -> None:
        """Create scatter plot for entropy vs activations with Nordic Ocean styling."""
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        # Use primary color from Nordic Ocean palette
        ax.scatter(df[entropy_type], df["activations"], alpha=0.2,
                  color=PlotStyle.COLORS['primary'], s=20)
        
        PlotStyle.style_axis(
            ax,
            title=f"{entropy_label} vs Activations for {layer_str} {transformer_part} {author}",
            xlabel=entropy_label,
            ylabel="Activations",
            grid_axis='both'
        )
        
        filename = f"{entropy_type}_vs_activations_{layer_str}_{transformer_part}_{author}.png"
        self._save_plot(fig, filename)

    def plot_token_positions(self, token_positions: Dict[int, List[int]], author: str, 
                           transformer_part: str, layer_str: str) -> None:
        """
        Plot token positions showing activation counts at each token position with Nordic Ocean styling.
        
        Args:
            token_positions: Dictionary mapping token indices to lists of activation counts
            author: Author name for the plot title
            transformer_part: Part of the transformer being analyzed
            layer_str: String representation of the layer
        """
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        for token_ind, activation_counts in token_positions.items():
            x_positions = np.full(len(activation_counts), token_ind)
            ax.scatter(x_positions, activation_counts, alpha=0.2,
                      color=PlotStyle.COLORS['primary'], s=20)
        
        PlotStyle.style_axis(
            ax,
            title=f"Token positions for {transformer_part} layer {layer_str} {author}",
            xlabel="Token Index",
            ylabel="Number of Activations",
            grid_axis='y'
        )
        
        filename = f"token_positions_{transformer_part}_{layer_str}_{author}.png"
        self._save_plot(fig, filename)


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
        self.visualizer = Visualizer(self.output_dir, self.color_manager)

    def analyze_single_file(self, filename: str) -> Tuple[np.ndarray, ActivationMetadata]:
        """
        Analyze a single SAE activation file.
        
        Args:
            filename: Name of the SAE activation file to analyze
            
        Returns:
            Flattened array of non-zero activation counts
        """
        title_suffix = filename.split('.')[0][22:]
        hist_filename = f"non_zeroes_hist_boxplots_{Path(filename).stem}.png"

        non_zeroes_count, metadata = self._load_or_compute_activations(filename)
        self.visualizer.plot_activation_histogram(non_zeroes_count, title_suffix, hist_filename)
        
        gc.collect()
        return non_zeroes_count, metadata

    
    def _load_or_compute_activations(self, filename: str) -> Tuple[np.ndarray, ActivationMetadata]:
        """Load precomputed activations or compute them from scratch."""
        # Use ActivationMetadata.get_metadata_filename to get the base name
        base_name = ActivationMetadata.get_metadata_filename(filename).replace('.meta.pkl', '')
        npy_path = self.output_dir / f"non_zeroes_count_{base_name}.npy"
        
        if npy_path.exists():
            non_zeroes_count = np.load(npy_path)
            # Pass the activation filename directly
            metadata = ActivationMetadata.load(str(self.data_dir / filename))
            return non_zeroes_count, metadata
        
        logger.info(f"Computing activations for {filename}")
        activations, metadata = self.data_loader.load_sae_activations(self.data_dir / filename)
        non_zeroes_count = self._compute_non_zero_activations(activations, metadata)
        
        # Save for future use
        np.save(npy_path, non_zeroes_count)
        return non_zeroes_count, metadata

    def _compute_non_zero_activations(self, activations: Union[np.ndarray, sp.spmatrix], 
                                    metadata: ActivationMetadata) -> np.ndarray:
        """Compute non-zero activation counts from activation data."""
        if sp.issparse(activations):
            return self._compute_sparse_activations(activations, metadata)
        else:
            return self._compute_dense_activations(activations)

    def _compute_sparse_activations(self, activations: sp.spmatrix, 
                                  metadata: ActivationMetadata) -> np.ndarray:
        """Compute non-zero counts for sparse activation matrix."""
        # Apply binary threshold
        activations.data = (activations.data >= ACTIVATION_THRESHOLD).astype(BINARY_ACTIVATION_TYPE)
        activations.eliminate_zeros()
        
        # Count non-zero elements per row (token position)
        non_zeroes_count = np.array(activations.getnnz(axis=1))

        # Reshape to (n_docs, max_seq_len) based on metadata
        n_docs, max_seq_len = metadata.original_shape[0], metadata.original_shape[1]
        return non_zeroes_count.reshape(n_docs, max_seq_len)

    def _compute_dense_activations(self, activations: np.ndarray) -> np.ndarray:
        """Compute non-zero counts for dense activation matrix."""
        binary_activations = (activations >= ACTIVATION_THRESHOLD).astype(BINARY_ACTIVATION_TYPE)
        return np.count_nonzero(binary_activations, axis=2)

    def analyze_all_files(self, filenames: Dict[str, Dict[str, Dict[str, str]]]) -> None:
        """
        Analyze all files for activation statistics.
        
        Args:
            filenames: List of filenames to analyze
        """

        aggregated_activations, mean_activations_per_token = self._process_files(filenames)
        
        # Force garbage collection after processing all files
        gc.collect()
        logger.info("Memory cleanup completed. Generating reports...")
        
        # Generate reports
        self._generate_reports_parallel(aggregated_activations, mean_activations_per_token)

    def _process_files(self, filenames: Dict[str, Dict[str, Dict[str, str]]]) -> Tuple[Dict, Dict]:
        """Process files in parallel and aggregate results."""
        aggregated_activations = defaultdict(lambda: defaultdict(dict))
        mean_activations_per_token = defaultdict(lambda: defaultdict(dict))

        for activation_type, layer_str_data in filenames.items():
            for layer_str, author_data in layer_str_data.items():
                for author, filename in author_data.items():
                    try:
                        non_zeroes_count, metadata = self.analyze_single_file(filename)
                        self._aggregate_file_results(non_zeroes_count, metadata,
                                                    aggregated_activations, mean_activations_per_token)
                    except Exception as e:
                        logger.error(f"Error processing file {filename}: {e}")

        return aggregated_activations, mean_activations_per_token

    def _aggregate_file_results(self, non_zeroes_count: np.ndarray,
                              metadata: ActivationMetadata, aggregated_activations: Dict, mean_activations_per_token: Dict) -> None:
        """Aggregate results from a single file."""
        activation_type = metadata.layer_type
        layer_str = metadata.layer_index
        author = metadata.author_id

        # Calculate mean activations for non-zero values
        non_zero_values = non_zeroes_count[non_zeroes_count > 0]
        averaged_activations = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0.0
        
        mean_activations_per_token[author][layer_str][activation_type] = averaged_activations
        aggregated_activations[activation_type][layer_str][author] = non_zeroes_count

    def _generate_reports_parallel(self, aggregated_activations: Dict, mean_activations_per_token: Dict) -> None:
        """Generate reports with parallel processing."""
        self._generate_visualization_reports(aggregated_activations)
        self._generate_author_reports(mean_activations_per_token)

    def _generate_visualization_reports(self, aggregated_activations: Dict) -> None:
        """Generate visualization reports in parallel."""
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = []
            
            for activation_type, data_items in aggregated_activations.items():
                for layer_str, author_data in data_items.items():
                    future = executor.submit(
                        self._create_aggregated_histogram,
                        author_data, activation_type, layer_str
                    )
                    futures.append(future)
            
            # Wait for all visualizations to complete
            for future in tqdm(futures, desc="Generating visualizations"):
                future.result()

    def _create_aggregated_histogram(self, author_data: Dict, activation_type: str, layer_str: str) -> None:
        """Create aggregated histogram for a specific layer and activation type."""
        # Flatten 2D arrays to 1D for DataFrame creation
        flattened_data = {}
        for author, activations in author_data.items():
            if isinstance(activations, np.ndarray):
                flattened_data[author] = activations.flatten()
            else:
                flattened_data[author] = activations
        
        df = pd.DataFrame.from_dict(flattened_data, orient='columns')
        logger.info(f"Creating histogram for {activation_type} layer {layer_str}: shape={df.shape}")
        
        df_long = df.melt(value_vars=flattened_data.keys(), var_name='author', value_name='activations')
        title_suffix = f"Layer {layer_str} {activation_type}"
        hist_filename = f"non_zeroes_hist_aggregated_{activation_type}_{layer_str}.png"
        
        self.visualizer.plot_activation_histogram(df_long, title_suffix, hist_filename)

    def _generate_author_reports(self, data: Dict) -> None:
        """Generate per-author analysis reports."""
        for author, layers_dict in data.items():
            df = self._prepare_author_dataframe(layers_dict)
            self._save_author_csv(df, author)
            self._create_author_heatmap(df, author)

    def _prepare_author_dataframe(self, layers_dict: Dict) -> pd.DataFrame:
        """Prepare DataFrame for author analysis."""
        df = pd.DataFrame.from_dict(layers_dict, orient='index')
        df.index.name = "layer"
        df.columns.name = "activation_type"
        return df.sort_index().sort_index(axis=1)

    def _save_author_csv(self, df: pd.DataFrame, author: str) -> None:
        """Save author data to CSV file."""
        csv_path = self.output_dir / f"{author}_non_zero_count_activations.csv"
        df.to_csv(csv_path)
        logger.info(f"Saved author report for {author} to {csv_path}")

    def _create_author_heatmap(self, df: pd.DataFrame, author: str) -> None:
        """Create heatmap visualization for author data."""
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
        for activation_type, layers_dict in cluster_metrics.items():
            df_metrics = pd.DataFrame.from_dict(layers_dict, orient='index')
            df_metrics.index.name = "layer"
            df_metrics = df_metrics.sort_index()
            df_metrics.to_csv(self.output_dir / f"clusters_metrics_{activation_type}.csv", index=False)

    def analyze_clusters(self, filenames: Dict[str, Dict[str, Dict[str, str]]]) -> None:
        """
        Perform clustering analysis across all files.
        
        Args:
            filenames: List of filenames to analyze for clustering
        """
        cluster_metrics = self._compute_cluster_metrics_parallel(filenames)
        self._save_cluster_metrics(cluster_metrics)


    def _compute_cluster_metrics_parallel(self, grouped_files: Dict) -> Dict:
        """Compute cluster metrics in parallel."""
        cluster_metrics = defaultdict(lambda: defaultdict(dict))
        jobs = self._create_cluster_jobs(grouped_files)

        with ProcessPoolExecutor(max_workers=CLUSTER_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._analyze_layer_cluster, authors_dict, activation_type, layer_str): (activation_type, layer_str)
                for activation_type, layer_str, authors_dict in jobs
            }

        for future in as_completed(futures):
            activation_type, layer_str = futures[future]
            try:
                metrics = future.result()
                cluster_metrics[activation_type][layer_str] = metrics.to_dict()
            except Exception as e:
                logger.error(f"Error computing cluster metrics for {activation_type} {layer_str}: {e}")

        return cluster_metrics

    def _create_cluster_jobs(self, grouped_files: Dict) -> List[Tuple[str, str, Dict]]:
        """Create jobs for parallel cluster analysis."""
        jobs = []
        for activation_type, layers_dict in grouped_files.items():
            for layer_str, authors_dict in layers_dict.items():
                jobs.append((activation_type, layer_str, authors_dict))
        return jobs

    def _analyze_layer_cluster(self, authors_dict: Dict[str, str], activation_type: str, layer_str: str) -> ClusterMetrics:
        """
        Analyze clustering for a specific layer.
        
        Args:
            authors_dict: Dictionary mapping author names to filenames
            activation_type: Type of activation being analyzed
            
        Returns:
            ClusterMetrics object containing clustering results
        """
        all_activations, all_activations_short, author_list, author_to_docs = self._load_author_activations(authors_dict)
        
        combined_activations = np.concatenate(all_activations, axis=0)
        combined_activations_short = np.concatenate(all_activations_short, axis=0)
        cluster_labels = self._create_cluster_labels(author_list, author_to_docs)
        
        features_dict = self._prepare_clustering_features(combined_activations)
        self._create_clustering_visualizations(features_dict, cluster_labels, activation_type, layer_str)
        
        return self.metrics_calculator.compute_cluster_metrics(combined_activations, cluster_labels)

    def _load_author_activations(self, authors_dict: Dict[str, str]) -> Tuple[List, List, List, Dict]:
        """Load and process activations for all authors."""
        all_activations, all_activations_short = [], []
        author_list, author_to_docs = [], {}

        for author, filename in authors_dict.items():
            activations, metadata = self.data_loader.load_sae_activations(self.data_dir / filename)
            activations_dense = self._process_activations_for_clustering(activations, metadata)
            
            # Process activations with different sequence lengths
            agg_activations = self.activation_processor.aggregate_normalized(activations_dense, metadata.doc_lengths)
            agg_activations_short = self.activation_processor.aggregate_normalized(
                activations_dense, metadata.doc_lengths, SHORT_SEQUENCE_LENGTH
            )

            all_activations.append(agg_activations)
            all_activations_short.append(agg_activations_short)
            author_list.append(metadata.author_id)
            author_to_docs[metadata.author_id] = metadata.doc_lengths

        return all_activations, all_activations_short, author_list, author_to_docs

    def _process_activations_for_clustering(self, activations: Union[np.ndarray, sp.spmatrix], 
                                          metadata: ActivationMetadata) -> np.ndarray:
        """Process activations for clustering analysis."""
        if sp.issparse(activations):
            return self._process_sparse_activations_for_clustering(activations, metadata)
        else:
            return self._process_dense_activations_for_clustering(activations)

    def _process_sparse_activations_for_clustering(self, activations: sp.spmatrix, 
                                                 metadata: ActivationMetadata) -> np.ndarray:
        """Process sparse activations for clustering."""
        activations.data = (activations.data >= ACTIVATION_THRESHOLD).astype(BINARY_ACTIVATION_TYPE)
        activations.eliminate_zeros()
        activations_dense = activations.toarray()
        return activations_dense.reshape(metadata.original_shape)

    def _process_dense_activations_for_clustering(self, activations: np.ndarray) -> np.ndarray:
        """Process dense activations for clustering."""
        return (activations >= ACTIVATION_THRESHOLD).astype(BINARY_ACTIVATION_TYPE)

    def _create_cluster_labels(self, author_list: List[str], author_to_docs: Dict[str, List]) -> List[str]:
        """Create cluster labels for all documents."""
        cluster_labels = []
        for author in author_list:
            num_docs = len(author_to_docs[author])
            cluster_labels.extend([author] * num_docs)
        return cluster_labels

    def _prepare_clustering_features(self, combined_activations: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare features for clustering analysis."""
        return {
            "original": combined_activations,
            "normalized": self.metrics_calculator.scaler.fit_transform(combined_activations),
        }

    def _create_clustering_visualizations(self, features_dict: Dict[str, np.ndarray], 
                                        cluster_labels: List[str], activation_type: str, layer_str: str) -> None:
        """Create t-SNE and UMAP visualizations for clustering analysis."""
        # Get SAE ID from metadata (assuming it's consistent across authors)
        
        self.visualizer.create_tsne_plots(features_dict, cluster_labels, layer_str, activation_type)
        self.visualizer.create_umap_plots(features_dict, cluster_labels, layer_str, activation_type)

    def analyze_detailed_counts(self, filenames: Dict[str, Dict[str, Dict[str, str]]]) -> None:
        """
        Analyze detailed activation counts per feature across all files.
        
        Args:
            filenames: List of filenames to analyze
        """
        activations_aggregated = defaultdict(lambda: defaultdict(dict))
        text_lengths = defaultdict()
        tokens = self._load_author_tokens(filenames)
        
        for activation_type, layer_str_data in filenames.items():
            for layer_str, author_data in layer_str_data.items():
                for author, filename_base in author_data.items():
                    filename = self._resolve_filename_extension(filename_base)
                    activations, metadata = self.data_loader.load_sae_activations(self.data_dir / filename)
                    activations_dense = self._process_activations_for_detailed_analysis(activations, metadata)
                    
                    activation_type = metadata.layer_type
                    layer_str = metadata.layer_index
                    author = metadata.author_id
                    
                    non_zeroes_per_feature = np.count_nonzero(activations_dense, axis=(0, 1))
                    logger.info(f"Shape of aggregated activations: {non_zeroes_per_feature.shape}")
                    
                    activations_aggregated[activation_type][layer_str][author] = non_zeroes_per_feature
                    text_lengths[author] = np.sum(metadata.doc_lengths)
                    
                    self._extract_token_counts(activations_dense, author, tokens)
        
        tokens_counter = self._create_token_counters(tokens)
        self._generate_detailed_reports(activations_aggregated, tokens_counter)

    def _load_author_tokens(self, filenames: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, List]:
        """Load tokens for all authors."""
        authors = list(set([author for activation_type, layer_str_data in filenames.items() for layer_str, author_data in layer_str_data.items() for author in author_data.keys()]))
        tokens_loader = TokenandFullTextFilenamesLoader(self.data_dir, include_authors=authors)
        tokens_structured = tokens_loader.get_structured_filenames()
        
        tokens = {}
        for author in authors:
            if author in tokens_structured.get('tokens', {}):
                tokens_file = tokens_structured['tokens'][author].get('baseline', '')
                if tokens_file:
                    with open(self.data_dir / tokens_file, 'r', encoding='utf-8') as f:
                        tokens[author] = json.load(f)
        return tokens

    def _process_activations_for_detailed_analysis(self, activations: Union[np.ndarray, sp.spmatrix], 
                                                 metadata: ActivationMetadata) -> np.ndarray:
        """Process activations for detailed analysis."""
        if sp.issparse(activations):
            return self._process_sparse_activations_for_detailed_analysis(activations, metadata)
        else:
            return self._process_dense_activations_for_detailed_analysis(activations)

    def _process_sparse_activations_for_detailed_analysis(self, activations: sp.spmatrix, 
                                                        metadata: ActivationMetadata) -> np.ndarray:
        """Process sparse activations for detailed analysis."""
        activations.data = (activations.data >= ACTIVATION_THRESHOLD).astype(BINARY_ACTIVATION_TYPE)
        activations.eliminate_zeros()
        activations_dense = activations.toarray()
        return activations_dense.reshape(metadata.original_shape)

    def _process_dense_activations_for_detailed_analysis(self, activations: np.ndarray) -> np.ndarray:
        """Process dense activations for detailed analysis."""
        return (activations >= ACTIVATION_THRESHOLD).astype(BINARY_ACTIVATION_TYPE)

    def _extract_token_counts(self, activations_dense: np.ndarray, author: str, tokens: Dict[str, List]) -> None:
        """Extract token counts for activated features."""
        # This method would need to be implemented to extract token counts
        # The original logic was complex and would need to be refactored
        pass

    def _create_token_counters(self, tokens: Dict[str, List]) -> Dict[str, Dict[int, Counter]]:
        """Create token counters for each author and feature."""
        # This method would need to be implemented based on the original logic
        return defaultdict(lambda: defaultdict(Counter))

    def _generate_detailed_reports(self, activations_aggregated: Dict, tokens_counter: Dict) -> None:
        """Generate detailed reports for each activation type and layer."""
        for activation_type, activation_type_items in activations_aggregated.items():
            for layer_str, layer_items in activation_type_items.items():
                if not layer_items:
                    logger.info(f"No data found for {activation_type} layer {layer_str}") 
                    continue
                    
                self._create_layer_detailed_report(layer_items, activation_type, layer_str, tokens_counter)

    def _create_layer_detailed_report(self, layer_items: Dict, activation_type: str, 
                                    layer_str: str, tokens_counter: Dict) -> None:
        """Create detailed report for a specific layer."""
        df = pd.DataFrame.from_dict(layer_items)
        if df.empty:
            logger.info(f"Empty DataFrame for {activation_type} layer {layer_str}")
            return
                    
        df["feature_ind"] = df.index
        df_long = df.melt(id_vars=['feature_ind'], value_vars=layer_items.keys())
                
        # Add feature counts with error handling
        feature_counts_list = self._add_feature_counts_with_error_handling(df_long, tokens_counter)
        df_long["feature_counts"] = feature_counts_list
        df_long["feature_counts_str"] = df_long["feature_counts"].apply(
            lambda x: str(dict(x.most_common())) if x else "{}"
        )
        
        # Save and visualize
        self._save_detailed_csv(df_long, activation_type, layer_str)
        self._create_detailed_visualization(df_long, activation_type, layer_str)

    def _add_feature_counts_with_error_handling(self, df_long: pd.DataFrame, 
                                              tokens_counter: Dict) -> List[Counter]:
        """Add feature counts with proper error handling."""
        feature_counts_list = []
        for ind, row in df_long.iterrows():
            try:
                author = row["variable"]
                feature_ind = row["feature_ind"]
                if author in tokens_counter and feature_ind in tokens_counter[author]:
                    feature_counts_list.append(tokens_counter[author][feature_ind])
                else:
                    feature_counts_list.append(Counter())
            except KeyError as e:
                logger.warning(f"Key error for row {ind}: {e}")
        #feature_counts_list.append(Counter())
        return feature_counts_list

    def _save_detailed_csv(self, df_long: pd.DataFrame, activation_type: str, layer_str: str) -> None:
        """Save detailed analysis to CSV."""
        csv_path = self.output_dir / f"frequencies_activated_per_feature_{activation_type}_{layer_str}.csv"
        df_long.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed analysis to {csv_path}")

    def _create_detailed_visualization(self, df_long: pd.DataFrame, activation_type: str, layer_str: str) -> None:
        """Create detailed visualization."""
        title_suffix = f"Layer {layer_str} {activation_type}"
        plot_filename = f"frequencies_activated_per_feature_{activation_type}_{layer_str}.png"
        self.visualizer.plot_activations_per_feature(df_long, title_suffix, plot_filename)
    
    def analyze_entropies(self, filenames: List[str], entropies_files: List[str], 
                         cross_entropies_files: List[str]) -> None:
        """
        Analyze entropy and cross-entropy statistics across all files.
        
        Args:
            filenames: List of activation filenames to analyze
            entropies_files: List of entropy file names
            cross_entropies_files: List of cross-entropy file names
        """
        aggregated_activations = self._load_activations_for_entropy_analysis(filenames)
        entropies = self._load_entropy_data(entropies_files)
        cross_entropies = self._load_cross_entropy_data(cross_entropies_files)
        
        self._generate_entropy_plots(aggregated_activations, entropies, cross_entropies)

    def _load_activations_for_entropy_analysis(self, filenames: Dict[str, Dict[str, Dict[str, str]]]) -> Dict:
        """Load activations for entropy analysis."""
        aggregated_activations = defaultdict(lambda: defaultdict(dict))
            
        for activation_type, layer_str_data in filenames.items():
            for layer_str, author_data in layer_str_data.items():
                for author, filename_base in author_data.items():
                    filename = self._resolve_filename_extension(filename_base)
                    non_zeroes_count, _metadata = self._load_or_compute_activations(filename)

                    aggregated_activations[activation_type][layer_str][author] = non_zeroes_count

        return aggregated_activations

    def _load_entropy_data(self, entropies_files: List[str]) -> Dict[str, np.ndarray]:
        """Load entropy data from files."""
        entropies = {}
        logger.info(f"Loading entropy files: {entropies_files}")
        
        for entropy_filename in entropies_files:
            author = self._extract_author_from_filename(entropy_filename)
            logger.info(f"Loading entropy file for author {author}")
            entropies[author] = np.load(self.data_dir / entropy_filename)
        
        logger.info(f"Loaded entropies for authors: {list(entropies.keys())}")
        return entropies

    def _load_cross_entropy_data(self, cross_entropies_files: List[str]) -> Dict[str, np.ndarray]:
        """Load cross-entropy data from files."""
        cross_entropies = {}
        logger.info(f"Loading cross-entropy files: {cross_entropies_files}")
        
        for cross_entropy_filename in cross_entropies_files:
            author = self._extract_author_from_filename(cross_entropy_filename)
            cross_entropies[author] = np.load(self.data_dir / cross_entropy_filename)

        return cross_entropies

    def _extract_author_from_filename(self, filename: str) -> str:
        """Extract author name from filename."""
        stem_filename = filename.split(".")[0]
        return stem_filename.split("_")[-1]

    def _generate_entropy_plots(self, aggregated_activations: Dict, entropies: Dict[str, np.ndarray], 
                              cross_entropies: Dict[str, np.ndarray]) -> None:
        """Generate entropy plots for all combinations."""
        for activation_type, activation_type_items in aggregated_activations.items():
            for layer_str, layer_items in activation_type_items.items():
                for author, activations in layer_items.items():
                    if self._has_entropy_data(author, entropies, cross_entropies):
                        self.visualizer.plot_entropies(
                            entropies[author], cross_entropies[author], 
                            layer_str, activation_type, author, activations
                        )
                    else:
                        logger.warning(f"Author {author} not found in entropies or cross_entropies")

    def _has_entropy_data(self, author: str, entropies: Dict[str, np.ndarray], 
                         cross_entropies: Dict[str, np.ndarray]) -> bool:
        """Check if entropy data exists for the given author."""
        return author in entropies and author in cross_entropies

    def _resolve_filename_extension(self, filename_base: str) -> str:
        """
        Resolve filename by adding the correct extension (.sparse.npz or .npz).
        
        Args:
            filename_base: Filename without extension
            
        Returns:
            Full filename with extension
        """
        # Check for sparse format first
        sparse_path = self.data_dir / f"{filename_base}.sparse.npz"
        if sparse_path.exists():
            return f"{filename_base}.sparse.npz"
        
        # Fall back to dense format
        dense_path = self.data_dir / f"{filename_base}.npz"
        if dense_path.exists():
            return f"{filename_base}.npz"
        
        # If neither exists, return with .npz extension (will fail later with clear error)
        logger.warning(f"Could not find file with base name: {filename_base}")
        return f"{filename_base}.npz"
    
    def analyze_token_positions(self, filenames: List[str]) -> None:
        """
        Analyze token positions and their activation patterns.
        
        Args:
            filenames: List of filenames to analyze for token positions
        """
        token_positions = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for activation_type, layer_ind_dict  in filenames.items():
            for layer_ind, author_dict in layer_ind_dict.items():
                for author, filename_base in author_dict.items():
                    # Add extension back - check which extension exists
                    filename = self._resolve_filename_extension(filename_base)
                    logger.info(f"Processing: {activation_type} {layer_ind} {author} - resolved to: {filename}")
                    
                    non_zeroes_count, _ = self._load_or_compute_activations(filename)
                    parsed = self.filename_parser.parse_filename(filename)
                    self.filename_parser.validate_activation_type(parsed["activation_type"])
                    layer_str = self.filename_parser.format_layer_string(parsed["layer_ind"])

                    self._extract_token_positions(non_zeroes_count, parsed, layer_str, token_positions)

        self._generate_token_position_plots(token_positions)

    def _extract_token_positions(self, non_zeroes_count: np.ndarray, parsed: Dict, 
                               layer_str: str, token_positions: Dict) -> None:
        """Extract token positions from activation data."""
        for doc_ind, doc_activations in enumerate(non_zeroes_count):
            for token_ind, number_activations in enumerate(doc_activations):
                if number_activations > 0:
                    token_positions[parsed["activation_type"]][layer_str][parsed["author"]][token_ind].append(number_activations)

    def _generate_token_position_plots(self, token_positions: Dict) -> None:
        """Generate token position plots for all combinations."""
        for activation_type, activation_type_items in token_positions.items():
            for layer_str, layer_items in activation_type_items.items():
                for author, author_items in layer_items.items():
                    self.visualizer.plot_token_positions(author_items, author, activation_type, layer_str)


def main() -> None:
    """Main entry point for SAE activation analysis."""
    args = _parse_arguments()
    
    # Get data and output paths
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type=args.mode,
        run_name_override=args.run_name
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type=args.mode,
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")
    
    analyzer = SAEAnalyzer(data_path, output_path, args.run_name or "")

    if args.filename:
        _analyze_single_file(analyzer, args.filename)
    else:
        _analyze_multiple_files(analyzer, data_path, args)

def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAE Activation Analysis Tool")
    
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
        "--filename",
        type=str,
        default=None,
        help="Specific file to analyze (if not provided, analyzes all files)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["explore", "cluster", "detailed", "entropies", "token_positions", "all"],
        default="entropies",
        help="Analysis mode: explore, cluster, detailed, entropies, token_positions, or all"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name override for output directory"
    )
    parser.add_argument(
        "--include_authors",
        type=str,
        nargs="+",
        default=None,
        help="Authors to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_types",
        type=str,
        nargs="+",
        default=None,
        choices=["res", "mlp", "att"],
        help="Layer types to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_inds",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to include in the analysis"
    )

    args = parser.parse_args()
    
    # Validate that either run_id or path_to_data is provided
    if not args.run_id and not args.path_to_data:
        parser.error("Either --run_id or --path_to_data must be provided")
    
    return args

def _analyze_single_file(analyzer: SAEAnalyzer, filename: str) -> None:
    """Analyze a single file."""
    analyzer.analyze_single_file(filename)

def _analyze_multiple_files(analyzer: SAEAnalyzer, data_dir: Path, args: argparse.Namespace) -> None:
    """Analyze multiple files based on mode."""
    filenames = _load_filenames(data_dir, args)

    logger.info(f"Loaded {len(filenames)} files") 
           
    
    if args.mode in ["explore", "all"]:
        analyzer.analyze_all_files(filenames)

    if args.mode in ["cluster", "all"]:
        analyzer.analyze_clusters(filenames)

    if args.mode in ["detailed", "all"]:
        analyzer.analyze_detailed_counts(filenames)

    if args.mode in ["entropies", "all"]:
        _analyze_entropies(analyzer, data_dir, args, filenames)

    if args.mode in ["token_positions", "all"]:
        analyzer.analyze_token_positions(filenames)

def _load_filenames(data_dir: Path, args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load filenames using the filename loader with filters."""
    filename_loader = ActivationFilenamesLoader(
        data_dir=data_dir,
        include_authors=args.include_authors,
        include_layer_types=args.include_layer_types,
        include_layer_inds=args.include_layer_inds
    )
    return filename_loader.get_structured_filenames()

def _analyze_entropies(analyzer: SAEAnalyzer, data_dir: Path, args: argparse.Namespace, filenames: List[str]) -> None:
    """Analyze entropy data."""
    entropy_loader = EntropyFilenamesLoader(
        data_dir=data_dir,
        include_authors=args.include_authors
    )
    entropy_structured = entropy_loader.get_structured_filenames()
    
    entropies_files, cross_entropies_files = _extract_entropy_files(entropy_structured)
    analyzer.analyze_entropies(filenames, entropies_files, cross_entropies_files)

def _extract_entropy_files(entropy_structured: Dict) -> Tuple[List[str], List[str]]:
    """Extract entropy and cross-entropy files from structured data."""
    entropies_files = []
    cross_entropies_files = []
    
    for entropy_type, authors_dict in entropy_structured.items():
        for author, prompted_dict in authors_dict.items():
            if 'baseline' in prompted_dict:
                if entropy_type == 'entropy':
                    entropies_files.append(prompted_dict['baseline'])
                elif entropy_type == 'cross_entropy_loss':
                    cross_entropies_files.append(prompted_dict['baseline'])
    
    return entropies_files, cross_entropies_files

if __name__ == "__main__":
    main()
