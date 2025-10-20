#!/usr/bin/env python3
"""
Visualization script for iterative variance threshold feature selection results.

This script creates interactive Altair visualizations for precision, recall, and F1-score
across different authors, layers, and layer types.

Usage:
    python visualize_iterative_results.py --data_dir path/to/json/files
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd
import altair as alt
import numpy as np
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Altair
alt.data_transformers.disable_max_rows()


class AuthorColorManager:
    """Manages consistent color assignments for authors using a predefined palette."""
    
    # Define a comprehensive color palette (from colorbrewer/d3)
    COLOR_PALETTE = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#1b9e77',  # Teal
        '#d95f02',  # Dark Orange
        '#7570b3',  # Light Purple
        '#e7298a',  # Magenta
        '#66a61e',  # Olive
        '#e6ab02',  # Gold
        '#a6761d',  # Tan
        '#666666',  # Gray
    ]
    
    def __init__(self):
        self.author_to_color = {}
        self.color_index = 0
        
    def get_author_color(self, author: str) -> str:
        """Get consistent color for an author."""
        if author not in self.author_to_color:
            self.author_to_color[author] = self.COLOR_PALETTE[self.color_index % len(self.COLOR_PALETTE)]
            self.color_index += 1
        return self.author_to_color[author]
    
    def get_color_with_tone(self, author: str, layer_ind: int, max_layers: int = 42) -> str:
        """
        Get a color with different tone based on layer index.
        Lower layer indices get darker tones, higher get lighter tones.
        """
        base_color = self.get_author_color(author)
        
        # Convert hex to RGB
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        # Calculate tone factor (0.6 to 1.4 range)
        # Lower layers are darker, higher layers are lighter
        tone_factor = 0.6 + (layer_ind / max_layers) * 0.8
        
        # Apply tone
        r = int(min(255, r * tone_factor))
        g = int(min(255, g * tone_factor))
        b = int(min(255, b * tone_factor))
        
        return f'#{r:02x}{g:02x}{b:02x}'


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize iterative variance threshold feature selection results"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output_data/AuthorMix/google_gemma-2-9b/politics_500_iterative_variance_threshold",
        help="Directory containing JSON result files"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix for output filenames"
    )
    
    return parser.parse_args()


def load_json_results(data_dir: Path) -> List[Dict]:
    """
    Load all iterative variance threshold JSON files from directory.
    
    Args:
        data_dir: Directory containing JSON files
        
    Returns:
        List of dictionaries with parsed data
    """
    logger.info(f"Loading JSON files from {data_dir}")
    
    # Find all relevant JSON files
    json_files = list(data_dir.glob("iterative_variance_threshold__*__*__*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return []
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    results = []
    
    for json_file in json_files:
        try:
            # Parse filename to extract info
            # Format: iterative_variance_threshold__{layer_type}__{layer_ind}__{author}.json
            filename = json_file.stem
            parts = filename.split("__")
            
            if len(parts) < 4:
                logger.warning(f"Skipping file with unexpected format: {json_file.name}")
                continue
            
            layer_type = parts[1]
            layer_ind = int(parts[2])
            author = parts[3]
            
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Add metadata
            result = {
                'file': json_file.name,
                'layer_type': layer_type,
                'layer_ind': layer_ind,
                'author': author,
                'data': data
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(results)} result files")
    return results


def create_dataframe(results: List[Dict], color_manager: AuthorColorManager) -> pd.DataFrame:
    """
    Convert loaded results into a pandas DataFrame for visualization.
    
    Args:
        results: List of result dictionaries
        color_manager: AuthorColorManager instance
        
    Returns:
        pandas DataFrame with all data
    """
    logger.info("Creating DataFrame from results")
    
    rows = []
    
    # Get max layer index for tone calculation
    max_layer_ind = max([r['layer_ind'] for r in results]) if results else 42
    
    for result in results:
        author = result['author']
        layer_type = result['layer_type']
        layer_ind = result['layer_ind']
        
        # Get color with tone
        color = color_manager.get_color_with_tone(author, layer_ind, max_layer_ind)
        base_color = color_manager.get_author_color(author)
        
        # Extract iterations
        iterations = result['data'].get('iterations', [])
        
        for iteration_data in iterations:
            row = {
                'author': author,
                'layer_type': layer_type,
                'layer_ind': layer_ind,
                'iteration': iteration_data['iteration'],
                'n_features': iteration_data['n_features'],
                'precision': iteration_data['precision_class_1'],
                'recall': iteration_data['recall_class_1'],
                'f1_score': iteration_data['f1_score_class_1'],
                'color': color,
                'base_color': base_color,
                'author_layer_type': f"{author}_{layer_type}_{layer_ind}"
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    logger.info(f"Created DataFrame with {len(df)} rows")
    logger.info(f"Authors: {df['author'].nunique()}")
    logger.info(f"Layer types: {df['layer_type'].unique().tolist()}")
    logger.info(f"Layer indices: {sorted(df['layer_ind'].unique().tolist())}")
    
    return df


def create_interactive_chart(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    color_manager: AuthorColorManager
) -> alt.Chart:
    """
    Create an interactive Altair chart for a specific metric.
    
    Args:
        df: DataFrame with data
        metric: Column name for the metric ('precision', 'recall', 'f1_score')
        metric_label: Display label for the metric
        color_manager: AuthorColorManager instance
        
    Returns:
        Altair Chart object
    """
    logger.info(f"Creating interactive chart for {metric_label}")
    
    # Define marker shapes for layer types
    shape_map = {
        'res': 'circle',
        'mlp': 'square',
        'att': 'triangle-up'
    }
    
    # Create selection for interactivity
    # Author selection - click on legend to toggle
    author_selection = alt.selection_multi(
        fields=['author'],
        bind='legend',
        name='author_select'
    )
    
    # Layer type selection - click on legend to toggle
    layer_type_selection = alt.selection_multi(
        fields=['layer_type'],
        bind='legend',
        name='layer_type_select'
    )
    
    # Layer index selection - click on legend to toggle
    layer_ind_selection = alt.selection_multi(
        fields=['layer_ind'],
        bind='legend',
        name='layer_ind_select'
    )
    
    # Create the base chart with conditional opacity based on all selections
    base = alt.Chart(df).mark_line(
        point=True,
        size=2
    ).encode(
        x=alt.X('n_features:Q',
                scale=alt.Scale(type='log'),
                axis=alt.Axis(title='Number of Features (log scale)', grid=True)),
        y=alt.Y(f'{metric}:Q',
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(title=metric_label, grid=True)),
        color=alt.Color('author:N',
                       scale=alt.Scale(
                           domain=list(color_manager.author_to_color.keys()),
                           range=list(color_manager.author_to_color.values())
                       ),
                       legend=alt.Legend(title='Author (click to toggle)')),
        shape=alt.Shape('layer_type:N',
                       scale=alt.Scale(
                           domain=list(shape_map.keys()),
                           range=list(shape_map.values())
                       ),
                       legend=alt.Legend(title='Layer Type (click to toggle)')),
        strokeDash=alt.StrokeDash('layer_ind:N',
                                  legend=alt.Legend(title='Layer Index (click to toggle)', 
                                                   symbolLimit=50)),
        detail='author_layer_type:N',
        opacity=alt.condition(
            author_selection & layer_type_selection & layer_ind_selection,
            alt.value(0.8),
            alt.value(0.05)
        ),
        tooltip=[
            alt.Tooltip('author:N', title='Author'),
            alt.Tooltip('layer_type:N', title='Layer Type'),
            alt.Tooltip('layer_ind:Q', title='Layer Index'),
            alt.Tooltip('iteration:Q', title='Iteration'),
            alt.Tooltip('n_features:Q', title='Features'),
            alt.Tooltip(f'{metric}:Q', title=metric_label, format='.4f')
        ]
    ).properties(
        width=900,
        height=500,
        title=f'{metric_label} vs Number of Features (Iterative Variance Threshold)'
    ).add_selection(
        author_selection,
        layer_type_selection,
        layer_ind_selection
    )
    
    # Add interactive zoom
    zoom = alt.selection_interval(
        bind='scales',
        encodings=['x', 'y']
    )
    base = base.add_selection(zoom)
    
    return base


def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for the data."""
    summary = df.groupby(['author', 'layer_type', 'layer_ind']).agg({
        'precision': ['mean', 'min', 'max', 'std'],
        'recall': ['mean', 'min', 'max', 'std'],
        'f1_score': ['mean', 'min', 'max', 'std'],
        'n_features': ['min', 'max'],
        'iteration': 'count'
    }).round(4)
    
    return summary


def create_comparison_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Create a comparison chart showing all three metrics together.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Altair Chart object
    """
    logger.info("Creating comparison chart for all metrics")
    
    # Reshape data for comparison
    df_melted = df.melt(
        id_vars=['author', 'layer_type', 'layer_ind', 'n_features', 'iteration', 'author_layer_type'],
        value_vars=['precision', 'recall', 'f1_score'],
        var_name='metric',
        value_name='value'
    )
    
    # Create selections
    metric_selection = alt.selection_multi(
        fields=['metric'],
        bind='legend',
        name='metric_select'
    )
    
    author_selection = alt.selection_multi(
        fields=['author'],
        bind='legend',
        name='author_select'
    )
    
    layer_type_selection = alt.selection_multi(
        fields=['layer_type'],
        bind='legend',
        name='layer_type_select'
    )
    
    layer_ind_selection = alt.selection_multi(
        fields=['layer_ind'],
        bind='legend',
        name='layer_ind_select'
    )
    
    # Create chart
    chart = alt.Chart(df_melted).mark_line(
        point=True,
        size=2
    ).encode(
        x=alt.X('n_features:Q',
                scale=alt.Scale(type='log'),
                axis=alt.Axis(title='Number of Features (log scale)')),
        y=alt.Y('value:Q',
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(title='Metric Value')),
        color=alt.Color('metric:N',
                       legend=alt.Legend(title='Metric (click to toggle)')),
        strokeDash=alt.StrokeDash('author:N',
                                  legend=alt.Legend(title='Author (click to toggle)')),
        shape=alt.Shape('layer_type:N',
                       legend=alt.Legend(title='Layer Type (click to toggle)')),
        detail='author_layer_type:N',
        opacity=alt.condition(
            metric_selection & author_selection & layer_type_selection & layer_ind_selection,
            alt.value(0.8),
            alt.value(0.05)
        ),
        tooltip=[
            alt.Tooltip('author:N', title='Author'),
            alt.Tooltip('layer_type:N', title='Layer Type'),
            alt.Tooltip('layer_ind:Q', title='Layer Index'),
            alt.Tooltip('metric:N', title='Metric'),
            alt.Tooltip('n_features:Q', title='Features'),
            alt.Tooltip('value:Q', title='Value', format='.4f')
        ]
    ).properties(
        width=900,
        height=500,
        title='All Metrics Comparison (Click legends to toggle)'
    ).add_selection(
        metric_selection,
        author_selection,
        layer_type_selection,
        layer_ind_selection
    )
    
    # Add zoom
    zoom = alt.selection_interval(bind='scales', encodings=['x', 'y'])
    chart = chart.add_selection(zoom)
    
    return chart


def create_feature_reduction_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Create a chart showing feature reduction progress over iterations.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Altair Chart object
    """
    logger.info("Creating feature reduction chart")
    
    # Create selections
    author_selection = alt.selection_multi(
        fields=['author'],
        bind='legend',
        name='author_select'
    )
    
    layer_type_selection = alt.selection_multi(
        fields=['layer_type'],
        bind='legend',
        name='layer_type_select'
    )
    
    layer_ind_selection = alt.selection_multi(
        fields=['layer_ind'],
        bind='legend',
        name='layer_ind_select'
    )
    
    chart = alt.Chart(df).mark_line(
        point=True
    ).encode(
        x=alt.X('iteration:Q',
                axis=alt.Axis(title='Iteration')),
        y=alt.Y('n_features:Q',
                scale=alt.Scale(type='log'),
                axis=alt.Axis(title='Number of Features (log scale)')),
        color=alt.Color('author:N',
                       legend=alt.Legend(title='Author (click to toggle)')),
        strokeDash=alt.StrokeDash('layer_type:N',
                                  legend=alt.Legend(title='Layer Type (click to toggle)')),
        shape=alt.Shape('layer_ind:N',
                       legend=alt.Legend(title='Layer Index (click to toggle)', symbolLimit=50)),
        detail='author_layer_type:N',
        opacity=alt.condition(
            author_selection & layer_type_selection & layer_ind_selection,
            alt.value(0.8),
            alt.value(0.05)
        ),
        tooltip=[
            alt.Tooltip('author:N', title='Author'),
            alt.Tooltip('layer_type:N', title='Layer Type'),
            alt.Tooltip('layer_ind:Q', title='Layer Index'),
            alt.Tooltip('iteration:Q', title='Iteration'),
            alt.Tooltip('n_features:Q', title='Features')
        ]
    ).properties(
        width=900,
        height=400,
        title='Feature Reduction Progress (Click legends to toggle)'
    ).add_selection(
        author_selection,
        layer_type_selection,
        layer_ind_selection
    )
    
    # Add zoom
    zoom = alt.selection_interval(bind='scales', encodings=['x', 'y'])
    chart = chart.add_selection(zoom)
    
    return chart


def main():
    """Main execution function."""
    args = parse_arguments()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    logger.info(f"Starting visualization generation")
    logger.info(f"Data directory: {data_dir}")
    
    # Load results
    results = load_json_results(data_dir)
    
    if not results:
        logger.error("No results loaded. Exiting.")
        return
    
    # Initialize color manager
    color_manager = AuthorColorManager()
    
    # Pre-populate color manager with all authors
    all_authors = sorted(set([r['author'] for r in results]))
    for author in all_authors:
        color_manager.get_author_color(author)
    
    # Create DataFrame
    df = create_dataframe(results, color_manager)
    
    if df.empty:
        logger.error("DataFrame is empty. Exiting.")
        return
    
    # Create summary statistics
    logger.info("\nSummary Statistics:")
    summary = create_summary_stats(df)
    summary_file = data_dir / f"summary_statistics{args.output_suffix}.csv"
    summary.to_csv(summary_file)
    logger.info(f"Saved summary statistics to {summary_file}")
    
    # Create individual metric charts
    metrics = [
        ('precision', 'Precision (Class 1)'),
        ('recall', 'Recall (Class 1)'),
        ('f1_score', 'F1-Score (Class 1)')
    ]
    
    for metric, label in metrics:
        logger.info(f"\nCreating chart for {label}")
        chart = create_interactive_chart(df, metric, label, color_manager)
        
        # Save chart
        output_file = data_dir / f"{metric}_visualization{args.output_suffix}.html"
        chart.save(str(output_file))
        logger.info(f"Saved {label} chart to {output_file}")
    
    # Create comparison chart
    logger.info("\nCreating comparison chart")
    comparison_chart = create_comparison_chart(df)
    comparison_file = data_dir / f"metrics_comparison{args.output_suffix}.html"
    comparison_chart.save(str(comparison_file))
    logger.info(f"Saved comparison chart to {comparison_file}")
    
    # Create feature reduction chart
    logger.info("\nCreating feature reduction chart")
    reduction_chart = create_feature_reduction_chart(df)
    reduction_file = data_dir / f"feature_reduction{args.output_suffix}.html"
    reduction_chart.save(str(reduction_file))
    logger.info(f"Saved feature reduction chart to {reduction_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Visualization generation complete!")
    logger.info("="*70)
    logger.info(f"\nGenerated files in {data_dir}:")
    logger.info(f"  - precision_visualization{args.output_suffix}.html")
    logger.info(f"  - recall_visualization{args.output_suffix}.html")
    logger.info(f"  - f1_score_visualization{args.output_suffix}.html")
    logger.info(f"  - metrics_comparison{args.output_suffix}.html")
    logger.info(f"  - feature_reduction{args.output_suffix}.html")
    logger.info(f"  - summary_statistics{args.output_suffix}.csv")
    logger.info("\nOpen the HTML files in a web browser to view interactive visualizations.")


if __name__ == "__main__":
    main()

