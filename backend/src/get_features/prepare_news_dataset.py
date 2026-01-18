#!/usr/bin/env python3
"""
Prepare news dataset to be used as AuthorMix dataset for SAE features extraction.
This script processes temporary category_{category}.csv files, extracts authors and 
combines headline + article_text to create datasets structured like AuthorMix.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.plot_styling import PlotStyle, apply_style, create_figure

# Set up logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Apply global plot styling
apply_style()


def load_category_data(category_file):
    """Load and process a single category CSV file."""
    logger.info(f"Processing {category_file}...")
    
    try:
        df = pd.read_csv(category_file)
        logger.info(f"  Loaded {len(df)} articles from {category_file}")
        
        # Clean and process the data
        df = df.dropna(subset=['headline', 'article_text'])
        logger.info(f"  After removing missing headline/article_text: {len(df)} articles")
        
        # Handle missing authors - use "Unknown" for empty author fields
        df['authors'] = df['authors'].fillna('Unknown')
        df['authors'] = df['authors'].replace('', 'Unknown')

        df = df[df['authors'] != 'Unknown']
        
        # Combine headline and article text
        df['text'] = df['headline'].astype(str) + '. ' + df['article_text'].astype(str)
        
        # Calculate text length in words
        df['text_length_words'] = df['text'].apply(lambda x: len(str(x).split()))
        
        # Filter out very short articles (less than 35 words)
        min_words = 35
        df = df[df['text_length_words'] >= min_words]
        logger.info(f"  After filtering articles < {min_words} words: {len(df)} articles")
        
        return df
        
    except Exception as e:
        logger.error(f"  Error processing {category_file}: {e}")
        return None


def create_authormix_structure(df, category_name):
    """Convert news dataset to AuthorMix-like structure."""
    authormix_data = []
    
    # Group by author
    author_groups = df.groupby('authors')
    
    logger.info(f"  Found {len(author_groups)} unique authors in {category_name}")
    
    for author, group in author_groups:
        # Skip if author has too few documents
        if len(group) < 10:
            continue
            
        for idx, row in group.iterrows():
            doc_data = {
                'style': author,
                'text':  row['headline'] + '. ' + row['text'],
                'category': category_name,
                'text_length_words': row['text_length_words']
            }
            authormix_data.append(doc_data)
    
    return authormix_data


def create_visualizations(authormix_data, category_name, save_dir):
    """Create and save modern, publication-ready visualizations for a single category."""
    
    df = pd.DataFrame(authormix_data)
    os.makedirs(save_dir, exist_ok=True)
    
    style = PlotStyle()
    
    # Create figure with white background
    fig, (ax1, ax2) = create_figure(1, 2, figsize=(12, 5))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.15, wspace=0.35)

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1: Documents per author (Horizontal bar chart with gradient)
    # ─────────────────────────────────────────────────────────────────────────
    author_counts = df['style'].value_counts().head(15)
    author_counts = author_counts.sort_values()
    
    n_bars = len(author_counts)
    colors = style.get_gradient_colors(n_bars)

    bars1 = ax1.barh(
        range(len(author_counts)),
        author_counts.values,
        color=colors,
        height=0.7,
        edgecolor='none'
    )
    
    # Style y-axis labels
    ax1.set_yticks(range(len(author_counts)))
    ax1.set_yticklabels(author_counts.index, fontsize=8)
    
    # Add value labels on bars
    style.add_bar_labels(ax1, bars1, author_counts.values, position='end')
    
    # Apply axis styling
    style.style_axis(
        ax1,
        title=f'Top Authors — {category_name.replace("_", " ").title()}',
        xlabel='Number of documents',
        grid_axis='x',
        title_loc='center'
    )
    
    ax1.set_xlim(0, max(author_counts.values) * 1.2)

    # Metadata badge
    style.add_metadata_badge(
        ax1,
        f'{df["style"].nunique()} authors · {len(df):,} docs',
        loc='lower right'
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2: Document length distribution (Histogram with gradient)
    # ─────────────────────────────────────────────────────────────────────────
    lengths = df['text_length_words']
    
    # Clip at 95th percentile for better visualization
    upper_bound = min(lengths.quantile(0.95), 2000)
    clipped_lengths = lengths.clip(upper=upper_bound)
    
    # Create histogram
    bins = np.linspace(0, upper_bound, 35)
    n, bins_out, patches = ax2.hist(
        clipped_lengths,
        bins=bins,
        color=style.COLORS['primary'],
        alpha=0.85,
        edgecolor=style.COLORS['bg_white'],
        linewidth=0.5
    )
    
    # Apply gradient to histogram bars
    hist_colors = style.get_gradient_colors(len(patches))
    for patch, color in zip(patches, hist_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Apply axis styling
    style.style_axis(
        ax2,
        title='Document Length Distribution (Words)',
        xlabel='Document length (words)',
        ylabel='Frequency',
        grid_axis='y',
        title_loc='center'
    )

    # Reference lines with accent color
    mean_val = lengths.mean()
    median_val = lengths.median()

    ax2.axvline(mean_val, linestyle='--', linewidth=2, 
                color=style.COLORS['accent'], alpha=0.9, 
                label=f'Mean: {mean_val:.0f}')
    ax2.axvline(median_val, linestyle=':', linewidth=2, 
                color=style.COLORS['accent_light'], alpha=0.9, 
                label=f'Median: {median_val:.0f}')
    
    # Legend
    legend = ax2.legend(
        loc='upper right',
        frameon=True,
        facecolor=style.COLORS['bg_white'],
        edgecolor=style.COLORS['border'],
        fontsize=9
    )
    legend.get_frame().set_alpha(0.95)

    # Save
    plot_path = os.path.join(save_dir, f"{category_name}_stats.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor=style.COLORS['bg_white'])
    plt.close()

    logger.info(f"  Saved visualization to {plot_path}")

    return {
        "category": category_name,
        "total_authors": df['style'].nunique(),
        "total_documents": len(df),
        "avg_docs_per_author": len(df) / df['style'].nunique(),
        "mean_text_length": float(mean_val),
        "median_text_length": float(median_val),
        "std_text_length": float(lengths.std()),
        "min_text_length": int(lengths.min()),
        "max_text_length": int(lengths.max()),
    }


def create_overall_summary_plot(all_stats, stats_dir, top_n: int = 20):
    """Create a modern summary dashboard across all categories."""

    if not all_stats:
        return

    style = PlotStyle()
    
    # Create DataFrame and keep only top N by total documents
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.nlargest(top_n, 'total_documents')
    stats_df = stats_df.sort_values('total_documents', ascending=False)
    
    os.makedirs(stats_dir, exist_ok=True)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(style.COLORS['bg_white'])
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.25,
                          left=0.08, right=0.95, top=0.88, bottom=0.12)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Prepare labels
    categories = [style.truncate_label(cat, max_len=14) 
                  for cat in stats_df['category']]
    x_pos = np.arange(len(categories))
    
    # Gradient colors for bars
    bar_colors = style.get_gradient_colors(len(categories))

    def plot_bars(ax, values, title, ylabel, show_error=False, error_vals=None):
        """Helper function for consistent bar styling."""
        bars = ax.bar(x_pos, values, color=bar_colors, width=0.75, edgecolor='none')
        
        if show_error and error_vals is not None:
            ax.errorbar(
                x_pos, values, yerr=error_vals * 0.5,
                fmt='none', ecolor=style.COLORS['text_light'],
                capsize=3, capthick=1, alpha=0.6
            )
        
        style.style_axis(ax, title=title, ylabel=ylabel, grid_axis='y', title_loc='center')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        
        # Add value labels
        style.add_bar_labels(ax, bars, values, fontsize=7)
        
        # Adjust y-limit for labels
        ax.set_ylim(0, max(values) * 1.18)
        
        return bars

    # Plot 1: Total documents
    plot_bars(ax1, stats_df['total_documents'].values, 
              'Total Documents', 'Count')

    # Plot 2: Total authors
    plot_bars(ax2, stats_df['total_authors'].values,
              'Total Authors', 'Count')

    # Plot 3: Average documents per author
    plot_bars(ax3, stats_df['avg_docs_per_author'].values,
              'Avg Documents per Author', 'Documents')

    # Plot 4: Mean document length (with std dev error bars)
    plot_bars(ax4, stats_df['mean_text_length'].values,
              'Mean Document Length', 'Words',
              show_error=True, error_vals=stats_df['std_text_length'].values)

    # Save
    output_path = os.path.join(stats_dir, "overall_summary_stats.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor=style.COLORS['bg_white'])
    plt.close()

    logger.info(f"Saved overall summary plot to {output_path}")


def main():
    """Main function to process all category files."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up paths
    intermediate_dir = "/cluster/home/mlynckat/mechanistic_interpretability_master/data/intermediate"
    output_dir = "/cluster/home/mlynckat/mechanistic_interpretability_master/data/news_authormix_datasets"
    stats_dir = "/cluster/home/mlynckat/mechanistic_interpretability_master/data/news_dataset_stats"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Find all category CSV files
    category_files = [f for f in os.listdir(intermediate_dir) if f.startswith('category_') and f.endswith('.csv')]
    
    logger.info(f"Found {len(category_files)} category files to process")
    
    all_stats = []
    all_categories_combined = []
    
    # Process each category
    for category_file in sorted(category_files):
        category_name = category_file.replace('category_', '').replace('.csv', '')
        category_path = os.path.join(intermediate_dir, category_file)
        
        logger.info(f"\n=== Processing Category: {category_name} ===")
        
        # Load and process the category data
        df = load_category_data(category_path)
        if df is None:
            continue
            
        # Convert to AuthorMix structure
        authormix_data = create_authormix_structure(df, category_name)
        
        if len(authormix_data) == 0:
            logger.warning(f"  No valid data for category {category_name}, skipping...")
            continue
            
        logger.info(f"  Created {len(authormix_data)} documents for category {category_name}")
        
        # Save category dataset
        output_file = os.path.join(output_dir, f"{category_name}_authormix.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(authormix_data, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved dataset to {output_file}")
        
        # Create visualizations and get stats
        stats = create_visualizations(authormix_data, category_name, stats_dir)
        all_stats.append(stats)
        
        # Add to combined dataset
        all_categories_combined.extend(authormix_data)
        
        logger.info(f"  Category {category_name} completed successfully!")
    
    # Save summary statistics
    stats_summary_file = os.path.join(stats_dir, "dataset_summary_stats.json")
    logger.info(all_stats)
    with open(stats_summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary statistics to {stats_summary_file}")
    
    # Print overall summary
    logger.info(f"\n=== PROCESSING COMPLETE ===")
    logger.info(f"Processed {len(category_files)} categories")
    logger.info(f"Total documents across all categories: {len(all_categories_combined)}")
    logger.info(f"Individual category datasets saved to: {output_dir}")
    logger.info(f"Visualizations and stats saved to: {stats_dir}")
    
    # Create overall summary plot (top 20 categories)
    create_overall_summary_plot(all_stats, stats_dir, top_n=20)


if __name__ == "__main__":
    main()
