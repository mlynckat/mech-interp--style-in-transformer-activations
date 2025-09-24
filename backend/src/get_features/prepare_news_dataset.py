#!/usr/bin/env python3
"""
Prepare news dataset to be used as AuthorMix dataset for SAE features extraction.
This script processes temporary category_{category}.csv files, extracts authors and 
combines headline + article_text to create datasets structured like AuthorMix.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import warnings
import logging

# Set up logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

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
                'style': author,  # Combine category and author for unique style
                'text':  row['headline'] + '. ' + row['text'],
                'category': category_name,
                'text_length_words': row['text_length_words']
            }
            authormix_data.append(doc_data)
    
    return authormix_data

def create_visualizations(authormix_data, category_name, save_dir):
    """Create and save visualizations for the category dataset."""
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(authormix_data)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Number of documents per author (bar plot)
    author_counts = df['style'].value_counts()
    
    # Limit to top 20 authors for readability
    top_authors = author_counts.head(20)
    # logger.debug(top_authors)
    
    ax1.bar(top_authors.index, top_authors.values, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Authors (ranked by document count)', fontsize=12)
    ax1.set_ylabel('Number of Documents', fontsize=12)
    ax1.set_title(f'Top 20 Authors by Document Count - {category_name}', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(top_authors)))
    ax1.set_xticklabels(top_authors.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add text with stats
    total_authors = len(author_counts)
    total_docs = len(df)
    avg_docs_per_author = total_docs / total_authors
    ax1.text(0.02, 0.98, f'Total Authors: {total_authors}\nTotal Documents: {total_docs}\nAvg Docs/Author: {avg_docs_per_author:.1f}', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Document length distribution (histogram)
    text_lengths = df['text_length_words']
    
    ax2.hist(text_lengths, bins=50, color='lightcoral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Document Length (words)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Document Length Distribution - {category_name}', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_length = text_lengths.mean()
    median_length = text_lengths.median()
    std_length = text_lengths.std()
    ax2.text(0.98, 0.98, f'Mean: {mean_length:.0f} words\nMedian: {median_length:.0f} words\nStd: {std_length:.0f} words', 
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{category_name}_stats.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved visualization to {plot_path}")
    
    # Return summary statistics
    stats = {
        'category': category_name,
        'total_authors': total_authors,
        'total_documents': total_docs,
        'avg_docs_per_author': avg_docs_per_author,
        'mean_text_length': float(mean_length),
        'median_text_length': float(median_length),
        'std_text_length': float(std_length),
        'min_text_length': int(text_lengths.min()),
        'max_text_length': int(text_lengths.max())
    }
    
    return stats

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
    
    # Save combined dataset
    """combined_output_file = os.path.join(output_dir, "all_categories_combined_authormix.json")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_categories_combined, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved combined dataset with {len(all_categories_combined)} documents to {combined_output_file}")
    """
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
    
    # Create overall summary plot
    create_overall_summary_plot(all_stats, stats_dir)

def create_overall_summary_plot(all_stats, stats_dir):
    """Create an overall summary plot across all categories."""
    
    if not all_stats:
        return
        
    # Create DataFrame from stats
    stats_df = pd.DataFrame(all_stats)
    
    # Create figure with 3 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Total documents per category
    ax1.bar(stats_df['category'], stats_df['total_documents'], color='lightblue', alpha=0.7)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Total Documents', fontsize=12)
    ax1.set_title('Total Documents per Category', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Total authors per category
    ax2.bar(stats_df['category'], stats_df['total_authors'], color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Total Authors', fontsize=12)
    ax2.set_title('Total Authors per Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Average docs per author by category
    ax3.bar(stats_df['category'], stats_df['avg_docs_per_author'], color='lightcoral', alpha=0.7)
    ax3.set_xlabel('Category', fontsize=12)
    ax3.set_ylabel('Avg Documents per Author', fontsize=12)
    ax3.set_title('Average Documents per Author by Category', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Mean text length by category
    ax4.bar(stats_df['category'], stats_df['mean_text_length'], color='gold', alpha=0.7)
    ax4.set_xlabel('Category', fontsize=12)
    ax4.set_ylabel('Mean Text Length (words)', fontsize=12)
    ax4.set_title('Mean Text Length by Category', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the overall summary plot
    summary_plot_path = os.path.join(stats_dir, "overall_summary_stats.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved overall summary plot to {summary_plot_path}")

if __name__ == "__main__":
    main()
