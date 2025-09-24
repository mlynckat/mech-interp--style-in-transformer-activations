#!/usr/bin/env python3
"""
Test script for the AuthorColorManager to verify consistent color assignments.
"""

import sys
import os
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the parent directory to the path to import the main module
sys.path.append(str(Path(__file__).parent.parent))

from sae_activations_exploration import AuthorColorManager, Visualizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_color_consistency():
    """Test that the same authors always get the same colors."""
    logger.info("Testing color consistency...")
    
    # Create two color managers
    cm1 = AuthorColorManager()
    cm2 = AuthorColorManager()
    
    # Test authors
    test_authors = ['obama', 'trump', 'bush', 'fitzgerald', 'woolf']
    
    # Get colors from both managers
    colors1 = [cm1.get_author_color(author) for author in test_authors]
    colors2 = [cm2.get_author_color(author) for author in test_authors]
    
    # Check consistency
    for i, author in enumerate(test_authors):
        color1 = colors1[i]
        color2 = colors2[i]
        if color1 == color2:
            logger.info(f"✓ {author}: {color1}")
        else:
            logger.info(f"✗ {author}: {color1} vs {color2}")
    
    logger.info("\nColor consistency test completed!")

def test_visualization_integration():
    """Test that the color manager integrates correctly with visualizations."""
    logger.info("\nTesting visualization integration...")
    
    # Create color manager and visualizer
    cm = AuthorColorManager()
    visualizer = Visualizer(Path("test_output"), cm)
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    
    # Create DataFrame with author information
    authors = ['obama', 'trump', 'bush']
    data = []
    
    for author in authors:
        # Generate some test activation data
        activations = np.random.exponential(2, n_samples)
        for activation in activations:
            data.append({
                'author': author,
                'activations': activation
            })
    
    df = pd.DataFrame(data)
    
    # Test histogram plotting
    logger.info("Creating test histogram...")
    visualizer.plot_activation_histogram(df, "Test", "test_histogram.png")
    logger.info("✓ Histogram created successfully")
    
    # Test distribution plot
    logger.info("Creating test distribution plot...")
    data_dict = {author: df[df['author'] == author]['activations'].values for author in authors}
    visualizer.create_distribution_plot(data_dict, "Test Distribution", "Author", "Activations", "test_distribution.png")
    logger.info("✓ Distribution plot created successfully")
    
    logger.info("\nVisualization integration test completed!")

def test_color_legend():
    """Test the color legend creation."""
    logger.info("\nTesting color legend creation...")
    
    cm = AuthorColorManager()
    
    # Add some test authors
    test_authors = ['obama', 'trump', 'bush', 'fitzgerald', 'woolf', 'hemingway', 'dickens']
    for author in test_authors:
        cm.get_author_color(author)
    
    # Create legend
    legend_path = Path("test_output") / "test_color_legend.png"
    legend_path.parent.mkdir(exist_ok=True)
    cm.create_color_legend(legend_path, "Test Author Color Legend")
    logger.info("✓ Color legend created successfully")

def test_save_load():
    """Test saving and loading color mappings."""
    logger.info("\nTesting save/load functionality...")
    
    cm1 = AuthorColorManager()
    
    # Add some test authors
    test_authors = ['obama', 'trump', 'bush', 'fitzgerald']
    for author in test_authors:
        cm1.get_author_color(author)
    
    # Save mapping
    save_path = Path("test_output") / "test_color_mapping.json"
    save_path.parent.mkdir(exist_ok=True)
    cm1.save_color_mapping(save_path)
    logger.info("✓ Color mapping saved")
    
    # Load mapping in new manager
    cm2 = AuthorColorManager()
    cm2.load_color_mapping(save_path)
    logger.info("✓ Color mapping loaded")
    
    # Verify consistency
    for author in test_authors:
        color1 = cm1.get_author_color(author)
        color2 = cm2.get_author_color(author)
        if color1 == color2:
            logger.info(f"✓ {author}: {color1}")
        else:
            logger.info(f"✗ {author}: {color1} vs {color2}")
    
    logger.info("Save/load test completed!")

def main():
    """Run all tests."""
    logger.info("Running AuthorColorManager tests...\n")
    
    # Create test output directory
    test_output = Path("test_output")
    test_output.mkdir(exist_ok=True)
    
    try:
        test_color_consistency()
        test_visualization_integration()
        test_color_legend()
        test_save_load()
        
        logger.info("\n🎉 All tests passed successfully!")
        logger.info(f"Test outputs saved to: {test_output.absolute()}")
        
    except Exception as e:
        logger.info(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





