import streamlit as st
import os
import glob
from pathlib import Path
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from PIL import Image
import io
import socket
import requests
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any
import json

# Import shared utilities
from brain.my_scripts.shared_utilities import AuthorColorManager

def get_output_directories():
    """Get all directories under ./sae_features/outputs"""
    outputs_path = Path("./sae_features/outputs")
    if not outputs_path.exists():
        return []
    
    directories = [d.name for d in outputs_path.iterdir() if d.is_dir()]
    return sorted(directories)

def get_files_by_type(directory_path: str, vis_type: str):
    """Get files based on visualization type"""
    if not os.path.exists(directory_path):
        return []
    
    files = []
    for file_path in Path(directory_path).glob("*"):
        if file_path.is_file():
            if vis_type in ['activation_frequencies', 'token_position']:
                if file_path.suffix.lower() == '.npy' and 'non_zeroes_count' in file_path.name:
                    files.append(file_path.name)
            elif vis_type == 'entropies':
                if file_path.suffix.lower() == '.npy' and ('non_zeroes_count' in file_path.name or 'entropy' in file_path.name):
                    files.append(file_path.name)
            elif vis_type == 'detailed_scatter':
                if file_path.suffix.lower() == '.csv':
                    files.append(file_path.name)
            elif vis_type == 'detailed_tfidf':
                if file_path.suffix.lower() == '.csv' and 'feature_importance_data' in file_path.name:
                    files.append(file_path.name)
    
    return sorted(files)

def parse_filenames(filenames):
    """Parse the filename to get the layer_type, layer_ind, and author"""
    files_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

    for filename in filenames:
        filename_stem = Path(filename).stem
        
        # Handle different filename patterns
        if 'non_zeroes_count' in filename:
            # Legacy pattern: non_zeroes_count_google_gemma-2-2b_att_activations_author_layer_ind.npy
            parts = filename_stem.split('_')
            layer_type = parts[-5]  # att, mlp, res
            author = parts[-3]   # author name
            layer_ind = parts[-1]      # layer number
            files_structure[layer_type][layer_ind][author] = filename
        elif 'activations' in filename:
            # New pattern: sae_baseline__google_gemma-2-9b__res__activations__bush__layer_0
            # Handle both .npz and .sparse.npz extensions
            if filename.endswith('.sparse.npz'):
                filename_stem = filename_stem[:-7]  # Remove .sparse from stem
            parts = filename_stem.split('__')
            if len(parts) >= 6:
                layer_type = parts[2]  # res, mlp, att
                author = parts[4]      # author name
                layer_part = parts[5]  # layer_0
                layer_ind = layer_part.split('_')[1]  # Extract number from layer_X
                files_structure[layer_type][layer_ind][author] = filename
        elif 'entropy' in filename and 'cross_entropy' not in filename:
            # New pattern: sae_baseline__google_gemma-2-9b__entropy__bush.npy
            parts = filename_stem.split('__')
            if len(parts) >= 4:
                author = parts[-1]  # author name
                files_structure['entropy'][author][filename] = filename
        elif 'cross_entropy' in filename:
            # New pattern: sae_baseline__google_gemma-2-9b__cross_entropy_loss__bush.npy
            parts = filename_stem.split('__')
            if len(parts) >= 4:
                author = parts[-1]  # author name
                files_structure['cross_entropy'][author][filename] = filename
        elif filename.endswith('.csv'):
            # CSV files for detailed scatter
            # Pattern: frequencies_activated_per_feature_att_0.csv
            parts = filename_stem.split('_')
            if len(parts) >= 4:
                layer_type = parts[-2]  # att, mlp, res
                layer_ind = parts[-1]   # layer number
                files_structure[layer_type][layer_ind]['csv'] = filename
    
    return files_structure

def load_data_for_cell(output_dir: str, selected_filenames: List[str], layer_type: str, layer_ind: str, selected_authors: List[str], vis_type: str):
    """Load data for a specific cell based on visualization type"""
    data = {}
    
    if vis_type in ['activation_frequencies', 'token_position']:
        # Load non_zeroes_count files for selected authors
        for author, filename in zip(selected_authors, selected_filenames):
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                data[author] = np.load(file_path)
                st.write(f"Loaded {filename} with shape: {data[author].shape}")
            else:
                st.warning(f"File not found: {filename}")
    
    elif vis_type == 'entropies':
        # Load both non_zeroes_count and entropy files
        for author, filename in zip(selected_authors, selected_filenames):
            # Load activations
            act_filename = filename
            act_file_path = os.path.join(output_dir, act_filename)
            if os.path.exists(act_file_path):
                data[f"{author}_activations"] = np.load(act_file_path)
            
            # Load entropy (try different possible patterns)
            entropy_filename = f"sae_google_gemma-2-2b_entropy_loss_{author}.npy"
            entropy_file_path = os.path.join(output_dir, entropy_filename)
            if os.path.exists(entropy_file_path):
                data[f"{author}_entropy"] = np.load(entropy_file_path)
            else:
                st.warning(f"No entropy file found for author {author} in {output_dir}")
            
            # Load cross entropy (try different possible patterns)
            cross_entropy_filename = f"sae_google_gemma-2-2b_cross_entropy_loss_{author}.npy"
            cross_entropy_file_path = os.path.join(output_dir, cross_entropy_filename)
            if os.path.exists(cross_entropy_file_path):
                data[f"{author}_cross_entropy"] = np.load(cross_entropy_file_path)
            else:
                st.warning(f"No cross-entropy file found for author {author} in {output_dir}")
    
    elif vis_type == 'detailed_scatter':
        # Load CSV file
        filename = f"frequencies_activated_per_feature_{layer_type}_{layer_ind}.csv"
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            data['csv'] = pd.read_csv(file_path)
            st.write(f"Loaded {filename} with shape: {data['csv'].shape}")
        else:
            st.warning(f"CSV file not found: {filename}")

    elif vis_type == 'detailed_tfidf':
        # Load CSV file
        filename = f"feature_importance_data__{layer_type}_{layer_ind}.csv"
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            data['tfidf_csv'] = pd.read_csv(file_path)

            # Filter for selected authors
            if selected_authors:
                data['tfidf_csv'] = data['tfidf_csv'][data['tfidf_csv']['author'].isin(selected_authors)]


            st.write(f"Loaded {filename} with shape: {data['tfidf_csv'].shape}")
    
    return data

def create_activation_histogram_altair(data: Dict[str, np.ndarray], title_suffix: str, color_manager: AuthorColorManager):
    """Create activation histogram using Altair"""
    # Prepare data for visualization
    chart_data = []
    
    for author, activations in data.items():
        # Flatten and filter non-zero activations
        flat_activations = activations.flatten()
        non_zero_activations = flat_activations[flat_activations > 0]
        
        # Create histogram data
        hist, bin_edges = np.histogram(non_zero_activations, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for bin_center, density in zip(bin_centers, hist):
            chart_data.append({
                'author': author,
                'activations': bin_center,
                'density': density
            })
    
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    # Get colors for authors
    authors = df['author'].unique().tolist()
    domain, range_colors = color_manager.get_altair_domain_range(authors)
    
    # Create overlay histogram chart with area marks
    histogram_chart = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=alt.X('activations:Q', title='Active SAE Features'),
        y=alt.Y('density:Q', title='Density'),
        color=alt.Color('author:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
        tooltip=['author', 'activations', 'density']
    ).properties(
        title=f'Distribution of Active SAE Features per Token - {title_suffix}',
        width=400,
        height=300
    )
    
    # Create density line chart
    density_line_chart = alt.Chart(df).mark_line(opacity=0.8, strokeWidth=2).encode(
        x=alt.X('activations:Q', title='Active SAE Features'),
        y=alt.Y('density:Q', title='Density'),
        color=alt.Color('author:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
        tooltip=['author', 'activations', 'density']
    )
    
    # Layer the charts together
    chart = alt.layer(histogram_chart, density_line_chart).interactive()
    
    return chart

def create_entropy_scatter_altair(data: Dict[str, np.ndarray], title_suffix: str, color_manager: AuthorColorManager):
    """Create entropy vs activations scatter plot using Altair"""
    chart_data = []
    
    # Extract unique authors from data keys
    authors = set()
    for key in data.keys():
        if key.endswith('_activations'):
            author = key.replace('_activations', '')
            authors.add(author)
    
    for author in authors:
        if f"{author}_activations" in data and f"{author}_entropy" in data:
            activations = data[f"{author}_activations"]
            entropies = data[f"{author}_entropy"]
            
            # Flatten arrays
            activations_flat = activations.flatten()
            entropies_flat = entropies.flatten()
            
            # Filter non-zero activations
            mask = activations_flat > 0
            activations_filtered = activations_flat[mask]
            entropies_filtered = entropies_flat[mask]
            
            # Sample data if too large for performance
            if len(activations_filtered) > 10000:
                indices = np.random.choice(len(activations_filtered), 10000, replace=False)
                activations_filtered = activations_filtered[indices]
                entropies_filtered = entropies_filtered[indices]
            
            for act, ent in zip(activations_filtered, entropies_filtered):
                chart_data.append({
                    'author': author,
                    'activations': act,
                    'entropy': ent
                })
    
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    # Get colors for authors
    authors = df['author'].unique().tolist()
    domain, range_colors = color_manager.get_altair_domain_range(authors)
    
    # Create scatter plot for entropy
    chart = alt.Chart(df).mark_circle(size=20, opacity=0.3).encode(
        x=alt.X('entropy:Q', title='Entropy'),
        y=alt.Y('activations:Q', title='Activations'),
        color=alt.Color('author:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
        tooltip=['author', 'entropy', 'activations']
    ).properties(
        title=f'Entropy vs Activations - {title_suffix}',
        width=600,
        height=500
    ).interactive()
    
    return chart

def create_cross_entropy_scatter_altair(data: Dict[str, np.ndarray], title_suffix: str, color_manager: AuthorColorManager):
    """Create cross-entropy vs activations scatter plot using Altair"""
    chart_data = []
    
    # Extract unique authors from data keys
    authors = set()
    for key in data.keys():
        if key.endswith('_activations'):
            author = key.replace('_activations', '')
            authors.add(author)
    
    for author in authors:
        if f"{author}_activations" in data and f"{author}_cross_entropy" in data:
            activations = data[f"{author}_activations"]
            cross_entropies = data[f"{author}_cross_entropy"]
            
            # Flatten arrays
            activations_flat = activations.flatten()
            cross_entropies_flat = cross_entropies.flatten()
            
            # Filter non-zero activations
            mask = activations_flat > 0
            activations_filtered = activations_flat[mask]
            cross_entropies_filtered = cross_entropies_flat[mask]
            
            # Sample data if too large for performance
            if len(activations_filtered) > 10000:
                indices = np.random.choice(len(activations_filtered), 10000, replace=False)
                activations_filtered = activations_filtered[indices]
                cross_entropies_filtered = cross_entropies_filtered[indices]
            
            for act, cent in zip(activations_filtered, cross_entropies_filtered):
                chart_data.append({
                    'author': author,
                    'activations': act,
                    'cross_entropy': cent
                })
    
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    # Get colors for authors
    authors = df['author'].unique().tolist()
    domain, range_colors = color_manager.get_altair_domain_range(authors)
    
    # Create scatter plot for cross-entropy
    chart = alt.Chart(df).mark_circle(size=20, opacity=0.3).encode(
        x=alt.X('cross_entropy:Q', title='Cross-Entropy'),
        y=alt.Y('activations:Q', title='Activations'),
        color=alt.Color('author:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
        tooltip=['author', 'cross_entropy', 'activations']
    ).properties(
        title=f'Cross-Entropy vs Activations - {title_suffix}',
        width=600,
        height=500
    ).interactive()
    
    return chart

def create_token_position_scatter_altair(data: Dict[str, np.ndarray], title_suffix: str, color_manager: AuthorColorManager):
    """Create token position scatter plot using Altair"""
    chart_data = []
    
    for author, activations in data.items():
        # activations shape: (num_docs, num_tokens)
        for doc_idx in range(activations.shape[0]):
            for token_idx in range(activations.shape[1]):
                num_activations = activations[doc_idx, token_idx]
                if num_activations > 0:
                    chart_data.append({
                        'author': author,
                        'token_index': token_idx,
                        'num_activations': num_activations
                    })
    
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    # Sample data if too large for performance
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
    
    # Get colors for authors
    authors = df['author'].unique().tolist()
    domain, range_colors = color_manager.get_altair_domain_range(authors)
    
    # Create scatter plot
    chart = alt.Chart(df).mark_circle(size=20, opacity=0.3).encode(
        x=alt.X('token_index:Q', title='Token Index'),
        y=alt.Y('num_activations:Q', title='Number of Activations'),
        color=alt.Color('author:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
        tooltip=['author', 'token_index', 'num_activations']
    ).properties(
        title=f'Token Positions - {title_suffix}',
        width=400,
        height=300
    ).interactive()
    
    return chart

def create_detailed_scatter_altair(data: Dict[str, pd.DataFrame], title_suffix: str, color_manager: AuthorColorManager):
    """Create detailed scatter plot using Altair"""
    if 'csv' not in data:
        return None
    
    df = data['csv']
    
    # Check if we have the required columns
    required_cols = ['feature_ind', 'value', 'variable']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Ensure feature_ind is numeric
    df['feature_ind'] = pd.to_numeric(df['feature_ind'], errors='coerce')
    
    # Get colors for authors
    authors = df['variable'].unique().tolist()
    domain, range_colors = color_manager.get_altair_domain_range(authors)
    
    # Create scatter plot
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X('feature_ind:Q', title='Feature Index'),
        y=alt.Y('value:Q', title='Activation Count'),
        color=alt.Color('variable:N', title='Author', scale=alt.Scale(domain=domain, range=range_colors)),
        tooltip=['feature_ind', 'value', 'variable']
    ).properties(
        title=f'Feature Activation Counts - {title_suffix}',
        width=400,
        height=300
    ).interactive()
    
    return chart

def create_detailed_tfidf_altair(data: Dict[str, pd.DataFrame], title_suffix: str, color_manager: AuthorColorManager):
    
    if 'tfidf_csv' not in data:
        return None
    
    df = data['tfidf_csv']
    
    # Check if we have the required columns
    required_cols = ['feature_ind', 'author', 'tfidf', 'input_tokens', 'predicted_tokens']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Ensure feature_ind is numeric
    df['feature_ind'] = pd.to_numeric(df['feature_ind'], errors='coerce')

    

    # Filter out points where tfidf value is 0
    df_filtered = df[df['tfidf'] > 0.0001].copy()
    
    # Get colors for authors
    authors = df_filtered['author'].unique().tolist()
    domain, range_colors = color_manager.get_altair_domain_range(authors)

    # Create scatter plot
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
        title=f'Feature Importance - {title_suffix}',
        width=1400,
        height=1200
    ).interactive()
    
    return chart
    


def display_file(data: Dict[str, Any], vis_type: str, title_suffix: str, color_manager: AuthorColorManager):
    """Display data as Altair chart based on visualization type"""
    if not data:
        st.info("No data available for this selection")
        return
    
    if vis_type == 'activation_frequencies':
        chart = create_activation_histogram_altair(data, title_suffix, color_manager)
    elif vis_type == 'entropies':
        # Create separate charts for entropy and cross-entropy
        entropy_chart = create_entropy_scatter_altair(data, title_suffix, color_manager)
        cross_entropy_chart = create_cross_entropy_scatter_altair(data, title_suffix, color_manager)
        
        # Display entropy chart
        if entropy_chart:
            st.subheader("Entropy vs Activations")
            st.altair_chart(entropy_chart, use_container_width=True)
        else:
            st.warning("No entropy data available")
        
        # Display cross-entropy chart
        if cross_entropy_chart:
            st.subheader("Cross-Entropy vs Activations")
            st.altair_chart(cross_entropy_chart, use_container_width=True)
        else:
            st.warning("No cross-entropy data available")
        
        return  # Return early since we've handled the display
    elif vis_type == 'token_position':
        chart = create_token_position_scatter_altair(data, title_suffix, color_manager)
    elif vis_type == 'detailed_scatter':
        chart = create_detailed_scatter_altair(data, title_suffix, color_manager)
    elif vis_type == 'detailed_tfidf':
        chart = create_detailed_tfidf_altair(data, title_suffix, color_manager)
    else:
        st.error(f"Unknown visualization type: {vis_type}")
        return
    
    if chart:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available for visualization")

def get_json_files_by_pattern(directory_path: str, pattern: str):
    """Get JSON files matching a specific pattern"""
    if not os.path.exists(directory_path):
        return []
    
    files = []
    for file_path in Path(directory_path).glob(f"{pattern}*.json"):
        if file_path.is_file():
            files.append(file_path.name)
    
    return sorted(files)

def parse_json_filename(filename: str):
    """Parse JSON filename to extract information type, layer type, and layer index"""
    # Remove .json extension
    filename_stem = Path(filename).stem
    
    # Split by double underscore to separate information type from layer info
    parts = filename_stem.split("__")
    if len(parts) != 2:
        return None, None, None
    
    information_type = parts[0]
    layer_info = parts[1]
    
    # Split layer info by underscore to get layer type and index
    layer_parts = layer_info.split("_")
    if len(layer_parts) != 2:
        return None, None, None
    
    layer_type = layer_parts[0]
    layer_ind = layer_parts[1]
    
    return information_type, layer_type, layer_ind

def load_json_data(file_path: str):
    """Load JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return None

def create_correlation_histogram(data: Dict[str, Dict[int, float]], title: str, color_manager: AuthorColorManager):
    """Create histogram of correlation values with tooltips showing feature indices"""
    
    # Prepare data for histogram
    chart_data = []
    feature_indices_by_bin = defaultdict(list)
    
    # Collect all correlation values
    all_values = []
    for author, feature_data in data.items():
        for feature_ind, value in feature_data.items():
            all_values.append(value)
    
    # Create histogram bins
    hist, bin_edges = np.histogram(all_values, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Group feature indices by bin
    for author, feature_data in data.items():
        for feature_ind, value in feature_data.items():
            # Find which bin this value belongs to
            bin_idx = np.digitize(value, bin_edges) - 1
            if 0 <= bin_idx < len(bin_centers):
                bin_center = bin_centers[bin_idx]
                feature_indices_by_bin[bin_center].append(feature_ind)
    
    # Create chart data
    for bin_center, feature_indices in feature_indices_by_bin.items():
        chart_data.append({
            'bin_center': bin_center,
            'count': len(feature_indices),
            'feature_indices': ', '.join(map(str, sorted(feature_indices)))
        })
    
    if not chart_data:
        return None
    
    df = pd.DataFrame(chart_data)
    
    # Create histogram chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('bin_center:Q', title='Correlation Value', bin=alt.Bin(step=0.1)),
        y=alt.Y('count:Q', title='Number of Features'),
        tooltip=[
            alt.Tooltip('bin_center:Q', title='Bin Center', format='.3f'),
            alt.Tooltip('count:Q', title='Feature Count'),
            alt.Tooltip('feature_indices:N', title='Feature Indices')
        ]
    ).properties(
        title=title,
        width=1200,
        height=800
    ).interactive()
    
    return chart

def create_feature_exploration_tab():
    """Create the Features exploration tab"""
    st.header("Features Exploration")
    
    # Check if directory is selected in sidebar
    if 'selected_directory' not in st.session_state or not st.session_state.selected_directory:
        st.info("Please select a directory in the sidebar first.")
        return
    
    selected_directory = st.session_state.selected_directory
    output_dir = f"./sae_features/outputs/{selected_directory}"
    
    st.success(f"Analyzing directory: {selected_directory}")
    
    # Load color mapping
    color_mapping_path = f"{output_dir}/author_color_mapping.json"
    if os.path.exists(color_mapping_path):
        try:
            st.session_state.color_manager.load_color_mapping(Path(color_mapping_path))
        except Exception as e:
            st.warning(f"Could not load color mapping: {e}")
    
    # Get available JSON files
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not json_files:
        st.warning("No JSON files found in the selected directory")
        return
    
    # Parse filenames to get available layer types and indices
    layer_info = {}
    for filename in json_files:
        info_type, layer_type, layer_ind = parse_json_filename(filename)
        if info_type and layer_type and layer_ind:
            if layer_type not in layer_info:
                layer_info[layer_type] = set()
            layer_info[layer_type].add(layer_ind)
    
    if not layer_info:
        st.warning("No valid layer information found in JSON filenames")
        return
    
    # Layer type and index selection
    st.subheader("1. Select Layer")
    col1, col2 = st.columns(2)
    
    with col1:
        layer_types = sorted(list(layer_info.keys()))
        selected_layer_type = st.selectbox(
            "Layer Type:",
            layer_types,
            key="features_layer_type"
        )
    
    with col2:
        if selected_layer_type:
            layer_inds = sorted(list(layer_info[selected_layer_type]), key=lambda x: int(x) if x.isdigit() else x)
            selected_layer_ind = st.selectbox(
                "Layer Index:",
                layer_inds,
                key="features_layer_ind"
            )
        else:
            selected_layer_ind = None
    
    if not selected_layer_type or not selected_layer_ind:
        return
    
    # Load and display correlation data
    st.subheader("2. Point Biserial Correlations")
    
    # Load point biserial correlations
    correlations_file = f"point_biserial_correlations__{selected_layer_type}_{selected_layer_ind}.json"
    correlations_path = os.path.join(output_dir, correlations_file)
    
    if os.path.exists(correlations_path):
        correlations_data = load_json_data(correlations_path)[selected_layer_type][selected_layer_ind]
        if correlations_data:
            # Get authors from the data
            authors = list(correlations_data.keys())
            selected_correlation_authors = st.multiselect(
                "Select authors for correlations:",
                authors,
                default=authors[:min(3, len(authors))],
                key="correlation_authors"
            )
            
            if selected_correlation_authors:
                # Filter data for selected authors
                filtered_correlations = {author: correlations_data[author] for author in selected_correlation_authors}
                
                # Create histogram
                correlation_chart = create_correlation_histogram(
                    filtered_correlations, 
                    f"Point Biserial Correlations - {selected_layer_type} {selected_layer_ind}",
                    st.session_state.color_manager
                )
                
                if correlation_chart:
                    st.altair_chart(correlation_chart, use_container_width=True)
                else:
                    st.warning("No correlation data available for visualization")
    else:
        st.warning(f"Correlations file not found: {correlations_file}")
    
    # Load and display cross-entropy correlations
    st.subheader("3. Point Biserial Cross-Entropy Correlations")
    
    cross_correlations_file = f"point_biserial_correlations_cross__{selected_layer_type}_{selected_layer_ind}.json"
    cross_correlations_path = os.path.join(output_dir, cross_correlations_file)
    
    if os.path.exists(cross_correlations_path):
        cross_correlations_data = load_json_data(cross_correlations_path)[selected_layer_type][selected_layer_ind]
        if cross_correlations_data: 
            # Get authors from the data
            cross_authors = list(cross_correlations_data.keys())   
            selected_cross_authors = st.multiselect(
                "Select authors for cross-entropy correlations:",
                cross_authors,
                default=cross_authors[:min(3, len(cross_authors))],
                key="cross_correlation_authors"
            )
            
            if selected_cross_authors:
                # Filter data for selected authors
                filtered_cross_correlations = {author: cross_correlations_data[author] for author in selected_cross_authors}
                
                # Create histogram
                cross_correlation_chart = create_correlation_histogram(
                    filtered_cross_correlations, 
                    f"Point Biserial Cross-Entropy Correlations - {selected_layer_type} {selected_layer_ind}",
                    st.session_state.color_manager
                )
                
                if cross_correlation_chart:
                    st.altair_chart(cross_correlation_chart, use_container_width=True)
                else:
                    st.warning("No cross-entropy correlation data available for visualization")
    else:
        st.warning(f"Cross-entropy correlations file not found: {cross_correlations_file}")
    
    # Load and display raw data
    st.subheader("4. Raw Data Analysis")
    
    raw_data_file = f"raw_data_aggregated_over_docs__{selected_layer_type}_{selected_layer_ind}.json"
    raw_data_path = os.path.join(output_dir, raw_data_file)
    
    if os.path.exists(raw_data_path):
        raw_data = load_json_data(raw_data_path)
        if raw_data:
            # Get authors from the data
            raw_authors = list(raw_data.keys())
            selected_raw_authors = st.multiselect(
                "Select authors for raw data analysis:",
                raw_authors,
                default=raw_authors[:min(3, len(raw_authors))],
                key="raw_data_authors"
            )
            
            if selected_raw_authors:
                # Get feature indices for selected authors
                all_feature_inds = set()
                for author in selected_raw_authors:
                    if author in raw_data:
                        all_feature_inds.update(raw_data[author].keys())
                
                # Sort feature indices numerically
                sorted_feature_inds = sorted(all_feature_inds, key=lambda x: int(x) if str(x).isdigit() else x)
                
                # Feature index selection
                selected_feature_inds = st.multiselect(
                    "Select feature indices to analyze:",
                    sorted_feature_inds,
                    default=sorted_feature_inds[:min(5, len(sorted_feature_inds))],
                    key="selected_feature_inds"
                )
                
                if selected_feature_inds:
                    # Display data for selected features
                    for feature_ind in selected_feature_inds:
                        st.write(f"**Feature {feature_ind}**")
                        
                        for author in selected_raw_authors:
                            if author in raw_data and feature_ind in raw_data[author]:
                                feature_data = raw_data[author][feature_ind]
                                
                                st.write(f"*Author: {author}*")
                                
                                # Display input tokens
                                if 'input_tokens' in feature_data:
                                    input_tokens = feature_data['input_tokens']
                                    if isinstance(input_tokens, dict):
                                        # Sort by frequency
                                        sorted_input = sorted(input_tokens.items(), key=lambda x: x[1], reverse=True)
                                        st.write("**Input tokens:**")
                                        for token, count in sorted_input[:10]:  # Show top 10
                                            st.write(f"  - {token}: {count}")
                                    else:
                                        st.write(f"**Input tokens:** {input_tokens}")
                                
                                # Display predicted tokens
                                if 'predicted_tokens' in feature_data:
                                    predicted_tokens = feature_data['predicted_tokens']
                                    if isinstance(predicted_tokens, dict):
                                        # Sort by frequency
                                        sorted_predicted = sorted(predicted_tokens.items(), key=lambda x: x[1], reverse=True)
                                        st.write("**Predicted tokens:**")
                                        for token, count in sorted_predicted[:10]:  # Show top 10
                                            st.write(f"  - {token}: {count}")
                                    else:
                                        st.write(f"**Predicted tokens:** {predicted_tokens}")
                                
                                # Display full texts
                                if 'full_texts' in feature_data:
                                    full_texts = feature_data['full_texts']
                                    if isinstance(full_texts, list):
                                        # Remove duplicates by converting to set
                                        unique_texts = list(set(full_texts))
                                        st.write(f"**Full texts ({len(unique_texts)} unique):**")
                                        for i, text in enumerate(unique_texts): 
                                            st.write(f"  {i+1}. {text}")
                                    else:
                                        st.write(f"**Full texts:** {full_texts}")
                                
                                st.write("---")
    else:
        st.warning(f"Raw data file not found: {raw_data_file}")

def main():
    st.set_page_config(page_title="SAE Features Explorer", layout="wide")
    
    st.title("SAE Features Explorer")
    st.markdown("Browse and visualize SAE feature outputs in a customizable grid layout.")
    
    # Initialize color manager
    if 'color_manager' not in st.session_state:
        st.session_state.color_manager = AuthorColorManager()
    
    # Sidebar for controls (available for both tabs)
    with st.sidebar:
        st.header("Controls")
        
        # Directory selection
        st.subheader("1. Select Output Directory")
        directories = get_output_directories()
        
        if not directories:
            st.error("No output directories found in ./sae_features/outputs")
            return
        
        selected_directory = st.selectbox(
            "Choose a directory:",
            directories,
            index=0,
            key="selected_directory"
        )
        
        if selected_directory:
            output_dir = f"./sae_features/outputs/{selected_directory}"
            st.success(f"Selected: {output_dir}")
            
            # Load color mapping from the selected directory
            color_mapping_path = f"{output_dir}/author_color_mapping.json"
            if os.path.exists(color_mapping_path):
                try:
                    st.session_state.color_manager.load_color_mapping(Path(color_mapping_path))
                    st.success(f"Loaded color mapping from {color_mapping_path}")
                except Exception as e:
                    st.warning(f"Could not load color mapping: {e}")

            vis_type = st.selectbox(
                "Choose a visualization type:",
                ["activation_frequencies", "entropies", "token_position", "detailed_scatter", "detailed_tfidf", "features_exploration"],
                index=0,
                key="vis_type"
            )

            if vis_type != "features_exploration":
                create_feature_exploration_tab()
            
            
                # Get files in the selected directory
                files = get_files_by_type(output_dir, vis_type)
                files_structure = parse_filenames(files)
                
                # Debug information
                st.write(f"Found {len(files)} files for visualization type: {vis_type}")
                if files:
                    st.write("Sample files:", files[:5])
                
                if not files:
                    st.warning("No files found in the selected directory")
                    return
                
                st.subheader("2. Grid Configuration")
                
                # Grid size controls
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'num_rows' not in st.session_state:
                        st.session_state.num_rows = 1
                    
                    if st.button("Add Row"):
                        st.session_state.num_rows += 1
                    
                    if st.button("Remove Row") and st.session_state.num_rows > 1:
                        st.session_state.num_rows -= 1
                    
                    st.write(f"Rows: {st.session_state.num_rows}")
                
                with col2:
                    if 'num_cols' not in st.session_state:
                        st.session_state.num_cols = 1
                    
                    if st.button("Add Column"):
                        st.session_state.num_cols += 1
                    
                    if st.button("Remove Column") and st.session_state.num_cols > 1:
                        st.session_state.num_cols -= 1
                    
                    st.write(f"Columns: {st.session_state.num_cols}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Grid Visualization", "Features Exploration"])
    
    with tab1:
        # Main content area
        if selected_directory and 'num_rows' in st.session_state and 'num_cols' in st.session_state:
            st.subheader(f"Grid Display: {selected_directory}")
            
            # Create the grid
            for row in range(st.session_state.num_rows):
                cols = st.columns(st.session_state.num_cols)
                
                for col_idx in range(st.session_state.num_cols):
                    with cols[col_idx]:
                        # Create a unique key for each cell
                        cell_key = f"cell_{row}_{col_idx}"
                        
                        st.write(f"**Cell ({row+1}, {col_idx+1})**")
                        
                        # Layer type selection
                        layer_types = ["None"] + list(files_structure.keys())
                        selected_layer_type = st.selectbox(
                            "Layer Type:",
                            layer_types,
                            key=f"layer_type_{cell_key}"
                        )

                        # Layer index selection (depends on layer type)
                        if selected_layer_type != "None":
                            layer_inds = ["None"] + list(files_structure[selected_layer_type].keys())
                            selected_layer_ind = st.selectbox(
                                "Layer Index:",
                                layer_inds,
                                key=f"layer_ind_{cell_key}"
                            )
                        else:
                            selected_layer_ind = "None"

                        # Author selection (depends on layer type and index)
                        if selected_layer_type != "None" and selected_layer_ind != "None":
                            if vis_type != 'detailed_scatter' and vis_type != 'detailed_tfidf':
                                authors = list(files_structure[selected_layer_type][selected_layer_ind].keys())
                            else:
                                df = pd.read_csv(f"{output_dir}/feature_importance_data__{selected_layer_type}_{selected_layer_ind}.csv")
                                authors = df['author'].unique().tolist()
                            selected_authors = st.multiselect(
                                "Authors:",
                                authors,
                                key=f"authors_{cell_key}"
                            )
                        else:
                            selected_authors = []
                        
                        # Load and display data
                        if selected_layer_type != "None" and selected_layer_ind != "None":
                            if selected_authors:
                                # Load data for this cell
                                selected_filenames = []
                                if vis_type != 'detailed_scatter' and vis_type != 'detailed_tfidf':
                                    for author in selected_authors:
                                        selected_filenames.append(files_structure[selected_layer_type][selected_layer_ind][author])
                                data = load_data_for_cell(output_dir, selected_filenames, selected_layer_type, selected_layer_ind, selected_authors, vis_type)
                                
                                # Display the data
                                title_suffix = f"Layer {selected_layer_ind} {selected_layer_type}"
                                display_file(data, vis_type, title_suffix, st.session_state.color_manager)
                            else:
                                st.info("Select authors to display data")
                        else:
                            st.info("Select layer type and index to display data")
    
    with tab2:
        create_feature_exploration_tab()

if __name__ == "__main__":
    main()
