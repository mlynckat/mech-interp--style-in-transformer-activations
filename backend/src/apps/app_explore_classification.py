import streamlit as st
import os
import sys
import glob
from pathlib import Path
import pandas as pd
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import re
from collections import Counter
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared utilities
from backend.src.utils.shared_utilities import ActivationFilenamesLoader, DataLoader


def get_output_directories():
    """Get all directories under data/output_data"""
    outputs_path = Path("data/output_data")
    if not outputs_path.exists():
        return []
    
    directories = [d.name for d in outputs_path.iterdir() if d.is_dir()]
    return sorted(directories)

def get_raw_features_folders():
    """Get all directories under data/raw_features"""
    raw_features_path = Path("data/raw_features")
    if not raw_features_path.exists():
        return []
    
    directories = [d.name for d in raw_features_path.iterdir() if d.is_dir()]
    return sorted(directories)

def get_most_important_features_files(directory_path: str):
    """Get all most_important_features JSON files in the selected directory"""
    if not os.path.exists(directory_path):
        return []
    
    files = []
    for file_path in Path(directory_path).glob("most_important_features__*.json"):
        if file_path.is_file():
            files.append(file_path.name)
    
    return sorted(files)

def parse_most_important_features_filename(filename: str):
    """Parse most_important_features filename to extract layer_type and layer_ind"""
    # Remove .json extension
    filename_stem = Path(filename).stem
    
    # Expected format: most_important_features__{layer_type}__{layer_ind}
    parts = filename_stem.split("__")
    if len(parts) != 3:
        return None, None
    
    layer_type = parts[1]
    layer_ind = parts[2]
    
    return layer_type, layer_ind

def load_most_important_features(file_path: str):
    """Load most important features data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return None

def extract_feature_ind(feature_name: str) -> int:
    """Extract feature index from feature name (e.g., 'x123' -> 123)"""
    match = re.search(r'x(\d+)', feature_name)
    if match:
        return int(match.group(1))
    return None

def load_token_data(author: str, data_dir: str):
    """Load token and full text data for an author"""
    tokens_file = f"sae_baseline__google_gemma-2-2b__tokens__{author}.json"
    full_texts_file = f"sae_baseline__google_gemma-2-2b__full_texts__{author}.json"
    
    tokens_path = os.path.join(data_dir, tokens_file)
    full_texts_path = os.path.join(data_dir, full_texts_file)
    
    tokens_data = None
    full_texts_data = None
    
    if os.path.exists(tokens_path):
        try:
            with open(tokens_path, 'r', encoding='utf-8') as f:
                tokens_data = json.load(f)
        except Exception as e:
            st.warning(f"Error loading tokens for {author}: {e}")
    
    if os.path.exists(full_texts_path):
        try:
            with open(full_texts_path, 'r', encoding='utf-8') as f:
                full_texts_data = json.load(f)
        except Exception as e:
            st.warning(f"Error loading full texts for {author}: {e}")
    
    return tokens_data, full_texts_data

def load_all_authors_token_data(authors: List[str], data_dir: str):
    """Load token and full text data for all authors"""
    all_tokens_data = {}
    all_full_texts_data = {}
    
    for author in authors:
        tokens_data, full_texts_data = load_token_data(author, data_dir)
        all_tokens_data[author] = tokens_data
        all_full_texts_data[author] = full_texts_data
    
    return all_tokens_data, all_full_texts_data


def load_activation_data(layer_type: str, layer_ind: str, author: str, data_dir: str = None):
    """Load activation data for specified layer and authors"""
    try:
        if data_dir is None:
            # Use selected folder from session state or default to AuthorMix
            selected_folder = st.session_state.get('selected_raw_features_folder', 'AuthorMix')
            data_dir = f"data/raw_features/{selected_folder}"
        
        
        file_path = Path(data_dir) / f"sae_baseline__google_gemma-2-2b__{layer_type}__activations__{author}__layer_{layer_ind}.npz"
        if file_path.exists():
            activations = DataLoader().load_sae_activations_simple(file_path)
            st.write(f"Loaded activation data for {author}: shape {activations.shape}")
            return activations
        else:
            st.warning(f"Activation file not found for {author}: {f"{data_dir} / sae_baseline__google_gemma-2-2b__{layer_type}__activations__{author}__layer_{layer_ind}.npz"}")
            return None
        
    except Exception as e:
        st.error(f"Error loading activation data: {e}")
        return {}

def find_feature_activations(activations: np.ndarray, feature_ind: int):
    """Find document and token indices where a feature is activated"""
    # activation_data shape: (num_docs, num_tokens, num_features)
    # Find where feature is activated above threshold
    feature_activations = activations[:, :, feature_ind]
    activated_positions = np.argwhere(feature_activations > 1)
    logger.debug(f"Activated positions: {activated_positions}")
    
    doc_inds = activated_positions[:, 0]
    token_inds = activated_positions[:, 1]
    
    return doc_inds, token_inds

def display_feature_context(author: str, feature_ind: int, doc_ind: int, token_ind: int, 
                          activation_value: float, tokens_data: List[str], full_texts_data: List[str]):
    """Display the context around a feature activation"""
    st.write(f"**Author: {author}, Feature: {feature_ind}, Doc: {doc_ind}, Token: {token_ind}, Activation: {activation_value:.4f}**")
    
    # Display input token
    if tokens_data and doc_ind < len(tokens_data) and token_ind < len(tokens_data[doc_ind]):
        input_token = tokens_data[doc_ind][token_ind]
        st.write(f"**Input token:** {input_token}")
    
    # Display predicted token (next token)
    if tokens_data and doc_ind < len(tokens_data) and token_ind + 1 < len(tokens_data[doc_ind]):
        predicted_token = tokens_data[doc_ind][token_ind + 1]
        st.write(f"**Predicted token:** {predicted_token}")
    
    # Display full text context
    if full_texts_data and doc_ind < len(full_texts_data):
        full_text = full_texts_data[doc_ind]
        st.write(f"**Full text:** {full_text}")
    
    st.write("---")

def aggregate_tokens_and_texts(doc_inds: List[int], token_inds: List[int], tokens_data: List[str], full_texts_data: List[str]):
    """Aggregate tokens and texts for a given set of doc and token indices"""
    input_tokens = []
    predicted_tokens = []
    texts = []
    for doc_ind, token_ind in zip(doc_inds, token_inds):
        input_tokens.append(tokens_data[doc_ind][token_ind])
        predicted_tokens.append(tokens_data[doc_ind][token_ind + 1])
        texts.append(full_texts_data[doc_ind])
    counter_input_tokens = Counter(input_tokens)
    counter_predicted_tokens = Counter(predicted_tokens)
    unique_texts = list(set(texts))
    return counter_input_tokens, counter_predicted_tokens, unique_texts


def create_feature_exploration_interface():
    """Create the main feature exploration interface"""
    st.header("Feature Classification Exploration")
    
    # Check if both directories are selected in sidebar
    if 'selected_output_directory' not in st.session_state or not st.session_state.selected_output_directory:
        st.info("Please select an output directory in the sidebar first.")
        return
    
    if 'selected_raw_features_folder' not in st.session_state or not st.session_state.selected_raw_features_folder:
        st.info("Please select a raw features folder in the sidebar first.")
        return
    
    selected_directory = st.session_state.selected_output_directory
    selected_raw_features_folder = st.session_state.selected_raw_features_folder
    output_dir = f"data/output_data/{selected_directory}"
    raw_features_dir = f"data/raw_features/{selected_raw_features_folder}"
    
    st.success(f"Analyzing directory: {selected_directory}")
    st.info(f"Loading activation data from: {raw_features_dir}")
    
    # Get available most_important_features files
    features_files = get_most_important_features_files(output_dir)
    
    if not features_files:
        st.warning("No most_important_features JSON files found in the selected directory")
        return
    
    # Parse filenames to get available layer types and indices
    layer_info = {}
    for filename in features_files:
        layer_type, layer_ind = parse_most_important_features_filename(filename)
        if layer_type and layer_ind:
            if layer_type not in layer_info:
                layer_info[layer_type] = set()
            layer_info[layer_type].add(layer_ind)
    
    if not layer_info:
        st.warning("No valid layer information found in most_important_features filenames")
        return
    
    # Layer type and index selection
    st.subheader("3. Select Layer")
    col1, col2 = st.columns(2)
    
    with col1:
        layer_types = sorted(list(layer_info.keys()))
        selected_layer_type = st.selectbox(
            "Layer Type:",
            layer_types,
            key="classification_layer_type"
        )
    
    with col2:
        if selected_layer_type:
            layer_inds = sorted(list(layer_info[selected_layer_type]), key=lambda x: int(x) if x.isdigit() else x)
            selected_layer_ind = st.selectbox(
                "Layer Index:",
                layer_inds,
                key="classification_layer_ind"
            )
        else:
            selected_layer_ind = None
    
    if not selected_layer_type or not selected_layer_ind:
        return
    
    # Load most important features data
    features_file = f"most_important_features__{selected_layer_type}__{selected_layer_ind}.json"
    features_path = os.path.join(output_dir, features_file)
    
    if not os.path.exists(features_path):
        st.error(f"Features file not found: {features_file}")
        return
    
    most_important_features = load_most_important_features(features_path)
    if not most_important_features:
        return
    
    st.subheader("4. Feature Selection Methods")
    
    # Get all authors and selectors
    authors = list(most_important_features.keys())
    all_selectors = set()
    for author_data in most_important_features.values():
        all_selectors.update(author_data.keys())
    selectors = sorted(list(all_selectors))
    
    # Create tabular display
    st.write("**Select features to explore:**")
    
    # Create a table-like display with separators
    selected_features = {}
    
    # Add custom CSS for table styling
    st.markdown("""
    <style>
    .feature-table {
        border-collapse: collapse;
        width: 100%;
    }
    .feature-table th, .feature-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .feature-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .table-separator {
        border-top: 2px solid #333;
        margin: 10px 0;
    }
    .row-separator {
        border-bottom: 1px solid #ccc;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header row with separator
    st.markdown('<div class="table-separator"></div>', unsafe_allow_html=True)
    header_cols = st.columns([2] + [1] * len(authors))
    with header_cols[0]:
        st.markdown("**Selector**")
    for i, author in enumerate(authors):
        with header_cols[i + 1]:
            st.markdown(f"**{author}**")
    
    # Add separator after header
    st.markdown('<div class="row-separator"></div>', unsafe_allow_html=True)
    
    # Data rows with separators
    for idx, selector in enumerate(selectors):
        if selector == "VarianceThreshold":
            continue
        
        # Add separator between rows
        if idx > 0:
            st.markdown('<div class="row-separator"></div>', unsafe_allow_html=True)
        
        row_cols = st.columns([1] + [1] * len(authors))
        
        with row_cols[0]:
            st.markdown(f"**{selector}**")
        
        for i, author in enumerate(authors):
            with row_cols[i + 1]:
                # Add CSS class for centering buttons
                st.markdown('<div class="button-cell">', unsafe_allow_html=True)
                
                if author in most_important_features and selector in most_important_features[author]:
                    features = most_important_features[author][selector]
                    
                    # Create buttons for each feature
                    for feature_name in features:
                        button_key = f"feature_{author}_{selector}_{feature_name}"
                        is_selected = (st.session_state.get('selected_feature', {}).get('feature_name') == feature_name and 
                                     st.session_state.get('selected_feature', {}).get('author') == author and
                                     st.session_state.get('selected_feature', {}).get('selector') == selector)
                        
                        # Use primary type for selected button, secondary for unselected
                        button_type = "primary" if is_selected else "secondary"
                        
                        if st.button(feature_name, key=button_key, type=button_type):
                            feature_ind = extract_feature_ind(feature_name)
                            if feature_ind is not None:
                                selected_features[f"{author}_{selector}_{feature_name}"] = {
                                    'author': author,
                                    'selector': selector,
                                    'feature_name': feature_name,
                                    'feature_ind': feature_ind
                                }
                                st.session_state.selected_feature = selected_features[f"{author}_{selector}_{feature_name}"]
                                st.rerun()  # Rerun to update button styling
                
                # Close the div
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Add final separator
    st.markdown('<div class="table-separator"></div>', unsafe_allow_html=True)
    
    # Display all authors' tokens and texts in columns
    all_tokens_data, all_full_texts_data = load_all_authors_token_data(authors, raw_features_dir)
    
    # Display selected feature details
    if 'selected_feature' in st.session_state and st.session_state.selected_feature:
        
        selected_feature = st.session_state.selected_feature
        st.subheader("6. Feature Analysis")
        
        st.write(f"**Selected Feature:** {selected_feature['feature_name']} (index: {selected_feature['feature_ind']})")
        st.write(f"**Author:** {selected_feature['author']}")
        st.write(f"**Selector:** {selected_feature['selector']}")
        

        
        # Display activation/feature data per author

        author_cols_activations = st.columns(len(authors))
        for i, author in enumerate(authors):
            with author_cols_activations[i]:
                st.subheader(f"7. Loading Activation Data for {author}")
                activation_data = load_activation_data(
                    selected_layer_type, 
                    selected_layer_ind, 
                    author,
                    raw_features_dir
                )
        
                # Find feature activations
                st.subheader("8. Feature Activations")
                doc_inds, token_inds = find_feature_activations(activation_data, selected_feature['feature_ind'])
        
                    
        
                st.write(f"Found {len(doc_inds)} activations for feature {selected_feature['feature_ind']}")
        
                # Display activation contexts
                st.subheader("9. Activation Contexts")

                aggregated_tokens_and_texts = aggregate_tokens_and_texts(doc_inds, token_inds, all_tokens_data[author], all_full_texts_data[author])
                counter_input_tokens, counter_predicted_tokens, unique_texts = aggregated_tokens_and_texts
                
                # Display counters in a more distinct format
                st.markdown("**Input tokens:**")
                if counter_input_tokens:
                    # Create a more visually distinct display for input tokens
                    st.markdown("🔹" + " 🔹 ".join([f"**{token}**: {count} " for token, count in counter_input_tokens.most_common()]))
                        
                else:
                    st.write("No input tokens found")
                
                st.markdown("**Predicted tokens:**")
                if counter_predicted_tokens:
                    # Create a more visually distinct display for predicted tokens
                    st.markdown("🔸" + " 🔸 ".join([f"**{token}**: {count} " for token, count in counter_predicted_tokens.most_common()]))
                else:
                    st.write("No predicted tokens found")
                
                st.markdown("**Full texts:**")
                if unique_texts:
                    for i, text in enumerate(unique_texts, 1):
                        st.markdown(f"📄 **Text {i}:** {text}")
                else:
                    st.write("No full texts found")
        
       
        
        

def main():
    st.set_page_config(page_title="Feature Classification Explorer", layout="wide")
    
    # Add custom CSS for button styling and improved visual elements
    st.markdown("""
    <style>
    .selected-button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: 2px solid #ff4757 !important;
        font-weight: bold !important;
    }
    .unselected-button {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border: 1px solid #dee2e6 !important;
    }
    .counter-item {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .text-item {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .section-divider {
        border-top: 3px solid #6c757d;
        margin: 20px 0;
        padding-top: 10px;
    }
    .button-cell {
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 40px;
    }
    .button-cell button {
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Feature Classification Explorer")
    st.markdown("Explore most important features from classification analysis and their activation contexts.")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # Raw features folder selection
        st.subheader("1. Select Raw Features Folder")
        raw_features_folders = get_raw_features_folders()
        
        if not raw_features_folders:
            st.error("No folders found in data/raw_features")
            return
        
        selected_raw_features_folder = st.selectbox(
            "Choose a raw features folder:",
            raw_features_folders,
            index=0,
            key="selected_raw_features_folder"
        )
        
        if selected_raw_features_folder:
            raw_features_path = f"data/raw_features/{selected_raw_features_folder}"
            st.success(f"Selected: {raw_features_path}")
        
        # Directory selection
        st.subheader("2. Select Directory with Relevant Files")
        directories = get_output_directories()
        
        if not directories:
            st.error("No output directories found in data/output_data")
            return
        
        selected_directory = st.selectbox(
            "Choose a directory:",
            directories,
            index=0,
            key="selected_output_directory"
        )
        
        if selected_directory:
            output_dir = f"data/output_data/{selected_directory}"
            st.success(f"Selected: {output_dir}")
    
    # Main content area
    create_feature_exploration_interface()

if __name__ == "__main__":
    main()

