import streamlit as st
import os
import glob
from pathlib import Path
import pandas as pd
import altair as alt
from PIL import Image
import io
import socket
import requests

def get_output_directories():
    """Get all directories under ./sae_features/outputs"""
    outputs_path = Path("./sae_features/outputs")
    if not outputs_path.exists():
        return []
    
    directories = [d.name for d in outputs_path.iterdir() if d.is_dir()]
    return sorted(directories)

def get_files_in_directory(directory_path):
    """Get all image and visualization files in the selected directory"""
    if not os.path.exists(directory_path):
        return []
    
    # Get all files
    files = []
    for file_path in Path(directory_path).glob("*"):
        if file_path.is_file():
            # Include PNG images, CSV files, and HTML files (for Altair visualizations)
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.html']:
                files.append(file_path.name)
    
    return sorted(files)

def display_file(file_path):
    """Display a file as image, Altair chart, or HTML"""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.png':
        # Display PNG image
        try:
            image = Image.open(file_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    elif file_extension == '.html':
        # Get the width of the container
        
        # Display HTML file (Altair interactive charts)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Display the HTML content using st.components.v1.html
            st.components.v1.html(html_content, width=1200, height=1200, scrolling=True)
        except Exception as e:
            st.error(f"Error loading HTML file: {e}")
    
    elif file_extension == '.csv':
        # Try to display CSV as Altair chart
        try:
            df = pd.read_csv(file_path)
            
            # Create a simple visualization based on the data
            if len(df.columns) >= 2:
                # If we have at least 2 columns, create a scatter plot
                chart = alt.Chart(df).mark_circle().encode(
                    x=df.columns[0],
                    y=df.columns[1],
                    tooltip=list(df.columns)
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                # If only one column, create a bar chart
                chart = alt.Chart(df).mark_bar().encode(
                    x=df.columns[0],
                    y='count()'
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating visualization from CSV: {e}")
            # Fallback: show the raw data
            try:
                df = pd.read_csv(file_path)
                st.dataframe(df)
            except:
                st.text("Could not display file content")

def main():
    st.set_page_config(page_title="SAE Features Explorer", layout="wide")
    
    st.title("SAE Features Explorer")
    st.markdown("Browse and visualize SAE feature outputs in a customizable grid layout.")
    
    # Sidebar for controls
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
            index=0
        )
        
        if selected_directory:
            output_dir = f"./sae_features/outputs/{selected_directory}"
            st.success(f"Selected: {output_dir}")
            
            # Get files in the selected directory
            files = get_files_in_directory(output_dir)
            
            if not files:
                st.warning("No image, CSV, or HTML files found in the selected directory")
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
                    
                    # File selection dropdown for this cell
                    selected_file = st.selectbox(
                        f"Select file for cell ({row+1}, {col_idx+1}):",
                        ["None"] + files,
                        key=f"file_{cell_key}"
                    )
                    
                    # Display the selected file
                    if selected_file and selected_file != "None":
                        file_path = os.path.join(output_dir, selected_file)
                        display_file(file_path)
                    else:
                        st.info("Select a file to display")

if __name__ == "__main__":
    main()
