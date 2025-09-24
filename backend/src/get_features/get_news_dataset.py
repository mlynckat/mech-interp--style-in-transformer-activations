import numpy as np 
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import kagglehub
import os
from pathlib import Path
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Download latest version
path = kagglehub.dataset_download("rmisra/news-category-dataset")

logger.info(f"Path to dataset files: {path}")

    
# Read the JSON file (assuming it's in JSON Lines format)
df = pd.read_json(Path(f"{path}/News_Category_Dataset_v3.json"), lines=True)

# Display the first 5 rows
logger.info(df.head())

# Create directories for saving intermediate files
os.makedirs("data", exist_ok=True)
os.makedirs("data/intermediate", exist_ok=True)

# File to track processed categories
TRACKING_FILE = "data/intermediate/processed_categories.json"
FINAL_OUTPUT_FILE = "data/news_dataset.csv"

def load_processed_categories():
    """Load the list of already processed categories from tracking file"""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_categories(processed_categories):
    """Save the list of processed categories to tracking file"""
    with open(TRACKING_FILE, 'w') as f:
        json.dump(list(processed_categories), f)

def get_unique_categories(df):
    """Get unique categories from the dataset"""
    return df['category'].unique()

def process_category(df, category, processed_categories):
    """Process a single category by scraping all articles in that category"""
    logger.info(f"\nProcessing category: {category}")
    
    # Filter dataset for this category
    category_df = df[df['category'] == category].copy()
    logger.info(f"Found {len(category_df)} articles in category '{category}'")
    
    # Initialize columns if they don't exist
    if 'scraped_title' not in category_df.columns:
        category_df['scraped_title'] = None
    if 'article_text' not in category_df.columns:
        category_df['article_text'] = None
    
    # Process each article in the category
    for idx, row in category_df.iterrows():
        if pd.isna(row['scraped_title']) or pd.isna(row['article_text']):
            logger.info(f"Scraping article {idx} of {len(category_df)} in category '{category}'")
            title, text = scrape_news_article(row['link'])
            category_df.at[idx, 'scraped_title'] = title if title is not None else "No title found"
            category_df.at[idx, 'article_text'] = text if text is not None else "No content found"
    
    # Save this category's data
    category_file = f"data/intermediate/category_{category.replace(' ', '_').replace('/', '_')}.csv"
    category_df.to_csv(category_file, index=False)
    logger.info(f"Saved category '{category}' data to {category_file}")
    
    # Mark category as processed
    processed_categories.add(category)
    save_processed_categories(processed_categories)
    
    return category_df

def scrape_news_article(url):
    try:
        # Fetch the webpage
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})  # Add headers to mimic a browser
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the article title - try multiple selectors for better coverage
        title = None
        title_selectors = [
            'h1.article-title',
            'h1.entry-title', 
            'h1.post-title',
            'h1.headline',
            'h1',
            'title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.text.strip():
                title = title_elem.text.strip()
                break
        
        if not title:
            title = "No title found"
        
        # Find the main article content - try multiple common selectors
        article_content = None
        content_selectors = [
            'article',
            '.article-content',
            '.entry-content', 
            '.post-content',
            '.article-body',
            '.story-body',
            '.content',
            'main',
            '[role="main"]'
        ]
        
        for selector in content_selectors:
            article_elem = soup.select_one(selector)
            if article_elem:
                article_content = article_elem
                break
        
        # If no main content area found, try to find the largest text block
        if not article_content:
            # Find all divs with substantial text content
            divs = soup.find_all('div')
            max_text_length = 0
            for div in divs:
                text_length = len(div.get_text().strip())
                if text_length > max_text_length and text_length > 200:  # Minimum length threshold
                    max_text_length = text_length
                    article_content = div
        
        # Extract paragraphs from the main article content
        if article_content:
            paragraphs = article_content.find_all('p')
            # Filter out very short paragraphs (likely navigation/ads)
            article_text = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 20])
        else:
            # Fallback: get all paragraphs but filter out very short ones
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 20])

        logger.info(f"Title: {title}")
        logger.info(f"First 100 characters: {article_text[:200]}...\n")
    except Exception as e:
        logger.info(f"Failed to scrape {url}: {e}\n")
        return None, None
    return title, article_text


def combine_category_files():
    """Combine all category CSV files into the final dataset"""
    logger.info("\nCombining all category files into final dataset...")
    
    # Get all category files
    category_files = [f for f in os.listdir("data/intermediate") if f.startswith("category_") and f.endswith(".csv")]
    
    if not category_files:
        logger.info("No category files found to combine!")
        return None
    
    # Read and combine all category files
    combined_dfs = []
    for file in category_files:
        file_path = os.path.join("data/intermediate", file)
        df_cat = pd.read_csv(file_path)
        combined_dfs.append(df_cat)
        logger.info(f"Loaded {len(df_cat)} articles from {file}")
    
    # Combine all dataframes
    final_df = pd.concat(combined_dfs, ignore_index=True)
    logger.info(f"Combined dataset contains {len(final_df)} total articles")
    
    # Save final dataset
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    logger.info(f"Final dataset saved to {FINAL_OUTPUT_FILE}")
    
    return final_df

# Main execution logic
if __name__ == "__main__":
    # Load already processed categories
    processed_categories = load_processed_categories()
    logger.info(f"Already processed categories: {processed_categories}")
    
    # Get unique categories from dataset
    unique_categories = get_unique_categories(df)
    logger.info(f"Total unique categories: {len(unique_categories)}")
    logger.info(f"Categories: {unique_categories}")
    
    # Process each category
    for category in unique_categories:
        if category in processed_categories:
            logger.info(f"Skipping already processed category: {category}")
            continue
        
        try:
            process_category(df, category, processed_categories)
        except Exception as e:
            logger.info(f"Error processing category '{category}': {e}")
            continue
    
    # Combine all category files into final dataset
    final_dataset = combine_category_files()
    
    if final_dataset is not None:
        logger.info(f"\nProcessing complete! Final dataset saved to {FINAL_OUTPUT_FILE}")
        logger.info(f"Total articles processed: {len(final_dataset)}")
    else:
        logger.info("No data was processed or combined.")