import pandas as pd
from typing import Tuple
from pathlib import Path
import os
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DataReader:
    """Handle data loading from news JSON file"""
    
    @staticmethod
    def read_news_json_data(category_name: str = "politics") -> Tuple[pd.Series, pd.Series]:
        """Read news JSON file and split features and target"""
        logger.info(f"Loading news {category_name} dataset...")
        dir_path = "data/news_authormix_datasets"
        file_path = os.path.join(dir_path, f"{category_name.upper()}_authormix.json")

        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if category_name == "politics":
            min_length_doc = 35
            max_n_docs_per_author = 500
            author_list = ["Sam Levine", "Paige Lavender", "Lee Moran", "Amanda Terkel"]
        else:
            raise NotImplementedError(f"Category {category_name} is not supported.")

        author_to_docs = {}

        X = []
        y = []

        for doc in dataset:


            author = doc['style']  # 'style' column contains author info
            if author not in author_list:
                continue

            len_doc = len(doc['text'].split())
            if len_doc < min_length_doc:
                continue

            # Add new author to the list if we haven't seen them before
            if author not in author_to_docs:
                author_to_docs[author] = []
            else:
                # Skip if we already have enough documents for this author
                if len(author_to_docs[author]) >= max_n_docs_per_author:
                    continue

            # Add document to author's list
            author_to_docs[author].append(doc['text'])
            X.append(doc['text'])
            y.append(author)

        del author_to_docs

        X = pd.Series(X)
        y = pd.Series(y)

        return X, y
