import os
import json
import datasets
from collections import defaultdict
import logging

# Set up logging
logger = logging.getLogger(__name__)


def load_AuthorMix_data(dataset_config, n_docs=None, exclude_authors=None):
    """
    Load the dataset by author and collect document indices.
    Returns: author_to_docs dict and author_list
    """

    logger.info("Loading dataset...")
    DATASET = "hallisky/AuthorMix"
    dataset = datasets.load_dataset(DATASET, streaming=True, split="train")

    min_length_doc = dataset_config.min_length_doc
    max_n_docs_per_author = dataset_config.max_n_docs_per_author

    author_to_docs = defaultdict(list)
    author_list = []
    total_docs_processed = 0

    for i, doc in enumerate(dataset):
        if n_docs is not None and total_docs_processed >= n_docs:
            break

        author = doc['style']  # 'style' column contains author info

        len_doc = len(doc['text'].split())
        if len_doc < min_length_doc:
            continue

        if exclude_authors is not None and author in exclude_authors:
            continue

        # Add new author to the list if we haven't seen them before
        if author not in author_to_docs:
            author_list.append(author)
        else:
            # Skip if we already have enough documents for this author
            if max_n_docs_per_author is not None and len(author_to_docs[author]) >= max_n_docs_per_author:
                continue

        # Add document to author's list
        author_to_docs[author].append((i, doc))
        total_docs_processed += 1

    # Log some info about the data organization
    logger.info(f"Total documents processed: {total_docs_processed}")
    logger.info(f"Number of unique authors: {len(author_list)}")

    dataset_config.max_n_docs_per_author = max(len(docs) for docs in author_to_docs.values())
    dataset_config.author_list = author_list

    return author_to_docs, dataset_config

def load_news_data(dataset_config, category_name, n_docs=None, exclude_authors=None):
    """
    Load the dataset by author and collect document indices.
    Returns: author_to_docs dict and author_list
    """

    logger.info("Loading dataset...")
    dir_path = "data/news_authormix_datasets"
    file_path = os.path.join(dir_path, f"{category_name.upper()}_authormix_cleaned.json")

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    min_length_doc = dataset_config.min_length_doc
    max_n_docs_per_author = dataset_config.max_n_docs_per_author

    author_to_docs = defaultdict(list)
    author_list = []

    for i, doc in enumerate(dataset):

        author = doc['style']  # 'style' column contains author info

        len_doc = len(doc['text'].split())
        if min_length_doc is not None and len_doc < min_length_doc:
            continue

        if exclude_authors is not None and author in exclude_authors:
            continue

        # Add new author to the list if we haven't seen them before
        if author not in author_to_docs:
            author_list.append(author)
        else:
            # Skip if we already have enough documents for this author
            if max_n_docs_per_author is not None and len(author_to_docs[author]) >= max_n_docs_per_author:
                continue

        # Add document to author's list
        author_to_docs[author].append((i, doc))
        
    
    # Sort authors list by number of documents
    author_list.sort(key=lambda x: len(author_to_docs[x]), reverse=True)

    # Keep only the top n authors until we have n_docs documents
    if n_docs is not None:
        filtered_author_list = []
        total_docs_processed = 0
        while total_docs_processed < n_docs and len(author_list) > 0:
            current_author = author_list.pop(0)
            filtered_author_list.append(current_author)
            total_docs_processed += len(author_to_docs[current_author])
        author_list = filtered_author_list
        author_to_docs = {author: docs for author, docs in author_to_docs.items() if author in filtered_author_list}

    dataset_config.max_n_docs_per_author = max(len(docs) for docs in author_to_docs.values())
    dataset_config.author_list = author_list

    # Log some info about the data organization
    logger.info(f"Number of unique authors: {len(author_list)}")

    return author_to_docs, dataset_config