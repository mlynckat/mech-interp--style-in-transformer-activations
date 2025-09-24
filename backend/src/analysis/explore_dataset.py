import datasets
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer
import pandas as pd
import seaborn as sns
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

DATASET = "hallisky/AuthorMix"
save_dir = Path("sae_features/outputs/dataset/")
save_dir.mkdir(parents=True, exist_ok=True)

author_list = []
author_to_num_docs = defaultdict(int)
author_to_num_docs_over35 = defaultdict(int)
author_to_seq_len = defaultdict(list)

logger.info("Loading dataset...")
dataset = datasets.load_dataset(DATASET, streaming=True, split="train")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        #n_devices=4
    ).to(device)


for i, doc in enumerate(dataset):

    author = doc['style']  # 'style' column contains author info



    # Add new author to the list if we haven't seen them before
    if author not in author_to_num_docs:
        author_list.append(author)


    # Add document to author's list
    author_to_num_docs[author] += 1

    encoded_inputs = model.tokenizer(
                doc["text"],
                return_tensors="pt",
                add_special_tokens=True,
                max_length=1024,
                truncation=True,
                padding=False
            ).to(device)

    author_to_seq_len[author].append(len(encoded_inputs["input_ids"][0]))
    if len(encoded_inputs["input_ids"][0]) < 35:
        logger.debug(doc["text"])
        logger.debug(len(encoded_inputs["input_ids"][0]))
        logger.debug(encoded_inputs["input_ids"][0])
        logger.debug(encoded_inputs["attention_mask"][0])

    if len(encoded_inputs["input_ids"][0]) > 35:
        author_to_num_docs_over35[author] += 1

# Log some info about the data organization
logger.info(f"Number of unique authors: {len(author_list)}")
logger.info(author_to_num_docs)
logger.info(author_to_num_docs_over35)


plt.bar(author_to_num_docs.keys(), author_to_num_docs.values())
plt.xticks(rotation=45)
plt.savefig(save_dir / "author_to_docs.png")
plt.close()

df = pd.concat(pd.DataFrame({'author':author, 'length':lengths}) for author, lengths in author_to_seq_len.items())
logger.info(df.describe())
sns.boxplot(df, x="author", y="length")
plt.xticks(rotation=45)
plt.savefig(save_dir / "author_to_seq_len.png")
plt.close()







