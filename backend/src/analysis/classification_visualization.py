import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import glob
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

path_to_outputs = "data/output_data/feature_selection_for_classification_politics500"
path_to_json = f"{path_to_outputs}/classification_results.json"

with open(path_to_json, "r") as f:
    classification_results = json.load(f)

n_layer_types = len(classification_results)

layer_types = list(classification_results.keys())

layer_inds = list(classification_results[layer_types[0]].keys())

authors = list(classification_results[layer_types[0]][layer_inds[0]].keys())

n_authors = len(authors)
logger.info(f"n_authors: {n_authors}, n_layer_types: {n_layer_types}")

fig, ax = plt.subplots(n_authors, n_layer_types, figsize=(15, 15))

reshaped_data = defaultdict(lambda: defaultdict(pd.DataFrame))

for layer_type, layer_ind_dict in classification_results.items():
    for layer_ind, author_dict in layer_ind_dict.items():
        for author, model_dict in author_dict.items():
            score_dict = model_dict["LogisticRegression"]
            scores_df = pd.DataFrame(score_dict, index=[int(layer_ind)])
            scores_df["layer_ind"] = layer_ind
            reshaped_data[layer_type][author] = pd.concat([reshaped_data[layer_type][author], scores_df])



for layer_type_ind, (layer_type, author_dict) in enumerate(reshaped_data.items()):
    for author_ind, (author, scores_df) in enumerate(author_dict.items()):
        # sort scores_df by layer_ind
        scores_df = scores_df.sort_values(by="layer_ind")
        logger.debug(scores_df.head())
        sns.heatmap(scores_df[["accuracy", "precision_macro", "recall_macro", "f1_macro"]], cmap="YlGn", annot=True, fmt=".2f", ax=ax[author_ind, layer_type_ind])
        ax[author_ind, layer_type_ind].set_title(f"{layer_type} {author}")
        ax[author_ind, layer_type_ind].set_xlabel("Metric")
        ax[author_ind, layer_type_ind].set_ylabel("Layer Index")
        plt.tight_layout()
plt.tight_layout()

plt.savefig(f"data/output_data/feature_selection_for_classification_politics500/classification_visualization.png")
plt.close()
        

files_with_most_important_features = glob.glob(f"{path_to_outputs}/most_important_features__*.json")

most_important_features_confusion_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(pd.DataFrame)))
columns = ["SelectKBest_f_classif", "SelectKBest_mutual_info_classif", "SequentialFeatureSelector_LogisticRegression"]
for file_with_most_important_features in files_with_most_important_features:
    with open(file_with_most_important_features, "r") as f:
        most_important_features = json.load(f)
    layer_type, layer_ind = file_with_most_important_features.split("__")[-2:]
    logger.info(f"Processing layer_type: {layer_type}, layer_ind: {layer_ind}")
    for author, features in most_important_features.items():
        most_important_features_confusion_matrix[layer_type][layer_ind][author] = pd.DataFrame(columns=columns, index=columns, dtype=int)
        for column in columns:
            for index in columns:
                most_important_features_confusion_matrix[layer_type][layer_ind][author].loc[column, index] = len(set(features[column]) & set(features[index]))

fig, ax = plt.subplots(len(layer_inds), n_layer_types, figsize=(15, 15))

for layer_type_ind, (layer_type, layer_ind_dict) in enumerate(most_important_features_confusion_matrix.items()):
    for layer_ind_ind, (layer_ind, author_dict) in enumerate(layer_ind_dict.items()):
        aggregated_confusion_matrix_per_author = pd.DataFrame(columns=columns, index=columns, dtype=int)
        for author, confusion_matrix in author_dict.items():
            aggregated_confusion_matrix_per_author = aggregated_confusion_matrix_per_author.add(confusion_matrix, fill_value=0)
            logger.debug(confusion_matrix)
            logger.debug(aggregated_confusion_matrix_per_author)
        # Ensure the DataFrame contains numeric data
        aggregated_confusion_matrix_per_author = aggregated_confusion_matrix_per_author.astype(float)
        sns.heatmap(aggregated_confusion_matrix_per_author, annot=True, ax=ax[layer_type_ind, layer_ind_ind])
        ax[layer_type_ind, layer_ind_ind].set_title(f"{layer_type} {layer_ind} {author}")
        ax[layer_type_ind, layer_ind_ind].set_xlabel("Feature")
        ax[layer_type_ind, layer_ind_ind].set_ylabel("Feature")
        plt.tight_layout()
plt.tight_layout()
plt.savefig(f"data/output_data/feature_selection_for_classification_politics500/most_important_features_confusion_matrix.png")
plt.close()


                




