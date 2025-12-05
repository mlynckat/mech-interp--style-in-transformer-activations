import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import glob
import logging
from pathlib import Path

import altair as alt


# Set up logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

path_to_outputs = "data/output_data/news/politics/google_gemma-2-9b-it/news_500_politics/feature_selection_aggregated"
path_to_json = f"{path_to_outputs}/classification_results.json"

with open(path_to_json, "r") as f:
    classification_results = json.load(f)

n_layer_types = len(classification_results)

layer_types = list(classification_results.keys())

layer_inds = list(classification_results[layer_types[0]].keys())

authors = list(classification_results[layer_types[0]][layer_inds[0]].keys())

n_authors = len(authors)
logger.info(f"n_authors: {n_authors}, n_layer_types: {n_layer_types}")



dict_reshaped = {
    "layer_type": [],
    "layer_ind": [],
    "author": [],
    "data_transformation": [],
    "classification_model": [],
    "metric": [],
    "value": []
}

for layer_type, layer_ind_dict in classification_results.items():
    for layer_ind, author_dict in layer_ind_dict.items():
        for author, transformation_dict in author_dict.items():
            for data_transformation, classification_model_dict in transformation_dict.items():
                for classification_model, metric_dict in classification_model_dict.items():
                    for metric, value in metric_dict.items():
                        dict_reshaped["layer_type"].append(layer_type)
                        dict_reshaped["layer_ind"].append(layer_ind)
                        dict_reshaped["author"].append(author)
                        dict_reshaped["data_transformation"].append(data_transformation)
                        dict_reshaped["classification_model"].append(classification_model)
                        dict_reshaped["metric"].append(metric)
                        dict_reshaped["value"].append(value)

df_reshaped = pd.DataFrame(dict_reshaped)

assert df_reshaped["layer_type"].nunique() == 1, "At the moment only res layer type is supported"

# Fill subplots with scatter plots (layer ind on X axis, values on Y axis, one plot per metric)
authors_to_exclude = ["Sam Levine", "Igor Bobic", "Marina Fang"]
df_reshaped_filtered = df_reshaped[~df_reshaped["author"].isin(authors_to_exclude)]

for metric in df_reshaped_filtered["metric"].unique():

    df_metric = df_reshaped_filtered[df_reshaped_filtered["metric"] == metric]
    
    # create subplots, one plot per author, no more than three plots per row
    fig, ax = plt.subplots(n_authors // 3 + 1, 3, figsize=((n_authors // 3 + 1) * 10, 30))


    chart = alt.Chart(df_metric).mark_line().encode(
        x=alt.X("layer_ind", title="Layer Index"),
        y=alt.Y("value", title=metric),
        color=alt.Color("classification_model", legend=alt.Legend(title="Classification Model")),
        shape=alt.Shape("data_transformation", legend=alt.Legend(title="Data Transformation")),
        strokeDash=alt.StrokeDash("data_transformation", legend=alt.Legend(title="Data Transformation Dash")),
        tooltip=["classification_model", "data_transformation", "value"]
    ).properties(
        title=f"{metric}",
        width=1200,
        height=1000
    ).interactive().facet(
    column='author:N'
)
    chart.save(str(Path(path_to_outputs) / f'classification_visualization__{metric}.html'))

sys.exit()

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


                




