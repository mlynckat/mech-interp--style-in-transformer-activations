import argparse
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import logging

import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, ShuffleSplit
import pandas as pd
from sklearn.metrics import classification_report


# Import shared utilities
from backend.src.utils.shared_utilities import (
    ActivationFilenamesLoader,
    DataLoader
)

# Set up logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments for feature importance analysis"""
    parser = argparse.ArgumentParser(description="SAE Feature Importance Analysis Tool")
    parser.add_argument(
        "--path_to_data",
        type=str,
        default="data/raw_features/AuthorMixPolitics500canonical-2b",
        help="Directory containing raw features"
    )
    parser.add_argument(
        "--path_to_outputs",
        type=str,
        default="data/output_data",
        help="Output directory for results"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="feature_selection_politics_500_2b",
        help="The name of the run to create a folder in outputs"
    )
    parser.add_argument(
        "--include_authors",
        type=str,
        nargs="+",
        default=None,
        help="The authors to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_types",
        type=str,
        nargs="+",
        default=None,
        choices=["res", "mlp", "att"],
        help="The layer types to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_inds",
        type=int,
        nargs="+",
        default=None,
        help="The layers to include in the analysis"
    )
    parser.add_argument(
        "--include_prompted",
        type=str,
        default="baseline",
        choices=["prompted", "baseline"],
        help="The prompted to include in the analysis"
    )
    
    args = parser.parse_args()

    return args

@dataclass
class FeaturesData:
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_token_indices: np.ndarray
    test_token_indices: np.ndarray

class FeatureSelectionForClassification:
    def __init__(self):
        pass

    def run_feature_selection(self):
        """Run feature selection"""
        pass

class VarianceThresholdFeatureSelection(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, threshold=0.01):
        self.threshold = threshold
        self.features_data = features_data

    def run_feature_selection(self):
        """Run feature selection"""
        logger.info(f"Applying VarianceThreshold")
        variance_threshold = VarianceThreshold(threshold=self.threshold)
        train_activations_reduced = variance_threshold.fit_transform(self.features_data.train_data)

        logger.info(f"VarianceThreshold selected {train_activations_reduced.shape[1]} features")
        most_important_features_author = variance_threshold.get_feature_names_out()
        logger.debug(f"VarianceThreshold features: {most_important_features_author}") 

        non_zero_doctok_indices = np.argwhere(train_activations_reduced.sum(axis=1) > 0)[:, 0]
        train_activations_reduced = train_activations_reduced[non_zero_doctok_indices]
        train_labels_one_vs_all = self.features_data.train_labels[non_zero_doctok_indices]
        train_token_indices_reduced = self.features_data.train_token_indices[non_zero_doctok_indices]

        test_activations_reduced = variance_threshold.transform(self.features_data.test_data)
        non_zero_doctok_indices_test = np.argwhere(test_activations_reduced.sum(axis=1) > 0)[:, 0]
        test_activations_reduced = test_activations_reduced[non_zero_doctok_indices_test]
        test_labels_one_vs_all = self.features_data.test_labels[non_zero_doctok_indices_test]
        test_token_indices_reduced = self.features_data.test_token_indices[non_zero_doctok_indices_test]

        out = FeaturesData(train_data=train_activations_reduced, test_data=test_activations_reduced, train_labels=train_labels_one_vs_all, test_labels=test_labels_one_vs_all, train_token_indices=train_token_indices_reduced, test_token_indices=test_token_indices_reduced)

        return out, most_important_features_author

class SelectKBestFeatureSelection(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, inherited_features_from_previous_step, approach=f_classif, k=10):
        self.inherited_features_from_previous_step = inherited_features_from_previous_step
        self.approach = approach
        self.features_data = features_data
        self.k = k

    def run_feature_selection(self):
        """Run feature selection"""
        # Apply SelectKBest on features chosen with VarianceThreshold
        select_k_best = SelectKBest(self.approach, k=self.k)
        logger.info(f"Fitting SelectKBest")
        train_data_reduced = select_k_best.fit_transform(self.features_data.train_data, self.features_data.train_labels)
        current_feature_names = train_data_reduced.get_feature_names_out()
        original_feature_names = [self.inherited_features_from_previous_step[int(i.replace("x", ""))] for i in current_feature_names]
        logger.debug(f"SelectKBest features: {original_feature_names}")

        test_data_reduced = select_k_best.transform(self.features_data.test_data)
        out = FeaturesData(train_data=train_data_reduced, test_data=test_data_reduced, train_labels=self.features_data.train_labels, test_labels=self.features_data.test_labels, train_token_indices=self.features_data.train_token_indices, test_token_indices=self.features_data.test_token_indices)
        return out, original_feature_names

class SequentialFeatureSelector(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, inherited_features_from_previous_step, approach=LogisticRegression(), k=10):
        self.inherited_features_from_previous_step = inherited_features_from_previous_step
        self.approach = approach
        self.features_data = features_data
        self.k = k
        self.approach = approach

    def run_feature_selection(self):
        """Run feature selection"""
        # Apply SequentialFeatureSelector
        sequential_feature_selector = SequentialFeatureSelector(self.approach, n_features_to_select=self.k, n_jobs=-1)
        logger.info(f"Applying SequentialFeatureSelector..")
        train_data_reduced = sequential_feature_selector.fit_transform(self.features_data.train_data, self.features_data.train_labels.ravel())

        current_feature_names = sequential_feature_selector.get_feature_names_out()
        original_feature_names = [self.inherited_features_from_previous_step[int(i.replace("x", ""))] for i in current_feature_names]
        logger.debug(f"SequentialFeatureSelector features: {original_feature_names}")

        test_data_reduced = sequential_feature_selector.transform(self.features_data.test_data)
        out = FeaturesData(train_data=train_data_reduced, test_data=test_data_reduced, train_labels=self.features_data.train_labels, test_labels=self.features_data.test_labels, train_token_indices=self.features_data.train_token_indices, test_token_indices=self.features_data.test_token_indices)
        
        return out, original_feature_names

        

def save_histogram_of_data(data, save_path):
    """Save histogram of data"""
    zero_count = np.sum(data == 0)
    data = data[data > 0]
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=100)
    plt.title(f"Histogram of raw activations data for {save_path.split('.')[0]}. {zero_count} tokens with no activations")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def retrieve_and_combine_author_data(author_filename_dict, path_to_data, binary=True):
    """Retrieve and combine author data"""

    train_activations = []
    test_activations = []

    train_labels = []
    test_labels = []

    # Initialize train and test token indices to track token positions (author_ind,doc_ind, tok_in_seq_ind)
    train_token_indices = []
    test_token_indices = []

    int_to_author = {}

    for author_ind, (author, filename) in enumerate(author_filename_dict.items()):
        int_to_author[author_ind] = author
        logger.info(f"Loading data for author {author_ind} {author} from {filename}")
        metadata, data = DataLoader().load_sae_activations(Path(path_to_data) / filename)
        doc_lengths = metadata.tokens_per_doc

        n_docs = len(doc_lengths)
        n_docs_train = int(n_docs * 0.8)
        
        for i in range(len(doc_lengths)):
            if i < n_docs_train:
                train_activations.extend(data[i, :doc_lengths[i], :].reshape((-1, data.shape[2])))
                train_token_indices.extend([(author_ind, i, j) for j in range(doc_lengths[i])])
                train_labels.extend([author_ind] * doc_lengths[i])
            else:
                test_activations.extend(data[i, :doc_lengths[i], :].reshape((-1, data.shape[2])))
                test_token_indices.extend([(author_ind, i, j) for j in range(doc_lengths[i])])
                test_labels.extend([author_ind] * doc_lengths[i])
            

        
    train_activations = np.array(train_activations)
    test_activations = np.array(test_activations)
    train_token_indices = np.array(train_token_indices)
    test_token_indices = np.array(test_token_indices)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    logger.info(f"Shape of train data filtered and reshaped: {train_activations.shape}")
    logger.info(f"Shape of test data filtered and reshaped: {test_activations.shape}")
    
    if binary:
        train_activations = (train_activations > 1).astype(np.int8)
        test_activations = (test_activations > 1).astype(np.int8)
    
    return train_activations, test_activations, train_token_indices, test_token_indices, train_labels, test_labels, int_to_author


def get_author_data_numerical_and_labels(author_data):
    """Get author data numerical and labels, ready to use for classification with scikit-learn"""
    author_data_numerical_dense = None
    all_non_zero_feature_names = []
    labels = []
    for author, data in author_data.items():
        logger.debug(f"Author {author} data shape: {data.shape}")
        non_zero_feature_names = get_non_zero_feature_names(data)
        logger.debug(f"Extending all non zero feature names with {len(non_zero_feature_names)} non zero feature names")
        all_non_zero_feature_names.extend(non_zero_feature_names)
    all_non_zero_feature_names = sorted(list(set(all_non_zero_feature_names)))
    logger.debug(f"First 10 all non zero feature names: {all_non_zero_feature_names[:10]}")
    active_feature_indices = [int(feature_name.replace("x", "")) for feature_name in all_non_zero_feature_names]
    logger.debug(f"First 10 active feature indices: {active_feature_indices[:10]}")

    for author, data in author_data.items():
        logger.debug(f"Shape of initial reshaped data: {data.shape}")
        data = data[:, active_feature_indices]
        logger.debug(f"Shape of reshaped data after features filtering: {data.shape}")
        activated_documents_indices = np.argwhere(data.sum(axis=1) > 0)[:, 0] # here anyway zero activation tokens get discarded
        logger.info(f"Activated documents for author {author}: {len(activated_documents_indices)} out of {data.shape[0]}")
        data_filtered = data[activated_documents_indices]
        if author_data_numerical_dense is None:
            author_data_numerical_dense = data_filtered.copy()
        else:
            author_data_numerical_dense = np.concatenate([author_data_numerical_dense, data_filtered.copy()])
        

        logger.info(f"Size of ready data for author {author}: {data_filtered.nbytes / (1024 ** 2):.2f} MB")
        labels.extend([author] * data_filtered.shape[0])

    logger.info(f"Starting to encode labels")
    binarizer = LabelBinarizer()
    labels_encoded  = binarizer.fit_transform(labels)
    logger.debug(f"Labels encoded shape: {labels_encoded.shape}")
    logger.debug(f"First 10 labels encoded: {labels_encoded[:10]}")
    

    authors_classes = binarizer.classes_
    logger.info(f"Authors classes: {authors_classes}")
    logger.info(f"Samples per author: {labels_encoded.sum(axis=0)}")

    all_non_zero_feature_names_map = {i: name for i, name in enumerate(all_non_zero_feature_names)}

    return author_data_numerical_dense, labels_encoded, authors_classes, all_non_zero_feature_names_map

def get_non_zero_feature_names(author_data_numerical: np.ndarray) -> list:
    """Get non zero feature names
    
    Args:
        author_data_numerical: np.ndarray, shape (n_docs x seq_len, n_features)
    
    Returns:
        list
    """
    feature_names = [f"x{i}" for i in range(author_data_numerical.shape[1])]
    summed_up_activations = author_data_numerical.sum(axis=0)
    non_zero_activation_indices = np.argwhere(summed_up_activations > 0)[:, 0]
    logger.debug(f"First 10 non zero activation indices: {non_zero_activation_indices[:10]}")
    non_zero_feature_names = [feature_names[i] for i in range(len(feature_names)) if i in non_zero_activation_indices]
    logger.debug(f"First 10 non zero feature names: {non_zero_feature_names[:10]}")

    return non_zero_feature_names
    
def visualize_results_with_heatmap(predicted_labels, true_labels, tokens_inds, title, save_path):
    """Visualize results with heatmap"""
    
    # construct back to matrix doc x tok
    max_docs = max(doc_ind for _, doc_ind, _ in tokens_inds)
    max_toks = max(tok_ind for _, _, tok_ind in tokens_inds)
    predictions = np.zeros((max_docs, max_toks))

    for i, (auth_ind, doc_ind, tok_ind) in enumerate(tokens_inds):
        predictions[doc_ind, tok_ind] = predicted_labels[i] == true_labels[i]

    # set the rest to nan
    predictions[predictions == 0] = np.nan

    # trnsfrom boolean to int binary

    predictions = predictions.astype(int)

    sns.heatmap(predictions, cmap=["red", "green"], 
                    mask=np.isnan(predictions),  # mask so the nan cells aren’t coloured
                    cbar=True)

    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Document")

    plt.savefig(save_path)
    plt.close()

def run_classification_on_all_authors(author_data_numerical, labels_encoded, authors_classes, layer_type, layer_ind, classification_results):
    """Run classification on all authors"""
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    for author_ind, author in enumerate(authors_classes):
        logger.info(f"Applying classification for {author}")
        if len(authors_classes) > 2:
            labels_encoded_one_vs_all = labels_encoded[:, author_ind]
        else:
            labels_encoded_one_vs_all = labels_encoded if author_ind == 0 else (labels_encoded-1)*-1
        
        logger.debug(f"Labels encoded one vs all shape: {labels_encoded_one_vs_all.shape}")
        logger.debug(f"First 10 labels encoded one vs all: {labels_encoded_one_vs_all[:10]}")
        logger.debug(f"Labels encoded one vs all sum: {labels_encoded_one_vs_all.sum()}")
     
        # Apply LogisticRegression
        logistic_regression = LogisticRegression()
        logger.info(f"Fitting LogisticRegression")
        cv_classification_logreg_score = cross_validate(logistic_regression, author_data_numerical, labels_encoded_one_vs_all, cv=cv, scoring=scoring, n_jobs=-1)
        logger.info(f"LogisticRegression classification scores: {cv_classification_logreg_score}")
        for score in scoring:
            classification_results[layer_type][layer_ind][author]["LogisticRegression"][score] = np.mean(cv_classification_logreg_score[f"test_{score}"])

def main():
    """Main entry point for feature importance analysis"""
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(Path(args.path_to_outputs) / f"{args.run_name}", exist_ok=True)

    activation_filenames_structured = ActivationFilenamesLoader(data_dir=args.path_to_data, include_authors=args.include_authors, include_layer_types=args.include_layer_types, include_layer_inds=args.include_layer_inds, include_prompted=args.include_prompted).get_structured_filenames()

    classification_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))))
    
    for layer_type, layer_ind_dict in activation_filenames_structured.items():
        for layer_ind, author_filename_dict in layer_ind_dict.items():


            if os.path.exists(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_results.json"):
                with open(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_results.json", "r+") as f:
                    classification_results_loaded = json.load(f)

                if layer_type in classification_results_loaded and layer_ind in classification_results_loaded[layer_type]:
                    logger.info(f"Classification results already computed for {layer_type} {layer_ind}")
                    continue

            train_activations, test_activations, train_token_indices, test_token_indices, train_labels, test_labels, int_to_author = retrieve_and_combine_author_data(author_filename_dict, args.path_to_data)

            most_important_features = defaultdict(lambda: defaultdict(list))

            # Apply dfferent feature selection methods one vs all
            for author_ind, author in int_to_author.items():
                logger.info(f"Applying feature selection for {author}")

                # Get labels encoded one vs all
                labels_encoded_one_vs_all = train_labels.copy()
                labels_encoded_one_vs_all[train_labels == author_ind] = 1
                labels_encoded_one_vs_all[train_labels != author_ind] = 0
                
                labels_encoded_one_vs_all_test = test_labels.copy()
                labels_encoded_one_vs_all_test[test_labels == author_ind] = 1
                labels_encoded_one_vs_all_test[test_labels != author_ind] = 0

                logger.debug(f"Train data labels shape: {labels_encoded_one_vs_all.shape}")
                logger.debug(f"Train data labels sum: {labels_encoded_one_vs_all.sum()}")

                logger.debug(f"Test data labels shape: {labels_encoded_one_vs_all_test.shape}")
                logger.debug(f"Test data labels sum: {labels_encoded_one_vs_all_test.sum()}")

                features_data_initial = FeaturesData(train_data=train_activations, test_data=test_activations, train_labels=labels_encoded_one_vs_all, test_labels=labels_encoded_one_vs_all_test, train_token_indices=train_token_indices, test_token_indices=test_token_indices)

                features_data_variance_threshold, most_important_features_variance_threshold = VarianceThresholdFeatureSelection(features_data_initial).run_feature_selection()
                most_important_features[author]["variance_threshold"] = most_important_features_variance_threshold

                svm = SVC()
                logger.info(f"Fitting SVM")
                svm.fit(features_data_variance_threshold.train_data, features_data_variance_threshold.train_labels)
                labels_predicted = svm.predict(features_data_variance_threshold.test_data)
                visualize_results_with_heatmap(labels_predicted, features_data_variance_threshold.test_labels, features_data_variance_threshold.test_token_indices, f"SVM classification results for {author} {layer_type} {layer_ind} Variance Threshold features", Path(args.path_to_outputs) / f"{args.run_name}" / f"svm_classification_results__variance_threshold__{layer_type}__{layer_ind}__{author}___{args.include_prompted}.png")
                classification_report_svm = classification_report(features_data_variance_threshold.test_labels, labels_predicted, output_dict=True)
                logger.info(f"SVM classification report: {classification_report_svm}")
                classification_report_df = pd.DataFrame(classification_report_svm)
                classification_report_df.to_csv(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__variance_threshold.csv")

                classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["precision__class_1"] = classification_report_svm["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["recall__class_1"] = classification_report_svm["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["f1__class_1"] = classification_report_svm["1"]["f1-score"]

                features_data_select_k_best, most_important_features_select_k_best = SelectKBestFeatureSelection(features_data_variance_threshold, most_important_features_variance_threshold, approach=f_classif, k=10).run_feature_selection()
                most_important_features[author]["select_k_best"] = most_important_features_select_k_best

                features_data_sequential_feature_selector, most_important_features_sequential_feature_selector = SequentialFeatureSelector(features_data_variance_threshold, most_important_features_variance_threshold, approach=LogisticRegression(), k=10).run_feature_selection()
                most_important_features[author]["sequential_feature_selector"] = most_important_features_sequential_feature_selector

                # Apply LogisticRegression
                logistic_regression = LogisticRegression()
                logger.info(f"Fitting LogisticRegression")
                logistic_regression.fit(features_data_sequential_feature_selector.train_data, features_data_sequential_feature_selector.train_labels)
                labels_predicted = logistic_regression.predict(features_data_sequential_feature_selector.test_data)
                classification_report_logreg = classification_report(features_data_sequential_feature_selector.test_labels, labels_predicted, output_dict=True)
                logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
                classification_report_df = pd.DataFrame(classification_report_logreg)
                classification_report_df.to_csv(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv")

                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["precision__class_1"] = classification_report_logreg["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["recall__class_1"] = classification_report_logreg["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["f1__class_1"] = classification_report_logreg["1"]["f1-score"]
                
                svm = SVC()
                logger.info(f"Fitting SVM")
                svm.fit(features_data_sequential_feature_selector.train_data, features_data_sequential_feature_selector.train_labels)
                labels_predicted = svm.predict(features_data_sequential_feature_selector.test_data)
                classification_report_svm = classification_report(features_data_sequential_feature_selector.test_labels, labels_predicted, output_dict=True)
                logger.info(f"SVM classification report: {classification_report_svm}")
                classification_report_df = pd.DataFrame(classification_report_svm)
                classification_report_df.to_csv(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv")

                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["precision__class_1"] = classification_report_svm["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["recall__class_1"] = classification_report_svm["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["f1__class_1"] = classification_report_svm["1"]["f1-score"]


                """random_forest = RandomForestClassifier()
                logger.info(f"Fitting RandomForest")
                cv_classification_random_forest_score = cross_validate(random_forest, author_data_numerical_sequential_feature_selector_filtered, labels_encoded_one_vs_all_variance_threshold_filtered, cv=cv, scoring=scoring, n_jobs=-1)
                logger.info(f"RandomForest classification scores: {cv_classification_random_forest_score}")
                for score in scoring:
                    classification_results[layer_type][layer_ind][author]["RandomForest"][score] = np.mean(cv_classification_random_forest_score[f"test_{score}"])"""

            # Save most important features
            with open(Path(args.path_to_outputs) / f"{args.run_name}" / f"most_important_features__{layer_type}__{layer_ind}.json", "w") as f:
                json.dump(most_important_features, f, indent=4)

                    

            # Save classification results
            with open(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_results.json", "w") as f:
                json.dump(classification_results, f, indent=4)
    


if __name__ == "__main__":
    main()
