"""
Further methods for exploring style-responsible layers in transformer activations.

This module implements linear residualization to disentangle content from style
in activation spaces, helping identify which layers encode stylistic information.

Method: Linear Residualization
------------------------------
For each layer activation vector A, fit a linear regression:
    A = CW + ε

Where:
- C is the content representation matrix (SBERT embeddings, TF-IDF, or LDA topics)
- W is the learned weight matrix
- ε (residuals) represents the component orthogonal to content (style)

The residuals can be used for downstream style analyses:
- Linear probing (linear_probs_on_classic_activations.py)
- Clustering analysis (sae_activations_exploration.py)

Usage:
------
    python -m backend.src.analysis.further_layer_style_search_methods \\
        --path_to_data /path/to/activations \\
        --content_type tfidf \\
        --activation_type classic \\
        --include_layer_types res mlp

    # Or using run_id:
    python -m backend.src.analysis.further_layer_style_search_methods \\
        --run_id 0001 \\
        --content_type sbert

Output:
-------
- Residual activations saved to residuals/ subdirectory
- Explained variance plots per layer
- Layer comparison plots showing which layers have more/less content
- Summary markdown and JSON files

Interpretation:
---------------
- Lower mean R² = less content explained = more style information
- Higher mean R² = more content explained = less style information
"""

import argparse
import re
import gc
import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer

from backend.src.utils.plot_styling import PlotStyle, apply_style
from backend.src.utils.shared_utilities import (
    AuthorColorManager, DataLoader, ActivationFilenamesLoader, TokenandFullTextFilenamesLoader
)
from backend.src.analysis.analysis_run_tracking import get_data_and_output_paths, AnalysisRunTracker
from backend.src.analysis.linear_probs_on_classic_activations import (
    ClassicActivationFilenamesLoader, load_classic_activations, aggregate_activations_per_document
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
apply_style()

TRAIN_SPLIT_RATIO = 0.8
RANDOM_STATE = 42
FIGURE_DPI = 300

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


class ContentRepresentationType(Enum):
    SBERT = "sbert"
    TFIDF = "tfidf"
    LDA = "lda"


class ContentRepresentationBuilder:
    """Builds content representation matrices (SBERT, TF-IDF, LDA)."""

    def __init__(self, rep_type: ContentRepresentationType, **kwargs):
        self.rep_type = rep_type
        self.params = kwargs
        self.vectorizer = None
        self.model = None
        self._fitted = False

    
    @staticmethod
    def preprocess_texts(texts: List[str], do_lemmatize: bool = True) -> List[str]:
        """
        Clean and normalize texts for content-focused TF-IDF:
        - lowercase
        - remove URLs/emails/non-alphanumeric
        - optional lemmatization
        - keep mainly content words (NOUN, VERB, ADJ, ADV)
        """
        cleaned = []
        for doc in texts:
            # Lowercase
            text = doc.lower()

            # Remove URLs and emails
            text = re.sub(r"http\S+|www\.\S+", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)

            # Remove non-alphanumeric chars (keep spaces)
            text = re.sub(r"[^a-z0-9\s]", " ", text)

            if do_lemmatize:
                spacy_doc = nlp(text)
                tokens = [
                    tok.lemma_
                    for tok in spacy_doc
                    if not tok.is_stop
                    and not tok.is_punct
                    and tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
                ]
                text = " ".join(tokens)

            cleaned.append(text)
        return cleaned

    def fit(self, texts: List[str]):
        if self.rep_type == ContentRepresentationType.SBERT:
            self._fit_sbert()
        elif self.rep_type == ContentRepresentationType.TFIDF:
            self._fit_tfidf(texts)
        elif self.rep_type == ContentRepresentationType.LDA:
            self._fit_lda(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.rep_type == ContentRepresentationType.SBERT:
            return self.model.encode(texts, show_progress_bar=True)
        elif self.rep_type == ContentRepresentationType.TFIDF:
            return self.vectorizer.transform(texts).toarray()
        elif self.rep_type == ContentRepresentationType.LDA:
            return self.model.transform(self.vectorizer.transform(texts))

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def _fit_sbert(self):
        
        self.model = SentenceTransformer(self.params.get('model_name', 'all-MiniLM-L6-v2'))

    def _fit_tfidf(self, texts):
        preprocessed_texts = self.preprocess_texts(texts)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            stop_words=None,        # already removed via spaCy; keep None here
            sublinear_tf=True,
            smooth_idf=True,
            norm="l2",
            max_features=100_000,
        )
        self.vectorizer.fit(preprocessed_texts)

    def _fit_lda(self, texts):
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, stop_words='english')
        dtm = self.vectorizer.fit_transform(texts)
        self.model = LatentDirichletAllocation(
            n_components=self.params.get('n_topics', 50), random_state=RANDOM_STATE
        )
        self.model.fit(dtm)

    def get_params(self) -> Dict:
        return {"type": self.rep_type.value, **self.params}


class LinearResidualization:
    """Linear residualization: A = CW + ε, ε is the style component."""

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization
        self.models: Dict[int, Ridge] = {}
        self.content_scaler = StandardScaler()
        self._fitted = False
        self.explained_variance_ratios: Dict[int, float] = {}

    def fit(self, activations: np.ndarray, content: np.ndarray):
        n_features = activations.shape[1]
        C = self.content_scaler.fit_transform(content)
        for i in tqdm(range(n_features), desc="Fitting"):
            y = activations[:, i]
            if np.allclose(y, 0):
                continue
            m = Ridge(alpha=self.regularization)
            m.fit(C, y)
            self.models[i] = m
            pred = m.predict(C)
            ss_res = np.sum((y - pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            self.explained_variance_ratios[i] = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        self._fitted = True
        return self

    def transform(self, activations: np.ndarray, content: np.ndarray) -> np.ndarray:
        residuals = activations.copy()
        C = self.content_scaler.transform(content)
        for i, m in self.models.items():
            residuals[:, i] = activations[:, i] - m.predict(C)
        return residuals

    def get_variance_summary(self) -> Dict[str, float]:
        v = list(self.explained_variance_ratios.values())
        return {"mean": np.mean(v), "std": np.std(v), "min": np.min(v), "max": np.max(v)}


def load_texts(data_dir: Path, authors: List[str], setting: str = "baseline"):
    loader = TokenandFullTextFilenamesLoader(data_dir, include_authors=authors)
    files = loader.get_structured_filenames()
    result = {}
    for author in authors:
        if author in files.get('full_texts', {}):
            f = files['full_texts'][author].get(setting, '')
            if f and (data_dir / f).exists():
                with open(data_dir / f, 'r') as fp:
                    result[author] = json.load(fp)
    return result


class ResidualVisualization:
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def plot_variance_dist(self, variances: Dict[int, float], layer_type: str, layer_ind: str):
        v = list(variances.values())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(v, bins=50, alpha=0.7, color=PlotStyle.COLORS['primary'])
        PlotStyle.style_axis(ax, title=f'R² Distribution ({layer_type} L{layer_ind})',
                             xlabel='R²', ylabel='Features')
        plt.savefig(self.save_dir / f"variance__{layer_type}__{layer_ind}.png", dpi=FIGURE_DPI)
        plt.close()

    def plot_layer_comparison(self, layer_vars: Dict[str, Dict], layer_type: str):
        layers = sorted(layer_vars.keys(), key=int)
        means = [layer_vars[l]['mean'] for l in layers]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(layers)), means, color=PlotStyle.COLORS['primary'])
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{l}' for l in layers])
        PlotStyle.style_axis(ax, title=f'Mean R² by Layer ({layer_type})', xlabel='Layer', ylabel='R²')
        plt.savefig(self.save_dir / f"layers__{layer_type}.png", dpi=FIGURE_DPI)
        plt.close()


class StyleLayerAnalyzer:
    """Main analyzer for finding style-responsible layers via linear residualization."""

    def __init__(self, data_dir: Path, output_dir: Path,
                 content_type: ContentRepresentationType = ContentRepresentationType.TFIDF,
                 activation_type: str = "classic", **content_params):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.content_type = content_type
        self.activation_type = activation_type
        self.content_params = content_params
        self.color_manager = AuthorColorManager()
        self.visualizer = ResidualVisualization(self.output_dir)
        self.data_loader = DataLoader()
        self.layer_variance_summary: Dict[str, Dict[str, Dict]] = defaultdict(dict)

    def _aggregate_sae(self, acts: np.ndarray, doc_lens: np.ndarray, agg: str = "mean"):
        """Aggregate SAE activations to document level."""
        n_docs, _, n_feat = acts.shape
        result = np.zeros((n_docs, n_feat), dtype=np.float32)
        for i in range(n_docs):
            if doc_lens[i] > 0:
                d = acts[i, :doc_lens[i], :]
                if agg == "mean":
                    result[i] = np.mean(d, axis=0)
                elif agg == "max":
                    result[i] = np.max(d, axis=0)
                elif agg == "last":
                    result[i] = d[-1, :]
        return result

    def run_analysis(self, include_authors=None, include_layer_types=None,
                     include_layer_inds=None, include_setting="baseline",
                     aggregation="mean", save_residuals=True):
        # Load filenames
        if self.activation_type == "classic":
            loader = ClassicActivationFilenamesLoader(
                self.data_dir, include_authors, include_layer_types, include_layer_inds, include_setting
            )
        else:
            loader = ActivationFilenamesLoader(
                self.data_dir, include_authors, include_layer_types, include_layer_inds, include_setting
            )
        files = loader.get_structured_filenames()
        if not files:
            logger.error("No activation files found!")
            return {}

        # Get authors
        all_authors = set()
        for lt in files:
            for li in files[lt]:
                all_authors.update(files[lt][li].keys())
        all_authors = sorted(all_authors)

        # Load texts
        author_texts = load_texts(self.data_dir, all_authors, include_setting)
        if not author_texts:
            logger.error("No texts found!")
            return {}

        # Build content representation
        content_builder = ContentRepresentationBuilder(self.content_type, **self.content_params)
        all_texts, text_map = [], []
        for author in all_authors:
            if author in author_texts:
                for i, t in enumerate(author_texts[author]):
                    all_texts.append(t)
                    text_map.append((author, i))

        content_full = content_builder.fit_transform(all_texts)
        author_doc_to_idx = {k: i for i, k in enumerate(text_map)}

        results = defaultdict(lambda: defaultdict(dict))

        for layer_type, layer_dict in files.items():
            logger.info(f"\nProcessing {layer_type}")
            for layer_ind, author_dict in layer_dict.items():
                logger.info(f"Layer {layer_ind}")
                combined_acts, combined_content, boundaries = [], [], {}
                all_author_labels = []  # Track author labels for each sample
                idx = 0

                for author, fname in author_dict.items():
                    if author not in author_texts:
                        continue
                    try:
                        fp = self.data_dir / fname
                        if self.activation_type == "classic":
                            acts, meta = load_classic_activations(fp)
                            agg = aggregate_activations_per_document(acts, meta.doc_lengths, aggregation)
                        else:
                            acts, meta = self.data_loader.load_sae_activations(self.data_dir / fname)
                            if sp.issparse(acts):
                                acts = acts.toarray().reshape(meta.original_shape)
                            agg = self._aggregate_sae(acts, meta.doc_lengths, aggregation)

                        n_docs = agg.shape[0]
                        content = np.array([content_full[author_doc_to_idx.get((author, i), 0)]
                                           for i in range(n_docs)])
                        boundaries[author] = (idx, idx + n_docs)
                        all_author_labels.extend([author] * n_docs)  # Add author labels
                        idx += n_docs
                        combined_acts.append(agg)
                        combined_content.append(content)
                    except Exception as e:
                        logger.error(f"Error loading {author}: {e}")

                if len(combined_acts) < 2:
                    continue

                X = np.vstack(combined_acts)
                C = np.vstack(combined_content)
                n_total = X.shape[0]

                # Train/test split
                np.random.seed(RANDOM_STATE)
                perm = np.random.permutation(n_total)
                n_train = int(n_total * TRAIN_SPLIT_RATIO)
                train_idx, test_idx = perm[:n_train], perm[n_train:]

                # Fit residualization
                residualizer = LinearResidualization()
                residualizer.fit(X[train_idx], C[train_idx])
                residuals_train = residualizer.transform(X[train_idx], C[train_idx])
                residuals_test = residualizer.transform(X[test_idx], C[test_idx])

                var_summary = residualizer.get_variance_summary()
                self.layer_variance_summary[layer_type][layer_ind] = var_summary
                logger.info(f"  R² mean={var_summary['mean']:.3f}")

                results[layer_type][layer_ind] = {
                    "variance_summary": var_summary,
                    "n_train": len(train_idx), "n_test": len(test_idx)
                }

                self.visualizer.plot_variance_dist(
                    residualizer.explained_variance_ratios, layer_type, layer_ind
                )

                if save_residuals:
                    self._save_residuals(residuals_train, residuals_test, train_idx, test_idx,
                                         boundaries, layer_type, layer_ind, content_builder, var_summary,
                                         all_author_labels)
                del X, C, residualizer
                gc.collect()

        # Layer comparison plots
        for lt in self.layer_variance_summary:
            self.visualizer.plot_layer_comparison(self.layer_variance_summary[lt], lt)

        self._save_summary(results)
        return results

    def _save_residuals(self, res_train, res_test, train_idx, test_idx, boundaries,
                        layer_type, layer_ind, content_builder, var_summary, all_author_labels):
        """
        Save residualized activations with author labels for downstream analysis.
        
        Args:
            res_train: Train residual activations
            res_test: Test residual activations
            train_idx: Original indices used for training
            test_idx: Original indices used for testing
            boundaries: Dict mapping author to (start_idx, end_idx)
            layer_type: Type of layer (res, mlp, att)
            layer_ind: Layer index
            content_builder: ContentRepresentationBuilder instance
            var_summary: Variance summary dict
            all_author_labels: List of author labels for all samples (before train/test split)
        """
        res_dir = self.output_dir / "residuals"
        res_dir.mkdir(exist_ok=True, parents=True)
        fname = f"residuals__{layer_type}__layer_{layer_ind}"
        
        # Get author labels for train and test sets
        all_author_labels_arr = np.array(all_author_labels)
        train_author_labels = all_author_labels_arr[train_idx]
        test_author_labels = all_author_labels_arr[test_idx]
        
        # Save boundaries as serializable format
        boundaries_serializable = {author: list(bounds) for author, bounds in boundaries.items()}
        
        np.savez_compressed(res_dir / f"{fname}.npz",
                            residuals_train=res_train, residuals_test=res_test,
                            train_indices=train_idx, test_indices=test_idx,
                            train_author_labels=train_author_labels,
                            test_author_labels=test_author_labels)
        with open(res_dir / f"{fname}__meta.json", 'w') as f:
            json.dump({
                "layer_type": layer_type, "layer_index": layer_ind,
                "content_rep": content_builder.get_params(),
                "variance_summary": {k: float(v) for k, v in var_summary.items()},
                "authors": list(boundaries.keys()),
                "boundaries": boundaries_serializable,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": res_train.shape[1] if len(res_train.shape) > 1 else 0
            }, f, indent=2)

    def _save_summary(self, results):
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump({
                "content_type": self.content_type.value,
                "layer_variance": {lt: {li: {k: float(v) for k, v in s.items()}
                                        for li, s in lvs.items()}
                                   for lt, lvs in self.layer_variance_summary.items()}
            }, f, indent=2)

        with open(self.output_dir / "summary.md", 'w') as f:
            f.write("# Linear Residualization Analysis\n\n")
            f.write(f"Content: {self.content_type.value}\n\n")
            f.write("| Layer Type | Layer | Mean R² | Std R² |\n|---|---|---|---|\n")
            for lt in sorted(self.layer_variance_summary.keys()):
                for li in sorted(self.layer_variance_summary[lt].keys(), key=int):
                    vs = self.layer_variance_summary[lt][li]
                    f.write(f"| {lt} | {li} | {vs['mean']:.4f} | {vs['std']:.4f} |\n")


def parse_args():
    p = argparse.ArgumentParser(description="Linear Residualization for Style Layers")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--path_to_data", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--content_type", type=str, default="tfidf", choices=["sbert", "tfidf", "lda"])
    p.add_argument("--activation_type", type=str, default="classic", choices=["classic", "sae"])
    p.add_argument("--include_authors", type=str, nargs="+", default=None)
    p.add_argument("--include_layer_types", type=str, nargs="+", default=None, choices=["res", "mlp", "att"])
    p.add_argument("--include_layer_inds", type=int, nargs="+", default=None)
    p.add_argument("--include_setting", type=str, default="baseline", choices=["baseline", "prompted"])
    p.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max", "last"])
    p.add_argument("--no_save_residuals", action="store_true")
    p.add_argument("--tfidf_max_features", type=int, default=5000)
    p.add_argument("--lda_n_topics", type=int, default=50)
    p.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    args = p.parse_args()
    if not args.run_id and not args.path_to_data:
        p.error("Either --run_id or --path_to_data required")
    return args


def main():
    args = parse_args()
    data_path, output_path, run_info = get_data_and_output_paths(
        args.run_id, args.path_to_data, "residualization", args.run_name
    )

    tracker = AnalysisRunTracker()
    if run_info and run_info.get('id'):
        tracker.register_analysis(run_info['id'], "residualization", str(data_path), str(output_path))

    content_type = ContentRepresentationType(args.content_type)
    params = {}
    if content_type == ContentRepresentationType.TFIDF:
        params['max_features'] = args.tfidf_max_features
    elif content_type == ContentRepresentationType.LDA:
        params['n_topics'] = args.lda_n_topics
    elif content_type == ContentRepresentationType.SBERT:
        params['model_name'] = args.sbert_model

    analyzer = StyleLayerAnalyzer(data_path, output_path, content_type, args.activation_type, **params)
    results = analyzer.run_analysis(
        args.include_authors, args.include_layer_types, args.include_layer_inds,
        args.include_setting, args.aggregation, not args.no_save_residuals
    )

    if results:
        logger.info("\n" + "="*50 + "\nSUMMARY\n" + "="*50)
        for lt, ld in results.items():
            for li, r in sorted(ld.items(), key=lambda x: int(x[0])):
                vs = r.get('variance_summary', {})
                logger.info(f"{lt} L{li}: R²={vs.get('mean', 0):.4f}")


if __name__ == "__main__":
    main()
