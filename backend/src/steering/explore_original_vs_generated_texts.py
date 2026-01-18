"""
Quick exploration script to compare original news articles with baseline generated
texts. Computes lightweight distributional differences and produces a few plots.

Metrics:
- Mean sentence length (in word tokens)
- Mean token length (characters)
- Proportion of punctuation tokens
- Type/token ratio
- JS/KL divergence of top-100 word distributions
- Proportion of named-entity tokens (best-effort; requires a NER model)

Outputs:
- JSON summary of metrics
- PNG figures comparing distributions
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import nltk
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Ensure required NLTK resources are present (downloads are silent/no-op if cached)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# Regex patterns for simple tokenization/sentence splitting
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CorpusStats:
    sentence_lengths: List[int]
    token_lengths: List[int]
    punctuation_tokens: int
    total_tokens: int
    word_counter: Counter
    entity_tokens: int
    stopword_tokens: int
    function_word_counter: Counter
    pos_counter: Counter
    pronoun_person_counter: Counter
    tense_counter: Counter
    capitalization_tokens: int
    quote_tokens: int
    parenthesis_tokens: int
    ellipsis_tokens: int
    per_doc_punct_rates: List[float]
    sentiment_scores: List[float]
    subjectivity_scores: List[float]
    mtld: float

    @property
    def mean_sentence_length(self) -> float:
        return float(np.mean(self.sentence_lengths)) if self.sentence_lengths else 0.0

    @property
    def mean_token_length(self) -> float:
        return float(np.mean(self.token_lengths)) if self.token_lengths else 0.0

    @property
    def punctuation_proportion(self) -> float:
        return (self.punctuation_tokens / self.total_tokens) if self.total_tokens else 0.0

    @property
    def type_token_ratio(self) -> float:
        total_word_tokens = sum(self.word_counter.values())
        return (len(self.word_counter) / total_word_tokens) if total_word_tokens else 0.0

    @property
    def named_entity_proportion(self) -> Optional[float]:
        if self.total_tokens == 0:
            return None
        return self.entity_tokens / self.total_tokens

    @property
    def stopword_proportion(self) -> float:
        total_word_tokens = sum(self.word_counter.values())
        return (self.stopword_tokens / total_word_tokens) if total_word_tokens else 0.0

    @property
    def capitalization_rate(self) -> float:
        total_word_tokens = sum(self.word_counter.values())
        return (self.capitalization_tokens / total_word_tokens) if total_word_tokens else 0.0

    @property
    def punctuation_rate_variance(self) -> float:
        return float(np.var(self.per_doc_punct_rates)) if self.per_doc_punct_rates else 0.0

    @property
    def mean_sentiment(self) -> Optional[float]:
        return float(np.mean(self.sentiment_scores)) if self.sentiment_scores else None

    @property
    def mean_subjectivity(self) -> Optional[float]:
        return float(np.mean(self.subjectivity_scores)) if self.subjectivity_scores else None


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Return tokens along with start/end character spans."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def split_sentences(text: str) -> List[str]:
    """Very lightweight sentence splitter based on punctuation."""
    sentences = re.split(SENTENCE_SPLIT_PATTERN, text.strip())
    return [s for s in sentences if s.strip()]


def is_punctuation_token(token: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", token))


def word_tokens(tokens_with_spans: Sequence[Tuple[str, int, int]]) -> List[str]:
    """Lowercased word tokens (punctuation removed)."""
    return [tok.lower() for tok, _, _ in tokens_with_spans if not is_punctuation_token(tok)]


def count_entity_tokens(
    tokens_with_spans: Sequence[Tuple[str, int, int]],
    entities: Sequence[dict],
) -> int:
    """Count tokens overlapping any named-entity span."""
    if not entities:
        return 0
    spans: List[Tuple[int, int]] = []
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            continue
        spans.append((int(start), int(end)))

    if not spans:
        return 0

    entity_tokens = 0
    for _, tok_start, tok_end in tokens_with_spans:
        for span_start, span_end in spans:
            # Overlap check
            if not (tok_end <= span_start or tok_start >= span_end):
                entity_tokens += 1
                break
    return entity_tokens


def compute_mtld(tokens: Sequence[str], ttr_threshold: float = 0.72, min_segment_length: int = 10) -> float:
    """Compute Measure of Textual Lexical Diversity (MTLD)."""

    def mtld_pass(seq: Sequence[str]) -> float:
        factors = 0
        types = Counter()
        start = 0
        for i, tok in enumerate(seq, start=1):
            types[tok] += 1
            ttr = len(types) / i
            if ttr <= ttr_threshold and i - start >= min_segment_length:
                factors += 1
                types.clear()
                start = i
        residual = len(types) / max(1, len(seq) - start)
        if residual > 0:
            factors += (1 - ttr_threshold) / (residual - ttr_threshold) if residual != ttr_threshold else 0
        return len(seq) / max(factors, 1e-9)

    if not tokens:
        return 0.0
    forward = mtld_pass(tokens)
    backward = mtld_pass(list(reversed(tokens)))
    return float(np.mean([forward, backward]))


def compute_corpus_stats(
    texts: Iterable[str],
    ner_pipeline=None,
    max_texts: Optional[int] = None,
) -> CorpusStats:
    sentence_lengths: List[int] = []
    token_lengths: List[int] = []
    punctuation_tokens = 0
    total_tokens = 0
    word_counter: Counter = Counter()
    entity_tokens = 0
    stopword_tokens = 0
    function_word_counter: Counter = Counter()
    pos_counter: Counter = Counter()
    pronoun_person_counter: Counter = Counter()
    tense_counter: Counter = Counter()
    capitalization_tokens = 0
    quote_tokens = 0
    parenthesis_tokens = 0
    ellipsis_tokens = 0
    per_doc_punct_rates: List[float] = []
    sentiment_scores: List[float] = []
    subjectivity_scores: List[float] = []

    stopwords = set(nltk.corpus.stopwords.words("english"))
    try:
        sid = SentimentIntensityAnalyzer()
    except Exception:
        sid = None
    try:
        from textblob import TextBlob
    except Exception:
        TextBlob = None

    for idx, text in enumerate(texts):
        if max_texts is not None and idx >= max_texts:
            break
        tokens_spans = tokenize_with_spans(text)
        doc_total_tokens = len(tokens_spans)
        total_tokens += doc_total_tokens
        doc_punct = sum(1 for tok, _, _ in tokens_spans if is_punctuation_token(tok))
        punctuation_tokens += doc_punct
        per_doc_punct_rates.append(doc_punct / doc_total_tokens if doc_total_tokens else 0.0)

        words = word_tokens(tokens_spans)
        token_lengths.extend([len(tok) for tok in words])
        word_counter.update(words)
        stopword_tokens += sum(1 for tok in words if tok in stopwords)
        function_word_counter.update([tok for tok in words if tok in stopwords])

        # Style counts
        capitalization_tokens += sum(1 for tok in words if tok and tok[0].isupper() and tok.lower() != tok)
        quote_tokens += sum(1 for tok, _, _ in tokens_spans if tok in {"'", '"', "``", "''"})
        parenthesis_tokens += sum(1 for tok, _, _ in tokens_spans if tok in {"(", ")"})
        ellipsis_tokens += sum(1 for tok, _, _ in tokens_spans if "..." in tok or tok == "…")

        # Sentiment / subjectivity
        if text.strip() and sid is not None:
            polarity = sid.polarity_scores(text)["compound"]
            sentiment_scores.append(polarity)
            if TextBlob is not None:
                subjectivity_scores.append(TextBlob(text).sentiment.subjectivity)

        for sent in split_sentences(text):
            sent_tokens = word_tokens(tokenize_with_spans(sent))
            if sent_tokens:
                sentence_lengths.append(len(sent_tokens))

        # POS / tense / pronouns
        if words:
            tags = nltk.pos_tag(words)
            pos_counter.update(tag for _, tag in tags)
            for tok, tag in tags:
                lower_tok = tok.lower()
                if tag.startswith("V"):
                    if tag in {"VBD", "VBN"}:
                        tense_counter["past"] += 1
                    elif tag in {"VB", "VBP", "VBZ"}:
                        tense_counter["present"] += 1
                    elif tag == "VBG":
                        tense_counter["progressive"] += 1
                    else:
                        tense_counter["other"] += 1
                if lower_tok in {"i", "me", "we", "us", "my", "our", "mine", "ours"}:
                    pronoun_person_counter["first"] += 1
                elif lower_tok in {"you", "your", "yours", "ya", "u"}:
                    pronoun_person_counter["second"] += 1
                elif lower_tok in {
                    "he",
                    "she",
                    "it",
                    "they",
                    "him",
                    "her",
                    "them",
                    "his",
                    "hers",
                    "its",
                    "their",
                    "theirs",
                }:
                    pronoun_person_counter["third"] += 1

        if ner_pipeline is not None:
            try:
                entities = ner_pipeline(text)
                entity_tokens += count_entity_tokens(tokens_spans, entities)
            except Exception:
                # If NER fails for a given doc, skip without stopping the run.
                continue

    return CorpusStats(
        sentence_lengths=sentence_lengths,
        token_lengths=token_lengths,
        punctuation_tokens=punctuation_tokens,
        total_tokens=total_tokens,
        word_counter=word_counter,
        entity_tokens=entity_tokens,
        stopword_tokens=stopword_tokens,
        function_word_counter=function_word_counter,
        pos_counter=pos_counter,
        pronoun_person_counter=pronoun_person_counter,
        tense_counter=tense_counter,
        capitalization_tokens=capitalization_tokens,
        quote_tokens=quote_tokens,
        parenthesis_tokens=parenthesis_tokens,
        ellipsis_tokens=ellipsis_tokens,
        per_doc_punct_rates=per_doc_punct_rates,
        sentiment_scores=sentiment_scores,
        subjectivity_scores=subjectivity_scores,
        mtld=compute_mtld(list(word_counter.elements())),
    )


def get_top_word_probs(
    counter: Counter,
    vocab: List[str],
    smoothing: float = 1e-9,
) -> np.ndarray:
    counts = np.array([counter.get(w, 0) for w in vocab], dtype=float)
    counts += smoothing
    return counts / counts.sum()


def compute_word_divergences(
    orig_counter: Counter,
    gen_counter: Counter,
    top_n: int = 100,
) -> Tuple[float, float, float, List[str]]:
    combined = orig_counter + gen_counter
    vocab = [w for w, _ in combined.most_common(top_n)]
    if not vocab:
        return 0.0, 0.0, 0.0, []

    p = get_top_word_probs(orig_counter, vocab)
    q = get_top_word_probs(gen_counter, vocab)

    # JS divergence (scipy returns the square root, so square it to get divergence)
    js = float(jensenshannon(p, q) ** 2)
    kl_pq = float(entropy(p, q))
    kl_qp = float(entropy(q, p))
    return js, kl_pq, kl_qp, vocab


def compute_js_from_arrays(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)


def compute_function_word_divergence(
    orig_counter: Counter, gen_counter: Counter, vocab: Sequence[str]
) -> Tuple[float, float, float]:
    if not vocab:
        return 0.0, 0.0, 0.0
    p = get_top_word_probs(orig_counter, list(vocab))
    q = get_top_word_probs(gen_counter, list(vocab))
    js = compute_js_from_arrays(p, q)
    kl_pq = float(entropy(p, q))
    kl_qp = float(entropy(q, p))
    return js, kl_pq, kl_qp


def counter_js(counter_a: Counter, counter_b: Counter, labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    p = np.array([counter_a.get(lbl, 0) for lbl in labels], dtype=float) + 1e-9
    q = np.array([counter_b.get(lbl, 0) for lbl in labels], dtype=float) + 1e-9
    return compute_js_from_arrays(p, q)


def binned_js_divergence(values_a: Sequence[float], values_b: Sequence[float], bins: np.ndarray) -> Optional[float]:
    if not values_a or not values_b:
        return None
    hist_a, _ = np.histogram(values_a, bins=bins, density=True)
    hist_b, _ = np.histogram(values_b, bins=bins, density=True)
    hist_a += 1e-9
    hist_b += 1e-9
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()
    return compute_js_from_arrays(hist_a, hist_b)


def init_ner_pipeline(model_name: str, disable: bool = False):
    if disable:
        return None

    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline(
            task="token-classification",
            model=model_name,
            aggregation_strategy="simple",
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not initialize NER pipeline ({model_name}): {exc}")
        return None


def plot_sentence_lengths(
    original_lengths: Sequence[int],
    generated_lengths: Sequence[int],
    output_path: Path,
):
    plt.figure(figsize=(8, 5))
    sns.histplot(original_lengths, color="#1f77b4", label="Original", stat="density", bins=30, alpha=0.45)
    sns.histplot(generated_lengths, color="#ff7f0e", label="Generated", stat="density", bins=30, alpha=0.45)
    plt.xlabel("Sentence length (word tokens)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Sentence Length Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_token_level_metrics(
    orig_stats: CorpusStats,
    gen_stats: CorpusStats,
    output_path: Path,
):
    metrics = [
        ("Mean sentence length", orig_stats.mean_sentence_length, gen_stats.mean_sentence_length),
        ("Mean token length", orig_stats.mean_token_length, gen_stats.mean_token_length),
        ("Punctuation proportion", orig_stats.punctuation_proportion, gen_stats.punctuation_proportion),
        ("Type/Token ratio", orig_stats.type_token_ratio, gen_stats.type_token_ratio),
    ]
    if orig_stats.named_entity_proportion is not None and gen_stats.named_entity_proportion is not None:
        metrics.append(("Named-entity proportion", orig_stats.named_entity_proportion, gen_stats.named_entity_proportion))

    df = pd.DataFrame(
        [
            {"metric": name, "corpus": "Original", "value": orig_val}
            for name, orig_val, _ in metrics
        ]
        + [
            {"metric": name, "corpus": "Generated", "value": gen_val}
            for name, _, gen_val in metrics
        ]
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="metric", y="value", hue="corpus")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Value")
    plt.xlabel("")
    plt.title("Token-level Metrics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_words(
    vocab: Sequence[str],
    orig_counter: Counter,
    gen_counter: Counter,
    output_path: Path,
    top_k: int = 20,
):
    vocab = list(vocab[:top_k])
    if not vocab:
        return

    records = []
    orig_total = sum(orig_counter[w] for w in vocab) or 1
    gen_total = sum(gen_counter[w] for w in vocab) or 1
    for word in vocab:
        records.append({"word": word, "corpus": "Original", "freq": orig_counter[word] / orig_total})
        records.append({"word": word, "corpus": "Generated", "freq": gen_counter[word] / gen_total})

    df = pd.DataFrame(records)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="word", y="freq", hue="corpus")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Normalized frequency (top words)")
    plt.xlabel("")
    plt.title(f"Top {top_k} Word Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pos_distribution(
    orig_pos: Counter,
    gen_pos: Counter,
    output_path: Path,
):
    tags = sorted(set(orig_pos.keys()) | set(gen_pos.keys()))
    if not tags:
        return
    records = []
    orig_total = sum(orig_pos.values()) or 1
    gen_total = sum(gen_pos.values()) or 1
    for tag in tags:
        records.append({"pos": tag, "corpus": "Original", "freq": orig_pos[tag] / orig_total})
        records.append({"pos": tag, "corpus": "Generated", "freq": gen_pos[tag] / gen_total})
    df = pd.DataFrame(records)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="pos", y="freq", hue="corpus")
    plt.title("POS Tag Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized frequency")
    plt.xlabel("POS tag")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sentiment_distribution(
    orig_sentiments: Sequence[float],
    gen_sentiments: Sequence[float],
    output_path: Path,
    title: str = "Sentiment Polarity Distribution",
):
    if not orig_sentiments and not gen_sentiments:
        return
    plt.figure(figsize=(8, 5))
    sns.kdeplot(orig_sentiments, fill=True, alpha=0.3, label="Original")
    sns.kdeplot(gen_sentiments, fill=True, alpha=0.3, label="Generated")
    plt.xlabel("Sentiment score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare distributional properties of original vs generated texts.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/steering/tests/generated_training_texts__baseline.json",
        help="Path to baseline dataset JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/steering/tests/analysis_original_vs_generated",
        help="Where to store plots and summary JSON.",
    )
    parser.add_argument(
        "--ner-model",
        type=str,
        default="dslim/bert-base-NER",
        help="Hugging Face model id for NER (set --disable-ner to skip).",
    )
    parser.add_argument(
        "--disable-ner",
        action="store_true",
        help="Skip NER computation if you only want faster lexical stats.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on number of documents (useful for quick checks).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    sns.set_theme(style="whitegrid")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Expected dataset to be a list of dicts.")

    original_texts = []
    generated_texts = []
    for doc in dataset:
        # Gracefully handle missing keys
        if "original_article" in doc:
            original_texts.append(doc["original_article"])
        if "generated_text" in doc:
            generated_texts.append(doc["generated_text"])

    ner = init_ner_pipeline(args.ner_model, disable=args.disable_ner)

    orig_stats = compute_corpus_stats(original_texts, ner_pipeline=ner, max_texts=args.max_docs)
    gen_stats = compute_corpus_stats(generated_texts, ner_pipeline=ner, max_texts=args.max_docs)

    js, kl_pq, kl_qp, vocab = compute_word_divergences(orig_stats.word_counter, gen_stats.word_counter, top_n=100)

    # Function words
    function_vocab = list(set(orig_stats.function_word_counter.keys()) | set(gen_stats.function_word_counter.keys()))
    func_js, func_kl_pq, func_kl_qp = compute_function_word_divergence(
        orig_stats.function_word_counter, gen_stats.function_word_counter, function_vocab
    )

    # Sentiment divergence
    sentiment_bins = np.linspace(-1, 1, 21)
    sentiment_js = binned_js_divergence(orig_stats.sentiment_scores, gen_stats.sentiment_scores, sentiment_bins)
    subjectivity_bins = np.linspace(0, 1, 21)
    subjectivity_js = binned_js_divergence(orig_stats.subjectivity_scores, gen_stats.subjectivity_scores, subjectivity_bins)

    # POS / tense / pronoun divergences
    pos_tags = sorted(set(orig_stats.pos_counter.keys()) | set(gen_stats.pos_counter.keys()))
    pos_js = counter_js(orig_stats.pos_counter, gen_stats.pos_counter, pos_tags) if pos_tags else 0.0
    pronoun_labels = ["first", "second", "third"]
    pronoun_js = counter_js(orig_stats.pronoun_person_counter, gen_stats.pronoun_person_counter, pronoun_labels)
    tense_labels = ["past", "present", "progressive", "other"]
    tense_js = counter_js(orig_stats.tense_counter, gen_stats.tense_counter, tense_labels)

    summary = {
        "n_docs_used": min(len(dataset), args.max_docs) if args.max_docs else len(dataset),
        "original": {
            "mean_sentence_length": orig_stats.mean_sentence_length,
            "mean_token_length": orig_stats.mean_token_length,
            "punctuation_proportion": orig_stats.punctuation_proportion,
            "type_token_ratio": orig_stats.type_token_ratio,
            "named_entity_proportion": orig_stats.named_entity_proportion,
            "stopword_proportion": orig_stats.stopword_proportion,
            "capitalization_rate": orig_stats.capitalization_rate,
            "punctuation_rate_variance": orig_stats.punctuation_rate_variance,
            "quote_rate": orig_stats.quote_tokens / orig_stats.total_tokens if orig_stats.total_tokens else 0.0,
            "parenthesis_rate": orig_stats.parenthesis_tokens / orig_stats.total_tokens if orig_stats.total_tokens else 0.0,
            "ellipsis_rate": orig_stats.ellipsis_tokens / orig_stats.total_tokens if orig_stats.total_tokens else 0.0,
            "mtld": orig_stats.mtld,
            "mean_sentiment": orig_stats.mean_sentiment,
            "mean_subjectivity": orig_stats.mean_subjectivity,
            "pos_distribution": orig_stats.pos_counter,
            "tense_distribution": orig_stats.tense_counter,
            "pronoun_person_distribution": orig_stats.pronoun_person_counter,
        },
        "generated": {
            "mean_sentence_length": gen_stats.mean_sentence_length,
            "mean_token_length": gen_stats.mean_token_length,
            "punctuation_proportion": gen_stats.punctuation_proportion,
            "type_token_ratio": gen_stats.type_token_ratio,
            "named_entity_proportion": gen_stats.named_entity_proportion,
            "stopword_proportion": gen_stats.stopword_proportion,
            "capitalization_rate": gen_stats.capitalization_rate,
            "punctuation_rate_variance": gen_stats.punctuation_rate_variance,
            "quote_rate": gen_stats.quote_tokens / gen_stats.total_tokens if gen_stats.total_tokens else 0.0,
            "parenthesis_rate": gen_stats.parenthesis_tokens / gen_stats.total_tokens if gen_stats.total_tokens else 0.0,
            "ellipsis_rate": gen_stats.ellipsis_tokens / gen_stats.total_tokens if gen_stats.total_tokens else 0.0,
            "mtld": gen_stats.mtld,
            "mean_sentiment": gen_stats.mean_sentiment,
            "mean_subjectivity": gen_stats.mean_subjectivity,
            "pos_distribution": gen_stats.pos_counter,
            "tense_distribution": gen_stats.tense_counter,
            "pronoun_person_distribution": gen_stats.pronoun_person_counter,
        },
        "top_word_divergences": {
            "vocab_size": len(vocab),
            "jensen_shannon": js,
            "kl_original_to_generated": kl_pq,
            "kl_generated_to_original": kl_qp,
            "vocab_preview": vocab[:10],
        },
        "function_word_divergence": {
            "jensen_shannon": func_js,
            "kl_original_to_generated": func_kl_pq,
            "kl_generated_to_original": func_kl_qp,
            "vocab_size": len(function_vocab),
        },
        "sentiment_subjectivity": {
            "sentiment_js": sentiment_js,
            "subjectivity_js": subjectivity_js,
            "mean_sentiment_original": orig_stats.mean_sentiment,
            "mean_sentiment_generated": gen_stats.mean_sentiment,
            "mean_subjectivity_original": orig_stats.mean_subjectivity,
            "mean_subjectivity_generated": gen_stats.mean_subjectivity,
        },
        "pos_pronoun_tense_divergence": {
            "pos_js": pos_js,
            "pronoun_js": pronoun_js,
            "tense_js": tense_js,
        },
    }

    with open(output_dir / "distributional_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_sentence_lengths(
        orig_stats.sentence_lengths,
        gen_stats.sentence_lengths,
        output_dir / "sentence_length_distribution.png",
    )
    plot_token_level_metrics(
        orig_stats,
        gen_stats,
        output_dir / "token_level_metrics.png",
    )
    plot_top_words(
        vocab,
        orig_stats.word_counter,
        gen_stats.word_counter,
        output_dir / "top_word_distribution.png",
        top_k=20,
    )
    plot_pos_distribution(
        orig_stats.pos_counter,
        gen_stats.pos_counter,
        output_dir / "pos_distribution.png",
    )
    plot_sentiment_distribution(
        orig_stats.sentiment_scores,
        gen_stats.sentiment_scores,
        output_dir / "sentiment_polarity_distribution.png",
        title="Sentiment Polarity Distribution",
    )
    plot_sentiment_distribution(
        orig_stats.subjectivity_scores,
        gen_stats.subjectivity_scores,
        output_dir / "subjectivity_distribution.png",
        title="Subjectivity Distribution",
    )

    print(f"Saved summary and plots to: {output_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

