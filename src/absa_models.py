"""Aspect-Based Sentiment Analysis (ABSA) models, evaluation, and baselines.

Includes:
1. Seed-guided aspect detection using canonical smartphone synonym dictionary
2. Sentence-level context extraction to avoid DeBERTa 512-token truncation
3. PyABSA APC (FAST-LSA-T-V2 on DeBERTa-v3-base) inference runner with batching
4. TF-IDF (word + char n-grams) + Logistic Regression 4-class baseline
5. Stratified grouped 5-fold cross-validation on gold labels with McNemar test & bootstrap CIs
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import sem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, KFold
import torch

from .utils import (
    ASPECT_SYNONYMS,
    CANONICAL_ASPECTS,
    MODELS_DIR,
    RESULTS_DIR,
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
    SENTIMENT_SCORE_MAP,
    _KEYWORD_TO_ASPECT,
    set_seed,
)

# ---------------------------------------------------------------------------
# Compatibility patch for transformers >= 5.0 with PyABSA
# ---------------------------------------------------------------------------
try:
    from transformers.models.deberta_v2.configuration_deberta_v2 import (
        DebertaV2Config,
    )
    if not hasattr(DebertaV2Config, "is_decoder"):
        DebertaV2Config.is_decoder = False
except ImportError:
    pass

ASP_BEG = "[B-ASP]"
ASP_END = "[E-ASP]"

# Label space for baseline (includes 'none' for absent aspects)
BASELINE_LABELS = ["none", "positive", "negative", "neutral"]
_MODEL_FILES = ("tfidf_word.joblib", "tfidf_char.joblib", "clf.joblib")


# ---------------------------------------------------------------------------
# Aspect Detection & Context Extraction
# ---------------------------------------------------------------------------
def detect_aspects(text: str) -> list[dict]:
    """Identify product aspect mentions using longest-match-first keyword search.

    Returns a list of dicts: {"aspect": str, "keyword": str, "start": int, "end": int}.
    Deduplicates multiple mentions of the same canonical aspect to the first hit.
    """
    text_lower = text.lower()
    found: list[dict] = []
    seen_spans: list[tuple[int, int]] = []

    for kw, aspect in sorted(_KEYWORD_TO_ASPECT.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            start, end = m.start(), m.end()
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            found.append({
                "aspect": aspect,
                "keyword": kw,
                "start": start,
                "end": end,
            })
            seen_spans.append((start, end))

    seen_aspects: set[str] = set()
    unique: list[dict] = []
    for hit in sorted(found, key=lambda x: x["start"]):
        if hit["aspect"] not in seen_aspects:
            unique.append(hit)
            seen_aspects.add(hit["aspect"])
    return unique


def inject_aspect_tags(text: str, aspect_hit: dict) -> str:
    """Wrap detected aspect keyword with PyABSA aspect markers [B-ASP] and [E-ASP]."""
    s, e = aspect_hit["start"], aspect_hit["end"]
    return text[:s] + ASP_BEG + text[s:e] + ASP_END + text[e:]


def _split_sentences(text: str) -> list[str]:
    """Tokenize review into sentences, with NLTK if available or regex fallback."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except (ImportError, LookupError):
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [p.strip() for p in parts if p.strip()]


def _extract_aspect_context(
    text: str,
    hit: dict,
    max_chars: int = 400,
) -> tuple[str, dict]:
    """Extract target sentence containing aspect plus +/- 1 neighbor.

    Guarantees that input fits within DeBERTa's 512-token limit without truncating tags.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text, hit

    kw_start, kw_end = hit["start"], hit["end"]
    char_offset = 0
    target_idx = 0

    for i, sent in enumerate(sentences):
        sent_start = text.find(sent, char_offset)
        if sent_start == -1:
            sent_start = char_offset
        sent_end = sent_start + len(sent)

        if sent_start <= kw_start < sent_end:
            target_idx = i
            break
        char_offset = sent_end

    lo = max(0, target_idx - 1)
    hi = min(len(sentences), target_idx + 2)
    context_text = " ".join(sentences[lo:hi])

    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    kw_text = text[kw_start:kw_end]
    new_start = context_text.lower().find(kw_text.lower())
    if new_start == -1:
        return text, hit
    new_end = new_start + len(kw_text)

    adjusted_hit = {**hit, "start": new_start, "end": new_end}
    return context_text, adjusted_hit


# ---------------------------------------------------------------------------
# PyABSA APC Inference Runner
# ---------------------------------------------------------------------------
_CLASSIFIER = None
_PYABSA_AVAILABLE = None
_HF_PIPELINE = None


def _load_classifier(checkpoint: str = "english", device: str = "auto") -> Any:
    """Load PyABSA APC FAST-LSA-T-V2 sentiment classifier with caching."""
    global _CLASSIFIER, _PYABSA_AVAILABLE
    if _PYABSA_AVAILABLE is False:
        return None
    if _CLASSIFIER is not None:
        return _CLASSIFIER

    try:
        from pyabsa import AspectPolarityClassification as APC
        auto_dev = device == "auto"
        _CLASSIFIER = APC.SentimentClassifier(checkpoint, auto_device=auto_dev)
        _PYABSA_AVAILABLE = True
        return _CLASSIFIER
    except Exception as e:
        warnings.warn(f"PyABSA failed to load ({e}). Using HuggingFace fallback if enabled.")
        _PYABSA_AVAILABLE = False
        return None


def _load_hf_fallback():
    """Load HuggingFace sentiment pipeline as fallback when PyABSA is unavailable."""
    global _HF_PIPELINE
    if _HF_PIPELINE is not None:
        return _HF_PIPELINE

    from transformers import pipeline
    device_id = 0 if torch.cuda.is_available() else -1
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _HF_PIPELINE = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device_id,
        torch_dtype=dtype,
        truncation=True,
        max_length=512,
        batch_size=64,
    )
    return _HF_PIPELINE


def _normalize_pyabsa_output(raw: dict, expected_n: int) -> list[dict]:
    """Convert PyABSA's dict-of-lists output into standard per-aspect dictionaries."""
    sentiments = raw.get("sentiment", [])
    confidences = raw.get("confidence", [])
    probs_list = raw.get("probs", [])

    n = len(sentiments)
    results: list[dict] = []
    for i in range(min(n, expected_n)):
        sent = sentiments[i]
        label = sent.lower() if isinstance(sent, str) else SENTIMENT_MAP.get(int(sent), "neutral")
        conf = float(confidences[i]) if i < len(confidences) else 0.0

        p_neg, p_neu, p_pos = 0.0, 0.0, 0.0
        if i < len(probs_list):
            p = probs_list[i]
            if isinstance(p, np.ndarray):
                p = p.astype(float).tolist()
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                p_neg, p_neu, p_pos = float(p[0]), float(p[1]), float(p[2])

        results.append({
            "label": label,
            "confidence": conf,
            "p_neg": p_neg,
            "p_neu": p_neu,
            "p_pos": p_pos,
        })

    for _ in range(max(0, expected_n - n)):
        results.append({
            "label": "neutral", "confidence": 0.0,
            "p_neg": 0.0, "p_neu": 0.0, "p_pos": 0.0,
        })

    return results


def predict_single(
    text: str,
    checkpoint: str = "english",
    allow_hf_fallback: bool = True,
) -> list[dict]:
    """Predict aspect sentiments for a single customer review."""
    hits = detect_aspects(text)
    if not hits:
        return []

    results: list[dict] = []
    classifier = _load_classifier(checkpoint)

    if classifier is not None:
        for hit in hits:
            context, adj_hit = _extract_aspect_context(text, hit)
            tagged = inject_aspect_tags(context, adj_hit)
            try:
                raw = classifier.predict(tagged, print_result=False)
                parsed = _normalize_pyabsa_output(raw, expected_n=1)[0]
                results.append({
                    "canonical_aspect": hit["aspect"],
                    "keyword": hit["keyword"],
                    "sentiment_label": parsed["label"],
                    "sentiment_score": parsed["p_pos"] - parsed["p_neg"],
                    "confidence": parsed["confidence"],
                    "p_neg": parsed["p_neg"],
                    "p_neu": parsed["p_neu"],
                    "p_pos": parsed["p_pos"],
                    "source_model": "pyabsa_apc",
                })
            except Exception:
                results.append({
                    "canonical_aspect": hit["aspect"],
                    "keyword": hit["keyword"],
                    "sentiment_label": "neutral",
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "p_neg": 0.0,
                    "p_neu": 0.0,
                    "p_pos": 0.0,
                    "source_model": "pyabsa_apc_error",
                })
    elif allow_hf_fallback:
        pipe = _load_hf_fallback()
        for hit in hits:
            window, _ = _extract_aspect_context(text, hit)
            pred = pipe(window)[0]
            lbl = pred["label"].lower()
            score = float(pred["score"])
            if lbl == "positive":
                p_neg, p_neu, p_pos = 0.0, 0.0, score
            elif lbl == "negative":
                p_neg, p_neu, p_pos = score, 0.0, 0.0
            else:
                p_neg, p_neu, p_pos = 0.0, score, 0.0

            results.append({
                "canonical_aspect": hit["aspect"],
                "keyword": hit["keyword"],
                "sentiment_label": lbl,
                "sentiment_score": p_pos - p_neg,
                "confidence": round(score, 4),
                "p_neg": p_neg,
                "p_neu": p_neu,
                "p_pos": p_pos,
                "source_model": "hf_sentiment_fallback",
            })
    return results


def predict_batch(
    texts: list[str],
    review_metadata: Optional[pd.DataFrame] = None,
    checkpoint: str = "english",
    infer_batch_size: int = 128,
    allow_hf_fallback: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run batched aspect sentiment prediction across a list of reviews."""
    from tqdm import tqdm

    classifier = _load_classifier(checkpoint)
    tagged_inputs: list[str] = []
    mapping: list[dict] = []

    iter_texts = tqdm(enumerate(texts), desc="Detecting aspects", total=len(texts)) if verbose else enumerate(texts)
    for i, text in iter_texts:
        hits = detect_aspects(text)
        for hit in hits:
            context, adj_hit = _extract_aspect_context(text, hit)
            tagged_inputs.append(inject_aspect_tags(context, adj_hit))
            mapping.append({"text_idx": i, "hit": hit})

    if not tagged_inputs:
        return pd.DataFrame()

    all_rows: list[dict] = []
    n_pairs = len(tagged_inputs)

    if classifier is not None:
        batch_iter = range(0, n_pairs, infer_batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, desc="PyABSA inference", total=(n_pairs + infer_batch_size - 1) // infer_batch_size)

        with torch.inference_mode():
            for start in batch_iter:
                end = min(start + infer_batch_size, n_pairs)
                batch_tagged = tagged_inputs[start:end]
                batch_mapping = mapping[start:end]

                try:
                    raw = classifier.predict(batch_tagged, print_result=False)
                    flat_preds: list[dict] = []
                    if isinstance(raw, list):
                        for r in raw:
                            n_asp = len(r.get("sentiment", []))
                            flat_preds.extend(_normalize_pyabsa_output(r, expected_n=n_asp))
                    else:
                        flat_preds = _normalize_pyabsa_output(raw, expected_n=len(batch_tagged))
                except Exception:
                    flat_preds = [{"label": "neutral", "confidence": 0.0, "p_neg": 0.0, "p_neu": 0.0, "p_pos": 0.0}] * len(batch_tagged)

                while len(flat_preds) < len(batch_mapping):
                    flat_preds.append({"label": "neutral", "confidence": 0.0, "p_neg": 0.0, "p_neu": 0.0, "p_pos": 0.0})

                for p, meta in zip(flat_preds[:len(batch_mapping)], batch_mapping):
                    is_err = p["confidence"] == 0.0 and p["label"] == "neutral"
                    row = {
                        "canonical_aspect": meta["hit"]["aspect"],
                        "keyword": meta["hit"]["keyword"],
                        "sentiment_label": p["label"],
                        "sentiment_score": p["p_pos"] - p["p_neg"],
                        "confidence": p["confidence"],
                        "p_neg": p["p_neg"],
                        "p_neu": p["p_neu"],
                        "p_pos": p["p_pos"],
                        "source_model": "pyabsa_apc_error" if is_err else "pyabsa_apc",
                    }
                    if review_metadata is not None:
                        meta_row = review_metadata.iloc[meta["text_idx"]]
                        for col in review_metadata.columns:
                            row[col] = meta_row[col]
                    all_rows.append(row)
    elif allow_hf_fallback:
        pipe = _load_hf_fallback()
        batch_iter = range(0, n_pairs, infer_batch_size)
        for start in batch_iter:
            end = min(start + infer_batch_size, n_pairs)
            batch_tagged = tagged_inputs[start:end]
            batch_mapping = mapping[start:end]
            preds = pipe(batch_tagged)
            for p, meta in zip(preds, batch_mapping):
                lbl = p["label"].lower()
                sc = float(p["score"])
                row = {
                    "canonical_aspect": meta["hit"]["aspect"],
                    "keyword": meta["hit"]["keyword"],
                    "sentiment_label": lbl,
                    "sentiment_score": sc if lbl == "positive" else (-sc if lbl == "negative" else 0.0),
                    "confidence": round(sc, 4),
                    "p_neg": sc if lbl == "negative" else 0.0,
                    "p_neu": sc if lbl == "neutral" else 0.0,
                    "p_pos": sc if lbl == "positive" else 0.0,
                    "source_model": "hf_sentiment_fallback",
                }
                if review_metadata is not None:
                    meta_row = review_metadata.iloc[meta["text_idx"]]
                    for col in review_metadata.columns:
                        row[col] = meta_row[col]
                all_rows.append(row)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression Baseline
# ---------------------------------------------------------------------------
def _sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pair_dataset(
    texts: list[str],
    gold_labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create full review x 9-aspect pairs for the baseline classifier.

    If gold_labels is provided, assigns actual polarity labels ('positive', 'negative', 'neutral')
    or 'none' if the aspect was absent in that review.
    """
    rows = []
    gold_lookup = {}
    if gold_labels is not None:
        for _, row in gold_labels.iterrows():
            gold_lookup[(int(row["review_idx"]), row["canonical_aspect"])] = row["sentiment_label"]

    for i, text in enumerate(texts):
        hits = {h["aspect"] for h in detect_aspects(text)} if gold_labels is None else set()
        for asp in CANONICAL_ASPECTS:
            if gold_labels is not None:
                lbl = gold_lookup.get((i, asp), "none")
            else:
                lbl = "neutral" if asp in hits else "none"
            rows.append({
                "review_idx": i,
                "review_text": text,
                "canonical_aspect": asp,
                "label": lbl,
            })
    return pd.DataFrame(rows)


class TfidfBaseline:
    """TF-IDF word & char n-grams + one-hot aspect indicator -> Logistic Regression baseline."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.tfidf_word = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=5000,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )
        self._is_fitted = False

    def _build_features(self, texts: list[str], aspects: list[str], fit: bool = False) -> csr_matrix:
        if fit:
            X_word = self.tfidf_word.fit_transform(texts)
            X_char = self.tfidf_char.fit_transform(texts)
        else:
            X_word = self.tfidf_word.transform(texts)
            X_char = self.tfidf_char.transform(texts)

        # One-hot encode canonical aspect (9 dims)
        aspect_mat = np.zeros((len(aspects), len(CANONICAL_ASPECTS)), dtype=np.float32)
        asp_to_idx = {a: i for i, a in enumerate(CANONICAL_ASPECTS)}
        for i, a in enumerate(aspects):
            if a in asp_to_idx:
                aspect_mat[i, asp_to_idx[a]] = 1.0
        X_aspect = csr_matrix(aspect_mat)

        return hstack([X_word, X_char, X_aspect]).tocsr()

    def fit(self, texts: list[str], aspects: list[str], labels: list[str]) -> TfidfBaseline:
        X = self._build_features(texts, aspects, fit=True)
        self.clf.fit(X, labels)
        self._is_fitted = True
        return self

    def predict(self, texts: list[str], aspects: list[str]) -> list[dict]:
        assert self._is_fitted, "Model must be fitted before predict"
        X = self._build_features(texts, aspects, fit=False)
        y_pred = self.clf.predict(X)
        y_proba = self.clf.predict_proba(X)
        score_map = {"none": 0.0, "positive": 1.0, "negative": -1.0, "neutral": 0.0}
        class_to_idx = {cls: j for j, cls in enumerate(self.clf.classes_)}

        results = []
        for i, (asp, lbl) in enumerate(zip(aspects, y_pred)):
            proba = y_proba[i]
            p_neg = float(proba[class_to_idx["negative"]]) if "negative" in class_to_idx else 0.0
            p_neu = float(proba[class_to_idx["neutral"]]) if "neutral" in class_to_idx else 0.0
            p_pos = float(proba[class_to_idx["positive"]]) if "positive" in class_to_idx else 0.0
            results.append({
                "canonical_aspect": asp,
                "sentiment_label": lbl,
                "sentiment_score": score_map.get(lbl, 0.0),
                "confidence": float(proba.max()),
                "p_neg": p_neg,
                "p_neu": p_neu,
                "p_pos": p_pos,
                "source_model": "tfidf_lr",
            })
        return results

    def evaluate(self, texts: list[str], aspects: list[str], labels: list[str], exclude_none: bool = True) -> dict:
        X = self._build_features(texts, aspects, fit=False)
        y_pred = self.clf.predict(X)
        y_true = np.array(labels)

        if exclude_none:
            mask = y_true != "none"
            y_true_eval = y_true[mask]
            y_pred_eval = y_pred[mask]
        else:
            y_true_eval = y_true
            y_pred_eval = y_pred

        macro_f1 = f1_score(y_true_eval, y_pred_eval, labels=SENTIMENT_LABELS, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_eval, y_pred_eval, labels=SENTIMENT_LABELS, average="weighted", zero_division=0)

        per_aspect_f1 = {}
        aspects_arr = np.array(aspects)
        for asp in CANONICAL_ASPECTS:
            asp_mask = (aspects_arr == asp) & (y_true != "none") if exclude_none else (aspects_arr == asp)
            if asp_mask.sum() == 0:
                per_aspect_f1[asp] = float("nan")
            else:
                per_aspect_f1[asp] = round(float(f1_score(
                    y_true[asp_mask], y_pred[asp_mask],
                    labels=SENTIMENT_LABELS, average="macro", zero_division=0,
                )), 4)

        return {
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_aspect_f1": per_aspect_f1,
            "report": classification_report(y_true_eval, y_pred_eval, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true_eval, y_pred_eval, labels=SENTIMENT_LABELS).tolist(),
        }

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tfidf_word, model_dir / "tfidf_word.joblib")
        joblib.dump(self.tfidf_char, model_dir / "tfidf_char.joblib")
        joblib.dump(self.clf, model_dir / "clf.joblib")
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "aspects": CANONICAL_ASPECTS,
                "labels": BASELINE_LABELS,
                "sha256": {name: _sha256(model_dir / name) for name in _MODEL_FILES},
            }, f, indent=2)

    def load(self, model_dir: Path) -> TfidfBaseline:
        self.tfidf_word = joblib.load(model_dir / "tfidf_word.joblib")
        self.tfidf_char = joblib.load(model_dir / "tfidf_char.joblib")
        self.clf = joblib.load(model_dir / "clf.joblib")
        self._is_fitted = True
        return self


# ---------------------------------------------------------------------------
# Stratified Grouped 5-Fold Cross-Validation on Gold Labels
# ---------------------------------------------------------------------------
def evaluate_absa_5fold(
    gold_df: pd.DataFrame,
    n_splits: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Perform 5-fold cross-validation on gold labels comparing PyABSA vs. TF-IDF baseline.

    Uses grouped K-Fold on unique reviews (or parent_asins) to prevent data leakage.
    Computes McNemar test statistic and 95% bootstrap confidence intervals.
    """
    set_seed(seed)
    df = gold_df.copy().reset_index(drop=True)

    # Required columns: review_text, canonical_aspect, sentiment_label
    # Review grouping column: review_pk or review_idx or review_text
    group_col = "review_pk" if "review_pk" in df.columns else ("review_id" if "review_id" in df.columns else "parent_asin")
    if group_col not in df.columns:
        df["_grp"] = df["review_text"].factorize()[0]
        group_col = "_grp"

    groups = df[group_col].values
    unique_groups = np.unique(groups)
    gkf = GroupKFold(n_splits=min(n_splits, len(unique_groups)))

    y_true_all: list[str] = []
    dl_preds_all: list[str] = []
    lr_preds_all: list[str] = []
    group_keys_all: list[str] = []

    # Prepare PyABSA predictions upfront (zero-shot inference on gold set)
    dl_cache = {}
    for idx, row in df.iterrows():
        text = str(row["review_text"])
        asp = str(row["canonical_aspect"])
        key = (text, asp)
        if key not in dl_cache:
            preds = predict_single(text)
            matched = next((p for p in preds if p["canonical_aspect"] == asp), None)
            dl_cache[key] = matched["sentiment_label"] if matched else "neutral"

    # Cross-validation loop for TF-IDF baseline
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Train baseline on train fold
        texts_train = train_df["review_text"].tolist()
        aspects_train = train_df["canonical_aspect"].tolist()
        labels_train = train_df["sentiment_label"].tolist()

        baseline = TfidfBaseline()
        baseline.fit(texts_train, aspects_train, labels_train)

        # Predict test fold
        texts_test = test_df["review_text"].tolist()
        aspects_test = test_df["canonical_aspect"].tolist()
        labels_test = test_df["sentiment_label"].tolist()
        lr_preds = [p["sentiment_label"] for p in baseline.predict(texts_test, aspects_test)]

        for i, row in test_df.reset_index(drop=True).iterrows():
            yt = row["sentiment_label"]
            y_true_all.append(yt)
            dl_preds_all.append(dl_cache.get((row["review_text"], row["canonical_aspect"]), "neutral"))
            lr_preds_all.append(lr_preds[i])
            group_keys_all.append(str(row[group_col]))

    # Compute overall Macro and Weighted F1 scores
    dl_macro = float(f1_score(y_true_all, dl_preds_all, labels=SENTIMENT_LABELS, average="macro", zero_division=0))
    dl_weighted = float(f1_score(y_true_all, dl_preds_all, labels=SENTIMENT_LABELS, average="weighted", zero_division=0))
    lr_macro = float(f1_score(y_true_all, lr_preds_all, labels=SENTIMENT_LABELS, average="macro", zero_division=0))
    lr_weighted = float(f1_score(y_true_all, lr_preds_all, labels=SENTIMENT_LABELS, average="weighted", zero_division=0))

    # McNemar test
    dl_correct = np.array(dl_preds_all) == np.array(y_true_all)
    lr_correct = np.array(lr_preds_all) == np.array(y_true_all)
    b = int(np.sum(dl_correct & ~lr_correct))  # DL correct, LR wrong
    c = int(np.sum(~dl_correct & lr_correct))  # LR correct, DL wrong

    # Continuity corrected McNemar Chi2
    mcnemar_stat = (abs(b - c) - 1.0) ** 2 / (b + c) if (b + c) > 0 else 0.0
    from scipy.stats import chi2
    mcnemar_p = float(1.0 - chi2.cdf(mcnemar_stat, df=1))

    # Bootstrap 95% Confidence Intervals (grouped by review)
    rng = np.random.default_rng(seed)
    unique_pks = list(set(group_keys_all))
    pk_to_indices: dict[str, list[int]] = {}
    for i, pk in enumerate(group_keys_all):
        pk_to_indices.setdefault(pk, []).append(i)

    boot_dl_m, boot_lr_m = [], []
    for _ in range(n_bootstrap):
        sampled_pks = rng.choice(unique_pks, size=len(unique_pks), replace=True)
        idx = [i for pk in sampled_pks for i in pk_to_indices[pk]]
        yt = [y_true_all[i] for i in idx]
        dp = [dl_preds_all[i] for i in idx]
        lp = [lr_preds_all[i] for i in idx]
        boot_dl_m.append(f1_score(yt, dp, labels=SENTIMENT_LABELS, average="macro", zero_division=0))
        boot_lr_m.append(f1_score(yt, lp, labels=SENTIMENT_LABELS, average="macro", zero_division=0))

    ci_dl = [round(float(np.percentile(boot_dl_m, 2.5)), 4), round(float(np.percentile(boot_dl_m, 97.5)), 4)]
    ci_lr = [round(float(np.percentile(boot_lr_m, 2.5)), 4), round(float(np.percentile(boot_lr_m, 97.5)), 4)]

    # Per aspect breakdown
    aspects_arr = df["canonical_aspect"].values
    per_aspect_dl, per_aspect_lr = {}, {}
    for asp in CANONICAL_ASPECTS:
        mask = (aspects_arr == asp)
        if mask.sum() > 0:
            sub_y = [y_true_all[i] for i in range(len(y_true_all)) if mask[i]]
            sub_dl = [dl_preds_all[i] for i in range(len(dl_preds_all)) if mask[i]]
            sub_lr = [lr_preds_all[i] for i in range(len(lr_preds_all)) if mask[i]]
            per_aspect_dl[asp] = round(float(f1_score(sub_y, sub_dl, labels=SENTIMENT_LABELS, average="macro", zero_division=0)), 4)
            per_aspect_lr[asp] = round(float(f1_score(sub_y, sub_lr, labels=SENTIMENT_LABELS, average="macro", zero_division=0)), 4)

    return {
        "n_samples": len(y_true_all),
        "n_folds": n_splits,
        "dl_macro_f1": round(dl_macro, 4),
        "dl_weighted_f1": round(dl_weighted, 4),
        "dl_macro_ci_95": ci_dl,
        "lr_macro_f1": round(lr_macro, 4),
        "lr_weighted_f1": round(lr_weighted, 4),
        "lr_macro_ci_95": ci_lr,
        "mcnemar_stat": round(mcnemar_stat, 4),
        "mcnemar_p": round(mcnemar_p, 6),
        "b_dl_only": b,
        "c_lr_only": c,
        "per_aspect_dl": per_aspect_dl,
        "per_aspect_lr": per_aspect_lr,
        "winner": "PyABSA (FAST-LSA-T-V2)" if dl_macro > lr_macro else "TF-IDF + LR",
    }
