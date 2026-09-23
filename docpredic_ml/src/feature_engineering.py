"""
DocPredic Feature Engineering
TF-IDF and other feature extraction methods.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib

from .config import PROCESSED_DATA_DIR


def build_tfidf_features(
    texts: pd.Series,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    fit_on_train: bool = True,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[np.ndarray, Optional[TfidfVectorizer]]:
    if fit_on_train or vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )
        features = vectorizer.fit_transform(texts)
    else:
        features = vectorizer.transform(texts)

    return features, vectorizer


def build_char_tfidf_features(
    texts: pd.Series,
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (2, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    fit_on_train: bool = True,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[np.ndarray, Optional[TfidfVectorizer]]:
    if fit_on_train or vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            analyzer="char_wb",
            sublinear_tf=True,
        )
        features = vectorizer.fit_transform(texts)
    else:
        features = vectorizer.transform(texts)

    return features, vectorizer


def build_combined_features(
    texts: pd.Series,
    fit_on_train: bool = True,
    word_vec: Optional[TfidfVectorizer] = None,
    char_vec: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer, TfidfVectorizer]:
    word_features, word_vec = build_char_tfidf_features(
        texts, fit_on_train=fit_on_train, vectorizer=word_vec
    )
    char_features, char_vec = build_char_tfidf_features(
        texts, fit_on_train=fit_on_train, vectorizer=char_vec
    )
    combined = hstack([word_features, char_features])
    return combined, word_vec, char_vec


def extract_statistical_features(texts: pd.Series) -> np.ndarray:
    features = pd.DataFrame()
    features["text_length"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["avg_word_length"] = texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
    )
    features["unique_word_ratio"] = texts.apply(
        lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
    )
    features["digit_count"] = texts.str.count(r"\d")
    features["special_char_count"] = texts.str.count(r"[^a-zA-Z0-9\s]")
    return features.values


def save_vectorizer(vectorizer, name: str) -> str:
    path = PROCESSED_DATA_DIR / f"{name}.joblib"
    joblib.dump(vectorizer, path)
    return str(path)


def load_vectorizer(name: str):
    path = PROCESSED_DATA_DIR / f"{name}.joblib"
    return joblib.load(path)
