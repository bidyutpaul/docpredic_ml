"""
DocPredic Text Preprocessing
Clean and normalize symptom text descriptions.
"""
import re
import unicodedata
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from .config import SEED

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

_stop_words = set(stopwords.words("english"))
_medical_keep = {
    "not", "no", "nor", "neither", "never", "none",
    "few", "more", "most", "very", "much", "too",
    "above", "below", "up", "down", "off", "on",
    "against", "between", "into", "through", "during",
    "before", "after", "same", "different",
    "should", "could", "would", "might",
}
_stop_words -= _medical_keep

_lemmatizer = WordNetLemmatizer()

_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}

_MEDICAL_SYNONYMS = {
    "headech": "headache", "headach": "headache", "headack": "headache", "headake": "headache",
    "fevr": "fever", "feaver": "fever",
    "cofe": "cough", "koff": "cough",
    "dizy": "dizziness", "dizyness": "dizziness",
    "vometing": "vomiting", "vomited": "vomiting",
    "tummy": "stomach", "belly": "stomach", "tummy ache": "stomach pain",
    "head hurt": "headache", "throwing up": "vomiting",
    "pass out": "fainting", "blackout": "fainting",
    "out of breath": "breathing difficulty", "winded": "breathing difficulty",
    "can't breathe": "breathing difficulty",
    "heart racing": "heart palpitations", "heart pounding": "heart palpitations",
    "pins and needles": "tingling",
    "cant sleep": "sleep problem", "cant fall asleep": "sleep problem",
    "cant walk": "difficulty walking",
    "cant speak": "speech difficulty",
    "shaking": "tremors", "trembling": "tremors",
    "high temperature": "fever",
    "rigid": "muscle stiffness",
    "clumsy": "poor coordination",
}


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return text


def expand_contractions(text: str) -> str:
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def apply_medical_synonyms(text: str) -> str:
    text_lower = text.lower()
    for synonym, replacement in _MEDICAL_SYNONYMS.items():
        text_lower = re.sub(r"\b" + re.escape(synonym) + r"\b", replacement, text_lower)
    return text_lower


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = expand_contractions(text)
    text = apply_medical_synonyms(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stop_words]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    text = clean_text(text)
    text = tokenize_and_lemmatize(text, remove_stopwords=remove_stopwords)
    return text


NEGATION_TRIGGERS = frozenset([
    "no", "not", "without", "denies", "deny", "denied", "never",
    "neither", "nor", "none",
])


def mark_negation(text: str, scope: int = 4) -> str:
    """Append a `_NEG` suffix to words inside a negation scope.

    Rule: after a negation trigger (no/not/without/denies/...), the next
    `scope` content words are suffixed. The scope ends early at contrastive
    conjunctions (but/however/although/except/only) AND at clause punctuation
    (, ; . :), so "no joy in kids or food, guilt" marks only the words before
    the comma. Applied to RAW text (before cleaning) so triggers survive
    stopword removal.

    Example: "no fever but severe headache" -> "no fever_NEG but severe headache"

    Must be applied identically at training and inference time.
    """
    if not isinstance(text, str):
        return ""
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z'\s]", text)
    out = []
    remaining = 0
    for tok in tokens:
        low = tok.lower()
        if low in NEGATION_TRIGGERS or low.endswith("n't"):
            remaining = scope
            out.append(tok)
            continue
        if low in ("but", "however", "although", "though", "except", "only",
                   "rather", "instead"):
            remaining = 0
            out.append(tok)
            continue
        if re.fullmatch(r"[,;.:!?]", tok or ""):
            remaining = 0
            out.append(tok)
            continue
        if re.fullmatch(r"[A-Za-z']+", tok) and remaining > 0:
            if low not in ("a", "an", "the", "any", "of", "with", "and", "or",
                           "to", "in", "on", "for", "my", "i"):
                out.append(tok + "_NEG")
                remaining -= 1
            else:
                out.append(tok)
            continue
        out.append(tok)
    return " ".join(out)


def preprocess_text_negmarked(text: str) -> str:
    """Negation-marked variant of preprocess_text for the fused ML pipeline."""
    marked = mark_negation(text)
    # keep the _NEG marker through cleaning (underscore is stripped by
    # clean_text, so protect it first)
    marked = marked.replace("_NEG", " NEGTOKEN")
    cleaned = clean_text(marked)
    cleaned = re.sub(r"\bnegtoken\b", "NEG", cleaned)
    return tokenize_and_lemmatize(cleaned, remove_stopwords=True)


def preprocess_dataframe(df, text_col: str, remove_stopwords: bool = True):
    df = df.copy()
    df[text_col] = df[text_col].apply(lambda x: preprocess_text(x, remove_stopwords=remove_stopwords))
    empty_mask = df[text_col].str.strip() == ""
    if empty_mask.any():
        print(f"Warning: {empty_mask.sum()} empty texts after preprocessing")
        df = df[~empty_mask].reset_index(drop=True)
    return df
