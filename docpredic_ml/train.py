"""
DocPredic Unified Training Pipeline
Ingests MIMIC-IV Clinical EHR data + Knowledge Matrix.
Extracts high-fidelity TF-IDF features and trains calibrated medical specialty predictors.
Exports production artifacts for inference.
"""
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LABEL_DIR, METRICS_DIR,
    TEXT_COL, LABEL_COL, SEED
)
from src.data_loader import load_raw_data, load_combined_dataset, save_processed_data
from src.preprocessing import preprocess_text
from src.models import get_models, evaluate_model_performance


def train():
    np.random.seed(SEED)
    print("=" * 70)
    print("        DOCPREDIC ML: MIMIC-IV & CLINICAL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Ingest Data
    print("\n[Step 1/5] Loading Knowledge Base and MIMIC-IV ED Dataset...")
    raw_df = load_raw_data()
    df = load_combined_dataset(raw_df, n_samples_per_class=150, use_mimic=True, seed=SEED)

    # 2. Text Preprocessing
    print("\n[Step 2/5] Preprocessing clinical text...")
    df["clean_text"] = df[TEXT_COL].apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 2].reset_index(drop=True)
    save_processed_data(df[[TEXT_COL, LABEL_COL]], "processed.csv")

    # 3. Label Encoding & Splits
    print("\n[Step 3/5] Encoding department labels and splitting dataset...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL])
    num_classes = len(label_encoder.classes_)
    print(f"  Total departments ({num_classes}): {list(label_encoder.classes_)}")

    joblib.dump(label_encoder, LABEL_DIR / "label_encoder.joblib")
    print(f"  Saved label encoder to {LABEL_DIR / 'label_encoder.joblib'}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.18, random_state=SEED, stratify=y
    )
    print(f"  Training samples: {len(X_train_text)}, Test samples: {len(X_test_text)}")

    # 4. Feature Extraction
    print("\n[Step 4/5] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=10000,
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    joblib.dump(vectorizer, PROCESSED_DATA_DIR / "tfidf_vectorizer.joblib")
    print(f"  Vocabulary size: {X_train_vec.shape[1]} features")

    # 5. Model Training & Benchmarking
    print("\n[Step 5/5] Training and benchmarking classification models...")
    models = get_models(num_classes)
    best_model = None
    best_model_name = ""
    best_f1 = -1.0
    best_acc = 0.0
    results = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_vec, y_train)
        metrics = evaluate_model_performance(model, X_test_vec, y_test)
        results[name] = metrics
        print(f"    -> Acc: {metrics['accuracy']*100:5.2f}% | Macro F1: {metrics['f1_macro']*100:5.2f}% | Weighted F1: {metrics['f1_weighted']*100:5.2f}%")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_acc = metrics["accuracy"]
            best_model = model
            best_model_name = name

    print("\n" + "=" * 70)
    print(f"  BEST MODEL SELECTED: {best_model_name.upper()}")
    print(f"  Accuracy  : {best_acc * 100:.2f}%")
    print(f"  Macro F1  : {best_f1 * 100:.2f}%")
    print("=" * 70)

    # Save best model to artifacts
    model_save_path = MODELS_DIR / "best_model.joblib"
    compat_save_path = MODELS_DIR / "best_sklearn_model.joblib"
    joblib.dump(best_model, model_save_path)
    joblib.dump(best_model, compat_save_path)

    import json
    info = {
        "best_model": best_model_name,
        "accuracy": float(best_acc),
        "f1_macro": float(best_f1),
        "num_classes": int(num_classes),
        "classes": list(label_encoder.classes_),
    }
    with open(MODELS_DIR / "best_model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved best model artifacts to:\n  - {model_save_path}\n  - {compat_save_path}")
    return results


if __name__ == "__main__":
    train()
