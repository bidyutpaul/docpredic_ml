"""
DocPredic High-Accuracy Training Pipeline
=========================================
Trains the fused prediction system:

  P_final(d | text) = softmax( ( λ·log P_ML(d) + (1−λ)·log P_KB(d) ) / T )

* P_ML : soft-voting ensemble (LogReg + ComplementNB + calibrated LinearSVC)
         over stacked word + char TF-IDF features.
* P_KB : IDF-weighted Bernoulli Naive Bayes over the doctor knowledge matrix
         (Data.xlsx) with template pseudo-counts (KnowledgeScorer).
* λ    : fusion weight, grid-tuned on validation accuracy (then macro-F1).
* T    : temperature, fitted on validation NLL (see calibration.py).

Saves production artifacts alongside (never overwriting) the legacy ones.
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LABEL_DIR, METRICS_DIR,
    TEXT_COL, LABEL_COL, SEED,
)
from src.data_loader import load_raw_data
from src.training_text import generate_enriched_texts
from src.preprocessing import preprocess_text_negmarked as preprocess_fn
from src.models import build_vectorizers, stack_features, build_ensemble
from src.knowledge_scorer import KnowledgeScorer, extract_symptoms
from src.calibration import (
    find_optimal_temperature, temperature_scale,
    compute_calibration_metrics,
)
from src.challenge_set import CHALLENGE_SET

WEIGHT_GRID = [
    (0.5, 0.2, 0.3),
    (0.6, 0.15, 0.25),
    (0.4, 0.3, 0.3),
    (0.34, 0.33, 0.33),
    (0.7, 0.1, 0.2),
    (0.45, 0.25, 0.30),
]
LAMBDA_GRID = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]


def _kb_proba_matrix(scorer: KnowledgeScorer, texts, order) -> np.ndarray:
    """KB posterior rows aligned to label-encoder `order` of departments."""
    idx = [scorer.departments.index(d) for d in order]
    P = np.zeros((len(texts), len(order)))
    for i, t in enumerate(texts):
        post, _ = scorer.posterior(t)
        P[i] = post[idx]
    # renormalise (uniform prior rows already sum to 1; guard fp error)
    P = np.clip(P, 1e-12, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def _fused_log_proba(log_ml: np.ndarray, log_kb: np.ndarray,
                     lam: float) -> np.ndarray:
    return lam * log_ml + (1.0 - lam) * log_kb


def _kb_hit_mask(texts) -> np.ndarray:
    """True where the knowledge scorer matched >=1 asserted/denied symptom.

    Where False, the KB abstains (uniform posterior) and fusion backs off to
    pure ML (lam_eff = 1) so uninformative KB mass cannot dilute ML evidence.
    """
    mask = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        act, neg = extract_symptoms(t)
        mask[i] = bool(act or neg)
    return mask


def _fused_log_proba_backoff(log_ml: np.ndarray, log_kb: np.ndarray,
                             lam: float, hit_mask: np.ndarray) -> np.ndarray:
    lam_eff = np.where(hit_mask, lam, 1.0)[:, None]
    return lam_eff * log_ml + (1.0 - lam_eff) * log_kb


def train() -> dict:
    np.random.seed(SEED)
    print("=" * 70)
    print("   DOCPREDIC HIGH-ACCURACY FUSED TRAINING PIPELINE")
    print("=" * 70)

    # ---- 1. data ------------------------------------------------------
    print("\n[1/7] Loading knowledge matrix + building enriched corpus...")
    matrix_df = load_raw_data()
    synth = generate_enriched_texts(matrix_df, n_per_class=500, seed=SEED)
    print(f"  enriched synthetic: {len(synth)} rows")

    try:
        from src.mimic_loader import load_mimic_dataset
        mimic_df = load_mimic_dataset()
        print(f"  MIMIC-IV clinical : {len(mimic_df)} rows")
        df = pd.concat([synth, mimic_df], ignore_index=True)
    except Exception as e:
        print(f"  MIMIC-IV unavailable ({e}); synthetic only")
        df = synth
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

    print("[2/7] Preprocessing text...")
    df["clean_text"] = df[TEXT_COL].apply(preprocess_fn)
    df = df[df["clean_text"].str.len() > 2].reset_index(drop=True)
    print(f"  usable samples: {len(df)}")

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    classes = list(le.classes_)
    print(f"  departments ({len(classes)}): {classes}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        df["clean_text"], y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp)
    print(f"  train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # ---- 3. features --------------------------------------------------
    print("[3/7] Fitting word + char TF-IDF vectorizers...")
    word_vec, char_vec = build_vectorizers()
    Xtr_w = word_vec.fit_transform(X_train)
    Xtr_c = char_vec.fit_transform(X_train)
    from scipy.sparse import hstack
    Xtr = hstack([Xtr_w, Xtr_c], format="csr")
    Xva = stack_features(word_vec, char_vec, X_val)
    Xte = stack_features(word_vec, char_vec, X_test)
    print(f"  word feats: {Xtr_w.shape[1]}, char feats: {Xtr_c.shape[1]}, "
          f"total: {Xtr.shape[1]}")

    # ---- 4. ensemble + weight tuning ----------------------------------
    print("[4/7] Training ensemble members + tuning vote weights...")
    best_w, best_acc, best_model = None, -1.0, None
    for w in WEIGHT_GRID:
        m = build_ensemble(weights=w)
        m.fit(Xtr, y_train)
        acc = accuracy_score(y_val, m.predict(Xva))
        print(f"    weights={w} -> val acc {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc, best_w, best_model = acc, w, m
    print(f"  best weights: {best_w} (val acc {best_acc * 100:.2f}%)")
    ensemble = best_model

    # ---- 5. knowledge scorer + fusion tuning --------------------------
    print("[5/7] Fitting Bayesian knowledge scorer + tuning fusion lambda...")
    kb = KnowledgeScorer(alpha=0.5).fit(matrix_df)
    log_ml_va = np.log(np.clip(ensemble.predict_proba(Xva), 1e-12, 1.0))
    Pkb_va = _kb_proba_matrix(kb, X_val.tolist(), classes)
    log_kb_va = np.log(np.clip(Pkb_va, 1e-12, 1.0))

    ml_acc = accuracy_score(y_val, log_ml_va.argmax(1))
    kb_acc = accuracy_score(y_val, log_kb_va.argmax(1))
    print(f"    ML-only val acc: {ml_acc * 100:.2f}% | "
          f"KB-only val acc: {kb_acc * 100:.2f}%")
    va_texts = X_val.tolist()
    va_hit = _kb_hit_mask(va_texts)
    print(f"    KB matched symptoms in {va_hit.sum()}/{len(va_hit)} val texts")

    best_lam, best_key = 0.5, (-1.0, -1.0)
    for lam in LAMBDA_GRID:
        pred = _fused_log_proba_backoff(
            log_ml_va, log_kb_va, lam, va_hit).argmax(1)
        key = (accuracy_score(y_val, pred),
               f1_score(y_val, pred, average="macro", zero_division=0))
        if key > best_key:
            best_key, best_lam = key, lam
    print(f"  best lam={best_lam} "
          f"(val acc {best_key[0] * 100:.2f}%, macro-F1 {best_key[1] * 100:.2f}%)")

    print("[6/7] Fitting posterior temperature T on validation NLL...")
    fused_log_va = _fused_log_proba_backoff(log_ml_va, log_kb_va, best_lam, va_hit)
    T = float(find_optimal_temperature(fused_log_va, np.asarray(y_val)))
    print(f"  optimal T = {T:.3f}")

    # ---- 7. test evaluation -------------------------------------------
    print("[7/7] Final evaluation on held-out TEST split...")
    log_ml_te = np.log(np.clip(ensemble.predict_proba(Xte), 1e-12, 1.0))
    te_texts = X_test.tolist()
    Pkb_te = _kb_proba_matrix(kb, te_texts, classes)
    log_kb_te = np.log(np.clip(Pkb_te, 1e-12, 1.0))
    te_hit = _kb_hit_mask(te_texts)
    fused_log_te = _fused_log_proba_backoff(log_ml_te, log_kb_te, best_lam, te_hit)
    P_final = temperature_scale(fused_log_te, T)
    y_pred = P_final.argmax(1)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    top2 = np.mean([y_test[i] in np.argsort(P_final[i])[-2:]
                    for i in range(len(y_test))])
    ece = compute_calibration_metrics(P_final, np.asarray(y_test))[
        "expected_calibration_error"]
    nll = log_loss(y_test, P_final)
    print(f"  FUSED TEST  acc {acc * 100:.2f}% | macro-F1 {f1m * 100:.2f}% | "
          f"weighted-F1 {f1w * 100:.2f}% | top-2 {top2 * 100:.2f}% | "
          f"ECE {ece:.4f} | NLL {nll:.4f}")
    for name, P in [("ML-only", np.exp(log_ml_te)),
                    ("KB-only", np.exp(log_kb_te))]:
        a = accuracy_score(y_test, P.argmax(1))
        f = f1_score(y_test, P.argmax(1), average="macro", zero_division=0)
        print(f"  {name:8s} TEST  acc {a * 100:.2f}% | macro-F1 {f * 100:.2f}%")

    # challenge set
    ch_raw = [t for t, _ in CHALLENGE_SET]
    ch_texts = [preprocess_fn(t) for t in ch_raw]
    ch_true = [le.transform([d])[0] for _, d in CHALLENGE_SET]
    ch_hit = _kb_hit_mask(ch_raw)
    Pml_ch = np.clip(ensemble.predict_proba(
        stack_features(word_vec, char_vec, ch_texts)), 1e-12, 1.0)
    Pkb_ch = _kb_proba_matrix(kb, ch_raw, classes)
    Pch = temperature_scale(
        _fused_log_proba_backoff(np.log(Pml_ch),
                                 np.log(np.clip(Pkb_ch, 1e-12, 1.0)),
                                 best_lam, ch_hit), T)
    ch_pred = Pch.argmax(1)
    ch_acc = float(np.mean(ch_pred == np.asarray(ch_true)))
    ch_top2 = float(np.mean([ch_true[i] in np.argsort(Pch[i])[-2:]
                             for i in range(len(ch_true))]))
    print(f"  CHALLENGE-SET (n={len(ch_true)}): "
          f"top-1 {ch_acc * 100:.1f}% | top-2 {ch_top2 * 100:.1f}%")

    # ---- save artifacts (new names; legacy files untouched) -----------
    print("\nSaving fused-system artifacts...")
    joblib.dump(ensemble, MODELS_DIR / "fused_ensemble.joblib")
    joblib.dump(word_vec, PROCESSED_DATA_DIR / "tfidf_word_vectorizer.joblib")
    joblib.dump(char_vec, PROCESSED_DATA_DIR / "tfidf_char_vectorizer.joblib")
    joblib.dump(le, LABEL_DIR / "label_encoder.joblib")
    kb.save(MODELS_DIR / "knowledge_scorer.joblib")
    fusion = {"lambda": float(best_lam), "temperature": float(T),
              "vote_weights": list(best_w), "classes": classes,
              "kb_backoff_no_match": True, "negation_marked": True,
              "low_confidence_threshold": 0.35, "low_margin_threshold": 0.10}
    with open(MODELS_DIR / "fusion_params.json", "w") as f:
        json.dump(fusion, f, indent=2)
    metrics = {"test_accuracy": float(acc), "test_f1_macro": float(f1m),
               "test_f1_weighted": float(f1w), "test_top2_accuracy": float(top2),
               "test_ece": float(ece), "test_nll": float(nll),
               "val_ml_accuracy": float(ml_acc), "val_kb_accuracy": float(kb_acc),
               "val_fused_accuracy": float(best_key[0]),
               "val_fused_f1_macro": float(best_key[1]),
               "challenge_top1": ch_acc, "challenge_top2": ch_top2,
               "classes": classes}
    with open(METRICS_DIR / "fused_system_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved to {MODELS_DIR} (+ processed vectorizers, metrics)")
    print("=" * 70)
    return metrics


if __name__ == "__main__":
    train()
