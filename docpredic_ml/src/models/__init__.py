"""
DocPredic Model Architectures and Evaluators
"""
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib


def get_models(num_classes: int, class_weight: Optional[str] = "balanced") -> Dict[str, Any]:
    """Return dictionary of classification models optimized for medical specialty prediction."""
    return {
        "logistic_regression": LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
        ),
        "calibrated_linear_svc": CalibratedClassifierCV(
            LinearSVC(
                C=1.5,
                max_iter=2000,
                class_weight=class_weight,
                random_state=42,
            ),
            cv=3,
        ),
        "mlp_neural_net": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            early_stopping=True,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=4,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model_performance(model, X_test, y_test) -> Dict[str, float]:
    """Compute accuracy, macro F1, and weighted F1 on test split."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
    }


# ---------------------------------------------------------------------------
# High-accuracy pipeline: word + char TF-IDF stacked features and a soft
# voting ensemble (Logistic Regression + ComplementNB + calibrated LinearSVC).
# ---------------------------------------------------------------------------

def build_vectorizers(max_word_features: int = 14000,
                      max_char_features: int = 5000
                      ) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.9,
        max_features=max_word_features,
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_features=max_char_features,
    )
    return word_vec, char_vec


def stack_features(word_vec: TfidfVectorizer,
                   char_vec: TfidfVectorizer,
                   texts) -> csr_matrix:
    return hstack([word_vec.transform(texts), char_vec.transform(texts)],
                  format="csr")


def build_ensemble(weights: Tuple[float, float, float] = (0.5, 0.2, 0.3)
                   ) -> VotingClassifier:
    """Soft-voting ensemble tuned for short clinical utterances.

    - LogisticRegression: strong calibrated linear baseline on TF-IDF.
    - ComplementNB: excels on imbalanced text with many classes.
    - Calibrated LinearSVC: max-margin separation, sigmoid-calibrated.
    """
    lr = LogisticRegression(
        C=4.0, max_iter=2000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )
    cnb = ComplementNB(alpha=0.3)
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=3000, class_weight="balanced",
                  random_state=42),
        cv=3,
    )
    return VotingClassifier(
        estimators=[("lr", lr), ("cnb", cnb), ("svc", svc)],
        voting="soft",
        weights=list(weights),
    )
