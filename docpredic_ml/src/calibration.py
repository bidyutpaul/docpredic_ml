"""
DocPredic Model Calibration
Ensures prediction probabilities are well-calibrated.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F


def calibrate_sklearn_model(model, X_val, y_val, method: str = "isotonic", cv: int = 3):
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X_val, y_val)
    return calibrated


def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    from sklearn.metrics import brier_score_loss

    max_probs = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_data = []

    for i in range(n_bins):
        mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_confidence = max_probs[mask].mean()
            bin_accuracy = correct[mask].mean()
            bin_count = mask.sum()
            calibration_data.append({
                "bin": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                "confidence": float(bin_confidence),
                "accuracy": float(bin_accuracy),
                "count": int(bin_count),
                "gap": float(abs(bin_confidence - bin_accuracy)),
            })

    ece = 0.0
    total_samples = len(labels)
    for data in calibration_data:
        weight = data["count"] / total_samples
        ece += weight * data["gap"]

    return {
        "expected_calibration_error": float(ece),
        "num_bins": n_bins,
        "bin_data": calibration_data,
    }


def temperature_scale(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll_loss(temp):
        probs = temperature_scale(logits, temp)
        nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
        return nll

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    return result.x


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return F.softmax(logits / temperature, dim=-1)
