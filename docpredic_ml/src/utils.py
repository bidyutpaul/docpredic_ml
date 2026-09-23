"""
DocPredic Utilities
Helper functions for the project.
"""
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any
import json
import hashlib
from datetime import datetime

from .config import SEED


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def update(self, epoch: int, **metrics):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append({"epoch": epoch, "value": value})

    def get_best(self, metric: str, mode: str = "max"):
        if metric not in self.metrics:
            return None
        values = self.metrics[metric]
        if mode == "max":
            best = max(values, key=lambda x: x["value"])
        else:
            best = min(values, key=lambda x: x["value"])
        return best

    def to_json(self, path: Path):
        save_json(self.metrics, path)
