"""
DocPredic Training Pipeline
Handles training for all model types.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from .config import (
    SEED, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    NUM_CLASSES, TEXT_COL, LABEL_COL, MODELS_DIR, PROCESSED_DATA_DIR
)
from .dataset import SymptomDataset, TfidfDataset, create_data_loaders


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            if "input_ids" in batch:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids, attention_mask)
            elif "features" in batch:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                logits = model(features)
            elif "text" in batch:
                continue
            else:
                continue
        else:
            continue

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                if "input_ids" in batch:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(input_ids, attention_mask)
                elif "features" in batch:
                    features = batch["features"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(features)
                elif "text" in batch:
                    continue
                else:
                    continue
            else:
                continue

            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: str = None,
    model_name: str = "model",
    save_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_state = None

    print(f"\nTraining {model_name} on {device}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        save_path = save_dir / f"{model_name}_best.pt"
        torch.save(best_model_state, save_path)
        print(f"  Best model saved to {save_path} (val_acc={best_val_acc:.4f})")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "model_name": model_name,
    }


def train_sklearn_model(model, X_train, y_train, X_val, y_val, model_name: str = "sklearn_model"):
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_acc = (train_pred == y_train).mean()
    val_acc = (val_pred == y_val).mean()

    print(f"  {model_name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ({elapsed:.1f}s)")

    save_path = MODELS_DIR / f"{model_name}.joblib"
    import joblib
    joblib.dump(model, save_path)
    print(f"  Model saved to {save_path}")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "time": elapsed,
    }
