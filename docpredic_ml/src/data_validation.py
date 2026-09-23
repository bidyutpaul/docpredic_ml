"""
DocPredic Data Validation
Comprehensive dataset audit and validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from collections import Counter

from .config import TEXT_COL, LABEL_COL


def compute_dataset_audit(df: pd.DataFrame, text_col: str = TEXT_COL, label_col: str = LABEL_COL) -> Dict[str, Any]:
    audit = {}
    audit["total_rows"] = len(df)
    audit["total_columns"] = len(df.columns)
    audit["columns"] = list(df.columns)
    audit["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    null_counts = df.isnull().sum().to_dict()
    audit["null_counts"] = null_counts
    audit["total_nulls"] = sum(null_counts.values())

    audit["duplicate_rows"] = int(df.duplicated().sum())

    if text_col in df.columns:
        audit["null_text"] = int(df[text_col].isnull().sum())
        empty_text = (df[text_col].astype(str).str.strip() == "").sum()
        audit["empty_text_rows"] = int(empty_text)
        lengths = df[text_col].astype(str).str.len()
        audit["text_length_stats"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "std": float(lengths.std()),
        }
    else:
        audit["null_text"] = 0
        audit["empty_text_rows"] = 0
        audit["text_length_stats"] = {}

    if label_col in df.columns:
        audit["unique_labels"] = int(df[label_col].nunique())
        audit["label_distribution"] = df[label_col].value_counts().to_dict()
        class_counts = df[label_col].value_counts()
        audit["min_class_size"] = int(class_counts.min())
        audit["max_class_size"] = int(class_counts.max())
        audit["class_imbalance_ratio"] = float(class_counts.max() / class_counts.min()) if class_counts.min() > 0 else float("inf")

        label_counts = Counter(df[label_col])
        total = len(df)
        entropy = -sum((c / total) * np.log2(c / total) for c in label_counts.values())
        max_entropy = np.log2(len(label_counts)) if len(label_counts) > 1 else 1
        audit["label_entropy"] = float(entropy)
        audit["label_normalized_entropy"] = float(entropy / max_entropy) if max_entropy > 0 else 0
    else:
        audit["unique_labels"] = 0
        audit["label_distribution"] = {}

    if text_col in df.columns and label_col in df.columns:
        dup_texts = df.groupby(text_col)[label_col].nunique()
        audit["texts_with_multiple_labels"] = int((dup_texts > 1).sum())

        text_dups = df[text_col].value_counts()
        audit["duplicate_text_count"] = int((text_dups > 1).sum())
        audit["unique_texts"] = int(df[text_col].nunique())

    return audit


def print_audit_report(audit: Dict[str, Any]) -> None:
    print("=" * 70)
    print("              DATASET AUDIT REPORT")
    print("=" * 70)

    print(f"\n1. DATASET SHAPE")
    print(f"   Rows: {audit['total_rows']}")
    print(f"   Columns: {audit['total_columns']}")

    print(f"\n2. COLUMN NAMES")
    for i, col in enumerate(audit["columns"]):
        print(f"   [{i}] {col} ({audit['dtypes'][col]})")

    print(f"\n3. MISSING VALUES")
    print(f"   Total nulls: {audit['total_nulls']}")
    for col, count in audit["null_counts"].items():
        if count > 0:
            print(f"   {col}: {count} ({count/audit['total_rows']*100:.1f}%)")

    print(f"\n4. DUPLICATE ROWS: {audit['duplicate_rows']}")

    if "unique_texts" in audit:
        print(f"\n5. TEXT ANALYSIS")
        print(f"   Unique texts: {audit['unique_texts']}")
        print(f"   Null text rows: {audit['null_text']}")
        print(f"   Empty text rows: {audit['empty_text_rows']}")
        print(f"   Duplicate text entries: {audit['duplicate_text_count']}")
        print(f"   Texts mapped to multiple labels: {audit['texts_with_multiple_labels']}")
        if audit["text_length_stats"]:
            stats = audit["text_length_stats"]
            print(f"   Text length (chars): mean={stats['mean']:.1f}, median={stats['median']:.1f}, min={stats['min']}, max={stats['max']}")
            print(f"   Text length std: {stats['std']:.1f}")

    print(f"\n6. LABEL ANALYSIS")
    print(f"   Unique departments: {audit['unique_labels']}")
    print(f"   Min class size: {audit['min_class_size']}")
    print(f"   Max class size: {audit['max_class_size']}")
    print(f"   Imbalance ratio: {audit['class_imbalance_ratio']:.2f}")
    print(f"   Label entropy: {audit['label_entropy']:.3f}")
    print(f"   Normalized entropy: {audit['label_normalized_entropy']:.3f}")

    print(f"\n7. CLASS DISTRIBUTION")
    for label, count in sorted(audit["label_distribution"].items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(count / audit["max_class_size"] * 30)
        print(f"   {label:25s} {count:5d} {bar}")

    print(f"\n8. QUALITY ASSESSMENT")
    issues = []
    if audit["total_rows"] < 100:
        issues.append("CRITICAL: Dataset is very small (< 100 rows)")
    if audit["class_imbalance_ratio"] > 10:
        issues.append("WARNING: Severe class imbalance (ratio > 10)")
    if audit["total_nulls"] > 0:
        issues.append(f"WARNING: {audit['total_nulls']} missing values found")
    if audit.get("duplicate_rows", 0) > 0:
        issues.append(f"WARNING: {audit['duplicate_rows']} duplicate rows")
    if audit.get("texts_with_multiple_labels", 0) > 0:
        issues.append(f"WARNING: {audit['texts_with_multiple_labels']} texts map to multiple labels")
    if audit["unique_labels"] < 5:
        issues.append("NOTE: Few classes (< 5)")

    if issues:
        for issue in issues:
            print(f"   [!] {issue}")
    else:
        print(f"   ✓ No major issues detected")

    print("\n" + "=" * 70)


def validate_binary_matrix(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("        BINARY MATRIX VALIDATION (Raw Excel)")
    print("=" * 70)

    label_col = df.columns[0]
    symptom_cols = df.columns[1:]

    print(f"\n1. Target column: {label_col}")
    print(f"   Unique values: {df[label_col].nunique()}")
    print(f"   Values: {list(df[label_col].unique())}")

    print(f"\n2. Symptom columns: {len(symptom_cols)}")
    binary_cols = [c for c in symptom_cols if set(df[c].unique()).issubset({0, 1})]
    print(f"   Binary columns: {len(binary_cols)}")
    non_binary = [c for c in symptom_cols if c not in binary_cols]
    if non_binary:
        print(f"   Non-binary columns: {non_binary}")

    print(f"\n3. Active symptoms per department:")
    for _, row in df.iterrows():
        dept = row[label_col]
        active = sum(1 for c in symptom_cols if row[c] == 1)
        print(f"   {dept:25s}: {active:3d} symptoms active")

    print(f"\n4. Symptom overlap analysis:")
    from itertools import combinations
    depts = df[label_col].tolist()
    for i, j in combinations(range(len(depts)), 2):
        row_i = df.iloc[i, 1:]
        row_j = df.iloc[j, 1:]
        overlap = int(((row_i == 1) & (row_j == 1)).sum())
        if overlap > 0:
            union = int(((row_i == 1) | (row_j == 1)).sum())
            jaccard = overlap / union if union > 0 else 0
            print(f"   {depts[i]:25s} & {depts[j]:25s}: {overlap} shared (Jaccard={jaccard:.2f})")

    print("\n" + "=" * 70)
