"""
Tests for Data Loader & Synthetic Data Generation
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_raw_data, generate_synthetic_text_data
from src.config import TEXT_COL, LABEL_COL


def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 19
    assert df.columns[0] == "Doctor's Department"
    # Check no trailing whitespace in department names
    for dept in df[df.columns[0]]:
        assert dept == dept.strip()


def test_generate_synthetic_text_data():
    raw_df = load_raw_data()
    text_df = generate_synthetic_text_data(raw_df, n_samples_per_class=10, seed=42)
    assert len(text_df) == 19 * 10
    assert TEXT_COL in text_df.columns
    assert LABEL_COL in text_df.columns
    assert text_df[LABEL_COL].nunique() == 19

    # Verify no department name leakage inside text
    for _, row in text_df.iterrows():
        dept = row[LABEL_COL]
        text = row[TEXT_COL]
        # Department name (e.g. DERMATOLOGY) should NOT appear directly in text
        assert dept.lower() not in text.lower()
