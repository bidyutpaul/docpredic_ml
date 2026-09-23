"""
Tests for Text Preprocessing & Symptom Normalization
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_text, expand_contractions, preprocess_text
from src.symptom_normalization import extract_symptoms, normalize_symptoms_to_vector


def test_clean_text():
    sample = "I can't sleep, I've had a severe headache and tummy pain."
    cleaned = clean_text(sample)
    assert "cannot" in cleaned or "not" in cleaned
    assert "headache" in cleaned
    assert "stomach" in cleaned


def test_preprocess_text():
    sample = "I am having severe headaches, dizzy feelings, and vomited twice."
    processed = preprocess_text(sample)
    assert isinstance(processed, str)
    assert len(processed) > 0


def test_extract_symptoms():
    sample = "I feel dizzy, nauseous, and have a fever."
    symptoms = extract_symptoms(sample)
    assert "dizziness" in symptoms
    assert "nausea" in symptoms
    assert "fever" in symptoms


def test_normalize_symptoms_to_vector():
    sample = "I have chest pain and shortness of breath."
    vec = normalize_symptoms_to_vector(sample)
    assert isinstance(vec, dict)
    assert vec.get("chest_pain") == 1
    assert vec.get("breathing_difficulty") == 1
