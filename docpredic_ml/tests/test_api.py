"""
Tests for FastAPI App Definitions & Routes
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app, SymptomInput, HealthResponse, PredictionResponse


def test_api_models():
    inp = SymptomInput(text="I have severe headaches and vomiting.")
    assert inp.text == "I have severe headaches and vomiting."

    health = HealthResponse(status="healthy", model_loaded=True)
    assert health.status == "healthy"
    assert health.model_loaded is True


def test_app_instance():
    assert app.title == "DocPredic API"
