"""
Tests for Model Architectures and Evaluators
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import get_models, evaluate_model_performance


class TestModels(unittest.TestCase):
    def test_get_models(self):
        models = get_models(num_classes=19)
        self.assertIn("logistic_regression", models)
        self.assertIn("calibrated_linear_svc", models)
        self.assertIn("random_forest", models)
        self.assertIn("mlp_neural_net", models)

        # Test fit and predict on dummy data
        X = np.random.rand(50, 100)
        y = np.random.randint(0, 19, size=50)

        model = models["logistic_regression"]
        model.fit(X, y)
        preds = model.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_evaluate_model_performance(self):
        X = np.random.rand(40, 20)
        y = np.random.randint(0, 5, size=40)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=100)
        clf.fit(X, y)
        metrics = evaluate_model_performance(clf, X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
