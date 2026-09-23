"""
DocPredic Inference Pipeline
Prediction from trained models.

Supports two modes:
  * legacy  (model_type="sklearn" | "transformer" | "tfidf") — unchanged
  * fused   (model_type="fused") — the high-accuracy system:
      P(d|text) = softmax(( lam*log P_ML(d) + (1-lam)*log P_KB(d)
                            with KB-abstention backoff ) / T)
    plus symptom-level explanations and a low-confidence triage flag.
"""
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import joblib

from .config import MODELS_DIR, LABEL_DIR, PROCESSED_DATA_DIR, MAX_LEN
from .preprocessing import preprocess_text, preprocess_text_negmarked


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Predictor:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        label_encoder=None,
        vectorizer=None,
        model_type: str = "transformer",
        device: str = None,
        # --- fused-system components (optional) ---
        word_vectorizer=None,
        char_vectorizer=None,
        knowledge_scorer=None,
        fusion_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.vectorizer = vectorizer
        self.model_type = model_type
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer
        self.knowledge_scorer = knowledge_scorer
        self.fusion_params = fusion_params or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        if self.model_type == "fused":
            return self.predict_fused(text, top_k=top_k)
        processed_text = preprocess_text(text)

        if self.model_type == "transformer" and self.tokenizer is not None:
            encoding = self.tokenizer(
                processed_text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        elif self.model_type == "tfidf" and self.vectorizer is not None:
            features = self.vectorizer.transform([processed_text])
            features_tensor = torch.tensor(
                features.toarray() if hasattr(features, "toarray") else features,
                dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(features_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        elif self.model_type == "sklearn" and self.model is not None:
            if self.vectorizer is not None:
                features = self.vectorizer.transform([processed_text])
            else:
                features = np.array([[0]])
            probs = self.model.predict_proba(features)[0]

        else:
            raise ValueError(f"Cannot run inference: model_type={self.model_type}, model={self.model is not None}")

        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = []
        for idx in top_indices:
            label = self.label_encoder.inverse_transform([idx])[0] if self.label_encoder is not None else str(idx)
            predictions.append({
                "department": label,
                "confidence": float(probs[idx]),
                "probability": float(probs[idx]),
            })

        return {
            "input_text": text,
            "processed_text": processed_text,
            "top_prediction": predictions[0]["department"],
            "top_confidence": predictions[0]["confidence"],
            "predictions": predictions,
        }

    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        return [self.predict(text, top_k) for text in texts]

    # ------------------------------------------------------------------
    # Fused high-accuracy path
    # ------------------------------------------------------------------
    def _ml_proba(self, text: str) -> Tuple[np.ndarray, str]:
        negmarked = bool(self.fusion_params.get("negation_marked", True))
        clean = (preprocess_text_negmarked(text) if negmarked
                 else preprocess_text(text))
        if self.word_vectorizer is not None and self.char_vectorizer is not None:
            from scipy.sparse import hstack
            X = hstack([self.word_vectorizer.transform([clean]),
                        self.char_vectorizer.transform([clean])],
                       format="csr")
        elif self.vectorizer is not None:
            X = self.vectorizer.transform([clean])
        else:
            raise ValueError("No vectorizer available for fused ML prediction")
        return np.clip(np.asarray(self.model.predict_proba(X)[0]),
                       1e-12, 1.0), clean

    def predict_fused(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """Fused posterior + explanation + triage safety flag."""
        classes = list(self.label_encoder.classes_)
        lam = float(self.fusion_params.get("lambda", 1.0))
        temp = float(self.fusion_params.get("temperature", 1.0))
        backoff = bool(self.fusion_params.get("kb_backoff_no_match", True))
        conf_thr = float(self.fusion_params.get("low_confidence_threshold", 0.35))
        marg_thr = float(self.fusion_params.get("low_margin_threshold", 0.10))

        p_ml, clean = self._ml_proba(text)

        kb_info: Dict[str, Any] = {"active": [], "negated": []}
        if self.knowledge_scorer is not None:
            kb_post_raw, kb_info = self.knowledge_scorer.posterior(text)
            order = [self.knowledge_scorer.departments.index(d) for d in classes]
            p_kb = np.clip(np.asarray(kb_post_raw)[order], 1e-12, 1.0)
            p_kb = p_kb / p_kb.sum()
            kb_hit = bool(kb_info.get("active") or kb_info.get("negated"))
        else:
            p_kb = np.full_like(p_ml, 1.0 / len(p_ml))
            kb_hit = False

        lam_eff = lam if (kb_hit or not backoff) else 1.0
        fused_log = lam_eff * np.log(p_ml) + (1.0 - lam_eff) * np.log(p_kb)
        probs = _softmax(fused_log / max(temp, 1e-6))

        top_idx = np.argsort(probs)[::-1][:top_k]
        predictions = [{"department": classes[i],
                        "confidence": float(probs[i]),
                        "probability": float(probs[i])} for i in top_idx]
        ml_top = classes[int(np.argmax(p_ml))]
        kb_top = classes[int(np.argmax(p_kb))] if self.knowledge_scorer else None
        margin = (float(probs[top_idx[0]]) - float(probs[top_idx[1]])
                  if len(top_idx) > 1 else 1.0)
        low_conf = bool(probs[top_idx[0]] < conf_thr or margin < marg_thr)

        return {
            "input_text": text,
            "processed_text": clean,
            "top_prediction": predictions[0]["department"],
            "top_confidence": predictions[0]["confidence"],
            "predictions": predictions,
            "model_type": "fused",
            "fusion": {"lambda": lam, "lambda_effective": lam_eff,
                       "temperature": temp, "kb_matched": kb_hit},
            "explanation": {
                "matched_symptoms": kb_info.get("active", []),
                "denied_symptoms": kb_info.get("negated", []),
                "ml_top": ml_top,
                "ml_top_confidence": float(p_ml.max()),
                "knowledge_top": kb_top,
                "models_agree": bool(ml_top == kb_top),
                "margin_top2": margin,
            },
            "low_confidence": low_conf,
            "triage_note": (
                "Low certainty — consider GENERAL MEDICINE review and "
                "in-person evaluation; this tool supports triage, not diagnosis."
                if low_conf else None
            ),
        }

    @classmethod
    def from_fused_artifacts(
        cls,
        models_dir: Path = MODELS_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        label_dir: Path = LABEL_DIR,
    ) -> "Predictor":
        """Load the full fused system (ensemble + KB + fusion params)."""
        from .knowledge_scorer import KnowledgeScorer

        models_dir, processed_dir, label_dir = (
            Path(models_dir), Path(processed_dir), Path(label_dir))
        model = joblib.load(models_dir / "fused_ensemble.joblib")
        label_encoder = joblib.load(label_dir / "label_encoder.joblib")
        word_vec = joblib.load(processed_dir / "tfidf_word_vectorizer.joblib")
        char_vec = joblib.load(processed_dir / "tfidf_char_vectorizer.joblib")
        kb = KnowledgeScorer.load(models_dir / "knowledge_scorer.joblib")
        with open(models_dir / "fusion_params.json") as f:
            fusion_params = json.load(f)
        return cls(model=model, label_encoder=label_encoder,
                   word_vectorizer=word_vec, char_vectorizer=char_vec,
                   knowledge_scorer=kb, fusion_params=fusion_params,
                   model_type="fused")

    @classmethod
    def from_artifacts(
        cls,
        model_path: Path,
        label_encoder_path: Path,
        vectorizer_path: Optional[Path] = None,
        tokenizer_name: Optional[str] = None,
        model_class=None,
        model_type: str = "transformer",
    ):
        label_encoder = joblib.load(label_encoder_path)

        if model_type == "sklearn":
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path) if vectorizer_path else None
            tokenizer = None
        elif model_class is not None:
            model = model_class(num_classes=len(label_encoder.classes_))
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            vectorizer = joblib.load(vectorizer_path) if vectorizer_path else None
            tokenizer = None
            if tokenizer_name:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            raise ValueError("Must provide model_class for non-sklearn models")

        return cls(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            vectorizer=vectorizer,
            model_type=model_type,
        )
