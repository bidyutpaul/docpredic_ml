"""
DocPredic Workspace Root Interactive Launcher
"""
import sys
from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parent
DOCPREDIC_DIR = ROOT_DIR / "docpredic_ml"

if str(DOCPREDIC_DIR) not in sys.path:
    sys.path.insert(0, str(DOCPREDIC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import warnings
warnings.filterwarnings("ignore")

try:
    # pyrefly: ignore [missing-import]
    from src.config import MODELS_DIR, LABEL_DIR, PROCESSED_DATA_DIR
    # pyrefly: ignore [missing-import]
    from src.inference import Predictor
except ImportError:
    from docpredic_ml.src.config import MODELS_DIR, LABEL_DIR, PROCESSED_DATA_DIR
    from docpredic_ml.src.inference import Predictor


def load_predictor() -> Predictor:
    """Prefer the fused high-accuracy system; fall back to legacy model."""
    try:
        predictor = Predictor.from_fused_artifacts(
            models_dir=MODELS_DIR,
            processed_dir=PROCESSED_DATA_DIR,
            label_dir=LABEL_DIR,
        )
        print("[OK] Fused high-accuracy system loaded "
              "(ensemble + Bayesian knowledge scorer).")
        return predictor
    except Exception as e:
        print(f"[INFO] Fused artifacts unavailable ({e}); using legacy model.")

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "best_sklearn_model.joblib"

    label_path = LABEL_DIR / "label_encoder.joblib"
    vectorizer_path = PROCESSED_DATA_DIR / "tfidf_vectorizer.joblib"

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_path)
    vectorizer = joblib.load(vectorizer_path)

    return Predictor(
        model=model,
        label_encoder=label_encoder,
        vectorizer=vectorizer,
        model_type="sklearn",
    )


def main():
    print("=" * 70)
    print("             DOCPREDIC MEDICAL SPECIALTY PREDICTOR")
    print("=" * 70)
    print("Loading model artifacts...")

    predictor = load_predictor()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("[OK] Model loaded successfully!")
    print("Enter patient symptom descriptions (or type 'exit' / 'q' to quit).")
    print("NOTE: triage support only — not a diagnosis. See a doctor.\n")

    while True:
        try:
            user_input = input("Patient Symptoms > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting DocPredic. Goodbye!")
                break

            result = predictor.predict(user_input, top_k=3)
            print("-" * 50)
            print(f"Top Recommendation : {result['top_prediction'].upper()}")
            print(f"Confidence         : {result['top_confidence'] * 100:.1f}%")
            print("\nTop 3 Department Options:")
            for i, p in enumerate(result["predictions"], 1):
                bar = "=" * int(p["confidence"] * 25)
                print(f"  {i}. {p['department']:25s} {p['confidence']*100:5.1f}%  [{bar}]")
            expl = result.get("explanation")
            if expl:
                if expl.get("matched_symptoms"):
                    print(f"\nDetected symptoms : "
                          f"{', '.join(expl['matched_symptoms']).lower()}")
                if expl.get("denied_symptoms"):
                    print(f"Denied symptoms   : "
                          f"{', '.join(expl['denied_symptoms']).lower()}")
            if result.get("low_confidence"):
                print(f"\n[!] {result.get('triage_note')}")
            print("-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting DocPredic. Goodbye!")
            break
        except Exception as e:
            print(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
