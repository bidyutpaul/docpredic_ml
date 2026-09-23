"""
DocPredic Configuration
Central configuration for paths, hyperparameters, and constants.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TOKENIZERS_DIR = ARTIFACTS_DIR / "tokenizers"
LABEL_DIR = ARTIFACTS_DIR / "label_encoders"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

RAW_EXCEL_PATH = RAW_DATA_DIR / "Data.xlsx"
if not RAW_EXCEL_PATH.exists():
    RAW_EXCEL_PATH = PROJECT_ROOT.parent / "Data.xlsx"
if not RAW_EXCEL_PATH.exists():
    RAW_EXCEL_PATH = Path(r"C:\Users\Bidyut Paul\Downloads\DocPredic ML\Data.xlsx")

MIMIC_DIR = PROJECT_ROOT.parent / "mimic-iv" / "mimic-iv-ed-demo-2.2" / "ed"
if not MIMIC_DIR.exists():
    MIMIC_DIR = Path(r"C:\Users\Bidyut Paul\Downloads\DocPredic ML\mimic-iv\mimic-iv-ed-demo-2.2\ed")

TEXT_COL = "text"
LABEL_COL = "department"
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
TRANSFORMER_MODEL = "bert-base-uncased"
NUM_CLASSES = 19
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_SAMPLES_PER_CLASS = 2

for d in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, TOKENIZERS_DIR, LABEL_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
