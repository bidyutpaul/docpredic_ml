"""
DocPredic MIMIC-IV Clinical Data Loader
Parses MIMIC-IV-ED triage, diagnosis, and stay data.
Maps ICD-9/10 codes and clinical presentations to Doctor Specialties.
"""
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from .config import MIMIC_DIR, TEXT_COL, LABEL_COL

# Clinical abbreviations commonly found in emergency triage
ABBREVIATIONS = {
    r"\bsob\b": "shortness of breath",
    r"\bcp\b": "chest pain",
    r"\bha\b": "severe headache",
    r"\babd\b": "abdominal",
    r"\bn/v/d\b": "nausea vomiting and diarrhea",
    r"\bn/v\b": "nausea and vomiting",
    r"\bbrbpr\b": "bright red blood per rectum",
    r"\bsi\b": "suicidal ideation",
    r"\bhi\b": "homicidal ideation",
    r"\bams\b": "altered mental status",
    r"\bdka\b": "diabetic ketoacidosis",
    r"\bs/p\b": "status post",
    r"\bmvc\b": "motor vehicle collision",
    r"\bsw\b": "stab wound",
    r"\bgsw\b": "gunshot wound",
    r"\bloc\b": "loss of consciousness",
    r"\bfx\b": "bone fracture",
    r"\blac\b": "laceration",
    r"\beval\b": "evaluation",
    r"\br leg\b": "right leg",
    r"\bl leg\b": "left leg",
    r"\br arm\b": "right arm",
    r"\bl arm\b": "left arm",
    r"\bw/o\b": "without",
    r"\bw/\b": "with",
    r"\bunsp\b": "unspecified",
    r"\bd/t\b": "due to",
    r"\bhx\b": "history of",
    r"\bhtn\b": "hypertension",
    r"\bdm\b": "diabetes mellitus",
    r"\buti\b": "urinary tract infection",
}


def clean_clinical_text(text: str) -> str:
    """Normalize clinical text and expand medical abbreviations."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    cleaned = text.lower().strip()
    for pattern, replacement in ABBREVIATIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def map_icd_to_department(icd_code: str, icd_version: int, icd_title: str) -> str:
    """
    Map an ICD-9 or ICD-10 code/title to one of DocPredic's 19 medical departments.
    """
    code = str(icd_code).strip().upper()
    title = str(icd_title).strip().upper() if pd.notna(icd_title) else ""

    # Priority 0: Exact ICD code mapping for common emergency ED codes
    if code in ["R079", "R0789", "R072", "R071", "78650", "78659", "78651", "78652", "410", "411", "412", "413", "414", "42731", "4280"]:
        return "CARDIOLOGY"
    if code in ["Z794", "V5867", "25000", "25001", "25002", "E119", "E109"]:
        return "ENDOCRINOLOGY"
    if code in ["78097", "R4182", "34889", "78039", "G40909", "431", "I6201"]:
        return "NEUROLOGY"
    if code in ["R1013", "R109", "78906", "78900", "5780", "5789"]:
        return "GASTROENTEROLOGY"
    if code in ["Z7901", "V5861", "280", "281", "2859", "D649"]:
        return "HEMATOLOGY"
    if code in ["R55", "7802"]:
        return "CARDIOLOGY"
    if code in ["5849", "5855", "N179", "N189"]:
        return "NEPHROLOGY"
    if code in ["5990", "N390", "R319", "R300"]:
        return "UROLOGY"

    # Priority 1: High-confidence keyword matching on ICD title
    if any(k in title for k in [
        "CHEST PAIN", "HEART", "CARDIAC", "CORONARY", "MYOCARDIAL", "ATRIAL", "VENTRICULAR",
        "HYPERTENSION", "AORT", "SYNCOPE", "TACHYCARDIA", "ANGINA", "ARRHYTHMIA",
        "HYPOTENSION", "PERICARD", "ENDOCARD", "VALVE", "CARDIOMYOPATHY", "PACEMAKER",
        "CONGESTIVE HEART", "NSTEMI", "STEMI", "ISCHEMIC HEART"
    ]):
        return "CARDIOLOGY"

    if any(k in title for k in [
        "PNEUMO", "PULMON", "RESPIRAT", "LUNG", "COPD", "ASTHMA", "BRONCH",
        "DYSPNEA", "PLEURAL", "EMPHYSEMA", "HYPOXEMIA", "TRACHEA", "SHORTNESS OF BREATH", "COUGH"
    ]):
        return "PULMONOLOGY"

    if any(k in title for k in [
        "BRAIN", "HEMORRHAGE", "SUBDURAL", "CEREBR", "STROKE", "SEIZURE",
        "CONVULSION", "EPILEPSY", "ALZHEIMER", "DEMENTIA", "NEURO", "PARALYSIS",
        "MENING", "HEAD INJURY", "DROOP", "COMA", "STUPOR", "NEUROPATHY",
        "ENCEPHAL", "MIGRAINE", "CRANIAL", "CONCUSSION", "ALTERED MENTAL"
    ]):
        return "NEUROLOGY"

    if any(k in title for k in [
        "FRACTURE", "FX ", "DISLOCATION", "SPRAIN", "BONE", "JOINT", "CERVICALGIA",
        "SPINE", "FEMUR", "TIBIA", "HUMERUS", "RADIUS", "RIBS", "FALL", "CONTUSION",
        "TRAUMA", "LACERATION", "LIGAMENT", "MENISCUS", "OSTEOPOROSIS", "TENDON",
        "BACK PAIN", "NECK PAIN", "LUMBAGO", "SCIATICA", "KNEE PAIN", "SHOULDER"
    ]):
        return "ORTHOPEDICS"

    if any(k in title for k in [
        "GASTRO", "STOMACH", "INTEST", "BOWEL", "COLON", "RECT", "HEMATEMESIS",
        "ESOPHAG", "ULCER", "LIVER", "HEPAT", "ABDOMINAL", "GALLBLADDER",
        "PANCREAS", "DIVERTICUL", "ILEUS", "APPENDIC", "CIRRHOSIS", "ASCITES",
        "HERNIA", "CHOLECYST", "MELENA", "EPIGASTRIC", "CONSTIPATION", "DIARRHEA", "TRANSAMNS"
    ]):
        return "GASTROENTEROLOGY"

    if any(k in title for k in [
        "KIDNEY", "RENAL", "NEPHR", "UREMIA", "DIALYSIS", "ACIDOSIS", "HYPERKALEMIA", "AZOTEMIA"
    ]):
        return "NEPHROLOGY"

    if any(k in title for k in [
        "URINARY", "BLADDER", "PROSTATE", "UTI", "HEMATURIA", "DYSURIA",
        "CYSTITIS", "URETHRA", "TESTIS", "SCROT", "HYDROCELE", "URINE"
    ]):
        return "UROLOGY"

    if any(k in title for k in [
        "SKIN", "RASH", "CELLULITIS", "ABSCESS", "DERM", "ECZEMA", "BURNS",
        "PRURITUS", "URTICARIA", "ULCERS", "WOUND", "ITCHING", "ERYTHEMA"
    ]):
        return "DERMATOLOGY"

    if any(k in title for k in [
        "EAR", "NOSE", "THROAT", "TINNITUS", "OTITIS", "SINUS", "PHARYNG",
        "TONSIL", "LARYNG", "EPISTAXIS", "RHINITIS", "EUSTACHIAN"
    ]):
        return "ENT"

    if any(k in title for k in [
        "EYE", "OPHTHALM", "VISION", "BLIND", "CONJUNCTIV", "CATARACT",
        "GLAUCOMA", "CORNEA", "DIPLOPIA", "RETIN", "VISUAL"
    ]):
        return "OPHTHALMOLOGY"

    if any(k in title for k in [
        "DIABETES", "THYROID", "HYPERLIPIDEMIA", "CHOLESTEROL", "OBESITY",
        "METABOLIC", "ENDOCRINE", "ADRENAL", "PITUITARY", "INSULIN", "HYPOGLYCEMIA", "HYPERGLYCEMIA"
    ]):
        return "ENDOCRINOLOGY"

    if any(k in title for k in [
        "ANEMIA", "HEMATOLOG", "LEUKEMIA", "LYMPH", "PLATELET", "COAGULOPATHY",
        "PURPURA", "BLEEDING DISORDER", "THROMBOCYTOPENIA", "NEUTROPENIA", "ANTICOAGULANT"
    ]):
        return "HEMATOLOGY"

    if any(k in title for k in [
        "ARTHRITIS", "RHEUMAT", "LUPUS", "GOUT", "FIBROMYALGIA", "ANKYLOSING",
        "POLYMYALGIA", "SCLERODERMA", "RHEUMATOID"
    ]):
        return "RHEUMATOLOGY"

    if any(k in title for k in [
        "NEOPLASM", "MALIG", "CANCER", "CARCINOMA", "TUMOR", "METASTA",
        "SARCOMA", "CHEMOTHERAPY", "LYMPHOMA"
    ]):
        return "ONCOLOGY"

    if any(k in title for k in [
        "PSYCH", "DEPRESSION", "BIPOLAR", "SCHIZO", "SUICID", "ANXIETY",
        "ALCOHOL", "SUBSTANCE", "ADDICTION", "OVERDOSE", "PANIC", "HALLUCIN",
        "NERVOUS", "EMOTIONAL", "MOOD"
    ]):
        return "PSYCHIATRY"

    if any(k in title for k in [
        "PREGNAN", "CHILDBIRTH", "UTER", "OVARY", "VAGIN", "CERVIX",
        "PELVIC PAIN", "OBSTETRIC", "LABOR", "MISCARRIAGE", "MENSTRUAL"
    ]):
        return "GYNECOLOGY & OBSTETRICS"

    if any(k in title for k in [
        "TOOTH", "TEETH", "DENTAL", "GINGIV", "PERIODON", "CARIES", "MANDIBLE", "JAW PAIN"
    ]):
        return "DENTISTRY"

    # Priority 2: Numerical & chapter ranges
    if icd_version == 9:
        digits = re.match(r"^(\d+)", code)
        if digits:
            num = int(digits.group(1)[:3])
            if 140 <= num <= 239: return "ONCOLOGY"
            if 240 <= num <= 279: return "ENDOCRINOLOGY"
            if 280 <= num <= 289: return "HEMATOLOGY"
            if 290 <= num <= 319: return "PSYCHIATRY"
            if 320 <= num <= 389: return "NEUROLOGY"
            if 390 <= num <= 459: return "CARDIOLOGY"
            if 460 <= num <= 519: return "PULMONOLOGY"
            if 520 <= num <= 579: return "GASTROENTEROLOGY"
            if 580 <= num <= 599: return "NEPHROLOGY"
            if 600 <= num <= 608: return "UROLOGY"
            if 614 <= num <= 679: return "GYNECOLOGY & OBSTETRICS"
            if 680 <= num <= 709: return "DERMATOLOGY"
            if 710 <= num <= 739: return "RHEUMATOLOGY"
            if 760 <= num <= 779: return "PEDIATRICS"
            if 800 <= num <= 999: return "ORTHOPEDICS"

    if icd_version == 10:
        ch = code[0].upper()
        if ch in ["C"]: return "ONCOLOGY"
        if ch in ["D"] and len(code) > 1 and code[1].isdigit() and int(code[1]) < 5: return "ONCOLOGY"
        if ch in ["D"]: return "HEMATOLOGY"
        if ch in ["E"]: return "ENDOCRINOLOGY"
        if ch in ["F"]: return "PSYCHIATRY"
        if ch in ["G"]: return "NEUROLOGY"
        if ch in ["H"] and len(code) > 1 and code[1].isdigit() and int(code[1]) <= 5: return "OPHTHALMOLOGY"
        if ch in ["H"]: return "ENT"
        if ch in ["I"]: return "CARDIOLOGY"
        if ch in ["J"]: return "PULMONOLOGY"
        if ch in ["K"]: return "GASTROENTEROLOGY"
        if ch in ["L"]: return "DERMATOLOGY"
        if ch in ["M"]: return "RHEUMATOLOGY"
        if ch in ["N"] and len(code) > 1 and code[1].isdigit() and int(code[1]) <= 2: return "NEPHROLOGY"
        if ch in ["N"]: return "UROLOGY"
        if ch in ["O"]: return "GYNECOLOGY & OBSTETRICS"
        if ch in ["P"]: return "PEDIATRICS"
        if ch in ["S", "T"]: return "ORTHOPEDICS"

    return "GENERAL MEDICINE"


def load_mimic_dataset(mimic_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load MIMIC-IV ED dataset and convert triage complaints + diagnoses
    into a standardized (text, department) dataframe for DocPredic.
    """
    path = Path(mimic_dir) if mimic_dir else MIMIC_DIR
    triage_file = path / "triage.csv.gz"
    diag_file = path / "diagnosis.csv.gz"
    stays_file = path / "edstays.csv.gz"

    if not (triage_file.exists() and diag_file.exists()):
        raise FileNotFoundError(f"MIMIC-IV files not found in {path}")

    triage = pd.read_csv(triage_file, compression="gzip")
    diag = pd.read_csv(diag_file, compression="gzip")
    stays = pd.read_csv(stays_file, compression="gzip") if stays_file.exists() else None

    merged = triage.merge(diag, on=["subject_id", "stay_id"], how="inner")
    if stays is not None and "gender" in stays.columns:
        merged = merged.merge(stays[["stay_id", "gender", "arrival_transport"]], on="stay_id", how="left")

    records: List[Dict[str, str]] = []

    for _, row in merged.iterrows():
        chief_raw = str(row.get("chiefcomplaint", ""))
        if not chief_raw or chief_raw.lower() in ["nan", "unknown-cc", ""]:
            continue

        chief_clean = clean_clinical_text(chief_raw)
        if len(chief_clean) < 3:
            continue

        icd_code = row.get("icd_code", "")
        icd_ver = row.get("icd_version", 9)
        icd_title = row.get("icd_title", "")
        dept = map_icd_to_department(icd_code, icd_ver, icd_title)

        records.append({
            TEXT_COL: f"Patient presents with {chief_clean}",
            LABEL_COL: dept
        })
        records.append({
            TEXT_COL: f"I have {chief_clean}",
            LABEL_COL: dept
        })

        if pd.notna(icd_title) and str(icd_title).strip():
            clean_diag = clean_clinical_text(str(icd_title))
            records.append({
                TEXT_COL: f"Patient experiencing {chief_clean} diagnosed with {clean_diag}",
                LABEL_COL: dept
            })

    df = pd.DataFrame(records).drop_duplicates()
    return df
