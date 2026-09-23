"""
DocPredic Knowledge-Matrix Bayesian Scorer
==========================================
Mathematical core of the high-accuracy pipeline.

Ground truth: the doctor-provided binary symptom matrix (Data.xlsx),
19 departments x 88 symptoms, M[d, j] = 1 if symptom j is characteristic
of department d.

Model: IDF-weighted Bernoulli Naive Bayes over the *observed* symptom set.

For input text we extract:
  active   = symptoms asserted present
  negated  = symptoms explicitly denied ("no fever", "without cough", ...)

For each department d the unnormalized log-score is::

    score(d) = log P(d)
             + Σ_{j ∈ active}   idf(j) · log θ(d, j)
             + Σ_{j ∈ negated}  idf(j) · log (1 - θ(d, j))

where
  θ(d, j) = (s(d, j) + α) / (n(d) + 2α)      (Laplace-smoothed likelihood)
  s(d, j) = M[d, j] + template_hits[d, j]    (matrix + template pseudo-counts)
  n(d)    = n_matrix(d) + n_templates(d)     (expert-labelled case count)
  idf(j)  = log((D + 1) / (df(j) + 1)) + 1   (discriminative weight;
              symptoms unique to one department count the most)

The posterior is softmax(score / T) with temperature T fitted on
validation data (see train_accurate.py).

Only *mentioned* symptoms contribute: unmentioned symptoms are treated as
unobserved (missing), NOT as absent. This is the correct probabilistic
treatment for short patient complaints and avoids the classic Naive Bayes
failure of over-penalizing departments with long symptom lists.
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import RAW_EXCEL_PATH
from .data_loader import (
    SYMPTOM_DESCRIPTIONS,
    DEPARTMENT_SYMPTOM_TEMPLATES,
)

# ---------------------------------------------------------------------------
# Extra colloquial / clinical phrase variants per symptom column.
# Keys are the exact Data.xlsx column names (upper-case).
# ---------------------------------------------------------------------------
EXTRA_PHRASES: Dict[str, List[str]] = {
    "FEVER": ["febrile", "running temperature", "burning up", "temperature is high",
              "feverish feeling", "pyrexia", "have temperature"],
    "COUGH": ["coughing fits", "hacky cough", "productive cough", "cough with phlegm",
              "coughing up mucus", "whooping cough"],
    "FATIGUE": ["tiredness", "lethargy", "lethargic", "worn out", "drained",
                "low energy", "lack of energy", "tuckered out"],
    "WEAKNESS": ["weak", "general weakness", "feeling faint", "lack of strength"],
    "HEADACHE": ["head ache", "headaches", "migraine", "head is pounding",
                 "pounding head", "splitting headache", "cephalgia"],
    "DIZZINESS": ["dizzy spells", "lightheadedness", "vertigo", "giddiness",
                  "feeling faint", "unsteady feeling"],
    "BODY_PAIN": ["body ache", "body aches", "all-over pain", "generalized pain",
                  "everywhere hurts", "body hurts"],
    "ABDOMINAL_PAIN": ["abdominal cramps", "belly ache", "tummy pain", "gut pain",
                       "pain in abdomen", "epigastric pain", "colicky pain"],
    "NAUSEA": ["nauseated", "feel sick", "sicky feeling", "queasiness", "morning sickness"],
    "VOMITING": ["vomits", "vomitted", "puking", "throw up", "threw up", "emesis",
                 "retching", "being sick"],
    "CHEST_PAIN": ["angina", "crushing chest", "chest discomfort", "chest ache",
                   "pain in chest", "tight chest", "heaviness in chest"],
    "BREATHING_DIFFICULTY": ["dyspnea", "breathlessness", "short breath",
                             "trouble breathing", "hard to breathe", "gasping",
                             "wheezing", "wheeze", "chest congestion"],
    "HEART_PALPITATIONS": ["palpitation", "fluttering heart", "pounding heart",
                           "racing pulse", "skipped beats", "arrhythmia"],
    "IRREGULAR_PULSE": ["irregular heartbeat", "uneven pulse", "erratic heartbeat"],
    "HIGH_BP": ["hypertension", "bp is high", "blood pressure high", "elevated bp"],
    "FAINTING": ["faint", "fainted", "blacked out", "passed out", "syncope",
                 "passing out", "collapse episode"],
    "FAST_HEARTBEAT": ["tachycardia", "rapid heartbeat", "racing heart",
                       "pulse is fast", "heart beating fast"],
    "SLOW_HEARTBEAT": ["bradycardia", "pulse is slow", "heart beating slow"],
    "EXTREME_FATIGUE": ["total exhaustion", "complete exhaustion", "chronic fatigue",
                        "always tired", "tired all the time"],
    "LEG_SWELLING": ["swollen legs", "leg edema", "edema in legs", "puffy legs"],
    "CHEST_TIGHTNESS": ["tightness in chest", "constricted chest", "heavy chest"],
    "ANKLE_SWELLING": ["swollen ankles", "ankle edema", "puffy ankles"],
    "EXCESS_SWEATING": ["sweaty", "sweats", "perspiring heavily", "cold sweats",
                        "night sweat", "diaphoresis"],
    "BACK_PAIN": ["backache", "back ache", "lumbar pain", "spine pain", "lumbago",
                  "sciatica", "slipped disc", "disc pain", "low back ache",
                  "chronic back pain", "back stiffness"],
    "NERVE_PAIN": ["neuralgia", "neuropathy", "neuropathic pain", "shooting pain",
                   "sciatic pain", "radiating down", "radiates to leg",
                   "radiates down", "shooting down", "pain shoots",
                   "radiating to leg", "radiating to arm", "radiates to arm",
                   "radiating down leg", "radiating down arm", "shooting leg pain",
                   "electric shock pain", "burning shooting pain"],
    "SEIZURES": ["tongue bite", "bit my tongue", "tongue bitten", "bitten tongue",
                   "jerky movements", "jerking movements", "body jerks",
                   "myoclonic jerks", "frothing mouth", "froth at mouth",
                   "fall unconscious shaking", "staring spells", "absence spells",
                   "epileptic fit", "shaking fit"],
    "BONE_FRACTURE": ["deformity", "bent wrong", "bone out of place",
                        "joint out of place", "dislocation", "dislocated",
                        "cannot bear weight", "swelling deformity"],
    "DIZZINESS": ["giddiness", "giddy feeling", "reeling sensation"],
    "FAINTING": ["swooning", "near faint", "presyncope"],
    "SEIZURES": ["seizure", "convulsions", "convulsion", "epileptic fit", "fits",
                 "epilepsy", "shaking fit"],
    "NUMBNESS": ["numb", "numb arm", "numb leg", "numbness in limbs",
                 "loss of sensation", "dead feeling in limbs"],
    "TINGLING": ["pins & needles", "prickling", "tingly", "tingle"],
    "MUSCLE_WEAKNESS": ["muscular weakness", "weak muscles", "muscle fatigue",
                        "loss of strength"],
    "TREMORS": ["tremor", "shaky hands", "hand tremor", "shakiness", "shakes"],
    "MEMORY_LOSS": ["memory issues", "forgetful", "forgetfulness", "amnesia",
                    "losing memory", "dementia symptoms"],
    "CONFUSION": ["confused", "disoriented", "disorientation", "mental fog",
                  "brain fog", "altered mental status"],
    "BALANCE_PROBLEM": ["balance issues", "off balance", "unsteady gait",
                        "losing balance", "ataxia"],
    "DIFFICULTY_WALKING": ["trouble walking", "hard to walk", "cannot walk",
                           "unable to walk", "limping", "gait problem"],
    "SPEECH_DIFFICULTY": ["slurred speech", "trouble talking", "aphasia",
                          "difficulty speaking", "speech slurred"],
    "FACIAL_WEAKNESS": ["face droop", "facial droop", "facial palsy",
                        "drooping face", "bell's palsy"],
    "CONSCIOUSNESS_LOSS": ["unconscious", "blackout", "coma", "unresponsive",
                           "knocked out"],
    "NERVE_PAIN": ["neuralgia", "neuropathy", "neuropathic pain", "shooting pain",
                   "burning pain", "sciatic pain"],
    "MUSCLE_STIFFNESS": ["stiff muscles", "rigidity", "spasticity", "muscle spasm",
                         "cramps", "muscle tightness"],
    "POOR_COORDINATION": ["uncoordinated", "coordination problem", "clumsiness"],
    "SLEEP_PROBLEM": ["insomnia", "sleeplessness", "cannot sleep", "trouble sleeping",
                      "sleep disturbance", "waking up at night", "restless sleep"],
    "JOINT_PAIN": ["arthralgia", "aching joints", "painful joint", "arthritis pain",
                   "joint ache"],
    "NECK_PAIN": ["neck ache", "stiff neck", "cervical pain", "crick in neck"],
    "KNEE_PAIN": ["knee ache", "sore knee", "knee arthritis", "knee swelling"],
    "SHOULDER_PAIN": ["shoulder ache", "sore shoulder", "frozen shoulder",
                      "rotator cuff pain"],
    "HIP_PAIN": ["hip ache", "sore hip", "hip arthritis"],
    "BONE_PAIN": ["aching bones", "bone ache", "ostealgia"],
    "MUSCLE_PAIN": ["myalgia", "aching muscles", "muscle ache", "muscle soreness",
                    "muscle cramp"],
    "JOINT_SWELLING": ["swollen joint", "joint inflammation", "inflamed joint",
                       "fluid in joint"],
    "JOINT_STIFFNESS": ["stiff joints", "morning stiffness", "joint rigidity"],
    "LIMITED_MOVEMENT": ["restricted movement", "cannot move", "loss of mobility",
                         "stiff movement", "range of motion loss"],
    "BONE_FRACTURE": ["fracture", "fractured", "broken bone", "break in bone",
                      "cracked bone", "stress fracture"],
    "MUSCLE_WEAKNESS.1": ["progressive weakness", "muscle wasting", "muscle atrophy"],
    "TINGLING_SENSATION": ["tingling feeling", "numbness and tingling"],
    "HEEL_PAIN": ["heel spur", "plantar fasciitis", "sore heel", "heel ache"],
    "SPORTS_INJURY": ["sports sprain", "sports strain", "injured playing",
                      "torn ligament", "sprained ankle", "acl injury", "tennis elbow"],
    # ---- dermatology / gastro / pulmo / nephro / uro / ent / endo / eye /
    # ---- dental / psych / onco / hemato / rheuma (zero-rows in the matrix,
    # ---- grounded via template pseudo-counts + these phrases) ----
    "SKIN_RASH": ["skin rash", "rashes", "skin eruption", "red spots on skin",
                  "hives", "urticaria", "eczema", "dermatitis", "psoriasis",
                  "itchy skin", "skin itching", "skin irritation", "skin redness",
                  "skin lesions", "blisters on skin", "skin peeling",
                  "acne", "pimples", "zits", "boils on skin", "carbuncle",
                  "boils", "boil ", "pus filled", "pus ", "purulent",
                  "peeling skin", "scaly skin", "flaky skin", "dry patches",
                  "ringworm", "tinea", "fungal rash", "nail fungus",
                  "dandruff", "hair fall", "hair loss", "alopecia",
                  "pigmentation", "dark spots", "white patches", "vitiligo",
                  "warts", "corns ", "callus", "athlete foot"],
    "STOMACH_ISSUE": ["stomach pain", "gastritis", "acid reflux", "heartburn",
                      "acidity", "bloating", "gas trouble", "indigestion",
                      "diarrhea", "loose motions", "loose motion", "constipation",
                      "ulcer pain", "bowel problem", "irritable bowel",
                      "blood in stool", "vomiting blood", "gerd", "gastric trouble",
                      "food poisoning", "gallstone", "gall stone", "gall bladder",
                      "piles ", "hemorrhoids", "fissure", "rectal pain",
                      "sour burps", "burping", "belching", "nausea after food"],
    "RESPIRATORY": ["asthma", "bronchitis", "pneumonia symptoms", "tb cough",
                    "tuberculosis", "copd", "breathing problem", "lung problem",
                    "chest infection", "dry cough", "wet cough", "mucus cough",
                    "phlegm", "sputum"],
    "JOINT_PAIN": ["arthralgia", "aching joints", "painful joint", "arthritis pain",
                   "joint ache", "knee joint", "hip joint", "shoulder joint",
                   "ankle joint", "swollen knee", "stiff knee", "knee swelling",
                   "knee stiffness", "painful knee", "sore knee", "ankle pain",
                   "wrist pain", "elbow pain", "finger pain", "toe pain",
                   "arthritis", "arthritic", "stiff joints",
                   "joints stiff", "joints swollen", "swollen joints"],
    "KNEE_PAIN": ["knee joint", "swollen knee", "stiff knee", "knee swelling",
                  "painful knee", "sore knee", "knee arthritis", "knee stiffness"],
    "SHOULDER_PAIN": ["shoulder joint", "shoulder stiffness", "shoulder swelling"],
    "HIP_PAIN": ["hip joint", "hip stiffness", "hip arthritis"],
    "EAR_NOSE_THROAT": ["ear pain", "earache", "ear infection", "ear discharge",
                        "hearing loss", "blocked ear", "blocked ears", "tinnitus",
                        "ringing in ear", "ringing in ears", "ringing ears",
                        "ears ringing", "sound in ears", "sore throat",
                        "throat pain", "throat ache", "sore throats", "tonsillitis",
                        "swollen tonsils", "swallowing pain", "pain when swallowing",
                        "difficulty swallowing", "trouble swallowing",
                        "cannot swallow", "nasal congestion", "blocked nose",
                        "stuffy nose", "runny nose", "sinus pain", "sinusitis",
                        "sinus pressure", "sinus headache", "loss of smell",
                        "cannot smell", "nosebleed", "nose bleed", "bleeding nose",
                        "hoarse voice", "hoarseness", "cold and cough",
                        "ear fullness", "post nasal drip", "post-nasal drip",
                        "throat irritation", "itchy throat", "swollen glands",
                        "swollen neck glands", "ear pus", "pus from ear",
                        "smell from ear", "watery discharge nose"],
    "THYROID_HORMONE": ["thyroid", "goiter", "goitre", "weight gain", "weight loss",
                        "diabetes", "high sugar", "blood sugar high", "sugar problem",
                        "excessive thirst", "frequent hunger", "hormonal problem",
                        "hormone imbalance", "heat intolerance", "cannot bear heat",
                        "heat sensitivity", "cold intolerance", "hair loss",
                        "hyperthyroid", "hypothyroid", "neck swelling",
                        "swelling in neck", "swollen thyroid", "swelling front neck",
                        "neck front swelling", "short height", "growth retardation",
                        "delayed puberty", "prolactin"],
    "EYE_PROBLEM": ["eye pain", "eye redness", "red eyes", "eyes are red",
                    "red eye", "itchy eyes", "itchy eye", "watery eyes",
                    "watery eye", "eyes watering", "blurry vision", "blurred vision",
                    "vision loss", "vision problem", "double vision", "eye discharge",
                    "discharge from eye", "dry eyes", "dry eye", "burning eyes",
                    "burning eye", "sensitivity to light", "sensitive to light",
                    "cataract", "glaucoma", "conjunctivitis", "pink eye",
                    "stye in eye", "stye ", "floaters", "eye floaters",
                    "eye strain", "eye irritation", "gritty eyes",
                    "gritty feeling", "halos around lights", "seeing halos",
                    "eye itching", "swollen eyes", "puffy eyes", "eye swelling",
                    "cannot see clearly", "vision is blurry", "eyes hurt"],
    "DENTAL": ["tooth pain", "toothache", "tooth decay", "cavity in tooth",
               "bleeding gums", "bleed from gums", "gums bleed", "gum bleeding",
               "swollen gums", "gum pain", "gum swelling", "sore gums",
               "jaw pain", "jaw ache", "wisdom tooth", "mouth ulcer",
               "mouth sore", "bad breath", "tooth sensitivity", "sensitive teeth",
               "broken tooth", "chipped tooth", "cracked tooth", "dental abscess",
               "teeth pain", "teeth hurt", "tooth hurt", "tooth hurts",
               "hurting tooth", "aching tooth", "sore tooth", "rotten tooth",
               "tooth ", "teeth ", "dental ", "gum ", "gums ", "jaw ",
                "molar pain", "canine tooth pain", "chewing pain", "pain when chewing",
                "pain while eating", "braces pain", "tartar", "plaque on teeth",
                "bad smell", "mouth odor", "smelling breath", "tooth hole",
                "hole in tooth", "black spot tooth", "mobile tooth",
                "shaky tooth", "pyorrhea", "pericoronitis"],
    "MENTAL_HEALTH": ["depression", "depressed", "depressive", "anxiety", "anxious",
                      "panic attack", "panic attacks", "panic disorder", "stress",
                      "stressed", "insomnia", "sleepless", "mood swings", "bipolar",
                      "hallucination", "hallucinations", "paranoia", "paranoid",
                      "suicidal thoughts", "suicidal", "self harm", "self-harm",
                      "ocd ", "phobia", "hopelessness", "hopeless", "sad",
                      "sadness", "worthlessness", "worthless", "loss of interest",
                      "no interest", "schizophrenia", "ptsd", "trauma stress",
                      "overthinking", "restlessness", "feeling empty",
                      "crying spells", "want to die", "hurting myself",
                      "hearing voices", "seeing things", "delusions",
                      "social anxiety", "generalized anxiety", "worrying a lot",
                      "cannot concentrate", "lack of motivation", "low mood",
                      "feeling down", "feeling low", "mental breakdown",
                      "hearing whispers", "hears voices", "suspicious",
                      "plotting against", "delusion", "doubts family",
                      "no joy", "joyless", "guilt ", "guilty feeling",
                      "cries daily", "crying daily", "blank mind", "fearful",
                      "irritable ", "irritability", "aggressive outburst",
                      "anger issues", "addiction", "drinking problem"],
    "KIDNEY_ISSUE": ["kidney pain", "flank pain", "pain in flank", "kidney stones",
                     "renal pain", "renal colic", "foamy urine", "frothy urine",
                     "blood in urine", "dark urine", "cloudy urine", "low urine",
                     "low urine output", "reduced urination", "kidney failure symptoms",
                     "swelling in feet", "swollen feet", "puffy eyes",
                     "protein in urine", "urine ", "dialysis", "creatinine high",
                     "creatinine ", "stone pain", "stones ", "colicky pain",
                     "pain in waves", "loin pain"],
    "BLADDER_ISSUE": ["burning urination", "burning while urinating",
                      "burning when urinating", "burning sensation urination",
                      "burning when I urinate", "burning pee", "painful urination",
                      "pain while urinating", "pain when urinating",
                      "difficulty urinating", "trouble urinating",
                      "frequent urination", "need to urinate", "urinate frequently",
                      "urge to urinate", "urgent urination", "urge incontinence",
                      " incontinence ", "bedwetting", "weak urine stream",
                      "dribbling urine", "uti ", "urinary tract infection",
                      "bladder pain", "bladder pressure", "cystitis",
                      "urinate ", "urinating ", "urination ", "pee ", "peeing "],
    "CANCER_SIGNS": ["lump", "tumor", "tumour", "mass in body", "swollen lymph",
                     "lymph node swelling", "unexplained bleeding", "night sweats",
                     "sudden weight loss", "rapid weight loss", "non-healing sore",
                     "sore not healing", "mole change", "cancer", "carcinoma",
                     "chemotherapy", "malignancy", "rectal bleeding",
                     "thin stools", "altered bowel habit", "difficulty swallowing food",
                     "persistent hoarseness", "abnormal vaginal bleeding",
                     "postmenopausal bleeding", "bone pain night", "cachexia"],
    "BLOOD_DISORDER": ["anemia", "anaemia", "low hemoglobin", "hb low", "hb ",
                       "hemoglobin ", "pale skin", "paleness", "pallor",
                       "easy bruising", "bruise easily", "bruises easily",
                       "nosebleeds", "nose bleeds", "frequent nosebleeds",
                       "heavy bleeding", "prolonged bleeding", "blood clots",
                       "bleeding disorder", "bleeder", "clotting problem",
                       "low platelet", "platelets low", "hemophilia",
                       "thalassemia", "leukemia", "blood cancer",
                       "dizzy and pale", "cold hands and feet", "petechiae",
                       "purpura", "pica ", "eats chalk", "eats mud",
                       "mud eating", "chalk eating", "craves chalk",
                       "dengue ", "platelet drop"],
    "AUTOIMMUNE_JOINT": ["rheumatoid", "rheumatism", "lupus", "gout",
                         "symmetrical joint pain", "both sides joint",
                         "butterfly rash", "malar rash", "cheek rash",
                         "sun sensitive rash", "rash sun exposure", "autoimmune",
                         "fibromyalgia", "ankylosing", "morning stiffness",
                         "joint deformity", "symmetric swelling", "ra factor",
                         "anti ccp", "joint pain months", "chronic joint pains"],
}

# Map general-symptom columns onto their pediatric equivalents. Activated only
# when a pediatric context word is present (baby/infant/toddler/...).
PEDIATRIC_CONTEXT_WORDS = [
    "baby", "babies", "infant", "infants", "toddler", "toddlers", "newborn",
    "newborns", "neonate", "kid", "kids", "child", "children", "toddler",
    "son", "daughter", "month old", "month-old", "year old", "year-old",
    "breastfed", "breastfeed", "formula-fed", "teething",
]
# Strict infant words: only these trigger "crying / poor-feeding" as a *baby*
# symptom. "kid/child/son/daughter" still trigger the symptom bridge below,
# but a teenager saying "kids" (their children) must not map onto baby symptoms.
STRICT_INFANT_WORDS = [
    "baby", "babies", "infant", "infants", "toddler", "toddlers", "newborn",
    "newborns", "neonate", "month old", "month-old", "breastfed", "breastfeed",
    "formula-fed", "teething",
]
PEDIATRIC_BRIDGE: Dict[str, str] = {
    "FEVER": "BABY_FEVER",
    "COUGH": "BABY_COUGH",
    "BREATHING_DIFFICULTY": "BABY_BREATHING_DIFFICULTY",
    "VOMITING": "BABY_VOMITING",
    "ABDOMINAL_PAIN": "BABY_STOMACH_PAIN",
    "SEIZURES": "BABY_SEIZURES",
    "FATIGUE": "BABY_WEAKNESS",
    "WEAKNESS": "BABY_WEAKNESS",
    "NAUSEA": "BABY_VOMITING",
}
APPETITE_PHRASES = ["not eating", "refuses to eat", "refuse to eat", "poor feeding",
                    "not feeding", "poor appetite", "loss of appetite", "won't eat",
                    "does not eat", "refusing milk", "refuses milk", "not drinking milk"]
CRYING_PHRASES = ["crying", "cries ", "cries", "inconsolable", "fussy", "irritable baby",
                  "keeps crying", "crying a lot", "crying excessively"]

NEGATION_WORDS = ["no", "not", "without", "denies", "deny", "denied", "never",
                  "neither", "nor", "free of", "absence of", "lack of", "ruled out"]
# "lack of" collides with phrases like "lack of energy" (fatigue) — handled by
# requiring the negation word to appear *before* the symptom phrase with a
# short window, and exempting phrases that legitimately contain those words.
NEGATION_EXEMPT_SUBSTRINGS = ["lack of energy", "loss of"]

CONTRACTION_MAP = {
    "won't": "will not", "can't": "cannot", "cannot": "can not",
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}


def _normalize(text: str) -> str:
    t = text.lower()
    for c, e in CONTRACTION_MAP.items():
        t = t.replace(c, e)
    t = re.sub(r"[^a-z0-9\s\-&]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _build_lexicon() -> Dict[str, List[str]]:
    """ symptom column -> sorted list of matchable phrases (longest first)."""
    lex: Dict[str, Set[str]] = {}
    for col, variants in SYMPTOM_DESCRIPTIONS.items():
        key = col.strip().upper()
        phrases = set()
        for v in variants:
            vn = _normalize(v)
            phrases.add(vn)
            # also add the content without a leading "i/my ..." subject
            for prefix in ("i have ", "i am ", "i feel ", "i get ", "my ",
                           "i have been ", "i am experiencing ", "i experience "):
                if vn.startswith(prefix):
                    phrases.add(vn[len(prefix):])
                    break
        lex.setdefault(key, set()).update(phrases)
    for col, variants in EXTRA_PHRASES.items():
        key = col.strip().upper()
        lex.setdefault(key, set()).update(_normalize(v) for v in variants)
    # generic fallback: the column name itself as words
    for key in list(lex.keys()):
        generic = _normalize(key.replace("_", " ").replace(".1", ""))
        if len(generic) >= 3:
            lex[key].add(generic)
    return {k: sorted(v, key=len, reverse=True) for k, v in lex.items()}


LEXICON = _build_lexicon()


def _compile_patterns() -> Dict[str, List[re.Pattern]]:
    pats: Dict[str, List[re.Pattern]] = {}
    for col, phrases in LEXICON.items():
        col_pats = []
        for ph in phrases:
            if len(ph) < 3:
                continue
            # word-boundary regex; allow any whitespace inside multiword phrases
            core = r"\s+".join(re.escape(tok) for tok in ph.split())
            try:
                col_pats.append(re.compile(r"\b" + core + r"\b"))
            except re.error:
                continue
        pats[col] = col_pats
    return pats


PATTERNS = _compile_patterns()
_NEG_WINDOW = r"(?:not|no|without|denies|deny|denied|never|neither|nor|free\s+of|absence\s+of|ruled\s+out)\b[\w\s]{0,28}?"


# ---------------------------------------------------------------------------
# Order-free proximity fallback: matches multi-word phrases whose content
# words all appear close together regardless of order
# ("knee joint is swollen" matches "swollen joint").
# Morphology is handled by candidate-set stemming: "urinating"/"urination"/
# "urinate" share the candidate stem "urinat"/"urinate", so verb/noun forms
# match without a lemmatizer dependency.
# ---------------------------------------------------------------------------
def _stem(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "ches", "shes", "zzes", "men")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "os", "as")):
        return w[:-1]
    return w


def _stem_candidates(w: str) -> Set[str]:
    """All morphological variants a token may match under."""
    out = {_stem(w)}
    lw = w.lower()
    if len(lw) > 5 and lw.endswith("ing"):
        b = _stem(lw[:-3])
        out.update({b, b + "e"})
    elif len(lw) > 4 and lw.endswith("ed"):
        b = _stem(lw[:-2])
        out.update({b, b + "e"})
        if len(b) > 2 and b[-1] == b[-2]:
            out.add(b[:-1])
    if len(lw) > 5 and lw.endswith("ation"):
        b = _stem(lw[:-5])
        out.update({b, b + "e", b + "ate"})
    return out


_TOKEN_STOPWORDS = frozenset(
    "i my me a an the and or is are was were be been have has had having "
    "with of to in on for from at by as it its this that these those "
    "very much more most feel feeling feels felt suffer suffering "
    "also just so too quite rather really still already yet both either "
    "our your his her their all any each every few here there when "
    "while during after before since until ago today yesterday days day "
    "weeks week months month years year times time got getting get gets".split()
)


def _token_set_hit(phrase: str, words: List[str], stem_pos: Dict[str, List[int]]
                   ) -> Tuple[bool, int]:
    """Check whether all content words of `phrase` occur within a tight window.

    Returns (hit, earliest_word_index) for negation checks.
    """
    toks = [_stem(w) for w in phrase.split()
            if w not in _TOKEN_STOPWORDS and len(w) > 1]
    if len(toks) < 2:
        return False, -1
    cand_lists = []
    for t in toks:
        merged: List[int] = []
        for c in _stem_candidates(t):
            merged.extend(stem_pos.get(c, []))
        if not merged:
            return False, -1
        cand_lists.append(sorted(set(merged)))
    # brute-force over combinations is exponential; phrases are short (<=4
    # content tokens) and position lists are short — cap effort sensibly.
    best = None
    import itertools
    total = 1
    for L in cand_lists:
        total *= len(L)
    if total > 4000:
        # fall back: span from min-first to max-first occurrence
        lo = min(L[0] for L in cand_lists)
        hi = max(L[0] for L in cand_lists)
        span = hi - lo
        return (span <= len(toks) + 5, lo)
    for combo in itertools.product(*cand_lists):
        span = max(combo) - min(combo)
        if best is None or span < best[0]:
            best = (span, min(combo))
            if span == len(toks) - 1:
                break
    if best is None:
        return False, -1
    return (best[0] <= len(toks) + 5, best[1])


_NEG_WORDS_SET = frozenset(
    ["no", "not", "without", "denies", "deny", "denied", "never",
     "neither", "nor", "ruled"])
_NEG_PHRASES_SET = frozenset(["free of", "absence of", "ruled out"])


def extract_symptoms(text: str) -> Tuple[Set[str], Set[str]]:
    """Return (active_symptoms, negated_symptoms) as sets of symptom columns."""
    norm = _normalize(text)
    words = norm.split()
    stem_pos: Dict[str, List[int]] = {}
    for i, w in enumerate(words):
        for c in _stem_candidates(_stem(w)):
            stem_pos.setdefault(c, []).append(i)

    def _negated_before(word_idx: int) -> bool:
        lo = max(0, word_idx - 7)
        window = words[lo:word_idx]
        if any(w in _NEG_WORDS_SET for w in window):
            return True
        win_str = " ".join(window)
        return any(p in win_str for p in _NEG_PHRASES_SET)

    active: Set[str] = set()
    negated: Set[str] = set()
    for col, pats in PATTERNS.items():
        hit = False
        is_neg = False
        for pat in pats:
            m = pat.search(norm)
            if m:
                hit = True
                start = m.start()
                window = norm[max(0, start - 34):start]
                if (re.search(_NEG_WINDOW + r"$", window)
                        and "lack of energy" not in norm[max(0, start - 20):m.end()]):
                    is_neg = True
                break
        if not hit:
            # fallback: order-free proximity match on multi-word phrases
            for ph in LEXICON.get(col, []):
                if len(ph.split()) < 2:
                    continue
                ok, widx = _token_set_hit(ph, words, stem_pos)
                if ok:
                    hit = True
                    is_neg = _negated_before(widx)
                    break
        if hit:
            (negated if is_neg else active).add(col)

    # Pediatric bridge: general symptom + baby context => pediatric symptom too.
    if any(w in norm for w in PEDIATRIC_CONTEXT_WORDS):
        for gen, baby in PEDIATRIC_BRIDGE.items():
            if gen in active:
                active.add(baby)
    if any(w in norm for w in STRICT_INFANT_WORDS):
        if any(p in norm for p in APPETITE_PHRASES):
            active.update({"BABY_POOR_FEEDING", "BABY_APPETITE_LOSS"})
        if any(p in norm for p in CRYING_PHRASES):
            active.add("BABY_EXCESSIVE_CRYING")
    return active, negated


class KnowledgeScorer:
    """IDF-weighted Bernoulli Naive Bayes over the doctor knowledge matrix."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.departments: List[str] = []
        self.symptom_cols: List[str] = []
        self.theta: np.ndarray = np.zeros((0, 0))   # P(symptom | dept)
        self.idf: np.ndarray = np.zeros(0)
        self.log_prior: np.ndarray = np.zeros(0)

    # -- fitting ---------------------------------------------------------
    def fit(self, matrix_df: pd.DataFrame) -> "KnowledgeScorer":
        dept_col = matrix_df.columns[0]
        depts = [str(d).strip() for d in matrix_df[dept_col].tolist()]
        sym_cols = [c for c in matrix_df.columns[1:]]

        # canonical symptom key: strip + upper (handles 'MUSCLE_WEAKNESS.1')
        canon = [c.strip().upper() for c in sym_cols]
        M = (matrix_df.iloc[:, 1:].fillna(0).to_numpy(dtype=float) > 0).astype(float)

        # Extra virtual symptom groups (dermatology etc.) have no matrix
        # columns; they become pseudo symptom columns appended here.
        extra_cols = sorted({k for k in LEXICON if k not in canon})
        if extra_cols:
            M = np.hstack([M, np.zeros((len(depts), len(extra_cols)))])
            canon = canon + extra_cols
            # department affinity for the extra groups, mined from templates
            AFFINITY = {
                "SKIN_RASH": ["DERMATOLOGY", "PEDIATRICS"],
                "STOMACH_ISSUE": ["GASTROENTEROLOGY", "GENERAL MEDICINE"],
                "RESPIRATORY": ["PULMONOLOGY", "GENERAL MEDICINE", "PEDIATRICS"],
                "KIDNEY_ISSUE": ["NEPHROLOGY", "UROLOGY"],
                "BLADDER_ISSUE": ["UROLOGY"],
                "EAR_NOSE_THROAT": ["ENT", "PEDIATRICS"],
                "THYROID_HORMONE": ["ENDOCRINOLOGY"],
                "EYE_PROBLEM": ["OPHTHALMOLOGY"],
                "DENTAL": ["DENTISTRY"],
                "MENTAL_HEALTH": ["PSYCHIATRY", "NEUROLOGY"],
                "CANCER_SIGNS": ["ONCOLOGY"],
                "BLOOD_DISORDER": ["HEMATOLOGY"],
                "AUTOIMMUNE_JOINT": ["RHEUMATOLOGY", "ORTHOPEDICS"],
            }
            for j, ec in enumerate(extra_cols):
                jj = len(sym_cols) + j
                for d in AFFINITY.get(ec, []):
                    if d in depts:
                        M[depts.index(d), jj] = 1.0
        # NOTE: JOINT_PAIN / KNEE_PAIN / ... may already be matrix columns;
        # AFFINITY.get covers only extra cols, matrix cols keep matrix values.

        # template pseudo-counts: mine symptom hits in each dept's templates
        # over the FULL symptom set (matrix + virtual groups).
        n_tpl = np.zeros(len(depts))
        hits = np.zeros_like(M)
        for i, d in enumerate(depts):
            tpl_list = DEPARTMENT_SYMPTOM_TEMPLATES.get(d, [])
            n_tpl[i] = len(tpl_list)
            for tpl in tpl_list:
                act, _ = extract_symptoms(tpl)
                for j, c in enumerate(canon):
                    if c in act:
                        hits[i, j] += 1.0

        s = M + hits
        n = (M.sum(axis=1) > 0).astype(float) + n_tpl  # expert cases per dept
        n = np.maximum(n, 1.0)
        self.theta = (s + self.alpha) / (n[:, None] + 2.0 * self.alpha)
        df = (s > 0).sum(axis=0)
        D = len(depts)
        self.idf = np.log((D + 1.0) / (df + 1.0)) + 1.0
        # uniform prior (no prevalence data); avoids biasing to big classes
        self.log_prior = np.full(D, -np.log(D))
        self.departments = depts
        self.symptom_cols = canon
        self._index = {c: j for j, c in enumerate(canon)}
        return self

    # -- scoring ---------------------------------------------------------
    def log_scores(self, text: str) -> Tuple[np.ndarray, Dict]:
        active, negated = extract_symptoms(text)
        logp = self.log_prior.copy()
        for col in active:
            j = self._index.get(col)
            if j is not None:
                logp += self.idf[j] * np.log(self.theta[:, j] + 1e-12)
        for col in negated:
            j = self._index.get(col)
            if j is not None:
                logp += self.idf[j] * np.log(1.0 - self.theta[:, j] + 1e-12)
        info = {"active": sorted(active), "negated": sorted(negated)}
        return logp, info

    def posterior(self, text: str, temperature: float = 1.0) -> Tuple[np.ndarray, Dict]:
        logp, info = self.log_scores(text)
        z = logp / max(temperature, 1e-6)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum(), info

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        joblib.dump(
            {"departments": self.departments, "symptom_cols": self.symptom_cols,
             "theta": self.theta, "idf": self.idf, "log_prior": self.log_prior,
             "alpha": self.alpha},
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "KnowledgeScorer":
        blob = joblib.load(path)
        obj = cls(alpha=blob.get("alpha", 0.5))
        obj.departments = blob["departments"]
        obj.symptom_cols = blob["symptom_cols"]
        obj.theta = blob["theta"]
        obj.idf = blob["idf"]
        obj.log_prior = blob["log_prior"]
        obj._index = {c: j for j, c in enumerate(obj.symptom_cols)}
        return obj


def build_default_scorer(excel_path: Path = RAW_EXCEL_PATH) -> KnowledgeScorer:
    df = pd.read_excel(excel_path)
    return KnowledgeScorer(alpha=0.5).fit(df)
