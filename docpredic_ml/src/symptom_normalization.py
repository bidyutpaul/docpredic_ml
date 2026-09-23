"""
DocPredic Symptom Normalization
Maps free-text symptoms to canonical symptom categories.
"""
import re
from typing import Dict, List, Set
from collections import defaultdict


SYMPTOM_CATEGORIES: Dict[str, List[str]] = {
    "fever": ["fever", "temperature", "feverish", "hot", "high temperature"],
    "cough": ["cough", "coughing", "hack"],
    "fatigue": ["tired", "fatigue", "exhausted", "no energy", "lethargic", "drained", "weak"],
    "headache": ["headache", "head pain", "head hurt", "migraine", "head ache"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "vertigo", "spinning"],
    "nausea": ["nauseous", "nausea", "queasy", "sick to stomach"],
    "vomiting": ["vomit", "vomiting", "throwing up", "threw up"],
    "chest_pain": ["chest pain", "chest hurt", "chest ache", "chest tightness"],
    "breathing_difficulty": ["breathing difficulty", "shortness of breath", "short of breath", "breathless", "cannot breathe", "dyspnea"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain", "abdomen hurt", "stomach ache"],
    "heart_palpitations": ["palpitations", "heart racing", "heart pounding", "rapid heartbeat"],
    "joint_pain": ["joint pain", "joint ache", "arthralgia"],
    "back_pain": ["back pain", "back ache", "back hurt"],
    "numbness": ["numbness", "numb", "loss of sensation"],
    "tingling": ["tingling", "pins and needles", "prickling"],
    "seizures": ["seizure", "convulsion", "fit", "epilepsy"],
    "tremors": ["tremor", "shaking", "trembling"],
    "muscle_weakness": ["muscle weakness", "weak muscles", "muscle fatigue"],
    "rash": ["rash", "skin rash", "eruption", "skin irritation"],
    "swelling": ["swelling", "swollen", "edema", "puffiness", "inflammation"],
    "pain": ["pain", "ache", "sore", "hurts", "soreness"],
    "confusion": ["confusion", "confused", "disoriented", "muddled"],
    "memory_loss": ["memory loss", "forgetfulness", "forgetting", "memory problem"],
    "insomnia": ["insomnia", "sleep problem", "cannot sleep", "sleep difficulty", "trouble sleeping"],
    "appetite_loss": ["appetite loss", "not eating", "no appetite", "refusing food"],
    "weight_loss": ["weight loss", "losing weight", "underweight"],
    "dehydration": ["dehydration", "dehydrated", "dry mouth"],
    "period_pain": ["period pain", "menstrual cramp", "cramps"],
    "irregular_periods": ["irregular period", "irregular cycle", "inconsistent period"],
    "itching": ["itching", "itchy", "pruritus"],
    "dryness": ["dryness", "dry skin", "lack of moisture"],
}


def _build_symptom_index() -> Dict[str, str]:
    index = {}
    for category, keywords in SYMPTOM_CATEGORIES.items():
        for keyword in keywords:
            index[keyword.lower()] = category
    return index


_symptom_index = _build_symptom_index()


def extract_symptoms(text: str) -> Set[str]:
    text_lower = text.lower()
    found = set()
    for keyword, category in _symptom_index.items():
        if keyword in text_lower:
            found.add(category)
    return found


def normalize_symptoms_to_vector(text: str, all_categories: List[str] = None) -> Dict[str, int]:
    if all_categories is None:
        all_categories = sorted(SYMPTOM_CATEGORIES.keys())
    symptoms = extract_symptoms(text)
    return {cat: 1 if cat in symptoms else 0 for cat in all_categories}


def add_symptom_features(df, text_col: str):
    df = df.copy()
    all_categories = sorted(SYMPTOM_CATEGORIES.keys())
    symptom_vectors = df[text_col].apply(lambda x: normalize_symptoms_to_vector(x, all_categories))
    symptom_df = pd.DataFrame(symptom_vectors.tolist())
    df = pd.concat([df, symptom_df], axis=1)
    return df


import pandas as pd
