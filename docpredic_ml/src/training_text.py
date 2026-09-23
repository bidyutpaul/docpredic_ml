"""
DocPredic Enriched Training-Text Generator
==========================================
Produces diverse, realistic patient utterances for every department by
combining three sources:

1. The doctor knowledge matrix (binary symptom rows) — sampled 1-4 active
   symptoms, verbalised with colloquial variants from the knowledge lexicon.
2. Curated colloquial templates per department (slang, typos, abbreviations,
   varied grammar) — disjoint in style from the base templates so the model
   generalises instead of memorising.
3. Character-level typo noise on a fraction of samples (teaches char n-grams).

Deterministic given `seed`.
"""

import random
import re
from typing import Dict, List

from .data_loader import SYMPTOM_DESCRIPTIONS

# Colloquial, realistic patient phrasings per department. Deliberately varied:
# different sentence structure, slang, abbreviations, mild typos.
COLLOQUIAL_TEMPLATES: Dict[str, List[str]] = {
    "GENERAL MEDICINE": [
        "fevr since 3 days, full body pain and no energy doc",
        "feeling sick overall, hedake, runny tummy and vomiting",
        "high fever with shivering, cough and weakness all over",
        "sir i have bukhar, sar dard and loose motions",  # code-mixed realism
        "flu like symptoms, body ache, cold and mild fever",
        "tired all the time, low grade fever, headache on and off",
        "stomach upset, nausea after food, general weakness",
        "viral type fever, throat irritation, body pain, fatigue",
    ],
    "CARDIOLOGY": [
        "chest pain left side going to arm, heavy sweating",
        "heart beating very fast, breathless on climbing stairs",
        "bp high, chest tight, dizzy spells since morning",
        "palpitations at night, heart pounds, ankles swollen",
        "crushing pain in chest, pain radiating to jaw and shoulder",
        "short breath even at rest, legs swollen, extreme tiredness",
        "irregular heartbeat, feel faint, chest discomfort after walking",
        "tightness in chest with cold sweat, pls suggest cardiologist",
    ],
    "NEUROLOGY": [
        "one side headache with vomiting, light bothers me",
        "hands trembling while holding cup, forgetting names lately",
        "had a fit yesterday, bit my tongue, confused after",
        "numbness in left leg with burning nerve pain",
        "slurred speech since morning, face slightly drooping",
        "dizzy when standing, imbalance while walking straight",
        "memory getting weak, repeats same questions again and again",
        "stiffness in limbs, slow movement, trouble writing",
        "fainted with jerky body movements, tongue bitten",
        "only one side headache with vomiting, no fever no cold",
        "giddiness with blackouts for seconds, falls down",
        "one sided pounding headache with vomiting, migraine",
        "half side head pain with nausea, migraine attack",
    ],
    "ORTHOPEDICS": [
        "knee pain while climbing stairs, cracking sound in joint",
        "lower back sprain after lifting weight, pain goes to leg",
        "fractured wrist after bike skid, swelling and deformity",
        "shoulder dislocated during cricket, cannot raise arm",
        "heel pain first step in morning, eases later",
        "neck stiffness from desk job, radiates to right arm",
        "ankle sprain, swollen, cannot put weight on foot",
        "hip joint pain, limping while walking long distance",
        "wrist bent wrong after fall, deformed, severe pain",
        "collar bone fracture, shoulder drooping one side",
        "low back ache radiates to leg, chronic sciatica pain",
        "back pain with leg numbness while walking far",
    ],
    "PEDIATRICS": [
        "my 8 month baby has fever 102, cold and cough",
        "toddler vomiting everything including water, very dull",
        "newborn not latching, cries continuously, less urine",
        "my son has rashes on body with itching and mild fever",
        "3 yr child ear pain, pulling ear, fever at night",
        "baby motions loose green colour, 5 times today",
        "infant breathing fast with wheezing sound in chest",
        "my daughter underweight, not growing like other kids",
        "my 4yr kid loose motions and vomiting, no appetite",
        "baby cold cough no fever but breathing noisy",
        "toddler orange urine very less, cries while peeing",
        "child worms in stool, itching at night, eats mud",
        "newborn jaundice eyes yellow day 3, sleepy not feeding",
        "baby vaccine due, mild fever after shot, fussy",
        "kid vomiting with loose motions, started ors at home",
        "4 year old dull, vomiting and motions since night",
        "child motions plus vomiting, mild dehydration, needs doctor",
    ],
    "GYNECOLOGY & OBSTETRICS": [
        "periods delayed by 12 days, urine pregnancy test faint line",
        "heavy bleeding with clots, changing pads every hour",
        "pcod problem, weight gain, facial hair growth",
        "white discharge with itching, burning sensation",
        "pregnant 7 months, swelling in feet, bp 140/90",
        "severe period cramps, vomiting first day every month",
        "missed periods 2 months, nausea morning time",
        "lower abdomen pain with irregular spotting between periods",
    ],
    "DERMATOLOGY": [
        "red itchy patches on elbows with white scaling",
        "pimples with pus on face and back, painful",
        "ring shaped itchy patch on thigh spreading slowly",
        "skin allergy after new soap, hives all over body",
        "dark patches on cheeks, melasma type pigmentation",
        "excessive hair fall with dandruff and itchy scalp",
        "fungal infection between toes, peeling and itching",
        "small warts on neck increasing in number",
        "face full of painful boils leaving marks",
        "palms peeling intense itching worse at night",
        "pus filled bumps on thighs, recurrent boils",
    ],
    "GASTROENTEROLOGY": [
        "burning in chest after meals, sour burps, acidity",
        "loose motions 6-7 times, stomach cramps, dehydration",
        "constipation since a week, hard stools, bleeding piles",
        "upper stomach pain empty stomach, relieved after eating",
        "jaundice? eyes yellow, dark urine, loss of appetite",
        "bloating and gas after every meal, indigestion",
        "vomiting blood once, black stools, severe weakness",
        "chronic liver issue, ascites belly swelling, alcohol history",
    ],
    "PULMONOLOGY": [
        "asthma attack at night, nebulizer needed twice",
        "chronic smoker cough with phlegm every morning",
        "breathless on walking 100m, wheezing sound",
        "dry cough 3 weeks not going, weight loss also",
        "chest infection, green sputum, fever with chills",
        "copd patient, oxygen drops on exertion",
        "allergic cough worse at night and early morning",
        "tb contact history, evening fever, cough with blood streaks",
    ],
    "NEPHROLOGY": [
        "both feet swollen, urine less and frothy",
        "kidney stone pain right flank radiating to groin",
        "creatinine 2.4, swelling face in morning, bp high",
        "dialysis patient missed session, breathless and swollen",
        "blood in urine with lower back dull ache",
        "recurrent kidney infections, burning plus fever",
        "stone pain coming in waves right side with vomiting",
        "creatinine rising, feet swollen, pressure high",
        "face puffy morning, urine frothy and scanty",
    ],
    "UROLOGY": [
        "burning while peeing, going every 30 mins, urgent",
        "kidney stone 6mm, severe colicky pain vomiting",
        "prostate issue, weak stream, night 4-5 times urine",
        "blood drops after urination, lower abdomen heaviness",
        "bedwetting 9 yr child, plus daytime urgency",
        "incontinence after delivery, leaks on coughing",
        "pee burns urgency every 20 min with fever",
        "old father dribbles urine, up 5 times at night",
        "leaks urine on sneezing after childbirth",
    ],
    "ENT": [
        "ear discharge with pain, hearing reduced right side",
        "tonsils swollen, pain swallowing even water",
        "nose blocked one side, headache forehead, sinus",
        "vertigo episodes, room spins 5 mins, vomiting",
        "throat pain with voice change 2 weeks, smoker",
        "nosebleed recurrent left nostril, picks nose habit kid",
        "right ear pus coming, hearing less with pain",
        "food stuck feeling, throat hurts swallowing saliva",
        "one nostril blocked, forehead heaviness mornings",
    ],
    "ENDOCRINOLOGY": [
        "sugar 280 fasting, more thirst, urine frequently",
        "thyroid swelling neck, weight gain, always cold",
        "tired, hair falling, periods irregular, prolactin?",
        "height not increasing 14yr boy, growth concern",
        "sweating, tremors, weight loss, thyroid overactive?",
        "diabetic foot ulcer not healing since a month",
        "front neck swelling with palpitations, cannot bear heat",
        "short height boy bone age delayed, thyroid testing?",
    ],
    "OPHTHALMOLOGY": [
        "vision blurred right eye since 2 days, floaters seen",
        "eye redness with discharge sticking lids morning",
        "itchy eyes in dust, watery, allergic conjunctivitis?",
        "need specs power check, headache after screen time",
        "eye injury with cracker, pain and blood inside eye",
        "squint in child, eye turns inward while reading",
        "sudden black curtain over vision, urgent",
        "eyes stick with pus mornings, red, contagious?",
    ],
    "DENTISTRY": [
        "wisdom tooth coming, cheek swollen, cannot open mouth",
        "cavity black spot molar, pain on sweets and cold",
        "gum swelling with pus, bad smell from mouth",
        "braces wire poking cheek, ulcer formed",
        "tooth broken in accident, sharp edge cutting tongue",
        "sensitivity in all teeth on hot tea, enamel worn?",
        "gums bleed on brushing, bad smell, tartar on teeth",
        "molar hole jolts on cold water, keeps food stuck",
    ],
    "PSYCHIATRY": [
        "cannot sleep whole night, overthinking, restless mind",
        "panic in crowds, heartbeat fast, sweating, want to run",
        "lost interest in everything, cries without reason",
        "anger outbursts, beats kids, regrets later, wants help",
        "hearing voices commenting, suspicious of family",
        "exam fear, blank mind, sweaty palms, poor concentration",
        "cries daily no joy in kids or food, guilt heavy",
        "doubts wife plotting, hears whispers at night",
        "mind never stops no sleep many nights, fears future",
        "no interest in children or food, cries often with guilt",
        "lost joy in family and meals, weeps daily, feels guilty",
    ],
    "ONCOLOGY": [
        "breast lump painless growing, nipple retracted",
        "blood in stool with weight loss 8kg, altered habit",
        "lymph node neck hard fixed, night sweats",
        "mouth ulcer 3 weeks tobacco chewer, white patch",
        "post chemo weakness, counts low, fever spikes",
        "prostate psa high, difficulty urine plus back pain",
        "rectal bleeding with thin stools, 10kg down",
        "armpit lump hard fixed size growing since a month",
    ],
    "HEMATOLOGY": [
        "hb 7.2, breathless on exertion, pale, pica mud eating",
        "platelets 30k dengue, gum bleed, red spots legs",
        "thalassemia minor child, needs folic acid followup",
        "clots in leg veins, swelling one leg, dvt?",
        "hemophilia boy, knee swollen after small fall",
        "leukemia treatment, frequent infections, low counts",
        "hb 6.8 giddiness climbing stairs, craves chalk",
        "bleeder son knee balloons after trivial knock",
        "dengue platelets falling, red dots on shin",
    ],
    "RHEUMATOLOGY": [
        "small joints hands swollen symmetric, morning stiff 2hrs",
        "ra factor positive, deformity fingers starting",
        "lupus rash face butterfly, joint pains, hair loss",
        "gout big toe sudden night pain red hot swollen",
        "back stiffness morning improves movement, as?",
        "dry eyes dry mouth with joint pains, sjogren?",
        "cheek rash after sun plus wrist swelling",
        "joint pains all over months without swelling, tired",
        "fingers stiff mornings symmetric both hands",
        "big toe fire pain midnight after feast, gout?",
    ],
}

PREFIXES = [
    "I have ", "I am suffering from ", "I am having ", "Suffering from ",
    "Having ", "Got ", "Having trouble with ", "Problem of ",
    "My child has ", "My baby has ",
]
SUFFIX_URGENCY = [
    "", "", "", "",
    " please suggest which doctor to consult.",
    " which specialist should I see?",
    " kindly advise department.",
    " need doctor recommendation.",
]


def _typo_noise(text: str, rng: random.Random, prob: float = 0.06) -> str:
    """Inject light character-level noise (swap/drop/repeat) into long words."""
    words = text.split()
    out = []
    for w in words:
        if len(w) > 5 and rng.random() < prob:
            op = rng.choice(["swap", "drop", "repeat"])
            i = rng.randrange(1, len(w) - 1)
            if op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def _verbalise_matrix_row(active_cols: List[str], rng: random.Random) -> str:
    """Turn 1-4 matrix symptom columns into a natural utterance."""
    k = min(len(active_cols), rng.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0])
    sampled = rng.sample(active_cols, k)
    parts = []
    for col in sampled:
        key = col.strip().upper()
        if key in SYMPTOM_DESCRIPTIONS:
            parts.append(rng.choice(SYMPTOM_DESCRIPTIONS[key]))
        else:
            clean = key.lower().replace("_", " ").replace(".1", "")
            parts.append(f"I have {clean}")
    if len(parts) == 1:
        text = parts[0]
        if rng.random() < 0.3:
            text = rng.choice([
                "I have ", "I am having ", "Suffering from ", "Having ",
                "Problem of ", "Complaint of ",
            ]) + text[0].lower() + text[1:]
    else:
        conn = rng.choice([", along with ", " and ", ", plus ", ", and also ",
                           " with ", " along with ", " as well as "])
        first, rest = parts[0], parts[1:]
        lowered = [r[0].lower() + r[1:] if r[:2].lower() in ("i ", "my") else r
                   for r in rest]
        # strip leading "I have"/"my" duplicates for flow
        stripped = [re.sub(r"^(i have|i am|i feel|my)\s+", "", r) for r in lowered]
        text = first + conn + conn.join(stripped)
    if rng.random() < 0.35:
        text = rng.choice(PREFIXES).rstrip() + " " + text[0].lower() + text[1:] \
            if rng.random() < 0.3 else text
    if rng.random() < 0.25:
        text = text + rng.choice(SUFFIX_URGENCY)
    return text


def generate_enriched_texts(matrix_df, n_per_class: int = 400,
                             seed: int = 42) -> "pd.DataFrame":
    """Balanced enriched utterances for every department in the matrix."""
    import pandas as pd

    rng = random.Random(seed)
    dept_col = matrix_df.columns[0]
    sym_cols = list(matrix_df.columns[1:])
    records = []
    for _, row in matrix_df.iterrows():
        dept = str(row[dept_col]).strip()
        active = [c for c in sym_cols if row[c] == 1]
        colloq = COLLOQUIAL_TEMPLATES.get(dept, [])
        n_colloq = min(len(colloq) * 8, n_per_class // 3) if colloq else 0
        n_matrix = n_per_class - n_colloq
        # 1. matrix-verbalised utterances (grounded in doctor knowledge)
        for _ in range(n_matrix):
            if active:
                t = _verbalise_matrix_row(active, rng)
                if rng.random() < 0.35:
                    t = _typo_noise(t, rng, prob=0.04)
            else:
                t = rng.choice(colloq) if colloq else \
                    "I am feeling unwell and need to see a doctor."
                if rng.random() < 0.5:
                    t = _typo_noise(t, rng)
            records.append({"text": t, "department": dept})
        # 2. colloquial templates, each repeated with typo variants
        for _ in range(n_colloq):
            t = rng.choice(colloq)
            if rng.random() < 0.5:
                t = _typo_noise(t, rng)
            if rng.random() < 0.3:
                t = t + rng.choice(SUFFIX_URGENCY)
            records.append({"text": t, "department": dept})

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
