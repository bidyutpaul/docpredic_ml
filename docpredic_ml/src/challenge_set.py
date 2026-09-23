"""
DocPredic Challenge Set
=======================
Hand-written realistic patient complaints used ONLY for evaluation —
phrasing is deliberately disjoint from every training template.
Each item: (text, expected_department).
Covers: colloquial language, typos, negations, multi-department overlap,
pediatric references, and urgency suffixes.
"""

CHALLENGE_SET = [
    # GENERAL MEDICINE
    ("down with fever and body ache since yesterday, feeling weak", "GENERAL MEDICINE"),
    ("mild temperature, cold, cough and headache, what doctor?", "GENERAL MEDICINE"),
    ("vomiting twice today with stomach ache and dizziness", "GENERAL MEDICINE"),
    # CARDIOLOGY
    ("sharp pain center of chest while walking, breath short", "CARDIOLOGY"),
    ("heart racing suddenly, pulse irregular, felt faint", "CARDIOLOGY"),
    ("bp 170/100, chest heaviness and swelling in ankles", "CARDIOLOGY"),
    # NEUROLOGY
    ("half head pain with nausea, cannot tolerate light", "NEUROLOGY"),
    ("right hand shakes when writing, steps unsteady", "NEUROLOGY"),
    ("fainted with jerky movements, tongue bitten", "NEUROLOGY"),
    # ORTHOPEDICS
    ("twisted ankle playing football, swollen cannot walk", "ORTHOPEDICS"),
    ("slipped in bathroom, wrist bent wrong, severe pain", "ORTHOPEDICS"),
    ("chronic low back ache radiating down left leg", "ORTHOPEDICS"),
    # PEDIATRICS
    ("6 month old baby fever 101, runny nose, not feeding", "PEDIATRICS"),
    ("my 4yr kid has loose motions and vomiting, dull", "PEDIATRICS"),
    ("newborn crying nonstop, orange urine less, worried", "PEDIATRICS"),
    # GYNECOLOGY & OBSTETRICS
    ("periods 10 days late, morning vomiting, test positive?", "GYNECOLOGY & OBSTETRICS"),
    ("pcos weight gain chin hair, cycles 45 days apart", "GYNECOLOGY & OBSTETRICS"),
    ("8 months pregnant, leg swelling plus headache", "GYNECOLOGY & OBSTETRICS"),
    # DERMATOLOGY
    ("coin shaped itchy patch on arm spreading outward", "DERMATOLOGY"),
    ("face full of painful boils leaving marks", "DERMATOLOGY"),
    ("palms peeling with intense itching at night", "DERMATOLOGY"),
    # GASTROENTEROLOGY
    ("sour water comes to mouth after food, chest burns", "GASTROENTEROLOGY"),
    (" motions watery 8 times, cramps, feeling dehydrated", "GASTROENTEROLOGY"),
    ("piles bleeding fresh blood, constipation hard stool", "GASTROENTEROLOGY"),
    # PULMONOLOGY
    ("wheeze and cough attack midnight, inhaler helps", "PULMONOLOGY"),
    ("beedi smoker, morning cough with yellow sputum", "PULMONOLOGY"),
    ("tb patient contact, cough evening rise temp", "PULMONOLOGY"),
    # NEPHROLOGY
    ("face puffy morning, urine frothy and scanty", "NEPHROLOGY"),
    ("stone right side pain coming in waves, vomited", "NEPHROLOGY"),
    ("creatinine rising, feet swollen, pressure high", "NEPHROLOGY"),
    # UROLOGY
    ("pee burns and urgency every 20 min, feverish", "UROLOGY"),
    ("old father dribbles urine, gets up 5 times night", "UROLOGY"),
    ("leaks urine on sneezing after childbirth", "UROLOGY"),
    # ENT
    ("right ear pus coming, hearing less, pain", "ENT"),
    ("food stuck feeling, throat hurts swallowing saliva", "ENT"),
    ("one nostril blocked, forehead heaviness mornings", "ENT"),
    # ENDOCRINOLOGY
    ("drinks 5 litres water, sugar 320, losing weight", "ENDOCRINOLOGY"),
    ("neck front swelling, palpitations, heat cannot bear", "ENDOCRINOLOGY"),
    ("short height boy 13yrs, bone age delayed", "ENDOCRINOLOGY"),
    # OPHTHALMOLOGY
    ("sudden black curtain right eye vision, urgent", "OPHTHALMOLOGY"),
    ("eyes stick with pus mornings, red, contagious?", "OPHTHALMOLOGY"),
    ("child squints one eye inward when tired", "OPHTHALMOLOGY"),
    # DENTISTRY
    ("molar black hole, jolts on cold water", "DENTISTRY"),
    ("gums bleed brushing, smell bad, tartar visible", "DENTISTRY"),
    ("clip wire cut inside cheek, big ulcer pain", "DENTISTRY"),
    # PSYCHIATRY
    ("mind never stops, no sleep 4 nights, fear future", "PSYCHIATRY"),
    ("cries daily, no joy in kids or food, guilt heavy", "PSYCHIATRY"),
    ("doubts wife plotting, hears whispers at night", "PSYCHIATRY"),
    # ONCOLOGY
    ("armpit lump hard fixed size growing month", "ONCOLOGY"),
    ("betel chewer white patch cheek not healing", "ONCOLOGY"),
    ("rectal bleeding with thin stools, 10kg down", "ONCOLOGY"),
    # HEMATOLOGY
    ("hb 6.8 giddiness climbing, craves chalk", "HEMATOLOGY"),
    ("dengue platelets falling, red dots on shin", "HEMATOLOGY"),
    ("bleeder son knee balloons after trivial knock", "HEMATOLOGY"),
    # RHEUMATOLOGY
    ("fingers stiff claw-like mornings, symmetric both hands", "RHEUMATOLOGY"),
    ("cheek rash sun exposure plus wrist swelling", "RHEUMATOLOGY"),
    ("midnight big-toe fire pain after wedding feast", "RHEUMATOLOGY"),
    # Negation / overlap edge cases
    ("no fever, no cold, only one-sided headache vomiting", "NEUROLOGY"),
    ("chest pain going to left arm, feels like acidity but radiates", "CARDIOLOGY"),
    ("baby cold cough but no fever, active playful", "PEDIATRICS"),
    ("joint pain all over without swelling, tired months", "RHEUMATOLOGY"),
]
