"""
DocPredic Data Loader
Handles loading raw data and generating synthetic text from binary symptom matrix.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

from .config import RAW_EXCEL_PATH, PROCESSED_DATA_DIR, TEXT_COL, LABEL_COL, SEED


SYMPTOM_DESCRIPTIONS: Dict[str, List[str]] = {
    "FEVER": [
        "I have a fever", "my body temperature is high", "I feel feverish",
        "I have a high temperature", "I am running a fever", "I feel hot and feverish",
    ],
    "COUGH": [
        "I have a persistent cough", "I am coughing a lot", "I have a dry cough",
        "I have a wet cough", "I keep coughing", "I have been coughing for days",
    ],
    "FATIGUE": [
        "I feel very tired", "I have fatigue", "I feel exhausted",
        "I have no energy", "I feel weak and tired", "I am constantly fatigued",
    ],
    "WEAKNESS": [
        "I feel weak all over", "my body feels weak", "I have general weakness",
        "I feel physically weak", "I lack strength", "I feel drained and weak",
    ],
    "HEADACHE": [
        "I have a severe headache", "I have been having headaches", "my head hurts",
        "I have a throbbing headache", "I have persistent head pain", "I have migraine-like headaches",
    ],
    "DIZZINESS": [
        "I feel dizzy", "I am experiencing dizziness", "I feel lightheaded",
        "the room feels like it is spinning", "I feel unsteady and dizzy",
        "I have vertigo-like dizziness",
    ],
    "BODY_PAIN": [
        "I have body pain", "my whole body aches", "I have generalized body pain",
        "I feel achy all over", "my body is sore", "I have widespread body pain",
    ],
    "ABDOMINAL_PAIN": [
        "I have stomach pain", "my abdomen hurts", "I have abdominal cramps",
        "I feel pain in my belly", "my stomach is hurting", "I have lower abdominal pain",
    ],
    "NAUSEA": [
        "I feel nauseous", "I am feeling sick to my stomach", "I feel like I might throw up",
        "I have nausea", "I feel queasy", "I have an upset stomach with nausea",
    ],
    "VOMITING": [
        "I have been vomiting", "I vomited multiple times", "I am throwing up",
        "I have been throwing up", "I cannot keep food down", "I have persistent vomiting",
    ],
    "CHEST_PAIN": [
        "I have chest pain", "my chest hurts", "I feel a sharp pain in my chest",
        "I have pressure in my chest", "my chest feels tight and painful",
        "I have intermittent chest pain",
    ],
    "BREATHING_DIFFICULTY": [
        "I have difficulty breathing", "I am short of breath", "I feel breathless",
        "I cannot breathe properly", "I have trouble catching my breath",
        "I experience breathing difficulty",
    ],
    "HEART_PALPITATIONS": [
        "I feel my heart racing", "I have heart palpitations", "my heart is beating fast",
        "I feel my heart pounding", "I have irregular heartbeats", "my heart skips beats",
    ],
    "IRREGULAR_PULSE": [
        "I have an irregular pulse", "my heartbeat is irregular", "I feel pulse irregularity",
        "my pulse rate is inconsistent", "I have an uneven heartbeat",
    ],
    "HIGH_BP": [
        "I have high blood pressure", "my blood pressure is elevated", "I have hypertension",
        "my BP readings are high", "I have been diagnosed with high blood pressure",
    ],
    "FAINTING": [
        "I have been fainting", "I feel like I might faint", "I fainted recently",
        "I have episodes of fainting", "I feel like I am going to pass out",
    ],
    "FAST_HEARTBEAT": [
        "I have a fast heartbeat", "my heart rate is elevated", "I have tachycardia",
        "my heart beats very fast", "I have a rapid pulse",
    ],
    "SLOW_HEARTBEAT": [
        "I have a slow heartbeat", "my heart rate is low", "I have bradycardia",
        "my heart beats slowly", "I have a slow pulse rate",
    ],
    "EXTREME_FATIGUE": [
        "I have extreme fatigue", "I am incredibly tired", "I feel utterly exhausted",
        "I have debilitating fatigue", "I can barely stay awake",
    ],
    "LEG_SWELLING": [
        "my legs are swollen", "I have swelling in my legs", "my ankles and legs are puffy",
        "I have leg edema", "my legs feel heavy and swollen",
    ],
    "CHEST_TIGHTNESS": [
        "I feel chest tightness", "my chest feels tight", "I have a tight feeling in my chest",
        "my chest feels constricted", "I have pressure and tightness in my chest",
    ],
    "ANKLE_SWELLING": [
        "my ankles are swollen", "I have swollen ankles", "my ankles are puffy",
        "I have ankle edema", "my ankles feel swollen and painful",
    ],
    "EXCESS_SWEATING": [
        "I am sweating excessively", "I have excessive sweating", "I sweat a lot",
        "I have night sweats", "I am drenched in sweat",
    ],
    "BACK_PAIN": [
        "I have back pain", "my back hurts", "I have lower back pain",
        "I have upper back pain", "my back is very sore",
    ],
    "SEIZURES": [
        "I have had seizures", "I experienced a seizure", "I have epilepsy-like seizures",
        "I have convulsions", "I had a seizure episode",
    ],
    "NUMBNESS": [
        "I feel numbness in my body", "I have numbness", "I feel tingling and numb",
        "parts of my body feel numb", "I have numbness in my extremities",
    ],
    "TINGLING": [
        "I have tingling sensations", "I feel tingling in my hands", "I have pins and needles feeling",
        "I experience tingling in my feet", "I have a tingling sensation all over",
    ],
    "MUSCLE_WEAKNESS": [
        "I have muscle weakness", "my muscles feel weak", "I cannot move my muscles properly",
        "I have progressive muscle weakness", "my muscles feel like they are failing",
    ],
    "TREMORS": [
        "I have tremors", "my hands are shaking", "I experience involuntary shaking",
        "I have a tremor in my hands", "my body shakes involuntarily",
    ],
    "MEMORY_LOSS": [
        "I have memory problems", "I keep forgetting things", "I have memory loss",
        "I cannot remember recent events", "my memory is getting worse",
    ],
    "CONFUSION": [
        "I feel confused", "I have mental confusion", "I am disoriented",
        "I cannot think clearly", "I feel muddled and confused",
    ],
    "BALANCE_PROBLEM": [
        "I have balance problems", "I feel unsteady", "I have trouble maintaining balance",
        "I feel like I might fall", "I have poor balance and coordination",
    ],
    "DIFFICULTY_WALKING": [
        "I have difficulty walking", "I cannot walk properly", "I struggle to walk",
        "I have trouble walking", "I walk with difficulty",
    ],
    "SPEECH_DIFFICULTY": [
        "I have trouble speaking", "I have difficulty with speech", "my speech is slurred",
        "I cannot speak clearly", "I have speech problems",
    ],
    "FACIAL_WEAKNESS": [
        "I have facial weakness", "one side of my face feels weak", "my face droops",
        "I have facial paralysis", "my facial muscles feel weak",
    ],
    "CONSCIOUSNESS_LOSS": [
        "I have lost consciousness", "I fainted and lost consciousness", "I blacked out",
        "I had a loss of consciousness", "I was unconscious for a period",
    ],
    "NERVE_PAIN": [
        "I have nerve pain", "I experience shooting nerve pain", "I have neuropathic pain",
        "I feel burning nerve pain", "I have sharp nerve pain",
    ],
    "MUSCLE_STIFFNESS": [
        "I have muscle stiffness", "my muscles feel stiff", "I have rigid muscles",
        "my muscles are tight and stiff", "I experience muscle rigidity",
    ],
    "POOR_COORDINATION": [
        "I have poor coordination", "I am clumsy", "I cannot coordinate my movements well",
        "I have trouble with fine motor tasks", "I feel uncoordinated",
    ],
    "SLEEP_PROBLEM": [
        "I have sleep problems", "I cannot sleep well", "I have insomnia",
        "I have trouble falling asleep", "I wake up frequently at night",
    ],
    "JOINT_PAIN": [
        "I have joint pain", "my joints hurt", "I feel pain in my joints",
        "my joints are aching", "I have painful joints",
    ],
    "NECK_PAIN": [
        "I have neck pain", "my neck hurts", "I have a stiff neck",
        "I feel pain in my neck", "my neck is very painful",
    ],
    "KNEE_PAIN": [
        "I have knee pain", "my knee hurts", "I feel pain in my knee",
        "my knee is swollen and painful", "I have difficulty with my knee",
    ],
    "SHOULDER_PAIN": [
        "I have shoulder pain", "my shoulder hurts", "I feel pain in my shoulder",
        "I have a frozen shoulder", "my shoulder is very sore",
    ],
    "HIP_PAIN": [
        "I have hip pain", "my hip hurts", "I feel pain in my hip",
        "I have hip joint pain", "my hip is stiff and painful",
    ],
    "BONE_PAIN": [
        "I have bone pain", "my bones ache", "I feel deep bone pain",
        "I have persistent bone pain", "my bones feel sore",
    ],
    "MUSCLE_PAIN": [
        "I have muscle pain", "my muscles ache", "I feel muscle soreness",
        "I have muscle cramps and pain", "my muscles are painful",
    ],
    "JOINT_SWELLING": [
        "my joints are swollen", "I have swollen joints", "I have joint swelling",
        "my joints feel puffy and inflamed", "I have inflamed swollen joints",
    ],
    "JOINT_STIFFNESS": [
        "my joints are stiff", "I have joint stiffness", "I feel stiff in my joints",
        "I have trouble moving my joints", "my joints feel rigid",
    ],
    "LIMITED_MOVEMENT": [
        "I have limited range of motion", "I cannot move my joint fully",
        "I have restricted movement", "my movement is limited", "I have difficulty moving",
    ],
    "BONE_FRACTURE": [
        "I have a bone fracture", "my bone is broken", "I fractured a bone",
        "I have a stress fracture", "I have a broken bone",
    ],
    "MUSCLE_WEAKNESS.1": [
        "I have progressive muscle weakness", "my muscles are getting weaker",
        "I experience muscle fatigue and weakness", "my muscle strength is declining",
    ],
    "TINGLING_SENSATION": [
        "I have a constant tingling sensation", "I feel tingling throughout my body",
        "I experience unusual tingling", "I have persistent tingling",
    ],
    "HEEL_PAIN": [
        "I have heel pain", "my heel hurts when I walk", "I have pain in my heel",
        "I have plantar fasciitis-like heel pain", "my heel feels painful",
    ],
    "SPORTS_INJURY": [
        "I have a sports injury", "I got injured while playing sports",
        "I have a sprain from sports", "I injured myself during exercise",
    ],
    "BABY_FEVER": [
        "my baby has a fever", "my child has a high temperature",
        "my infant is feverish", "my baby feels very hot",
    ],
    "BABY_COUGH": [
        "my baby is coughing", "my child has a cough",
        "my baby has a persistent cough", "my infant is coughing a lot",
    ],
    "BABY_COLD": [
        "my baby has a cold", "my child has cold symptoms",
        "my baby is sneezing and has a runny nose", "my infant has a cold",
    ],
    "BABY_SORE_THROAT": [
        "my baby has a sore throat", "my child is complaining of throat pain",
        "my baby's throat is sore", "my infant has throat pain",
    ],
    "BABY_BREATHING_DIFFICULTY": [
        "my baby has breathing difficulty", "my child is struggling to breathe",
        "my baby is short of breath", "my infant has breathing problems",
    ],
    "BABY_VOMITING": [
        "my baby is vomiting", "my child has been throwing up",
        "my baby cannot keep milk down", "my infant is vomiting",
    ],
    "BABY_STOMACH_PAIN": [
        "my baby has stomach pain", "my child is crying from stomach pain",
        "my baby has a stomach ache", "my infant has abdominal pain",
    ],
    "BABY_SKIN_RASH": [
        "my baby has a skin rash", "my child has a rash on their body",
        "my baby's skin is irritated", "my infant has a red rash",
    ],
    "BABY_SEIZURES": [
        "my baby had a seizure", "my child is having convulsions",
        "my baby had a seizure episode", "my infant had a fit",
    ],
    "BABY_POOR_FEEDING": [
        "my baby is not feeding well", "my child refuses to eat",
        "my baby has poor appetite", "my infant is not drinking milk properly",
    ],
    "BABY_EXCESSIVE_CRYING": [
        "my baby cries excessively", "my child cries all the time",
        "my baby is inconsolable", "my infant cries uncontrollably",
    ],
    "BABY_WEAKNESS": [
        "my baby seems weak", "my child is lethargic",
        "my baby has low energy", "my infant appears weak and floppy",
    ],
    "BABY_APPETITE_LOSS": [
        "my baby has lost appetite", "my child is not eating",
        "my baby refuses to eat", "my infant has no appetite",
    ],
    "BABY_LOW_WEIGHT": [
        "my baby is underweight", "my child is not gaining weight properly",
        "my baby has low birth weight", "my infant is too small",
    ],
    "BABY_GROWTH_DELAY": [
        "my baby has delayed growth", "my child is not growing properly",
        "my baby is smaller than expected", "my infant has growth issues",
    ],
    "BABY_DEVELOPMENT_DELAY": [
        "my baby has developmental delays", "my child is not meeting milestones",
        "my baby is developing slowly", "my infant has delayed development",
    ],
    "BABY_EAR_PAIN": [
        "my baby has ear pain", "my child is pulling at their ear",
        "my baby has an ear infection", "my infant cries from ear pain",
    ],
    "BABY_YELLOW_SKIN": [
        "my baby has yellow skin", "my child looks jaundiced",
        "my baby's skin is turning yellow", "my infant has jaundice",
    ],
    "BABY_DEHYDRATION": [
        "my baby is dehydrated", "my child is not drinking enough fluids",
        "my baby has dry mouth and no tears", "my infant shows signs of dehydration",
    ],
    "BABY_WHEEZING": [
        "my baby is wheezing", "my child has a wheezing sound when breathing",
        "my baby wheezes when breathing", "my infant has wheezing",
    ],
    "PCOS": [
        "I have PCOS", "I have polycystic ovary syndrome",
        "I was diagnosed with PCOS", "I have cysts on my ovaries",
    ],
    "IRREGULAR PERIODS": [
        "I have irregular periods", "my menstrual cycle is irregular",
        "my periods are not regular", "I have inconsistent menstrual cycles",
    ],
    "PREGNANCY-RELATED PROBLEMS": [
        "I am having pregnancy complications", "I have pregnancy-related problems",
        "I am experiencing issues during pregnancy", "I have problems with my pregnancy",
    ],
    "MENSTRUAL PROBLEMS": [
        "I have menstrual problems", "I have issues with my periods",
        "I experience menstrual difficulties", "I have problems during menstruation",
    ],
    "MILD_PERIOD_PAIN": [
        "I have mild period pain", "I feel slight cramps during my period",
        "I have mild menstrual cramps", "my period pain is mild",
    ],
    "HEAVY_PERIOD": [
        "I have heavy periods", "my periods are very heavy",
        "I experience heavy menstrual bleeding", "I have menorrhagia",
    ],
    "LIGHT_PERIOD": [
        "I have very light periods", "my periods are barely there",
        "I have scanty periods", "my menstrual flow is very light",
    ],
    "DELAYED_PERIOD": [
        "my period is late", "I have a delayed period",
        "my period has been delayed", "my menstruation is overdue",
    ],
    "MISSED_PERIOD": [
        "I missed my period", "I have a missed period",
        "my period did not come this month", "I have amenorrhea",
    ],
    "VAGINAL_ITCHING": [
        "I have vaginal itching", "I feel itchy down there",
        "I have persistent vaginal itching", "my vaginal area is very itchy",
    ],
    "VAGINAL_DRYNESS": [
        "I have vaginal dryness", "I feel dry in the vaginal area",
        "I experience vaginal dryness", "my vaginal area lacks moisture",
    ],
    "HORMONAL_ACNE": [
        "I have hormonal acne", "I get acne due to hormones",
        "I have persistent acne from hormonal imbalance", "my acne is hormone-related",
    ],
    "LOWER_ABDOMINAL_PAIN": [
        "I have lower abdominal pain", "I feel pain in my lower abdomen",
        "my lower belly hurts", "I have pain in my lower stomach area",
    ],
}
DEPARTMENT_SYMPTOM_TEMPLATES: Dict[str, List[str]] = {
    "GENERAL MEDICINE": [
        "I have been having a high fever for three days along with severe body pain and weakness.",
        "I feel very tired, weak, have a persistent cough, and mild headache.",
        "I have stomach ache, nausea, vomiting, and general body fatigue.",
        "Running a temperature, feeling dizzy, and having general body soreness.",
        "I have a fever, cough, fatigue, and feeling weak all over.",
    ],
    "CARDIOLOGY": [
        "I feel a sharp chest pain and tightness, especially when walking, along with shortness of breath.",
        "My heart rate is very high, I feel heart palpitations, dizziness, and ankle swelling.",
        "I am having chest tightness, irregular pulse rate, high blood pressure, and extreme fatigue.",
        "I felt severe chest pain radiating to my left arm, sweating, and difficulty breathing.",
        "I experience shortness of breath, rapid heart rate, and swelling in my lower legs.",
    ],
    "NEUROLOGY": [
        "I have had severe throbbing headaches for two days, dizzy feeling, and vomited twice.",
        "I feel numbness and tingling sensation in my left arm, along with muscle weakness.",
        "I experienced a sudden seizure episode, confusion, and difficulty walking steadily.",
        "My hands have tremors, I suffer from memory loss, slurred speech, and poor coordination.",
        "I have persistent nerve pain, facial muscle weakness, and frequent loss of balance.",
    ],
    "ORTHOPEDICS": [
        "I have severe knee joint pain, stiffness in the morning, and difficulty walking.",
        "My lower back hurts intensely when standing, and I feel shooting pain down my leg.",
        "I sustained a shoulder injury during sports, with limited joint movement and swelling.",
        "I have deep bone pain, joint swelling, heel pain, and muscle stiffness.",
        "I suspect a bone fracture in my foot after a fall, accompanied by severe local pain.",
    ],
    "PEDIATRICS": [
        "My 2-year-old baby has a high fever, persistent coughing, and refuses to eat.",
        "My infant has been vomiting, crying uncontrollably, and shows signs of dehydration.",
        "My child has skin rashes all over, sore throat, baby cold, and mild breathing difficulty.",
        "My baby seems very lethargic, weak, crying excessively, and not feeding properly.",
        "My toddler has growth delay, low weight, and delayed developmental milestones.",
    ],
    "GYNECOLOGY & OBSTETRICS": [
        "I have severe period cramps, irregular menstrual cycles, and lower abdominal pain.",
        "I missed my period this month, experiencing morning nausea and hormonal acne.",
        "I have been diagnosed with PCOS, heavy period bleeding, and vaginal discomfort.",
        "I am experiencing complications during pregnancy with lower stomach pain.",
        "I have light, irregular periods, vaginal itching, and menstrual cycle irregularities.",
    ],
    "DERMATOLOGY": [
        "I have red itchy skin rashes on my arms and torso that feel inflamed.",
        "Persistent acne breakouts on my face, painful pimples, and skin redness.",
        "Dry, scaly patches on my skin that itch constantly and flake off.",
        "I have hives, intense skin itching, eczema flare-ups, and allergic skin spots.",
        "Psoriasis patches on my elbows with redness, scaling, and skin irritation.",
    ],
    "GASTROENTEROLOGY": [
        "I have severe stomach pain, heartburn, acid reflux, and stomach bloating.",
        "Persistent diarrhea, abdominal cramps, nausea, and indigestion after meals.",
        "I have been feeling burning in my upper stomach, nausea, and vomiting after eating.",
        "Chronic constipation, stomach pain, loss of appetite, and intestinal gas.",
        "Sharp abdominal pain, bloating, blood in stool, and frequent acid regurgitation.",
    ],
    "PULMONOLOGY": [
        "I have a chronic dry cough, wheezing when I breathe, and chest tightness.",
        "Severe shortness of breath, asthma attack flare-up, and coughing up mucus.",
        "Persistent coughing for weeks, breathless after minor exertion, and wheezing.",
        "I feel like I cannot catch my breath, chest feels constricted, coughing constantly.",
        "Difficulty breathing at night, heavy chest feeling, and noisy wheezing breath.",
    ],
    "NEPHROLOGY": [
        "I have swollen feet, puffy ankles, reduced urination, and flank pain.",
        "Foamy urine, blood in urine, persistent fatigue, and fluid retention in legs.",
        "Dull aching pain in my lower back near kidneys, low urine output, and puffiness.",
        "Swelling around eyes and feet, high blood pressure, and dark foamy urine.",
        "Kidney area soreness, low urine volume, fatigue, and systemic swelling.",
    ],
    "UROLOGY": [
        "Burning sensation when urinating, frequent urgent need to urinate, and bladder pain.",
        "Severe sharp kidney stone pain in the side and back, with cloudy urine.",
        "Difficulty starting urination, weak urine stream, and nighttime frequent urination.",
        "Incontinence, painful urination, lower abdominal bladder pressure, and blood in urine.",
        "Sharp lower flank pain radiating to groin, pain during urination, and urge incontinence.",
    ],
    "ENT": [
        "Severe earache, ringing sound in my ears, and partial hearing reduction.",
        "Intense sore throat, difficulty swallowing, tonsil pain, and swollen neck glands.",
        "Sinus pain, nasal congestion, runny nose, and severe pressure around eyes and forehead.",
        "Ear pain, ear discharge, blocked ears, and persistent throat irritation.",
        "Loss of smell, stuffed nose, post-nasal drip, sinus headache, and ear fullness.",
    ],
    "ENDOCRINOLOGY": [
        "Unexplained rapid weight loss, extreme thirst, frequent urination, and constant fatigue.",
        "Sudden weight gain, intolerance to cold, dry skin, and thyroid gland swelling.",
        "High blood sugar readings, excessive hunger, numbness in toes, and chronic exhaustion.",
        "Heat intolerance, sweating, rapid pulse, trembling hands, and weight loss.",
        "Hormonal imbalance symptoms, extreme tiredness, mood swings, and abnormal thirst.",
    ],
    "OPHTHALMOLOGY": [
        "Blurry vision in my left eye, severe eye pain, and sensitivity to light.",
        "Redness in both eyes, itching, watery discharge, and gritty feeling.",
        "Double vision, sudden reduction in visual clarity, and seeing halos around lights.",
        "Dry eye irritation, burning feeling in eyes, redness, and eye strain.",
        "Throbbing pain behind the eyes, blurry vision, and floaters in visual field.",
    ],
    "DENTISTRY": [
        "Severe throbbing toothache, sensitivity to hot and cold food, and swollen gums.",
        "Bleeding gums while brushing, jaw pain, and pain in a decayed tooth.",
        "Wisdom tooth pain, swollen cheek, difficulty opening mouth, and gum inflammation.",
        "Painful mouth ulcer, tooth abscess, severe tooth soreness when chewing.",
        "Chipped tooth with exposed nerve, intense sharp dental pain, and gum swelling.",
    ],
    "PSYCHIATRY": [
        "I feel persistent sadness, hopelessness, lack of energy, and loss of interest in everything.",
        "Severe anxiety attacks, sudden panic, racing heart, and uncontrollable worrying.",
        "Extreme mood swings between high energy and deep depression, intrusive thoughts.",
        "Chronic insomnia, inability to sleep, persistent restlessness, and mental exhaustion.",
        "Feeling overwhelmed by stress, panic attacks, social anxiety, and emotional distress.",
    ],
    "ONCOLOGY": [
        "Unexplained dramatic weight loss, persistent night sweats, and a hard painless lump.",
        "Chronic unexplained fatigue, swollen lymph nodes in neck, and persistent fever.",
        "Persistent deep bone pain, unexplained weight loss, and chronic fatigue.",
        "Firm growing mass lump, chronic night sweats, loss of appetite, and weakness.",
        "Unexplained persistent fatigue, swollen glands, night sweats, and weight loss.",
    ],
    "HEMATOLOGY": [
        "Easy unprovoked skin bruising, frequent nosebleeds, and pale skin with exhaustion.",
        "Severe anemia symptoms, dizziness, paleness, extreme fatigue, and shortness of breath.",
        "Frequent infections, easy bleeding from gums, petechiae purple skin spots.",
        "Swollen lymph nodes, pale mucous membranes, chronic fatigue, and abnormal bleeding.",
        "Unexplained low red blood cell count symptoms, weakness, dizziness, and cold limbs.",
    ],
    "RHEUMATOLOGY": [
        "Morning joint stiffness lasting over an hour, swollen painful finger joints.",
        "Symmetrical joint inflammation, wrist and knee pain, joint redness, and stiffness.",
        "Rheumatoid arthritis flare-up, joint deformity pain, chronic fatigue, and swelling.",
        "Widespread joint pain, butterfly facial rash, autoimmune fatigue, and joint warmth.",
        "Stiff painful joints in hands and feet every morning with visible swelling.",
    ],
}


def _reconstruct_text(row: pd.Series, symptom_cols: List[str], department: str) -> str:
    import random
    dept_clean = str(department).strip()
    active_symptoms = [col for col in symptom_cols if row[col] == 1]
    
    # 40% of the time, or if no binary active symptoms, use rich templates
    if (random.random() < 0.45 or not active_symptoms) and dept_clean in DEPARTMENT_SYMPTOM_TEMPLATES:
        return random.choice(DEPARTMENT_SYMPTOM_TEMPLATES[dept_clean])
    
    if active_symptoms:
        # Sample realistic subset of symptoms (1 to 3, matching real patient behavior)
        k = random.choices([1, 2, 3], weights=[0.40, 0.45, 0.15])[0]
        sampled = random.sample(active_symptoms, min(k, len(active_symptoms)))
        
        descriptions = []
        for symptom in sampled:
            sym_key = symptom.upper()
            if sym_key in SYMPTOM_DESCRIPTIONS:
                descriptions.append(random.choice(SYMPTOM_DESCRIPTIONS[sym_key]))
            else:
                clean_sym = symptom.lower().replace("_", " ").replace(".1", "")
                descriptions.append(f"I have {clean_sym}")
        
        if len(descriptions) == 1:
            return descriptions[0]
        
        connectors = [
            ", along with ", " and ", " as well as ",
            ", and I also have ", ", plus "
        ]
        text = descriptions[0]
        for desc in descriptions[1:]:
            text += random.choice(connectors) + desc[0].lower() + desc[1:]
        return text

    if dept_clean in DEPARTMENT_SYMPTOM_TEMPLATES:
        return random.choice(DEPARTMENT_SYMPTOM_TEMPLATES[dept_clean])
    
    return "I am feeling unwell with persistent health symptoms and need medical advice."


def load_raw_data(excel_path: Path = RAW_EXCEL_PATH) -> pd.DataFrame:
    possible_paths = [
        excel_path,
        Path("Data.xlsx"),
        Path("../Data.xlsx"),
        Path(r"C:\Users\Bidyut Paul\Downloads\DocPredic ML\Data.xlsx"),
    ]
    
    target_path = None
    for p in possible_paths:
        if p.exists():
            target_path = p
            break
            
    if target_path is None:
        raise FileNotFoundError(f"Could not find dataset Data.xlsx at specified locations")
        
    df = pd.read_excel(target_path)
    # Clean department trailing whitespace
    dept_col = df.columns[0]
    df[dept_col] = df[dept_col].astype(str).str.strip()
    
    print(f"Raw data loaded from {target_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def generate_synthetic_text_data(df: pd.DataFrame, n_samples_per_class: int = 200, seed: int = SEED) -> pd.DataFrame:
    np.random.seed(seed)
    import random
    random.seed(seed)

    label_col = df.columns[0]
    symptom_cols = df.columns[1:].tolist()

    records = []
    for _, row in df.iterrows():
        department = str(row[label_col]).strip()
        for _ in range(n_samples_per_class):
            text = _reconstruct_text(row, symptom_cols, department)
            records.append({TEXT_COL: text, LABEL_COL: department})

    result = pd.DataFrame(records)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Synthetic text data generated: {result.shape[0]} samples, {result[LABEL_COL].nunique()} classes")
    return result


def load_combined_dataset(
    raw_df: pd.DataFrame,
    n_samples_per_class: int = 200,
    use_mimic: bool = True,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate balanced synthetic text data from knowledge base and combine with
    real clinical Emergency Department data from MIMIC-IV.
    """
    # 1. Generate full-spectrum synthetic data from knowledge matrix
    synthetic_df = generate_synthetic_text_data(raw_df, n_samples_per_class=n_samples_per_class, seed=seed)

    if not use_mimic:
        return synthetic_df

    # 2. Ingest real-world MIMIC-IV ED data
    try:
        from .mimic_loader import load_mimic_dataset
        mimic_df = load_mimic_dataset()
        print(f"Loaded MIMIC-IV clinical dataset: {mimic_df.shape[0]} samples across {mimic_df[LABEL_COL].nunique()} departments")

        # Combine datasets
        combined = pd.concat([synthetic_df, mimic_df], ignore_index=True)
        # Drop duplicates and shuffle
        combined = combined.drop_duplicates(subset=[TEXT_COL]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"Combined Training Dataset: {len(combined)} total samples, {combined[LABEL_COL].nunique()} departments")
        return combined
    except Exception as e:
        print(f"Warning: Failed to load MIMIC-IV dataset ({e}). Using synthetic dataset only.")
        return synthetic_df


def save_processed_data(df: pd.DataFrame, name: str = "processed_data.csv") -> Path:
    path = PROCESSED_DATA_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved processed data to {path}")
    return path
