import pandas as pd
import re
import numpy as np
import random
from typing import Dict, Tuple, Optional

def get_patient_details(data):
    gender = "female" if data['female'] == 1 else "male"
    cvd_history = "a history of cardiovascular disease" if data['cvd_hx_baseline'] == 1 else "no history of cardiovascular disease"
    race = "Black" if data['black'] == 1 else "White"
    smoking_status = "smoker" if data['smoke'] == 1 else "non-smoker"
    bprx_status = "is on blood pressure medication" if data['bprx'] == 1 else "is not on blood pressure medication"
    statin_status = "is taking a statin" if data['statin'] == 1 else "is not taking a statin"
    return gender, cvd_history, race, smoking_status, bprx_status, statin_status

def create_patient_narrative(patient_data: pd.Series) -> str:
    """
    Creates a formatted text narrative from a single patient's data,
    including an instruction for a language model.

    Args:
        patient_data (pd.Series): A single row of patient data with a
                                  pre-defined set of columns.

    Returns:
        str: A formatted instruction-based narrative.
    """
    # Helper functions to convert boolean/categorical data to strings
    gender, cvd_history, race, smoking_status, bprx_status, statin_status = get_patient_details(patient_data)

    # Construct a more narrative-style prompt
    text_prompt = f"""
The patient is a {gender}, {patient_data['baseline_age']}-year-old {race} {smoking_status}, having {cvd_history}. This patient's vital signs show a blood pressure of {patient_data['sbp']}/{patient_data['dbp']} mmHg and a heart rate of {patient_data['hr']} bpm, with BMI being {patient_data['bmi']:.2f} kg/m².
Laboratory results indicate an HbA1c of {patient_data['hba1c']:.2f}%, total cholesterol of {patient_data['chol']:.1f} mg/dL, LDL of {patient_data['ldl']:.1f} mg/dL, HDL of {patient_data['hdl']:.1f} mg/dL, and triglycerides of {patient_data['trig']:.1f} mg/dL. The fasting plasma glucose is {patient_data['fpg']:.1f} mg/dL. Other lab values include a serum creatinine of {patient_data['screat']:.2f} mg/dL, a UACR of {patient_data['uacr']:.2f} mg/g, and a potassium level of {patient_data['potassium']:.1f} mmol/L.

As for intervention, the patient {bprx_status} and {statin_status}.
"""
    return text_prompt.strip()


# crms
def create_clinical_role_model_profile(patient_data: pd.Series) -> str:
    """
    Creates a detailed narrative for a "clinical role model" patient,
    highlighting the changes from pre- to post-intervention.

    Args:
        patient_data (pd.Series): A single row of patient data at baseline
                                  and 24-month follow-up. Columns should
                                  be distinguished by '_pre' and '_post' suffixes.

    Returns:
        str: A formatted instruction-based narrative for fine-tuning.
    """
    # Helper functions for pre- and post- data


    gender_pre, cvd_history_pre, race_pre, smoking_status_pre, bprx_status_pre, statin_status_pre = get_patient_details({k.replace('_pre', ''): v for k, v in patient_data.items() if '_pre' in k})
    gender_post, cvd_history_post, race_post, smoking_status_post, bprx_status_post, statin_status_post = get_patient_details({k.replace('_post', ''): v for k, v in patient_data.items() if '_post' in k})

    # Determine interventions
    bprx_change = "started" if bprx_status_pre == 0 and bprx_status_post == 1 else "continued" if bprx_status_pre == 1 else "did not start"
    statin_change = "started" if statin_status_pre == 0 and statin_status_post == 1 else "continued" if statin_status_post == 1 else "did not start"

    # Build the narrative
    text_prompt = f"""
The crm has been a previous patient. At the baseline visit, the crm was a {gender_pre}, {patient_data['baseline_age_pre']}-year-old {race_pre} {smoking_status_pre}, had {cvd_history_pre}. This crm's vital signs showed a blood pressure of {patient_data['sbp_pre']}/{patient_data['dbp_pre']} mmHg and a heart rate of {patient_data['hr_pre']} bpm, with BMI being {patient_data['bmi_pre']:.2f} kg/m² at the baseline visit.
Also, at the baseline visit, laboratory results indicated an HbA1c of {patient_data['hba1c_pre']:.2f}%, total cholesterol of {patient_data['chol_pre']:.1f} mg/dL, LDL of {patient_data['ldl_pre']:.1f} mg/dL, HDL of {patient_data['hdl_pre']:.1f} mg/dL, and triglycerides of {patient_data['trig_pre']:.1f} mg/dL. The fasting plasma glucose was {patient_data['fpg_pre']:.1f} mg/dL. Other lab values included a serum creatinine of {patient_data['screat_pre']:.2f} mg/dL, a UACR of {patient_data['uacr_pre']:.2f} mg/g, and a potassium level of {patient_data['potassium_pre']:.1f} mmol/L.
At the 24-month follow-up, the crm was a {gender_post}, {patient_data['baseline_age_post']}-year-old {race_post} {smoking_status_post}, had {cvd_history_post}. This crm's vital signs showed a blood pressure of {patient_data['sbp_post']}/{patient_data['dbp_post']} mmHg and a heart rate of {patient_data['hr_post']} bpm, with BMI being {patient_data['bmi_post']:.2f} kg/m² at the 24-month follow-up.
Also, at the 24-month follow-up, laboratory results indicated an HbA1c of {patient_data['hba1c_post']:.2f}%, total cholesterol of {patient_data['chol_post']:.1f} mg/dL, LDL of {patient_data['ldl_post']:.1f} mg/dL, HDL of {patient_data['hdl_post']:.1f} mg/dL, and triglycerides of {patient_data['trig_post']:.1f} mg/dL. The fasting plasma glucose was {patient_data['fpg_post']:.1f} mg/dL. Other lab values included a serum creatinine of {patient_data['screat_post']:.2f} mg/dL, a UACR of {patient_data['uacr_post']:.2f} mg/g, and a potassium level of {patient_data['potassium_post']:.1f} mmol/L.

As for intervention, the crm {bprx_status_pre} and {statin_status_pre} at the baseline visit. Then, the crm {bprx_status_post} and {statin_status_post} at the 24-month follow-up.
"""
    return text_prompt.strip()



def generate_prompts_from_crmdb(patients_record, CRMdb, tokenizer, create_both=True, lambda_lm=0.5, seed=42):
    """
    Generate prompts from pre-computed CRMdb.
    
    Args:
        patients_record: DataFrame with patient data including 'patient_text'
        CRMdb: Pre-computed DataFrame with K CRMs per patient (already sorted by feasibility then cost)
        tokenizer: Tokenizer for EOS token
        create_both: If True, create both LM and FSCE prompts for each patient (doubles dataset size)
        lambda_lm: Probability of simple LM prompt vs FSCE prompt (only used if create_both=False)
        seed: Random seed (only used if create_both=False)
    """
    if not create_both and seed:
        random.seed(seed)
    
    EOS_TOKEN = tokenizer.eos_token
    all_prompts = []
    
    # Only process patients that exist in BOTH dataframes
    valid_patient_ids = set(range(len(patients_record))) & set(CRMdb['PatientID'].unique())
    for patient_id in sorted(valid_patient_ids):
        patient_text = patients_record.iloc[patient_id]['patient_text']
        patient_crms = CRMdb[CRMdb['PatientID'] == patient_id].reset_index(drop=True)
        
        # Get data for this patient's CRMs
        crm_texts = patient_crms['crm_text'].tolist()
        feasibilities = patient_crms['crm_feasibility'].tolist()
        costs = patient_crms['trans_cost'].tolist()
        k = len(patient_crms)
        
        # Helper function to create FSCE prompt
        def create_fsce_prompt():
            prompt = f"""Patient: {patient_text}

All {k} candidate crms with feasibility assessment:
"""
            # Separate feasible and infeasible
            feas_crms = [(c, f, tc) for c, f, tc in zip(crm_texts, feasibilities, costs) if f]
            infeas_crms = [(c, f, tc) for c, f, tc in zip(crm_texts, feasibilities, costs) if not f]
            
            i = 0
            if feas_crms:
                prompt += "Note: PROMOTE generation of these feasible crms.\n"
                for (crm, _, tc) in feas_crms:
                    if i == 0:
                        prompt += f"{i+1}. [FEASIBLE, transformation costs {tc}, Optimal crm] {crm}\n"
                    else:
                        prompt += f"{i+1}. [FEASIBLE, transformation costs {tc}] {crm}\n"
                    i += 1
            
            if infeas_crms:
                prompt += "Note: AVOID generation of these infeasible crms.\n"
                for (crm, _, tc) in infeas_crms:
                    if i == 0:
                        prompt += f"{i+1}. [INFEASIBLE, transformation costs {tc}, current best but infeasible crm] {crm}\n"
                    else:
                        prompt += f"{i+1}. [INFEASIBLE, transformation costs {tc}] {crm}\n"
                    i += 1
            
            prompt += f"\nGenerate current best crm: {crm_texts[0]}" + EOS_TOKEN
            return prompt
        
        if create_both:
            # Create BOTH prompt types for each patient
            # 1. Simple LM prompt
            lm_prompt = f"""Patient: {patient_text}

Generate current best crm: {crm_texts[0]}""" + EOS_TOKEN
            
            all_prompts.append({
                'text': lm_prompt,
                'prompt_type': 'lm',
                'patient_idx': patient_id
            })
            
            # 2. FSCE prompt
            fsce_prompt = create_fsce_prompt()
            all_prompts.append({
                'text': fsce_prompt,
                'prompt_type': 'fsce',
                'patient_idx': patient_id
            })
            
        else:
            # Original behavior: randomly choose one prompt type
            if random.random() < lambda_lm:
                # Simple LM prompt
                prompt = f"""Patient: {patient_text}

Generate current best crm: {crm_texts[0]}""" + EOS_TOKEN
                prompt_type = 'lm'
            else:
                # FSCE prompt
                prompt = create_fsce_prompt()
                prompt_type = 'fsce'
            
            all_prompts.append({
                'text': prompt,
                'prompt_type': prompt_type,
                'patient_idx': patient_id
            })
    
    return all_prompts


def generate_inference_prompt(patient: str) -> str:
    """
    Create L_LM prompt: Simple generation matching inference format.
    This teaches standard language modeling.
    """
    return f"""Patient: {patient}

Generate current best crm: """

def extract_clinical_features(text: str, MaskID, PatientID) -> Dict:
    """
    Extracts clinical features from LLM output text containing patient and CRM data.

    Args:
        text (str): The LLM output text containing patient and CRM information

    Returns:
        Dict: Dictionary containing extracted features for patient and CRM
    """

    def extract_patient_features(patient_text: str) -> Dict:
        """Extract features from patient section"""
        features = {}

        # Demographics
        age_match = re.search(r'(\d+\.?\d*)-year-old', patient_text)
        features['baseline_age'] = float(age_match.group(1)) if age_match else None

        gender_match = re.search(r'(male|female)', patient_text, re.IGNORECASE)
        features['female'] = float(gender_match.group(1).lower() == 'female') if gender_match else None

        race_match = re.search(r'(\d+\.?\d*)-year-old\s+(\w+)\s+(non-)?smoker', patient_text)
        features['black'] = float(race_match.group(2).lower() == 'black' )if race_match else None

        smoking_match = re.search(r'(non-)?smoker', patient_text)
        features['smoke'] = 0. if smoking_match and smoking_match.group(1) else 1.

        cvd_match = re.search(r'(having no|had no|having|had)\s+history of cardiovascular disease', patient_text)
        features['cvd_hx_baseline'] = 0. if cvd_match and ('no' in cvd_match.group(1)) else 1.
        # Vital signs
        bp_match = re.search(r'blood pressure of\s+(\d+\.?\d*)/(\d+\.?\d*)\s+mmHg', patient_text)
        if bp_match:
            features['sbp'] = float(bp_match.group(1))
            features['dbp'] = float(bp_match.group(2))

        hr_match = re.search(r'heart rate of\s+(\d+\.?\d*)\s+bpm', patient_text)
        features['hr'] = float(hr_match.group(1)) if hr_match else None

        bmi_match = re.search(r'BMI being\s+(\d+\.?\d*)\s+kg/m²', patient_text)
        features['bmi'] = float(bmi_match.group(1)) if bmi_match else None

        # Laboratory results
        hba1c_match = re.search(r'HbA1c of\s+(\d+\.?\d*)%', patient_text)
        features['hba1c'] = float(hba1c_match.group(1)) if hba1c_match else None

        chol_match = re.search(r'total cholesterol of\s+(\d+\.?\d*)\s+mg/dL', patient_text)
        features['chol'] = float(chol_match.group(1)) if chol_match else None

        ldl_match = re.search(r'LDL of\s+(\d+\.?\d*)\s+mg/dL', patient_text)
        features['ldl'] = float(ldl_match.group(1)) if ldl_match else None

        hdl_match = re.search(r'HDL of\s+(\d+\.?\d*)\s+mg/dL', patient_text)
        features['hdl'] = float(hdl_match.group(1)) if hdl_match else None

        trig_match = re.search(r'triglycerides of\s+(\d+\.?\d*)\s+mg/dL', patient_text)
        features['trig'] = float(trig_match.group(1)) if trig_match else None

        fpg_match = re.search(r'fasting plasma glucose\s+(?:is|was)\s+(\d+\.?\d*)\s+mg/dL', patient_text)
        features['fpg'] = float(fpg_match.group(1)) if fpg_match else None

        screat_match = re.search(r'serum creatinine of\s+(\d+\.?\d*)\s+mg/dL', patient_text)
        features['screat'] = float(screat_match.group(1)) if screat_match else None

        uacr_match = re.search(r'UACR of\s+(\d+\.?\d*)\s+mg/g', patient_text)
        features['uacr'] = float(uacr_match.group(1)) if uacr_match else None

        potassium_match = re.search(r'potassium level of\s+(\d+\.?\d*)\s+mmol/L', patient_text)
        features['potassium'] = float(potassium_match.group(1)) if potassium_match else None

        # Interventions
        bprx_match = re.search(r'(is|is not)\s+on blood pressure medication', patient_text)
        features['bprx'] = 1 if bprx_match and bprx_match.group(1) == 'is' else 0

        statin_match = re.search(r'(is|is not)\s+taking a statin', patient_text)
        features['statin'] = 1 if statin_match and statin_match.group(1) == 'is' else 0

        return features

    def extract_crm_features(crm_text: str) -> Tuple[Dict, Dict]:
        """Extract features from CRM section for both baseline and follow-up"""
        baseline_features = {}
        followup_features = {}

        # Split by baseline and follow-up sections
        baseline_section = re.search(r'At the baseline visit,.*?(?=At the 24-month follow-up)', crm_text, re.DOTALL)
        followup_section = re.search(r'At the 24-month follow-up,.*?(?=As for intervention|$)', crm_text, re.DOTALL)
        intervention_section = re.search(r'As for intervention,.*', crm_text, re.DOTALL)

        if baseline_section:
            baseline_text = baseline_section.group(0)
            baseline_features = extract_patient_features(baseline_text)

        if followup_section:
            followup_text = followup_section.group(0)
            followup_features = extract_patient_features(followup_text)

        # Extract intervention status separately
        if intervention_section:
            intervention_text = intervention_section.group(0)

            # Baseline intervention
            baseline_int_match = re.search(r'the crm\s+(is|is not)\s+on blood pressure medication\s+and\s+(is|is not)\s+taking a statin\s+at the baseline visit', intervention_text)
            if baseline_int_match:
                baseline_features['bprx'] = 1 if baseline_int_match.group(1) == 'is' else 0
                baseline_features['statin'] = 1 if baseline_int_match.group(2) == 'is' else 0

            # Follow-up intervention
            followup_int_match = re.search(r'the crm\s+(is|is not)\s+on blood pressure medication\s+and\s+(is|is not)\s+taking a statin\s+at the 24-month follow-up', intervention_text)
            if followup_int_match:
                followup_features['bprx'] = 1 if followup_int_match.group(1) == 'is' else 0
                followup_features['statin'] = 1 if followup_int_match.group(2) == 'is' else 0

        return baseline_features, followup_features

    # Main extraction
    results = {
        'patient': {},
        'crm_baseline': {},
        'crm_followup': {}
    }

    # Split text into patient and CRM sections
    patient_match = re.search(r'Patient:.*?(?=Generate current best crm:|$)', text, re.DOTALL)
    crm_match = re.search(r'Generate current best crm:.*', text, re.DOTALL)

    if patient_match:
        results['patient'] = extract_patient_features(patient_match.group(0))

    if crm_match:
        results['crm_baseline'], results['crm_followup'] = extract_crm_features(crm_match.group(0))
        results['crm_baseline']['PatientID'] = PatientID
        results['crm_baseline']['MaskID'] = MaskID
        results['crm_baseline']['Visit'] = 'BLR'

        results['crm_followup']['PatientID'] = PatientID
        results['crm_followup']['MaskID'] = MaskID
        results['crm_followup']['Visit'] = 'F24'
    return results

def features_to_dataframe(extracted_features: Dict) -> pd.DataFrame:
    """
    Convert extracted features to a pandas DataFrame for easier viewing

    Args:
        extracted_features (Dict): Dictionary from extract_clinical_features

    Returns:
        pd.DataFrame: DataFrame with features as rows and categories as columns
    """
    df_data = []

    # Add patient features
    for key, value in extracted_features['patient'].items():
        df_data.append({
            'Category': 'Patient',
            'Feature': key,
            'Value': value
        })

    # Add CRM baseline features
    for key, value in extracted_features['crm_baseline'].items():
        df_data.append({
            'Category': 'CRM Baseline',
            'Feature': key,
            'Value': value
        })

    # Add CRM follow-up features
    for key, value in extracted_features['crm_followup'].items():
        df_data.append({
            'Category': 'CRM Follow-up',
            'Feature': key,
            'Value': value
        })

    return pd.DataFrame(df_data)