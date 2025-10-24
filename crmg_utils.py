# system packages
import os
import sys
import re
import warnings
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
import torch
# from torch.utils.data import DataLoader
from datasets import Dataset
# for savings
from pathlib import Path
import pickle

import random
### self-defined code modules
from ai_ml_feasibility_check import * # uncertainty feasibility checking

from utils import *
from config import ExperimentConfig
from crms_utils import *
from trl import SFTConfig, SFTTrainer

def generate_crms_for_patients(patients_record, model, tokenizer, config, 
                               batch_size=100, prompts_per_patient=6, verbose=True):
    """
    Generate CRMs for all patients using LLM inference with batch processing.
    
    Args:
        patients_record: DataFrame with patient data including 'patient_text'
        model: Fine-tuned language model for generation
        tokenizer: Tokenizer for the model
        config: ExperimentConfig object with K and other parameters
        batch_size: Number of prompts to process in one batch
        prompts_per_patient: Minimum prompts to generate per patient per iteration
        verbose: Whether to show progress bars and print statements
        
    Returns:
        tuple: (crm_dec_pre, crm_dec_post) - DataFrames with generated CRMs
    """
    from tqdm import tqdm
    import torch
    
    # Initialize output dataframes
    CRM_FEATURES = ['PatientID', 'MaskID', 'Visit'] + config.PT_FEATURES
    crm_dec_pre = pd.DataFrame(columns=CRM_FEATURES)
    crm_dec_post = pd.DataFrame(columns=CRM_FEATURES)
    
    # Initialize tracking
    total_patients = len(patients_record)
    patient_crm_counts = {pid: 0 for pid in range(total_patients)}
    patient_mask_ids = {pid: pid * config.K for pid in range(total_patients)}
    
    # Queue of patients that still need CRMs
    pending_patients = set(range(total_patients))
    
    iteration = 0
    pbar = tqdm(total=total_patients, desc="Generating CRMs") if verbose else None
    
    while pending_patients:
        iteration += 1
        if verbose:
            print(f"\nIteration {iteration}: {len(pending_patients)} patients still need CRMs")
        
        # Prepare prompts from pending patients
        all_prompts = []
        prompt_mapping = []  # Maps prompt index to patient_id
        
        for patient_id in list(pending_patients):
            needed = config.K - patient_crm_counts[patient_id]
            if needed <= 0:
                pending_patients.remove(patient_id)
                continue
            
            patient_text = patients_record.iloc[patient_id]['patient_text']
            
            # Generate proportional to how many CRMs they need
            num_prompts = min(prompts_per_patient, needed * 2)
            
            for _ in range(num_prompts):
                # Generate inference prompt (assuming this function exists)
                prompt = generate_inference_prompt(patient_text)
                all_prompts.append(prompt)
                prompt_mapping.append(patient_id)
            
            # Stop if we have enough prompts for a batch
            if len(all_prompts) >= batch_size:
                break
        
        if not all_prompts:
            break
            
        if pbar:
            pbar.set_description(f"Iter {iteration}: Gen {len(all_prompts)} CRMs")
        
        if verbose:
            print(f"  Generating {len(all_prompts)} CRMs for {len(set(prompt_mapping))} patients")
        
        # Batch generation
        inputs = tokenizer(
            all_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded_crms = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Process results
        completed_this_batch = 0
        for crm_text, patient_id in zip(decoded_crms, prompt_mapping):
            # Skip if patient already has enough CRMs
            if patient_crm_counts[patient_id] >= config.K:
                continue
            
            # Extract features from generated CRM text
            mask_id = patient_mask_ids[patient_id] + patient_crm_counts[patient_id]
            features = extract_clinical_features(crm_text, MaskID=mask_id, PatientID=patient_id)
            
            # Validate extracted features
            temp_pre = features['crm_baseline']
            temp_post = features['crm_followup']
            
            is_valid_pre = not any(pd.isna(val) or val == '' for val in temp_pre.values())
            is_valid_post = not any(pd.isna(val) or val == '' for val in temp_post.values())
            
            if is_valid_pre and is_valid_post:
                # Add valid CRM to dataframes
                crm_dec_pre.loc[len(crm_dec_pre)] = temp_pre
                crm_dec_post.loc[len(crm_dec_post)] = temp_post
                
                # Update count
                patient_crm_counts[patient_id] += 1
                
                # Remove from pending if complete
                if patient_crm_counts[patient_id] >= config.K:
                    pending_patients.discard(patient_id)
                    completed_this_batch += 1
        
        if pbar:
            pbar.update(completed_this_batch)
        
        # Progress update
        if verbose:
            completed = total_patients - len(pending_patients)
            print(f"  Completed: {completed}/{total_patients} patients")
    
    # Final stats
    if verbose:
        print(f"\nCompleted all patients in {iteration} iterations")
    if pbar:
        pbar.close()
    
    return crm_dec_pre, crm_dec_post

def predict_risks_for_crms(crm_dec_pre, crm_dec_post, config, model_dir='risk_models', verbose=True):
    """
    Load pre-fitted ML risk models and predict risks for generated CRMs.
    
    Args:
        crm_dec_pre: DataFrame with baseline CRM data
        crm_dec_post: DataFrame with follow-up CRM data
        config: ExperimentConfig object with RISK_MODEL and TOTAL_RISK_MODELS
        model_dir: Directory containing risk models
        verbose: Whether to print progress
        
    Returns:
        dict: {'crm_blr_risks': df_with_baseline_risks, 'crm_f24_risks': df_with_followup_risks}
    """
    import os
    import joblib
    from tqdm import tqdm
    
    # Setup data for inference
    df_to_inference = {
        'crm_blr_risks': crm_dec_pre,
        'crm_f24_risks': crm_dec_post
    }
    
    result_df = {}
    
    for name, feat_data in df_to_inference.items():
        if verbose:
            print(f"\nProcessing {name}...")
        
        # Prepare features
        df_full = feat_data.copy()
        df = df_full.drop(columns=["PatientID", "MaskID", "Visit"])
        
        # Dictionary to store predictions
        predictions_dict = {}
        
        # Progress tracking
        successful_runs = 0
        failed_runs = []
        
        # Load models for each run
        for run in range(1, config.TOTAL_RISK_MODELS + 1):
            try:
                # Construct model path using config.RISK_MODEL
                run_dir = f'run_{run:03d}'
                model_path = os.path.join(model_dir, run_dir, config.RISK_MODEL, f'{config.RISK_MODEL}.joblib')
                
                # Load model
                risk_prediction_model = joblib.load(model_path)
                
                # Get predictions
                y_pred_proba = risk_prediction_model.predict_proba(df)[:, 1]
                
                # Store with standardized column name
                col_name = f"risk_prediction_{config.RISK_MODEL}_run{run}"
                predictions_dict[col_name] = y_pred_proba
                successful_runs += 1
                
            except Exception as e:
                failed_runs.append(run)
                if verbose and len(failed_runs) <= 3:
                    print(f"  Error loading {config.RISK_MODEL} run {run}: {str(e)}")
        
        if verbose:
            print(f"  Successfully loaded {successful_runs}/{config.TOTAL_RISK_MODELS} models")
            if failed_runs:
                print(f"  Failed runs: {failed_runs}")
        
        # Create DataFrame with predictions
        risk_predictions = pd.DataFrame(predictions_dict)
        
        # Concatenate with original data
        result_df[name] = pd.concat([df_full, risk_predictions], axis=1)
    
    return result_df

def update_crmdb(CRMdb, patients_record, config):
    # compute 
    CRMdb['personal_feasibility'] = False
    CRMdb['trans_cost']  = 0.
    CRMdb['crm_feasibility'] = False

    for i in CRMdb['PatientID'].unique():
        # personal feasibility
        CRMdb.loc[CRMdb['PatientID'] == i,'personal_feasibility'] = (
            CRMdb.loc[CRMdb['PatientID'] == i,config.matching_columns_pre_db] == patients_record.loc[[i]][config.matching_columns_pre].values[0] 
        ).all(axis=1)
        
        # trans cost
        differences = (
            CRMdb.loc[CRMdb['PatientID'] == i,config.loss_condition_columns_db] -  patients_record.loc[[i]][config.loss_condition_columns].values
        )
        if config.COST_PARAMS['type'] == 'p-norm':
            CRMdb.loc[CRMdb['PatientID'] == i,'trans_cost'] = np.linalg.norm(differences, ord=config.COST_PARAMS['p'], axis=1)

        
    CRMdb['crm_feasibility'] = CRMdb['crm_ai_ml_feasibility'] & CRMdb['personal_feasibility']
    
    CRMdb = CRMdb.sort_values(
        by=['PatientID', 'crm_feasibility', 'trans_cost'], 
        ascending=[True, False, True]  # Keep patients together, feasible=True first, then low cost first
    )
    CRMdb['crm_text'] = CRMdb.apply(create_clinical_role_model_profile, axis=1)
    return CRMdb

def active_learning_select(metric_train, cur_iter, config):
    # different ways to build the weights

    # read previous sampling counts
    num_selections = pd.read_csv(
        f'{config.output_dir_result}/num_selected_for_training_iter_{cur_iter}.csv'
    )[['PatientID','num_selections']]

    # read previous metric first
    metric_prev = pd.read_csv(
        f'{config.output_dir_result}/metrics_train_iter_{cur_iter}.csv'
    )[['feasibility_rate', 'cur_best_trans_cost']]
    
    epsilon = 1e-10  # small value to avoid division by zero
    
    if config.active_learning == 'proposed':
        infeasibility_rate = (1 - metric_train['feasibility_rate']).to_numpy()
        # Convert to numpy arrays for consistent operations
        cur_best_cost = metric_train['cur_best_trans_cost'].to_numpy()
        prev_best_cost = metric_prev['cur_best_trans_cost'].to_numpy()


        # Calculate the base improvement
        cost_improv = 1 - (cur_best_cost / (prev_best_cost + epsilon))

        # Apply the max(0, ·) operation to match the math
        cost_improv = np.maximum(0, cost_improv)

        # Set to 0 when conditions aren't met (infeasible or invalid previous cost)
        cost_improv[prev_best_cost <= 0] = 0.
        cost_improv[metric_prev['feasibility_rate'].to_numpy() <= 0] = 0.
        cost_improv[metric_train['feasibility_rate'].to_numpy() <= 0] = 0.

        sampling_weights = config.lambda_ifr * infeasibility_rate + config.lambda_cir * cost_improv

    elif config.active_learning == 'feasibility_rate':
        sampling_weights = (1 - metric_train['feasibility_rate']).to_numpy()
        
    elif config.active_learning == 'cost_improvement':
        cur_best_cost = metric_train['cur_best_trans_cost'].to_numpy()
        prev_best_cost = metric_prev['cur_best_trans_cost'].to_numpy()

        # Calculate the base improvement
        cost_improv = 1 - (cur_best_cost / (prev_best_cost + epsilon))

        # Apply the max(0, ·) operation to match the math
        cost_improv = np.maximum(0, cost_improv)

        # Set to 0 when conditions aren't met (infeasible or invalid previous cost)
        cost_improv[prev_best_cost <= 0] = 0.
        cost_improv[metric_prev['feasibility_rate'].to_numpy() <= 0] = 0.
        cost_improv[metric_train['feasibility_rate'].to_numpy() <= 0] = 0.
        sampling_weights = cost_improv

    elif config.active_learning == 'uniform':
        sampling_weights = np.ones(len(metric_train))
    
    elif config.active_learning == 'count_inverse':
        sampling_weights = 1 / (num_selections['num_selections'].to_numpy() + 1)

    elif config.active_learning == 'proposed_new':
        feasibility_rate = (metric_train['feasibility_rate']).to_numpy()
        # Convert to numpy arrays for consistent operations
        cur_best_cost = metric_train['cur_best_trans_cost'].to_numpy()
        prev_best_cost = metric_prev['cur_best_trans_cost'].to_numpy()


        # Calculate the base improvement
        cost_improv = 1 - (cur_best_cost / (prev_best_cost + epsilon))

        # Apply the max(0, ·) operation to match the math
        cost_improv = np.maximum(0, cost_improv)

        # Set to 0 when conditions aren't met (infeasible or invalid previous cost)
        cost_improv[prev_best_cost <= 0] = 0.
        cost_improv[metric_prev['feasibility_rate'].to_numpy() <= 0] = 0.
        cost_improv[metric_train['feasibility_rate'].to_numpy() <= 0] = 0.

        sampling_weights = config.lambda_ifr * feasibility_rate + config.lambda_cir * cost_improv
    elif config.active_learning == 'proposed_new_new':
        feasibility_rate = (metric_train['feasibility_rate']).to_numpy()
        # Convert to numpy arrays for consistent operations
        cur_best_cost = metric_train['cur_best_trans_cost'].to_numpy()
        prev_best_cost = metric_prev['cur_best_trans_cost'].to_numpy()


        # Calculate the base improvement
        cost_improv = 1 - (cur_best_cost / (prev_best_cost + epsilon))

        # Apply the max(0, ·) operation to match the math
        cost_improv = np.maximum(0, cost_improv)

        # Set to 0 when conditions aren't met (infeasible or invalid previous cost)
        cost_improv[prev_best_cost <= 0] = 0.
        cost_improv[metric_prev['feasibility_rate'].to_numpy() <= 0] = 0.
        cost_improv[metric_train['feasibility_rate'].to_numpy() <= 0] = 0.

        sampling_weights = config.lambda0 + config.lambda_ifr * feasibility_rate + config.lambda_cir * cost_improv


    elif config.active_learning == 'feasibility_rate_new':
        sampling_weights = ( metric_train['feasibility_rate']).to_numpy()

    
    # done with creating weights, now do probabilities
    if sampling_weights.sum() <= 0:
        sampling_weights = np.ones_like(sampling_weights)
    
    # independent of weight methods
    probabilities = sampling_weights / sampling_weights.sum()
    
    # Get the indices (assuming these correspond to patient IDs or row indices)
    indices = np.arange(len(sampling_weights))
    n_samples = min(config.active_learning_budget, len(indices))
    
    # Check if we have enough non-zero weights
    num_nonzero = np.sum(sampling_weights > 0)
    
    if num_nonzero < n_samples:
        # Add uniform baseline to ensure we can sample
        print(f"Warning: Only {num_nonzero} non-zero weights but need {n_samples}. Adding uniform baseline.")
        probabilities += 1.0 / len(sampling_weights)
        probabilities = probabilities / probabilities.sum()

    sampled_indices = np.random.choice(
        indices,
        size=n_samples,
        replace=False,
        p=probabilities
    )

    # Since rows are aligned, we can directly use iloc with sampled_indices
    num_selections.iloc[sampled_indices, num_selections.columns.get_loc('num_selections')] += 1

    return sampled_indices, num_selections['num_selections'].to_numpy()



    



