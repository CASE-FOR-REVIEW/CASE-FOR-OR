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
from trl import SFTConfig, SFTTrainer

import random
### self-defined code modules
from ai_ml_feasibility_check import * # uncertainty feasibility checking

from utils import *
from config import ExperimentConfig
from crms_utils import *
from crmg_utils import generate_crms_for_patients, predict_risks_for_crms, update_crmdb, active_learning_select

import argparse

def prepare_crmdb_general_only_perc_low_risk(crm_pre, crm_post, mergeon, config):
    """
    Prepare general CRMdb with risk statistics and AI/ML feasibility
    
    Args:
        config: Config object with parameters (N_OBS, TOTAL_RISK_MODELS, RISK_MODEL, etc.)

        
    Returns:
        pd.DataFrame: Prepared CRMdb_general with added risk statistics and feasibility
    """
    # Load and merge data
    CRMdb_general = pd.merge(crm_pre, crm_post, on=mergeon, suffixes=('_pre', '_post'))
    
    # Define risk model columns
    observed_risk_models = [f'{config.RISK_MODEL}_run{i+1}' for i in range(config.N_OBS)]
    observed_risk_models_pre = [f'risk_prediction_{model_run}_pre' for model_run in observed_risk_models]
    observed_risk_models_post = [f'risk_prediction_{model_run}_post' for model_run in observed_risk_models]
    
    unobserved_risk_models = [f'{config.RISK_MODEL}_run{i+1}' for i in range(config.N_OBS, config.TOTAL_RISK_MODELS)]
    unobserved_risk_models_pre = [f'risk_prediction_{model_run}_pre' for model_run in unobserved_risk_models]
    unobserved_risk_models_post = [f'risk_prediction_{model_run}_post' for model_run in unobserved_risk_models]
    
    # Compute risk statistics
    CRMdb_general['in_sample_perc_low_risk_pred_pre'] = (CRMdb_general[observed_risk_models_pre]  < config.RISK_THRESHOLD).mean(axis=1)
    CRMdb_general['in_sample_perc_low_risk_pred_post'] = (CRMdb_general[observed_risk_models_post] < config.RISK_THRESHOLD).mean(axis=1)
    CRMdb_general['out_sample_perc_low_risk_pred_pre'] = (CRMdb_general[unobserved_risk_models_pre]< config.RISK_THRESHOLD).mean(axis=1)
    CRMdb_general['out_sample_perc_low_risk_pred_post'] = (CRMdb_general[unobserved_risk_models_post] < config.RISK_THRESHOLD).mean(axis=1)
    



    # Compute AI/ML feasibility
    crm_ai_ml_feasibility = config.solver.solve_batch(
        risks_pre_df=CRMdb_general[observed_risk_models_pre],
        risks_post_df=CRMdb_general[observed_risk_models_post],
        r=config.RISK_THRESHOLD,
        alpha=config.ALPHA
    )
    CRMdb_general['crm_ai_ml_feasibility'] = crm_ai_ml_feasibility
    
    return CRMdb_general
def collect_metrics_only_perc_low_risk(CRMdb, patients_record, iteration, use_type, exp_name):
    """
    Collect metrics from CRMdb for output
    
    Args:
        CRMdb: DataFrame with CRM data
        patients_record: DataFrame with patient records (for index)
        iteration: Current iteration number (0 for CRMs)
        use_type: 'train' or 'test'
        config: Config object with output_columns definition
        
    Returns:
        pd.DataFrame: Collected metrics
    """
    # Initialize output dataframe


    num_patients = len(patients_record)
    output_this_iter = pd.DataFrame({
        'exp_name': [exp_name] * num_patients,  # Repeat for each patient
        'PatientID': patients_record.index,
        'iteration': iteration,
        'use_type': use_type
    })


    # Aggregate metrics by PatientID
    patient_metrics = CRMdb.groupby('PatientID').agg({
        'crm_feasibility': 'mean',
        'trans_cost': 'first',
        'in_sample_perc_low_risk_pred_pre': 'mean',
        'in_sample_perc_low_risk_pred_post': 'mean',
        'out_sample_perc_low_risk_pred_pre': 'mean',
        'out_sample_perc_low_risk_pred_post': 'mean'        
    })
    

    # Map aggregated values to output columns
    output_this_iter['feasibility_rate'] = patient_metrics['crm_feasibility'].values
    output_this_iter['cur_best_trans_cost'] = patient_metrics['trans_cost'].values
    # pre and post risks
    output_this_iter['avg_in_sample_perc_low_risk_pred_pre'] = patient_metrics['in_sample_perc_low_risk_pred_pre'].values
    output_this_iter['avg_in_sample_perc_low_risk_pred_post'] = patient_metrics['in_sample_perc_low_risk_pred_post'].values
    output_this_iter['avg_out_sample_perc_low_risk_pred_pre'] = patient_metrics['out_sample_perc_low_risk_pred_pre'].values
    output_this_iter['avg_out_sample_perc_low_risk_pred_post'] = patient_metrics['out_sample_perc_low_risk_pred_post'].values
    
    patient_metrics = CRMdb.groupby('PatientID').agg({
        'crm_feasibility': 'mean',
        'trans_cost': 'first',
        'in_sample_perc_low_risk_pred_pre': 'std',
        'in_sample_perc_low_risk_pred_post': 'std',
        'out_sample_perc_low_risk_pred_pre': 'std',
        'out_sample_perc_low_risk_pred_post': 'std'
    })
    output_this_iter['std_in_sample_perc_low_risk_pred_pre'] = patient_metrics['in_sample_perc_low_risk_pred_pre'].values
    output_this_iter['std_in_sample_perc_low_risk_pred_post'] = patient_metrics['in_sample_perc_low_risk_pred_post'].values
    output_this_iter['std_out_sample_perc_low_risk_pred_pre'] = patient_metrics['out_sample_perc_low_risk_pred_pre'].values
    output_this_iter['std_out_sample_perc_low_risk_pred_post'] = patient_metrics['out_sample_perc_low_risk_pred_post'].values
    return output_this_iter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect result")
    parser.add_argument("crmg_experiment", type=str, 
                        choices=[
                            'crmg_experiment_v10', 
                            'crmg_experiment_v12',
                            'crmg_experiment_v13', 
                            'crmg_experiment_v20', 
                            'crmg_experiment_v21',
                            'crmg_experiment_v26',
                            'crmg_experiment_v27',
                            'crmg_experiment_v28',
                            'crmg_experiment_v29'
                            ],
                        help="The name of the system to process")
    args = parser.parse_args()


    exp_name = args.crmg_experiment
    Config = ExperimentConfig(exp_name)
    Config.CRMdb_columns = [
            'PatientID',
            'MaskID', 
            # pre
            'Visit_pre', 'female_pre', 'baseline_age_pre',
            'cvd_hx_baseline_pre', 'black_pre', 'smoke_pre', 'bmi_pre', 'sbp_pre',
            'dbp_pre', 'hr_pre', 'hba1c_pre', 'chol_pre', 'ldl_pre', 'hdl_pre',
            'trig_pre', 'fpg_pre', 'potassium_pre', 'screat_pre', 'uacr_pre',
            'bprx_pre', 'statin_pre',
            # post
            'Visit_post', 'female_post', 'baseline_age_post',
            'cvd_hx_baseline_post', 'black_post', 'smoke_post', 'bmi_post',
            'sbp_post', 'dbp_post', 'hr_post', 'hba1c_post', 'chol_post',
            'ldl_post', 'hdl_post', 'trig_post', 'fpg_post', 'potassium_post',
            'screat_post', 'uacr_post', 'bprx_post', 'statin_post',
            # metrics
            'crm_ai_ml_feasibility', 'personal_feasibility', 'crm_feasibility', 'trans_cost',
            'in_sample_perc_low_risk_pred_pre', 'in_sample_perc_low_risk_pred_post',
            'out_sample_perc_low_risk_pred_pre', 'out_sample_perc_low_risk_pred_post'
        ]
    pre_csv_path='data/accord_blr_risks.csv'
    post_csv_path='data/accord_f24_risks.csv'
    crm_pre = pd.read_csv(pre_csv_path)
    crm_post = pd.read_csv(post_csv_path)

    CRMdb_general = prepare_crmdb_general_only_perc_low_risk(crm_pre=crm_pre,crm_post=crm_post, mergeon='MaskID',config=Config)

    patients_record_train = load_patient_records('train', Config)  # loads 'training' file
    CRMdb_train = create_personal_crmdb(patients_record_train, CRMdb_general, Config)

    patients_record_test = load_patient_records('test', Config)   # loads 'test' file
    CRMdb_test= create_personal_crmdb(patients_record_test, CRMdb_general, Config)
    metric_train = collect_metrics_only_perc_low_risk(CRMdb_train, patients_record_train, iteration=0, use_type='train', exp_name=exp_name)
    metric_test  = collect_metrics_only_perc_low_risk(CRMdb_test,  patients_record_test, iteration=0, use_type='test', exp_name=exp_name)

    print('CRMS: percentage of low-risk prediction')
    print('Train: (averaged across training patients)')
    print(
    metric_train[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].mean(axis=0)
    )
    print('Train: (std across training patients)')
    print(
    metric_train[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].std(axis=0)
    )
    print()
    print()
    print('Test: (averaged across training patients)')
    print(
    metric_test[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].mean(axis=0)
    )
    print('Test: (std across training patients)')
    print(
    metric_test[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].std(axis=0)
    )


    print('done with CRMS')

    cur_iter = 6
    print('start with CRMG at iteration 6')

    model_prev, tokenizer_prev = FastLanguageModel.from_pretrained(
        model_name = f'{Config.output_dir_ft_model}/CRMG_iter_{cur_iter}',
        max_seq_length = Config.max_seq_length,
        dtype = Config.dtype,
        load_in_4bit = Config.load_in_4bit,
    )
    FastLanguageModel.for_inference(model_prev)

    ###### read patient
    # read patient data - do it every iteration to prevent overwriting
    patients_record_train = load_patient_records('train', Config)  # loads 'training' file
    patients_record_test = load_patient_records('test', Config)   # loads 'test' file

    print('Generating CRMs for training patients')
    crm_dec_pre_train, crm_dec_post_train = generate_crms_for_patients(
        patients_record_train, 
        model_prev, 
        tokenizer_prev, 
        Config, 
        batch_size=100, 
        prompts_per_patient=6, 
        verbose=False
    )
    print('Generating CRMs for testing patients')
    crm_dec_pre_test, crm_dec_post_test = generate_crms_for_patients(
        patients_record_test, 
        model_prev, 
        tokenizer_prev, 
        Config, 
        batch_size=100, 
        prompts_per_patient=6, 
        verbose=False
    )

    ###### compute risks for generated crms
    print('Compute risks for generated crms')
    risk_results_train = predict_risks_for_crms(
        crm_dec_pre=crm_dec_pre_train,
        crm_dec_post=crm_dec_post_train,
        config=Config
    )

    risk_results_test = predict_risks_for_crms(
        crm_dec_pre=crm_dec_pre_test,
        crm_dec_post=crm_dec_post_test,
        config=Config
    )

    ###### compute risks for generated crms
    # create CRMdb and ai_ml_feasibility
    print('Compute crm_ai_ml_feasibility')
    CRMdb_train = prepare_crmdb_general_only_perc_low_risk(crm_pre=risk_results_train['crm_blr_risks'],
                                    crm_post=risk_results_train['crm_f24_risks'],
                                    mergeon=['PatientID','MaskID'],
                                    config=Config)
    CRMdb_test = prepare_crmdb_general_only_perc_low_risk(crm_pre=risk_results_test['crm_blr_risks'],
                                    crm_post=risk_results_test['crm_f24_risks'],
                                    mergeon=['PatientID','MaskID'],
                                    config=Config)

    # update CRMdb to for personal metrics
    print('Compute feasibility and costs for each patient-crm pair')
    CRMdb_train = update_crmdb(
        CRMdb=CRMdb_train, 
        patients_record=patients_record_train, 
        config=Config
    )
    CRMdb_test = update_crmdb(
        CRMdb=CRMdb_test, 
        patients_record=patients_record_test, 
        config=Config
    )

    metric_train = collect_metrics_only_perc_low_risk(CRMdb_train, patients_record_train, iteration=cur_iter+1, use_type='train', exp_name=exp_name)
    metric_test  = collect_metrics_only_perc_low_risk(CRMdb_test,  patients_record_test, iteration=cur_iter+1, use_type='test', exp_name=exp_name)
    print('CRMS: percentage of low-risk prediction')
    print('Train: (averaged across training patients)')
    print(
    metric_train[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].mean(axis=0)
    )
    print('Train: (std across training patients)')
    print(
    metric_train[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].std(axis=0)
    )
    print()
    print()
    print('Test: (averaged across training patients)')
    print(
    metric_test[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].mean(axis=0)
    )
    print('Test: (std across training patients)')
    print(
    metric_test[[
        'avg_in_sample_perc_low_risk_pred_pre', 'avg_in_sample_perc_low_risk_pred_post',
        'avg_out_sample_perc_low_risk_pred_pre', 'avg_out_sample_perc_low_risk_pred_post'
                ]].std(axis=0)
    )
