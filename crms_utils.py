import pandas as pd
from scipy.stats import chi2
from utils import create_clinical_role_model_profile
from config import *
def load_patient_records(data_type='train', config=None, data_dir='data'):
    """
    Load patient records from NHANES data.
    
    Args:
        data_type: 'train' or 'test' to specify which dataset to load
        config: Config object with parameters (RISK_MODEL, PT_FEATURES)
        data_dir: Base directory for data files
        
    Returns:
        pd.DataFrame: Patient records with features and patient_text
    """
    import os
    
    # Determine file path based on data_type
    if data_type == 'train':
        file_name = 'nhanes_data_prediction_training.csv'
    elif data_type == 'test':
        file_name = 'nhanes_data_prediction_test.csv'
    else:
        raise ValueError(f"data_type must be 'train' or 'test', got '{data_type}'")
    
    # Construct full path
    nhanes_data = os.path.join(data_dir, file_name)
    
    # Read data
    df = pd.read_csv(nhanes_data)
    
    # Use 1 risk model for diagnosis
    first_observed_risk = [f'risk_prediction_{config.RISK_MODEL}_run1']
    
    # Select relevant columns
    patients_record = df[config.PT_FEATURES + first_observed_risk].copy()
    
    # Create patient narrative text
    patients_record['patient_text'] = patients_record.apply(create_patient_narrative, axis=1)
    
    return patients_record


def prepare_crmdb_general(crm_pre, crm_post, mergeon, config):
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
    CRMdb_general['in_sample_risk_pre_mean'] = CRMdb_general[observed_risk_models_pre].mean(axis=1)
    CRMdb_general['in_sample_risk_pre_std'] = CRMdb_general[observed_risk_models_pre].std(axis=1)
    CRMdb_general['out_sample_risk_pre_mean'] = CRMdb_general[unobserved_risk_models_pre].mean(axis=1)
    CRMdb_general['out_sample_risk_pre_std'] = CRMdb_general[unobserved_risk_models_pre].std(axis=1)
    
    CRMdb_general['in_sample_risk_post_mean'] = CRMdb_general[observed_risk_models_post].mean(axis=1)
    CRMdb_general['in_sample_risk_post_std'] = CRMdb_general[observed_risk_models_post].std(axis=1)
    CRMdb_general['out_sample_risk_post_mean'] = CRMdb_general[unobserved_risk_models_post].mean(axis=1)
    CRMdb_general['out_sample_risk_post_std'] = CRMdb_general[unobserved_risk_models_post].std(axis=1)
    
    CRMdb_general['in_sample_risk_reduction_mean'] = (
        CRMdb_general[observed_risk_models_pre].values - 
        CRMdb_general[observed_risk_models_post].values
    ).mean(axis=1)
    
    CRMdb_general['out_sample_risk_reduction_mean'] = (
        CRMdb_general[unobserved_risk_models_pre].values - 
        CRMdb_general[unobserved_risk_models_post].values
    ).mean(axis=1)
    
    CRMdb_general['in_sample_risk_reduction_std'] = (
        CRMdb_general[observed_risk_models_pre].values - 
        CRMdb_general[observed_risk_models_post].values
    ).std(axis=1)
    
    CRMdb_general['out_sample_risk_reduction_std'] = (
        CRMdb_general[unobserved_risk_models_pre].values - 
        CRMdb_general[unobserved_risk_models_post].values
    ).std(axis=1)

    # Compute AI/ML feasibility
    if config.beta > 0: # DRD case
        N_scenario = CRMdb_general[observed_risk_models_pre].shape[1]
        degrees_of_freedom = N_scenario - 1
        confidence_level = 1 - config.beta
        chi_squared_val = chi2.ppf(confidence_level, df=degrees_of_freedom)
        rho = chi_squared_val / (2 *N_scenario)
        crm_ai_ml_feasibility = config.solver.solve_batch(
            risks_pre_df=CRMdb_general[observed_risk_models_pre],
            risks_post_df=CRMdb_general[observed_risk_models_post],
            r=config.RISK_THRESHOLD,
            alpha=config.ALPHA,
            rho=rho
        )
    else:
        crm_ai_ml_feasibility = config.solver.solve_batch(
            risks_pre_df=CRMdb_general[observed_risk_models_pre],
            risks_post_df=CRMdb_general[observed_risk_models_post],
            r=config.RISK_THRESHOLD,
            alpha=config.ALPHA
        )
    CRMdb_general['crm_ai_ml_feasibility'] = crm_ai_ml_feasibility
    
    return CRMdb_general


def create_personal_crmdb(patients_record, CRMdb_general, config):
    """
    Create personalized CRMdb by selecting top K CRMs for each patient.
    
    Args:
        patients_record: DataFrame with patient data
        CRMdb_general: General CRMdb with all potential CRMs
        config: Config object with parameters (K, matching_columns_pre, etc.)
        
    Returns:
        pd.DataFrame: Personalized CRMdb with top K CRMs per patient
    """
    # Create personal CRMdb
    target_rows = config.K * len(patients_record)
    
    # Initialize empty CRMdb with correct dtypes
    CRMdb = pd.DataFrame(index=range(target_rows), columns=config.CRMdb_columns)
    
    # Track current row position in CRMdb
    current_row = 0
    
    # For each patient, loop through general crmdb and select top K by sorting feasibility then costs
    for patient_id in range(len(patients_record)):
        tmp_crmdb = CRMdb_general.copy()
        tmp_crmdb['PatientID'] = patient_id
        
        # Compute feasibility (all matching columns must match)
        personal_feasibility_mask = True
        for col in config.matching_columns_pre:
            col_db = col + '_pre'
            personal_feasibility_mask = personal_feasibility_mask & (
                tmp_crmdb[col_db] == patients_record.loc[patient_id, col]
            )
        tmp_crmdb['personal_feasibility'] = personal_feasibility_mask
        tmp_crmdb['crm_feasibility'] = tmp_crmdb['crm_ai_ml_feasibility'] & tmp_crmdb['personal_feasibility']
        
        # Compute cost
        differences = (
            tmp_crmdb[config.loss_condition_columns_db].values - 
            patients_record.loc[[patient_id]][config.loss_condition_columns].values
        )
        if config.COST_PARAMS['type'] == 'p-norm':
            tmp_crmdb['trans_cost'] = np.round(
                np.linalg.norm(differences, ord=config.COST_PARAMS['p'], axis=1), 4
            )
        
        # Rank tmp_crmdb: feasible first (True before False), then by cost (ascending)
        tmp_crmdb_sorted = tmp_crmdb.sort_values(
            by=['crm_feasibility', 'trans_cost'], 
            ascending=[False, True]  # feasible=True first (descending), then low cost first (ascending)
        )
        
        # Select top K rows
        top_k = tmp_crmdb_sorted.head(config.K)
        
        # Copy selected rows to CRMdb
        for col in config.CRMdb_columns:
            if col == 'PatientID':
                CRMdb.loc[current_row:current_row+config.K-1, col] = patient_id
            elif col in top_k.columns:
                CRMdb.loc[current_row:current_row+config.K-1, col] = top_k[col].values
        
        # Update current row position
        current_row += config.K
    
    # Create text for all CRMs
    CRMdb['crm_text'] = CRMdb.apply(create_clinical_role_model_profile, axis=1)
    
    return CRMdb


def collect_metrics(CRMdb, patients_record, iteration, use_type, exp_name):
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
        'in_sample_risk_pre_mean': 'mean',
        'in_sample_risk_pre_std': 'mean',
        'out_sample_risk_pre_mean': 'mean',
        'out_sample_risk_pre_std': 'mean',
        'in_sample_risk_post_mean': 'mean',
        'in_sample_risk_post_std': 'mean',
        'out_sample_risk_post_mean': 'mean',
        'out_sample_risk_post_std': 'mean',
        'in_sample_risk_reduction_mean': 'mean',
        'in_sample_risk_reduction_std':'mean', 
        'out_sample_risk_reduction_mean':'mean', 
        'out_sample_risk_reduction_std':'mean'
        
    })
    
    # Map aggregated values to output columns
    output_this_iter['feasibility_rate'] = patient_metrics['crm_feasibility'].values
    output_this_iter['cur_best_trans_cost'] = patient_metrics['trans_cost'].values
    # pre and post risks
    output_this_iter['avg_in_sample_risk_pre_mean'] = patient_metrics['in_sample_risk_pre_mean'].values
    output_this_iter['avg_in_sample_risk_pre_std'] = patient_metrics['in_sample_risk_pre_std'].values
    output_this_iter['avg_out_sample_risk_pre_mean'] = patient_metrics['out_sample_risk_pre_mean'].values
    output_this_iter['avg_out_sample_risk_pre_std'] = patient_metrics['out_sample_risk_pre_std'].values
    output_this_iter['avg_in_sample_risk_post_mean'] = patient_metrics['in_sample_risk_post_mean'].values
    output_this_iter['avg_in_sample_risk_post_std'] = patient_metrics['in_sample_risk_post_std'].values
    output_this_iter['avg_out_sample_risk_post_mean'] = patient_metrics['out_sample_risk_post_mean'].values
    output_this_iter['avg_out_sample_risk_post_std'] = patient_metrics['out_sample_risk_post_std'].values
    # risk reductions
    output_this_iter['avg_in_sample_risk_reduction_mean'] = patient_metrics['in_sample_risk_reduction_mean'].values
    output_this_iter['avg_in_sample_risk_reduction_std'] = patient_metrics['in_sample_risk_reduction_std'].values
    output_this_iter['avg_out_sample_risk_reduction_mean'] = patient_metrics['out_sample_risk_reduction_mean'].values
    output_this_iter['avg_out_sample_risk_reduction_std'] = patient_metrics['out_sample_risk_reduction_std'].values
    

    return output_this_iter

def update_selection_counts(patients_record, iteration, use_type, exp_name, num_selections=None):
    num_patients = len(patients_record)
    if use_type == 'test':
        num_selections = np.zeros(num_patients)
    else:
        if iteration == 0:
            num_selections = np.ones(num_patients)
    
    output_selection = pd.DataFrame({
        'exp_name': [exp_name] * num_patients,  # Repeat for each patient
        'PatientID': patients_record.index,
        'iteration': iteration,
        'use_type': use_type,
        'num_selections': num_selections
    })
    return output_selection
    
    
        
