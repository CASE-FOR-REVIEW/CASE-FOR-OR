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
from trl import SFTConfig, SFTTrainer

import random
### self-defined code modules
from ai_ml_feasibility_check import * # uncertainty feasibility checking

from utils import *
from config import ExperimentConfig
from crms_utils import *
from crmg_utils import generate_crms_for_patients, predict_risks_for_crms, update_crmdb, active_learning_select

def run_crms(exp_name):
    Config = ExperimentConfig(exp_name)
    
    ############ Init base LLM ############
    print(f"Loading base LLM model")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = "unsloth/Meta-Llama-3.1-8B",
        model_name = Config.base_model,
        max_seq_length = Config.max_seq_length,
        dtype = Config.dtype,
        load_in_4bit = Config.load_in_4bit,
    )

    ############ lora adaptor ############
    # add the adaptor
    print(f"Loading base LORA adaptor")
    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    ############ Read and Prepare data ############
    print(f"Loading data and preparing CRMdb")
    
    pre_csv_path='data/accord_blr_risks.csv'
    post_csv_path='data/accord_f24_risks.csv'
    crm_pre = pd.read_csv(pre_csv_path)
    crm_post = pd.read_csv(post_csv_path)

    CRMdb_general = prepare_crmdb_general(crm_pre=crm_pre,crm_post=crm_post, mergeon='MaskID',config=Config)

    patients_record_train = load_patient_records('train', Config)  # loads 'training' file
    CRMdb_train = create_personal_crmdb(patients_record_train, CRMdb_general, Config)

    patients_record_test = load_patient_records('test', Config)   # loads 'test' file
    CRMdb_test= create_personal_crmdb(patients_record_test, CRMdb_general, Config)
    
    
    ############ Collect and save metrics  ############
    print(f"Saving initial metrics")
    metric_train = collect_metrics(CRMdb_train, patients_record_train, iteration=0, use_type='train', exp_name=exp_name)
    metric_test  = collect_metrics(CRMdb_test,  patients_record_test, iteration=0, use_type='test', exp_name=exp_name)

    metric_train.to_csv(
        f'{Config.output_dir_result}/metrics_train_iter_0.csv',     
        index=False,           # Don't write row indices
        header=True 
    )
    metric_train.to_pickle(f'{Config.output_dir_result}/metrics_train_iter_0.pkl')

    metric_test.to_csv(
        f'{Config.output_dir_result}/metrics_test_iter_0.csv',     
        index=False,           # Don't write row indices
        header=True 
    )
    metric_test.to_pickle(f'{Config.output_dir_result}/metrics_test_iter_0.pkl')

    ############ Generate prompts  ############
    print(f"Geneating prompts for training")
    prompts = generate_prompts_from_crmdb(
        patients_record=patients_record_train,
        CRMdb=CRMdb_train,
        tokenizer=tokenizer,
        lambda_lm=Config.lambda_lm,
        seed=42
    )
    hf_dataset = Dataset.from_list(prompts)

    ############ Init Trainer ############
    trainer = SFTTrainer(
        model = base_model,
        tokenizer = tokenizer,
        train_dataset = hf_dataset,
        dataset_text_field = "text",
        max_seq_length = Config.max_length,
        packing = True, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            warmup_steps = 15,
            num_train_epochs = 3, # Set this for 1 full training run.
            lr_scheduler_type = "cosine",
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            seed = 3407,
            output_dir = Config.output_dir_checkpoint,
            report_to = "none", # Use this for WandB etc
        ),
    )

    # Training
    trainer_stats = trainer.train()

    # save model
    print(f"Saving trained model")
    base_model.save_pretrained(f'{Config.output_dir_ft_model}/CRMG_iter_0' )  # Local saving
    tokenizer.save_pretrained(f'{Config.output_dir_ft_model}/CRMG_iter_0' )
    
    # save number of selected for training
    num_selected_for_training = update_selection_counts(
        patients_record=patients_record_train, 
        iteration=0.,
        use_type='train', 
        exp_name=exp_name
    )
    num_selected_for_training.to_csv(
        f'{Config.output_dir_result}/num_selected_for_training_iter_0.csv',
        index=False,           # Don't write row indices
        header=True 
    )
    print(f"Done CRMS!")

def run_crmg(exp_name):
    ##### Read config files
    Config = ExperimentConfig(exp_name)

    # for each iteration
    for cur_iter in range(Config.N_iter):
        # read trained model
        print(f'Loading model iter_{cur_iter}')
    
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
        CRMdb_train = prepare_crmdb_general(crm_pre=risk_results_train['crm_blr_risks'],
                                           crm_post=risk_results_train['crm_f24_risks'],
                                           mergeon=['PatientID','MaskID'],
                                           config=Config)
        CRMdb_test = prepare_crmdb_general(crm_pre=risk_results_test['crm_blr_risks'],
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

        # Collect and save metrics
        print('Collect metrics')
        metric_train = collect_metrics(CRMdb_train, patients_record_train, iteration=cur_iter+1, use_type='train', exp_name=exp_name)
        metric_test  = collect_metrics(CRMdb_test,  patients_record_test, iteration=cur_iter+1, use_type='test', exp_name=exp_name)

        metric_train.to_csv(
            f'{Config.output_dir_result}/metrics_train_iter_{cur_iter+1}.csv',     
            index=False,           # Don't write row indices
            header=True 
        )
        metric_train.to_pickle(f'{Config.output_dir_result}/metrics_train_iter_{cur_iter+1}.pkl')

        metric_test.to_csv(
            f'{Config.output_dir_result}/metrics_test_iter_{cur_iter+1}.csv',     
            index=False,           # Don't write row indices
            header=True 
        )
        metric_test.to_pickle(f'{Config.output_dir_result}/metrics_test_iter_{cur_iter+1}.pkl')

        print(f'Perform active learning algorithm {Config.active_learning}')
        # active learning - get training data
        sampled_index, num_selected_for_training = active_learning_select(
            metric_train=metric_train, 
            cur_iter=cur_iter, 
            config=Config)
        
        # update selection counts
        print(f'Compute and update selection results')
        num_selected_for_training = update_selection_counts(
            patients_record=patients_record_train, 
            iteration=cur_iter+1,
            use_type='train', 
            exp_name=exp_name,
            num_selections=num_selected_for_training

        )
        num_selected_for_training.to_csv(
            f'{Config.output_dir_result}/num_selected_for_training_iter_{cur_iter+1}.csv',
            index=False,           # Don't write row indices
            header=True 
        )


        ##### Now, create new training samples using the sampled index
        print(f'Creating new training samples')
        # First get PatientIDs from the sampled indices
        sampled_patient_ids = metric_train.iloc[sampled_index]['PatientID'].values

        # Then filter CRMdb by those PatientIDs  
        CRMdb_train = CRMdb_train[CRMdb_train['PatientID'].isin(sampled_patient_ids)]
        

        # generate training prompts
        print(f"Geneating prompts for training")
        prompts = generate_prompts_from_crmdb(
            patients_record=patients_record_train,
            CRMdb=CRMdb_train,
            tokenizer=tokenizer_prev,
            lambda_lm=Config.lambda_lm,
            seed=42
        )
        hf_dataset = Dataset.from_list(prompts)


        # load again the previous model for training, since the model_prev was optimized for inference
        print(f'Load CRMG_iter_{cur_iter} model again since the previous one was set for inference')
        model_prev, tokenizer_prev = FastLanguageModel.from_pretrained(
            model_name = f'{Config.output_dir_ft_model}/CRMG_iter_{cur_iter}',
            max_seq_length = Config.max_seq_length,
            dtype = Config.dtype,
            load_in_4bit = Config.load_in_4bit,
        )

        # train model
        ############ Init Trainer ############
        trainer = SFTTrainer(
            model = model_prev,
            tokenizer = tokenizer_prev,
            train_dataset = hf_dataset,
            dataset_text_field = "text",
            max_seq_length = Config.max_length,
            packing = True, # Can make training 5x faster for short sequences.
            args = SFTConfig(
                per_device_train_batch_size = 6,
                gradient_accumulation_steps = 1,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                dataloader_prefetch_factor=2,
                warmup_steps = 5,
                num_train_epochs = 2, # Set this for 1 full training run.
                lr_scheduler_type = "cosine",
                learning_rate = 1e-4,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                seed = 3407,
                output_dir = Config.output_dir_checkpoint,
                report_to = "none", # Use this for WandB etc
            ),
        )

        # Training
        trainer_stats = trainer.train()

        # save model
        print(f"Saving trained model")
        model_prev.save_pretrained(f'{Config.output_dir_ft_model}/CRMG_iter_{cur_iter+1}' )  # Local saving
        tokenizer_prev.save_pretrained(f'{Config.output_dir_ft_model}/CRMG_iter_{cur_iter+1}' )
        
        print(f"Done CRMG iter {cur_iter}!")


