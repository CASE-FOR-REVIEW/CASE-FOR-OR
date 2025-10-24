#### Data Preprocessing
1. `run_accord_preprocess.ipynb` creates local CRM database
2. `run_generate_risk_models.ipynb` trains risk scoring tools
3. `run_risk_inference_<data_name>.ipynb` loads models trained in Step 2. and append the risk estimates
4. `split_train_test.ipynb` creates 200 high-risk patients for training and 100 for testing

#### Main Algorithm
- `run_jobs.py` executes warm-start and active self-improving learning
- `main_algorithms` stores wrapper functions for both warm-start and self-improving
- `utils.py` stores functions to process LLM texts and to create prompts from patient features
- `crms_utils.py` and `crmg_utlils.py` stores helper functions for warm-up and active self-improving
- `collect_perc_low_risks.py` loads checkpointed LLMs to create other metrics
