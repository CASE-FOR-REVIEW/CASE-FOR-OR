import sys
from main_algorithms import run_crms, run_crmg

if len(sys.argv) > 1:
    exp_name = sys.argv[1]
else:
    exp_name = ''  # default


print('running CRMS')
run_crms(exp_name)
print('running CRMG')
run_crmg(exp_name)