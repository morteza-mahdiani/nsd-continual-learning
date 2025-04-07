import os
import numpy as np
from nsd_access import NSDAccess
from nsd_get_data_light import get_conditions, load_or_compute_betas_average, get_betas, get_1000

subj = "subj01"
n_sessions = 40

 # Local directory to mimic NSD structure
data_dir = '/home/mahdiani/projects/rrg-charesti/data/natural-scenes-dataset/' 

# Extract a limited set of conditions
conditions = get_conditions(data_dir, subj, n_sessions=n_sessions)
conditions = np.asarray(conditions).ravel()
print(f"Loaded {len(conditions)} trials for {subj}")

# 1000 shared conditions 
conditions_test = np.asarray(get_1000(data_dir)).ravel()
# Unique 9000 conditions for each subject
conditions_train = np.unique(conditions[np.logical_not(np.isin(conditions, conditions_test))])

# Debugging: Ensure conditions are matched correctly
# unique_conditions, counts = np.unique(conditions, return_counts=True)
# for cond, count in zip(unique_conditions, counts):
#     print(f"Condition {cond} appears {count} times in unique set")


# Define target space and output path for betas
betas_file =  './data/fs_avg/subj01/'
local_base_dir = os.path.join(os.getcwd(), betas_file)
os.makedirs(local_base_dir, exist_ok=True)
betas_file += 'subj01_beta.npy'
targetspace = "fsaverage"
# Compute or load betas
betas = load_or_compute_betas_average(betas_file=betas_file, nsd_dir=data_dir, subj=subj, n_sessions=n_sessions, conditions=conditions, conditions_sampled=conditions_train, targetspace=targetspace)
print("Betas shape after averaging:", betas.shape)

# Define target space and output path for betas
betas_file =  './data/fs_avg/shared/'
local_base_dir = os.path.join(os.getcwd(), betas_file)
os.makedirs(local_base_dir, exist_ok=True)
betas_file += 'shared_beta.npy'
targetspace = "fsaverage"
# Compute or load betas
betas = load_or_compute_betas_average(betas_file=betas_file, nsd_dir=data_dir, subj=subj, n_sessions=n_sessions, conditions=conditions, conditions_sampled=conditions_test, targetspace=targetspace)
print("Betas shape after averaging:", betas.shape)
