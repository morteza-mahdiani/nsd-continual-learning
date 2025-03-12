import os
import numpy as np
# import boto3
# from botocore import UNSIGNED
# from botocore.config import Config
from nsd_access import NSDAccess
from nsd_get_data_light import get_conditions, load_or_compute_betas_average, get_betas

# Configure boto3 for anonymous access
# s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# Bucket and base path
# bucket_name = "natural-scenes-dataset"
# base_path = "data/nsddata/ppdata"

# Local base directory to store NSD data
# local_base_dir = os.path.join(os.getcwd(), "data/nsddata/ppdata")
# os.makedirs(local_base_dir, exist_ok=True)

# Define a single subject and small data subset
subj = "subj01"
n_sessions = 40

# Download behavioral response file for subj01
# remote_file_key = f"{base_path}/{subj}/behav/responses.tsv"
# local_file_dir = os.path.join(local_base_dir, subj, "behav")
# local_file_path = os.path.join(local_file_dir, "responses.tsv")

# os.makedirs(local_file_dir, exist_ok=True)

# try:
#     print(f"Downloading {remote_file_key} to {local_file_path}...")
#     s3.download_file(bucket_name, remote_file_key, local_file_path)
#     print(f"Downloaded {local_file_path}")
# except Exception as e:
#     print(f"Failed to download {remote_file_key}: {e}")

# Load NSD conditions
data_dir = './data/'  # Local directory to mimic NSD structure
nsda = NSDAccess(data_dir)

# Extract a limited set of conditions
conditions = get_conditions(data_dir, subj, n_sessions=n_sessions)
conditions = np.asarray(conditions).ravel()
print(f"Loaded {len(conditions)} trials for {subj}")

# Sample valid trials (limit scope to simplify)
conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
conditions_sampled = conditions[conditions_bool]
print(f"Filtered down to {len(conditions_sampled)} trials with 3 repetitions")

# Debugging: Ensure conditions are matched correctly
unique_conditions, counts = np.unique(conditions, return_counts=True)
for cond, count in zip(unique_conditions, counts):
    print(f"Condition {cond} appears {count} times")

# Check sampled conditions before averaging
print(f"Sampled conditions: {np.unique(conditions_sampled)}")

# Define target space and output path for betas
betas_file_path = './data/avg/subj01/'
local_base_dir = os.path.join(os.getcwd(), betas_file_path)
os.makedirs(local_base_dir, exist_ok=True)

betas_file =  './data/avg/subj01/subj01_beta.npy'
targetspace = "func1pt8mm"

# Load raw betas for inspection
raw_betas = get_betas(data_dir, subj, n_sessions=n_sessions, targetspace=targetspace)
raw_betas = np.concatenate(raw_betas, axis=-1)
print("Raw Betas shape:", raw_betas.shape)

# Inspect NaNs in raw betas before averaging
for cond in np.unique(conditions_sampled):
    conditions_bool = conditions == cond
    selected_data = raw_betas[:, :, :, conditions_bool]
    if np.isnan(selected_data).all():
        print(f"Warning: All data is NaN for condition {cond}")
    elif np.isnan(selected_data).any():
        print(f"Partial NaNs detected for condition {cond}")
    else:
        print(f"Condition {cond} has valid data.")

# Compute or load betas
betas = load_or_compute_betas_average(betas_file=betas_file, nsd_dir=data_dir, subj=subj, n_sessions=n_sessions, conditions=conditions, conditions_sampled=conditions_sampled, targetspace=targetspace)
print("Betas shape after averaging:", betas.shape)

