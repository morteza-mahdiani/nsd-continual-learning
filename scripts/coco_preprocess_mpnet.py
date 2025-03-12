import os
import torch
import numpy as np
import psutil
import time
import json
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer
from pycocotools.coco import COCO

# ==== CONFIGURATION ====
coco_annotation_file = './data/coco_data/annotations/captions_val2017.json'  # Path to COCO captions
output_dir = './data/mpnet_embeddings/'  # Directory to store embeddings
os.makedirs(output_dir, exist_ok=True)

# ==== GPU SETUP ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MPNet model on GPU from local path instead of downloading
model_path = "./data/models/mpnet_model"
model = SentenceTransformer(model_path).to(device)
# model = SentenceTransformer('all-roberta-large-v1').to(device)

# Load COCO dataset
coco = COCO(coco_annotation_file)

# List of COCO IDs for subject 01
exp_design = "./data/nsddata/experiments/nsd/nsd_expdesign.mat"
synth_design = "./data/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat"
nsd_stiminfo = './data/nsddata/experiments/nsd/nsd_stim_info_merged.pkl'
stiminfo = pd.read_pickle(nsd_stiminfo)
exp_design = loadmat(exp_design)
synth_design = loadmat(synth_design)

subject_idx  = exp_design['subjectim']
shared_idx   = exp_design['sharedix']

filtered_subject_idx = [np.setdiff1d(row, shared_idx) for row in subject_idx]
filtered_subject_idx = np.array(filtered_subject_idx, dtype=object)

cocoId_arr = np.zeros(shape=filtered_subject_idx.shape, dtype=int)

for j in range(len(filtered_subject_idx)):
    cocoId = np.array(stiminfo['cocoId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
    nsdId = np.array(stiminfo['nsdId'])[stiminfo['subject%d'%(j+1)].astype(bool)]
    imageId = filtered_subject_idx[j]-1
    for i,k in enumerate(imageId):
        if (k+1) in shared_idx:
          continue
        else:
          cocoId_arr[j,i] = (cocoId[nsdId==k])[0]  # COCO ID for each image

subject_one_coco_ids = cocoId_arr[0] # COCO IDs for subject 01

# ==== LOGGING FUNCTION ====
def log_gpu_cpu_usage(log_file):
    """Logs GPU and CPU usage to a file."""
    with open(log_file, 'a') as f:
        while True:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
                gpu_usage = torch.cuda.memory_reserved() / 1e9  # GB
            else:
                gpu_memory, gpu_usage = 0, 0

            log_entry = f"CPU: {cpu_usage}% | Memory: {memory_usage}% | GPU Memory: {gpu_memory}GB | GPU Usage: {gpu_usage}GB\n"
            f.write(log_entry)
            f.flush()
            time.sleep(10)  # Log every 10 seconds

# Start logging in a separate process
import multiprocessing
log_file_path = os.path.join(output_dir, "gpu_cpu_usage.log")
log_process = multiprocessing.Process(target=log_gpu_cpu_usage, args=(log_file_path,))
log_process.start()

# ==== PROCESS IMAGES ====
embeddings_dict = {}

for coco_id in subject_one_coco_ids:
    ann_ids = coco.getAnnIds(imgIds=[coco_id])
    anns = coco.loadAnns(ann_ids)

    if anns:
        first_caption = anns[0]['caption']
        print(f"Processing Image ID {coco_id}: {first_caption}")

        # Convert caption to embedding
        with torch.no_grad():
            embedding = model.encode([first_caption], convert_to_numpy=True, device=device)[0]

        # Save embedding
        np.save(os.path.join(output_dir, f"{coco_id}.npy"), embedding)
        embeddings_dict[coco_id] = first_caption  # Store captions for reference

# Save metadata
# Convert NumPy int64 keys to Python int
embeddings_dict = {int(k): v for k, v in embeddings_dict.items()}

# Save JSON file
with open(os.path.join(output_dir, "captions_metadata.json"), 'w') as f:
    json.dump(embeddings_dict, f, indent=4)

# Stop logging
log_process.terminate()

print("Processing complete. Embeddings saved in ./data/")
print(f"GPU and CPU usage logged in {log_file_path}")