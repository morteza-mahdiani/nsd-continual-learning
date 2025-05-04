# import os
# import torch
# import numpy as np
# import pandas as pd
# import json
# from scipy.io import loadmat
# from PIL import Image
# from tqdm import tqdm
# import requests
# from sentence_transformers import SentenceTransformer
# from pycocotools.coco import COCO

# # ==== CONFIG ====
# output_dir = './data/'
# os.makedirs(output_dir, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# model = SentenceTransformer('all-mpnet-base-v2').to(device)
# print("MPNet model loaded.")

# # ==== Load Shared COCO IDs ====
# expd_path = '/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat'
# stiminfo_path = '/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl'

# expd = loadmat(expd_path)
# shared = expd['sharedix'].ravel()
# stiminfo = pd.read_pickle(stiminfo_path)
# shared_df = stiminfo[stiminfo['nsdId'].isin(shared)]
# shared_ids = shared_df['cocoId'].values
# print(f"Total shared samples: {len(shared_ids)}")

# # ==== Load COCO captions ====
# coco_val = COCO('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/annotations/captions_val2017.json')
# coco_train = COCO('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/annotations/captions_train2017.json')

# def get_caption(coco_id):
#     for coco in [coco_val, coco_train]:
#         if coco_id in coco.getImgIds():
#             anns = coco.loadAnns(coco.getAnnIds(imgIds=[coco_id]))
#             if anns:
#                 return anns[0]['caption']
#     return None

# def download_image(coco_id):
#     padded_id = f"{int(coco_id):012d}"
#     urls = [
#         f"http://images.cocodataset.org/val2017/{padded_id}.jpg",
#         f"http://images.cocodataset.org/train2017/{padded_id}.jpg"
#     ]
#     for url in urls:
#         try:
#             r = requests.get(url, timeout=10)
#             if r.status_code == 200:
#                 return np.array(Image.open(BytesIO(r.content)).convert("RGB"))
#         except:
#             continue
#     return None

# # ==== Main loop ====
# embedding_list = []
# image_list = []
# caption_dict = {}
# used_ids = []

# for coco_id in tqdm(shared_ids, desc="Processing shared samples"):
#     caption = get_caption(coco_id)
#     if caption is None:
#         continue

#     image = download_image(coco_id)
#     if image is None:
#         continue

#     with torch.no_grad():
#         embedding = model.encode([caption], convert_to_numpy=True, device=device)[0]

#     embedding_list.append(embedding)
#     image_list.append(image)
#     caption_dict[int(coco_id)] = caption
#     used_ids.append(coco_id)

# # Convert to arrays
# embedding_array = np.vstack(embedding_list)
# image_array = np.array(image_list)
# coco_id_array = np.array(used_ids)

# # Save to file
# np.savez_compressed(os.path.join(output_dir, "mpnet_shared_embeddings_and_images.npz"),
#                     embeddings=embedding_array,
#                     images=image_array,
#                     coco_ids=coco_id_array)

# with open(os.path.join(output_dir, "captions_shared.json"), "w") as f:
#     json.dump(caption_dict, f, indent=4)

# print("Saved shared embeddings, images, and captions.")

import os
import torch
import numpy as np
import pandas as pd
import json
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor

# ==== CONFIG ====
output_dir = './data/'
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-mpnet-base-v2').to(device)
print("MPNet model loaded.")

# ==== Load Shared COCO IDs ====
expd_path = '/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat'
stiminfo_path = '/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl'
expd = loadmat(expd_path)
shared = expd['sharedix'].ravel()
stiminfo = pd.read_pickle(stiminfo_path)
shared_df = stiminfo[stiminfo['nsdId'].isin(shared)]
shared_ids = shared_df['cocoId'].values
print(f"Total shared samples: {len(shared_ids)}")

# ==== Load COCO captions ====
coco_val = COCO('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/annotations/captions_val2017.json')
coco_train = COCO('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/annotations/captions_train2017.json')

def get_caption(coco_id):
    for coco in [coco_val, coco_train]:
        if coco_id in coco.getImgIds():
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[coco_id]))
            if anns:
                return anns[0]['caption']
    return None

def download_image(coco_id):
    padded_id = f"{int(coco_id):012d}"
    urls = [
        f"http://images.cocodataset.org/val2017/{padded_id}.jpg",
        f"http://images.cocodataset.org/train2017/{padded_id}.jpg"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB").resize((224, 224))
                return coco_id, img
        except:
            continue
    return coco_id, None

# === Setup streaming memmap storage ===
H, W, C = 224, 224, 3
MAX_SAMPLES = 1000
memmap_path = os.path.join(output_dir, "shared_images_temp.dat")
images_memmap = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(MAX_SAMPLES, H, W, C))

# === Process data ===
embedding_list = []
image_list = []
caption_dict = {}
used_ids = []
idx = 0

print("Downloading and encoding...")
with ThreadPoolExecutor(max_workers=8) as executor:
    for coco_id, image in tqdm(executor.map(download_image, shared_ids), total=len(shared_ids)):
        if image is None:
            continue

        caption = get_caption(coco_id)
        if caption is None:
            continue

        with torch.no_grad():
            embedding = model.encode([caption], convert_to_numpy=True, device=device)[0]

        # Save image
        images_memmap[idx] = np.array(image) / 255.0
        embedding_list.append(embedding)
        used_ids.append(coco_id)
        caption_dict[int(coco_id)] = caption
        idx += 1

        if idx >= MAX_SAMPLES:
            break

images_memmap.flush()

# Final arrays
embedding_array = np.vstack(embedding_list)
coco_id_array = np.array(used_ids)
images_array = images_memmap[:idx]

# Save everything
np.savez_compressed(os.path.join(output_dir, "mpnet_shared_embeddings_and_images.npz"),
                    embeddings=embedding_array,
                    images=images_array,
                    coco_ids=coco_id_array)

with open(os.path.join(output_dir, "captions_shared.json"), "w") as f:
    json.dump(caption_dict, f, indent=4)

os.remove(memmap_path)
print(f"✅ Saved {idx} samples to .npz and captions to .json")
