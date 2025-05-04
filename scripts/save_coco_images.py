import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# === Paths ===
SAVE_PATH = "./data/coco_images_subject01.npz"
TEMP_MEMMAP_PATH = "./data/coco_images_temp.dat"

# === Load COCO IDs ===
embedding_data = np.load("./data/mpnet_embeddings/mpnet_embeddings_subject01.npz")
coco_ids = embedding_data['coco_ids']
print(f"Total COCO IDs: {len(coco_ids)}")

# === Settings ===
N = len(coco_ids)
H, W, C = 224, 224, 3
MAX_WORKERS = 16

# === Initialize memmap for streaming
images_memmap = np.memmap(TEMP_MEMMAP_PATH, dtype=np.float32, mode="w+", shape=(N, H, W, C))
valid_ids = []

def download_image(coco_id):
    padded_id = f"{int(coco_id):012d}"
    urls = [f"http://images.cocodataset.org/val2017/{padded_id}.jpg",
            f"http://images.cocodataset.org/train2017/{padded_id}.jpg"]
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                return coco_id, img
        except:
            continue
    return coco_id, None

# === Download and write to disk stream
idx = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for coco_id, img in tqdm(executor.map(download_image, coco_ids), total=len(coco_ids)):
        if img is not None:
            try:
                resized_img = np.array(img.resize((H, W))) / 255.0
                images_memmap[idx] = resized_img
                valid_ids.append(coco_id)
                idx += 1
                if idx % 100 == 0:
                    images_memmap.flush()
            except Exception as e:
                print(f"Failed to process image {coco_id}: {e}")

# Final flush and save
images_memmap.flush()
images_np = images_memmap[:idx]  # Truncate to actual size
coco_ids_np = np.array(valid_ids)

np.savez_compressed(SAVE_PATH, images=images_np, coco_ids=coco_ids_np)
os.remove(TEMP_MEMMAP_PATH)  # Clean up temp file if desired

print(f"\nSaved {len(images_np)} images to {SAVE_PATH}")
