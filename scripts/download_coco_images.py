import os
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_coco_image_array(coco_id, splits=("val2017", "train2017"), image_size=(224, 224)):
    """
    Download and return a resized COCO image as a NumPy array, trying multiple splits.

    Returns:
      np.ndarray or None
    """
    padded_id = f"{int(coco_id):012d}"
    for split in splits:
        url = f"http://images.cocodataset.org/{split}/{padded_id}.jpg"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = img.resize(image_size, resample=Image.BICUBIC)
                return np.array(img) / 255.0  # Normalize to [0,1]
        except Exception as e:
            print(f"Error fetching {coco_id} from {split}: {e}")
    return None


def download_and_save_coco_images(coco_ids, output_file="data/coco_images.npz", image_size=(224, 224)):
    """
    Download images by COCO IDs and save all into one compressed .npz file.

    Parameters:
      coco_ids (list): List of COCO image IDs
      output_file (str): Where to save the .npz file
    """
    images = []
    valid_ids = []

    for coco_id in tqdm(coco_ids, desc="Downloading COCO images"):
        img_array = download_coco_image_array(coco_id, image_size=image_size)
        if img_array is not None:
            if img_array.shape != (image_size[1], image_size[0], 3):
                print(f"Skipping image {coco_id} due to incorrect shape {img_array.shape}")
                continue
            images.append(img_array.astype(np.float32))
            valid_ids.append(coco_id)
        else:
            print(f"Skipping image {coco_id} (not found or error).")

    if not images:
        raise RuntimeError("No images were downloaded. Check network or coco_id list.")

    images_np = np.stack(images)
    coco_ids_np = np.array(valid_ids)

    np.savez_compressed(output_file, images=images_np, coco_ids=coco_ids_np)
    print(f"Saved {len(images_np)} images to {output_file}")


if __name__ == "__main__":
    input_file = './data/coco_ids.json'  # Your JSON with COCO IDs

    with open(input_file, 'r') as f:
        coco_ids = json.load(f)

    os.makedirs("data", exist_ok=True)
    download_and_save_coco_images(coco_ids, output_file="data/coco_images_subject01.npz")
