# import os
# import json
# import requests


# def download_coco_image(coco_id, dest_folder="coco_images", splits=("val2017", "train2017")):
#     """
#     Attempt to download a COCO image by trying multiple splits.

#     Parameters:
#       coco_id (int): The COCO image ID.
#       dest_folder (str): The folder where the image will be saved.
#       splits (tuple): Tuple of split names to try (e.g. ("val2017", "train2017")).
#     """
#     os.makedirs(dest_folder, exist_ok=True)
#     padded_id = f"{int(coco_id):012d}"

#     for split in splits:
#         url = f"http://images.cocodataset.org/{split}/{padded_id}.jpg"
#         try:
#             response = requests.get(url, timeout=10)
#             if response.status_code == 200:
#                 save_path = os.path.join(dest_folder, f"{coco_id}.jpeg")
#                 with open(save_path, "wb") as f:
#                     f.write(response.content)
#                 print(f"Downloaded image {coco_id} from {split} to {save_path}")
#                 return  # Exit once the image is successfully downloaded
#             else:
#                 print(f"Image {coco_id} not found in {split} (HTTP {response.status_code}). Trying next split.")
#         except Exception as e:
#             print(f"Error downloading image {coco_id} from {split}: {e}")

#     print(f"Image {coco_id} could not be downloaded from any split.")


# def download_coco_images(coco_ids, dest_folder="coco_images"):
#     """
#     Download multiple COCO images given a list of image IDs.

#     Parameters:
#       coco_ids (list): List of COCO image IDs.
#       dest_folder (str): Folder where images will be saved.
#     """
#     for coco_id in coco_ids:
#         download_coco_image(coco_id, dest_folder)


# # Example usage:
# if __name__ == "__main__":
#     # Replace with your actual list of COCO image IDs
#     # Load your JSON file
#     input_file = 'data/coco_meta/coco_ids.json'  # change to your actual file name

#     # Read the JSON content
#     with open(input_file, 'r') as f:
#         coco_ids = json.load(f)
#     # print(len(coco_ids))
#     download_coco_images(coco_ids, dest_folder="data/downloaded_images")


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
