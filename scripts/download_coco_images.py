import os
import json
import requests


def download_coco_image(coco_id, dest_folder="coco_images", splits=("val2017", "train2017")):
    """
    Attempt to download a COCO image by trying multiple splits.

    Parameters:
      coco_id (int): The COCO image ID.
      dest_folder (str): The folder where the image will be saved.
      splits (tuple): Tuple of split names to try (e.g. ("val2017", "train2017")).
    """
    os.makedirs(dest_folder, exist_ok=True)
    padded_id = f"{int(coco_id):012d}"

    for split in splits:
        url = f"http://images.cocodataset.org/{split}/{padded_id}.jpg"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                save_path = os.path.join(dest_folder, f"{coco_id}.jpeg")
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded image {coco_id} from {split} to {save_path}")
                return  # Exit once the image is successfully downloaded
            else:
                print(f"Image {coco_id} not found in {split} (HTTP {response.status_code}). Trying next split.")
        except Exception as e:
            print(f"Error downloading image {coco_id} from {split}: {e}")

    print(f"Image {coco_id} could not be downloaded from any split.")


def download_coco_images(coco_ids, dest_folder="coco_images"):
    """
    Download multiple COCO images given a list of image IDs.

    Parameters:
      coco_ids (list): List of COCO image IDs.
      dest_folder (str): Folder where images will be saved.
    """
    for coco_id in coco_ids:
        download_coco_image(coco_id, dest_folder)


# Example usage:
if __name__ == "__main__":
    # Replace with your actual list of COCO image IDs
    # Load your JSON file
    input_file = 'data/coco_meta/coco_ids.json'  # change to your actual file name

    # Read the JSON content
    with open(input_file, 'r') as f:
        coco_ids = json.load(f)
    # print(len(coco_ids))
    download_coco_images(coco_ids, dest_folder="data/downloaded_images")
