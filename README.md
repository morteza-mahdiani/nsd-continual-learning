# NSD Continual Learning

This repository contains code for investigating how well continual learning in vision-language models aligns with the human brain, using the Natural Scenes Dataset (NSD) and COCO dataset.

## Project Overview

The project aims to:
- Compare continual learning approaches in vision-language models with human brain responses
- Analyze how models adapt to new visual concepts over time
- Study the alignment between model representations and neural responses
- Visualize and analyze representational similarity between models and brain data

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv nsd_continual
source nsd_continual/bin/activate  # On Windows, use `nsd_continual\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Download

### NSD Beta Data
To download NSD beta data:
```bash
chmod +x scripts/download_nsd_betas.sh
./scripts/download_nsd_betas.sh
```

For all subjects:
```bash
chmod +x scripts/download_nsd_betas_all_subjects.sh
./scripts/download_nsd_betas_all_subjects.sh
```

### COCO Images
The repository includes scripts to download and process COCO images:
- `scripts/download_coco_images.py`: Downloads COCO images
- `scripts/save_coco_images.py`: Saves COCO images in a specific format
- `scripts/coco_preprocess_mpnet.py`: Preprocesses COCO images using MPNet
- `scripts/extract_coco_ids_from_captions_json.py`: Extracts COCO IDs from captions
- `scripts/shared_images_mpnet_embeddings.py`: Processes shared images with MPNet embeddings

## Scripts

The repository contains several key scripts:

### Training Scripts
- `training_CLIP_adapter.py`: Implements CLIP adapter training
- `training_AlexNet_naive.py`: Implements naive AlexNet training
- `sequential_binary_CL.py`: Implements sequential binary continual learning

### Data Processing
- `preprocessing_betas.py`: Preprocesses NSD beta data
- `nsd_get_data_light.py`: Lightweight NSD data access
- `coco_preprocess_mpnet.py`: Preprocesses COCO images with MPNet

### Analysis and Visualization
- `searchlight_rsa_visualization.py`: Implements searchlight RSA analysis and visualization
- Various visualization scripts in the `visualizations/` directory for:
  - Mean Squared Error (MSE) analysis across tasks
  - Representational Similarity Analysis (RSA) by region
  - Task-specific analyses (8 and 11 tasks)

## Dependencies

Key dependencies include:
- numpy
- pandas
- boto3
- nibabel
- scipy
- pillow
- tqdm
- matplotlib
- pycocotools
- nsd_access


## Citation

If you use this code in your research, please cite:
[TBD]