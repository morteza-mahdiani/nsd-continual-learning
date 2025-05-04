# # searchlight_rsa_visualization.py

# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from transformers import CLIPModel, CLIPProcessor
# from PIL import Image
# from nilearn import surface, plotting, datasets
# from torch import nn
# from torch.utils.data import Dataset
# from sklearn.metrics import mean_squared_error

# # === Dataset Loader ===
# class CLIPAdapterDataset(Dataset):
#     def __init__(self, embed_image_npz_path, processor):
#         print(f"Loading dataset from {embed_image_npz_path}")
#         data = np.load(embed_image_npz_path)
#         self.ids = data['coco_ids']
#         self.embeddings = data['embeddings']
#         self.images = data['images']
#         self.processor = processor
#         self.id_to_idx = {int(cid): i for i, cid in enumerate(self.ids)}
#         print(f"Loaded {len(self.ids)} items.")

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         coco_id = int(self.ids[idx])
#         img_idx = self.id_to_idx[coco_id]

#         image = self.images[img_idx]  # (H, W, 3)
#         image = Image.fromarray((image * 255).astype(np.uint8))

#         inputs = self.processor(images=image, return_tensors='pt')
#         pixel_values = inputs['pixel_values'].squeeze(0)
#         target = torch.tensor(self.embeddings[img_idx], dtype=torch.float32)
#         return pixel_values, target

# # === Adapter Module ===
# class Adapter(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.adapter = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )

#     def forward(self, x):
#         return self.adapter(x)

# # === Composite Model ===
# class CLIPAdapterModel(nn.Module):
#     def __init__(self, clip_model, adapter):
#         super().__init__()
#         self.clip = clip_model
#         for p in self.clip.parameters():
#             p.requires_grad = False
#         self.adapter = adapter

#     def forward(self, pixel_values):
#         image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
#         return self.adapter(image_embeds)

# # === Predict MPNet Embeddings ===
# def predict_embeddings(model_path, clip_model, processor, dataset, device='cuda'):
#     print(f"Loading model from {model_path}")
#     model = CLIPAdapterModel(clip_model, Adapter(clip_model.config.projection_dim, 768)).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     print("Model loaded and set to eval mode.")

#     dataloader = DataLoader(dataset, batch_size=32)
#     predictions = []
#     targets = []

#     print("Starting prediction loop...")
#     with torch.no_grad():
#         for i, (pixel_values, target) in enumerate(dataloader):
#             print(f"Processing batch {i+1}")
#             pixel_values = pixel_values.to(device)
#             pred = model(pixel_values).cpu().numpy()
#             predictions.append(pred)
#             targets.append(target.numpy())

#     pred_array = np.vstack(predictions)
#     target_array = np.vstack(targets)
#     mse = mean_squared_error(target_array, pred_array)
#     print(f"MSE between predicted and actual MPNet embeddings: {mse:.6f}")
#     return pred_array

# # === Searchlight RSA ===
# def searchlight_rsa(embedding_matrix, brain_matrix):
#     from sklearn.metrics.pairwise import cosine_similarity
#     print("Running full-region RSA...")

#     if embedding_matrix.shape[0] != brain_matrix.shape[0]:
#         raise ValueError(f"Mismatch in number of stimuli: {embedding_matrix.shape[0]} (embeddings) vs {brain_matrix.shape[0]} (brain)")

#     embed_rdm = 1 - cosine_similarity(embedding_matrix)
#     brain_rdm = 1 - cosine_similarity(brain_matrix)

#     if embed_rdm.shape != brain_rdm.shape:
#         raise ValueError(f"RDM shape mismatch: {embed_rdm.shape} vs {brain_rdm.shape}")

#     corr = np.corrcoef(embed_rdm.ravel(), brain_rdm.ravel())[0, 1]
#     print(f"RSA correlation: {corr:.4f}")
#     return np.array([corr])

# # === Multi-region Grid Visualization ===
# def visualize_all_regions(task_id=1, output_path="rsa_results/summary_task1.png"):
#     print("Generating summary bar plot of RSA scores...")
#     regions = {
#         'early': 1,
#         'midventral': 2,
#         'midlateral': 3,
#         'ventral': 5,
#         'lateral': 6
#     }

#     rsa_values = []
#     region_names = []

#     for region, region_id in regions.items():
#         rsa_scores = np.load(f"rsa_results/task{task_id}_region{region_id}_rsa.npz")['rsa']
#         rsa_values.append(rsa_scores[0])
#         region_names.append(region)

#     plt.figure(figsize=(8, 5))
#     plt.bar(region_names, rsa_values, color='skyblue')
#     plt.ylabel("RSA Correlation")
#     plt.title(f"RSA Scores by Region (Task {task_id})")
#     plt.ylim(0, max(rsa_values) + 0.05)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Saved RSA summary plot to {output_path}")
#     print(f"Saved visualization to {output_path}")

# # === Example Usage ===
# if __name__ == "__main__":
#     print("Initializing CLIP model and processor...")
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     dataset = CLIPAdapterDataset(
#         embed_image_npz_path="data/mpnet_shared_embeddings_and_images.npz",
#         processor=processor
#     )

#     pred_embeddings = predict_embeddings(
#         model_path="models/clip_adapter_sequential_task1.pth",
#         clip_model=clip_model,
#         processor=processor,
#         dataset=dataset
#     )

#     os.makedirs("rsa_results", exist_ok=True)

#     for region_id in [1, 2, 3, 5, 6]:
#         print(f"Loading betas for region {region_id}")
#         region_npz = np.load(f"/home/mahdiani/betas/region_{region_id}/region_{region_id}_shared.npz")
#         region_coco_ids = region_npz['coco_ids']  # shape (1000,)
#         region_betas = region_npz['masked_betas']  # shape (V, 1000)

#         # Convert betas to shape (1000, V) by transposing
#         brain_data = region_betas.T
#         rsa_scores = searchlight_rsa(pred_embeddings, brain_data)
#         np.savez(f"rsa_results/task1_region{region_id}_rsa.npz", rsa=rsa_scores)
#         print(f"Saved RSA results for region {region_id}")

#     visualize_all_regions(task_id=1)


# searchlight_rsa_visualization.py

# searchlight_rsa_visualization.py
# searchlight_rsa_visualization.py

# searchlight_rsa_visualization.py

# searchlight_rsa_visualization.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from nilearn import surface, plotting, datasets
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as mtick

# === Dataset Loader ===
class CLIPAdapterDataset(Dataset):
    def __init__(self, embed_image_npz_path, processor):
        print(f"Loading dataset from {embed_image_npz_path}")
        data = np.load(embed_image_npz_path)
        self.ids = data['coco_ids']
        self.embeddings = data['embeddings']
        self.images = data['images']
        self.processor = processor
        self.id_to_idx = {int(cid): i for i, cid in enumerate(self.ids)}
        print(f"Loaded {len(self.ids)} items.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco_id = int(self.ids[idx])
        img_idx = self.id_to_idx[coco_id]

        image = self.images[img_idx]  # (H, W, 3)
        image = Image.fromarray((image * 255).astype(np.uint8))

        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)
        target = torch.tensor(self.embeddings[img_idx], dtype=torch.float32)
        return pixel_values, target

# === Adapter Module ===
class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.adapter(x)

# === Composite Model ===
class CLIPAdapterModel(nn.Module):
    def __init__(self, clip_model, adapter):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.adapter = adapter

    def forward(self, pixel_values):
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        return self.adapter(image_embeds)

# === Predict MPNet Embeddings ===
def predict_embeddings(model_path, clip_model, processor, dataset, device='cuda'):
    print(f"Loading model from {model_path}")
    model = CLIPAdapterModel(clip_model, Adapter(clip_model.config.projection_dim, 768)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded and set to eval mode.")

    dataloader = DataLoader(dataset, batch_size=32)
    predictions = []
    targets = []

    print("Starting prediction loop...")
    with torch.no_grad():
        for i, (pixel_values, target) in enumerate(dataloader):
            print(f"Processing batch {i+1}")
            pixel_values = pixel_values.to(device)
            pred = model(pixel_values).cpu().numpy()
            predictions.append(pred)
            targets.append(target.numpy())

    pred_array = np.vstack(predictions)
    target_array = np.vstack(targets)
    mse = mean_squared_error(target_array, pred_array)
    print(f"MSE between predicted and actual MPNet embeddings: {mse:.6f}")
    np.savez("rsa_results/task1_mse.npz", mse=mse)
    return pred_array

# === Searchlight RSA ===
def searchlight_rsa(embedding_matrix, brain_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    print("Running full-region RSA...")

    if embedding_matrix.shape[0] != brain_matrix.shape[0]:
        raise ValueError(f"Mismatch in number of stimuli: {embedding_matrix.shape[0]} (embeddings) vs {brain_matrix.shape[0]} (brain)")

    embed_rdm = 1 - cosine_similarity(embedding_matrix)
    brain_rdm = 1 - cosine_similarity(brain_matrix)

    if embed_rdm.shape != brain_rdm.shape:
        raise ValueError(f"RDM shape mismatch: {embed_rdm.shape} vs {brain_rdm.shape}")

    corr = np.corrcoef(embed_rdm.ravel(), brain_rdm.ravel())[0, 1]
    print(f"RSA correlation: {corr:.4f}")
    return np.array([corr])

# === Multi-region Grid Visualization ===
# def visualize_all_regions(task_id=1, output_path="rsa_results/summary_task1.png", mse_path="rsa_results/task1_mse.npz"):
#     print("Generating summary bar plot of RSA scores...")
#     regions = {
#         'early': 1,
#         'midventral': 2,
#         'midlateral': 3,
#         'ventral': 5,
#         'lateral': 6
#     }

#     fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

#     rsa_values = []
#     region_names = []

#     for region, region_id in regions.items():
#         rsa_scores = np.load(os.path.join(os.path.dirname(output_path), f"region{region_id}_rsa.npz"))['rsa']
#         rsa_values.append(np.mean(rsa_scores))
#         region_names.append(region)

        

#     # Plot RSA bar chart summary
#     plt.figure(figsize=(8, 5))
#     plt.bar(region_names, rsa_values, color='skyblue')
#     plt.ylabel("Mean RSA Correlation")
#     plt.title(f"Mean RSA Scores by Region (Task {task_id})")
#     plt.ylim(0, max(rsa_values) + 0.05)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Saved RSA summary plot to {output_path}")

#     # Plot MSE
#     mse_data = np.load(mse_path)['mse']
#     plt.figure(figsize=(5, 4))
#     plt.bar([f"Task {task_id}"], [mse_data], color='orange')
#     plt.ylabel("MSE Loss")
#     plt.title("MSE between Predicted and Actual Embeddings")
#     plt.tight_layout()
#     plt.savefig(f"rsa_results/task{task_id}_mse.png")
#     plt.close()
#     print(f"Saved MSE plot to rsa_results/task{task_id}_mse.png")
#     plt.close()
#     print(f"Saved summary bar plot to {output_path}")
#     print(f"Saved RSA summary plot to {output_path}")
#     print(f"Saved visualization to {output_path}")
def visualize_all_regions(task_id=1, output_path="rsa_results/summary_task1.png", mse_path="rsa_results/task1_mse.npz"):
    print("Generating summary bar plot of RSA scores...")

    regions = {
        'Early': 1,
        'Midventral': 2,
        'Midlateral': 3,
        'Ventral': 5,
        'Lateral': 6
    }

    # Optional: fetch fsaverage just to ensure it's downloaded
    datasets.fetch_surf_fsaverage('fsaverage')

    rsa_values = []
    region_names = []

    for region, region_id in regions.items():
        rsa_file = os.path.join(os.path.dirname(output_path), f"region{region_id}_rsa.npz")
        rsa_scores = np.load(rsa_file)['rsa']
        rsa_values.append(np.mean(rsa_scores))
        region_names.append(region)

    # Plot RSA bar chart summary
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(region_names, rsa_values, color='#5DADE2', edgecolor='black')

    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # Offset text above bar
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='medium')

    ax.set_ylabel("Mean RSA Correlation", fontsize=12)
    ax.set_title(f"Mean RSA Scores by Brain Region (Task {task_id})", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rsa_values) + 0.1)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelrotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved RSA summary plot to {output_path}")

    # Plot MSE
    mse_data = np.load(mse_path)['mse']
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=150)
    bar = ax.bar([f"Task {task_id}"], [mse_data], color='#F5B041', edgecolor='black')

    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("MSE: Predicted vs Actual Embeddings", fontsize=13, fontweight='bold')
    ax.set_ylim(0, mse_data * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for b in bar:
        height = b.get_height()
        ax.annotate(f'{height:.3f}', xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='medium')

    plt.tight_layout()
    mse_out_path = f"rsa_results/task{task_id}_mse.png"
    plt.savefig(mse_out_path, dpi=300)
    plt.close()
    print(f"Saved MSE plot to {mse_out_path}")


# === Example Usage ===
if __name__ == "__main__":
    print("Initializing CLIP model and processor...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = CLIPAdapterDataset(
        embed_image_npz_path="data/mpnet_shared_embeddings_and_images.npz",
        processor=processor
    )

    os.makedirs("rsa_results", exist_ok=True)

    model_dir = "models"
    model_paths = sorted([f for f in os.listdir(model_dir) if f.startswith("clip_adapter") and f.endswith(".pth")])

    for model_file in model_paths:
        model_path = os.path.join(model_dir, model_file)
        task_name = model_file.replace(".pth", "")
        result_dir = os.path.join("rsa_results", task_name)
        os.makedirs(result_dir, exist_ok=True)

        print(f"Processing {task_name}...")
        pred_embeddings = predict_embeddings(
            model_path=model_path,
            clip_model=clip_model,
            processor=processor,
            dataset=dataset
        )

        for region_id in [1, 2, 3, 5, 6]:
            print(f"Loading betas for region {region_id}")
            region_npz = np.load(f"/home/mahdiani/betas/region_{region_id}/region_{region_id}_shared.npz")
            region_coco_ids = region_npz['coco_ids']  # shape (1000,)
            region_betas = region_npz['masked_betas']  # shape (V, 1000)

            # Convert betas to shape (1000, V) by transposing
            brain_data = region_betas.T
            rsa_scores = searchlight_rsa(pred_embeddings, brain_data)
            np.savez(f"{result_dir}/region{region_id}_rsa.npz", rsa=rsa_scores)
            print(f"Saved RSA results for region {region_id} in {result_dir}")

        # Save MSE path and summary plot
        os.rename("rsa_results/task1_mse.npz", f"{result_dir}/mse.npz")
        visualize_all_regions(task_id=task_name.split("_task")[-1],
                              output_path=f"{result_dir}/summary.png",
                              mse_path=f"{result_dir}/mse.npz")


