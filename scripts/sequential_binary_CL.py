import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pycocotools.coco import COCO
import scipy.io as sio
import pandas as pd
from itertools import combinations
import numpy as np

# 1. COCO utilities
def get_coco_category_ids(coco, class_names):
    cats = coco.loadCats(coco.getCatIds())
    return {c['name']: c['id'] for c in cats if c['name'] in class_names}

def build_binary_split(coco, cat1_id, cat2_id, restrict_ids=None):
    ids1 = set(coco.getImgIds(catIds=[cat1_id]))
    ids2 = set(coco.getImgIds(catIds=[cat2_id]))
    inter = ids1 & ids2
    only1 = ids1 - inter
    only2 = ids2 - inter
    if restrict_ids is not None:
        only1 &= set(restrict_ids)
        only2 &= set(restrict_ids)
    return list(only1) + list(only2)

# 2. Adapter dataset
class CLIPAdapterDataset(Dataset):
    def __init__(self, image_ids, embed_path, image_npz_path, processor):
        self.ids = image_ids
        self.processor = processor

        emb_data = np.load(embed_path)
        self.embeddings = emb_data['embeddings']
        self.emb_coco_ids = emb_data['coco_ids']
        self.emb_id_to_idx = {int(cid): i for i, cid in enumerate(self.emb_coco_ids)}

        img_data = np.load(image_npz_path)
        self.images = img_data['images']              # shape: (N, H, W, 3)
        self.img_coco_ids = img_data['coco_ids']
        self.img_id_to_idx = {int(cid): i for i, cid in enumerate(self.img_coco_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco_id = int(self.ids[idx])
        emb_idx = self.emb_id_to_idx[coco_id]
        img_idx = self.img_id_to_idx[coco_id]

        image = self.images[img_idx]  # (H, W, 3)
        image = Image.fromarray((image * 255).astype(np.uint8))

        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)
        target = torch.tensor(self.embeddings[emb_idx], dtype=torch.float32)
        return pixel_values, target

# 3. Adapter module and composite model
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

class CLIPAdapterModel(nn.Module):
    def __init__(self, clip_model, adapter):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad = False
        self.adapter = adapter

    def forward(self, pixel_values):
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        return self.adapter(image_embeds)

# 4. Training function for adapter
def train_adapter(model, train_loader, val_loader, epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.adapter.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} — Train MSE: {avg_train:.4f}, Val MSE: {avg_val:.4f}")
    return model

def run_task_training(task_id, class_pair, shared_ids, available, coco, embed_path, image_npz_path, processor, model, device):
    tr_ids = build_binary_split(coco, class_pair[0], class_pair[1])
    tr_ids = [cid for cid in tr_ids if cid in available]
    
    va_ids = build_binary_split(coco, class_pair[0], class_pair[1], restrict_ids=shared_ids)
    va_ids = [cid for cid in va_ids if cid in available]
    
    if not va_ids:
        print(f"No shared-only for Task{task_id} → using full set for val.")
        va_ids = [cid for cid in build_binary_split(coco, class_pair[0], class_pair[1]) if cid in available]

    print(f"T{task_id} images: train {len(tr_ids)}, val {len(va_ids)}")

    ds_tr = CLIPAdapterDataset(tr_ids, embed_path, image_npz_path, processor)
    ds_va = CLIPAdapterDataset(va_ids, embed_path, image_npz_path, processor)
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=32)

    print(f"\n### Training adapter on Task{task_id} ({class_pair}) ###")
    model = train_adapter(model, dl_tr, dl_va, epochs=10, device=device)

    save_path = f'models/clip_adapter_sequential_task{task_id}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Saved adapter for Task{task_id} → {save_path}")

    return model

# 5. Main pipeline
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load annotation and stimulus info
    coco = COCO('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/annotations/instances_val2017.json')
    # classes = ['person', 'chair', 'car', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'bench', 'book', 
    # 'backpack', 'sink', 'clock', 'dog', 'sports ball', 'cat', 'potted plant', 'cell phone', 'surfboard']
    classes = [
    "person", "chair", "car", "dining table", "cup", "bottle", "bowl", "handbag",
    "truck", "bench", "book", "backpack", "sink", "clock", "dog", "sports ball",
    "cat", "potted plant", "cell phone", "surfboard", "skis", "knife", "tie", "bus",
    "traffic light", "tv", "bed", "umbrella", "train", "toilet", "tennis racket",
    "couch", "spoon", "bird", "skateboard", "airplane", "boat", "motorcycle", "vase",
    "bicycle", "pizza", "fork", "oven", "giraffe", "laptop", "baseball glove", "cake",
    "horse", "banana", "remote", "elephant", "baseball bat", "kite", "wine glass",
    "frisbee", "suitcase", "refrigerator", "cow", "sandwich", "teddy bear", "stop sign",
    "zebra", "broccoli", "snowboard", "microwave", "fire hydrant", "keyboard", "donut",
    "sheep", "apple", "carrot", "mouse", "orange", "bear", "hot dog", "toothbrush",
    "scissors", "parking meter", "hair drier", "toaster"]
    cls2id = get_coco_category_ids(coco, classes)
    id2cls = {v: k for k, v in cls2id.items()}


    expd     = sio.loadmat('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat')
    shared   = expd['sharedix'].ravel()
    stiminfo = pd.read_pickle('/home/mahdiani/projects/def-charesti/data/natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    shared_ids = stiminfo['cocoId'][stiminfo['nsdId'].isin(shared)]

    embed_path = '/home/mahdiani/projects/def-charesti/mahdiani/data/mpnet_embeddings/mpnet_embeddings_subject01.npz'
    image_npz_path = '/home/mahdiani/projects/def-charesti/mahdiani/data/coco_images_subject01.npz'
    embed_data = np.load(embed_path)
    available = set(embed_data['coco_ids'])

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    adapter = Adapter(input_dim=clip_model.config.projection_dim, output_dim=768)
    model = CLIPAdapterModel(clip_model, adapter)

    # Generate class pairs
    pairs = list(combinations(classes, 2))
    selected_pairs = []
    used_classes = set()
    for p in sorted(pairs, key=lambda p: len(build_binary_split(coco, cls2id[p[0]], cls2id[p[1]])), reverse=True):
        if p[0] not in used_classes and p[1] not in used_classes:
            selected_pairs.append((cls2id[p[0]], cls2id[p[1]]))
            used_classes.update(p)
        if len(selected_pairs) >= 11:  # Change 4 to however many tasks you want
            break

    print(f"Selected tasks: {[ (id2cls[p[0]], id2cls[p[1]]) for p in selected_pairs ]}")

    # Sequential training over tasks
    for i, class_pair in enumerate(selected_pairs, 1):
        model = run_task_training(i, class_pair, shared_ids, available, coco, embed_path, image_npz_path, processor, model, device)


if __name__ == '__main__':
    main()
