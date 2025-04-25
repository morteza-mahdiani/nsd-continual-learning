#!/usr/bin/env python
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
    def __init__(self, image_ids, embed_dir, images_dir, processor):
        self.ids = image_ids
        self.embed_dir = embed_dir
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco_id = self.ids[idx]
        padded = f"{int(coco_id):012d}.jpg"
        img_path = os.path.join(self.images_dir, padded)
        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)
        emb_path = os.path.join(self.embed_dir, f"{int(coco_id)}.npy")
        target = torch.from_numpy(np.load(emb_path)).float()
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

# 5. Main pipeline
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coco = COCO('data/coco_data/annotations/instances_val2017.json')
    classes = ['person','chair','car','dining table','cup','bottle','bowl']
    cls2id = get_coco_category_ids(coco, classes)

    pairs = list(combinations(classes, 2))
    best1 = max(pairs, key=lambda p: len(build_binary_split(coco, cls2id[p[0]], cls2id[p[1]])))
    used  = set(best1)
    best2 = max([p for p in pairs if p[0] not in used and p[1] not in used],
                key=lambda p: len(build_binary_split(coco, cls2id[p[0]], cls2id[p[1]])))
    print(f"Task1: {best1}, Task2: {best2}")

    expd     = sio.loadmat('data/nsddata/experiments/nsd/nsd_expdesign.mat')
    shared   = expd['sharedix'].ravel()
    stiminfo = pd.read_pickle('data/nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    shared_ids = stiminfo['cocoId'][stiminfo['nsdId'].isin(shared)]

    img_dir   = 'data/coco_data/val2017'
    embed_dir = 'data/mpnet_embeddings'
    available = {int(os.path.splitext(f)[0]) for f in os.listdir(embed_dir) if f.endswith('.npy')}
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # Task1
    tr1 = build_binary_split(coco, cls2id[best1[0]], cls2id[best1[1]], restrict_ids=None)
    tr1 = [cid for cid in tr1 if cid in available]
    va1 = build_binary_split(coco, cls2id[best1[0]], cls2id[best1[1]], restrict_ids=shared_ids)
    va1 = [cid for cid in va1 if cid in available]
    if not va1:
        print("No shared-only for Task1 → using full set for val.")
        va1 = [cid for cid in build_binary_split(coco, cls2id[best1[0]], cls2id[best1[1]]) if cid in available]
    print(f"T1 images: train {len(tr1)}, val {len(va1)}")

    ds_tr1 = CLIPAdapterDataset(tr1, embed_dir, img_dir, processor)
    ds_va1 = CLIPAdapterDataset(va1, embed_dir, img_dir, processor)
    dl_tr1 = DataLoader(ds_tr1, batch_size=32, shuffle=True)
    dl_va1 = DataLoader(ds_va1, batch_size=32)

    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    adapter    = Adapter(input_dim=clip_model.config.projection_dim, output_dim=768)
    model      = CLIPAdapterModel(clip_model, adapter)

    print("\n### Training adapter on Task1 ###")
    model = train_adapter(model, dl_tr1, dl_va1, epochs=10, device=device)

    # Task2
    tr2 = build_binary_split(coco, cls2id[best2[0]], cls2id[best2[1]])
    tr2 = [cid for cid in tr2 if cid in available]
    va2 = build_binary_split(coco, cls2id[best2[0]], cls2id[best2[1]], restrict_ids=shared_ids)
    va2 = [cid for cid in va2 if cid in available]
    if not va2:
        print("No shared-only for Task2 → using full set for val.")
        va2 = [cid for cid in build_binary_split(coco, cls2id[best2[0]], cls2id[best2[1]]) if cid in available]
    print(f"T2 images: train {len(tr2)}, val {len(va2)}")

    ds_tr2 = CLIPAdapterDataset(tr2, embed_dir, img_dir, processor)
    ds_va2 = CLIPAdapterDataset(va2, embed_dir, img_dir, processor)
    dl_tr2 = DataLoader(ds_tr2, batch_size=32, shuffle=True)
    dl_va2 = DataLoader(ds_va2, batch_size=32)

    print("\n### Sequential adapter train on Task2 ###")
    model = train_adapter(model, dl_tr2, dl_va2, epochs=10, device=device)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/clip_adapter_sequential.pth')
    print("\nSaved sequential adapter → models/clip_adapter_sequential.pth")
    
if __name__ == '__main__':
    main()
