#!/usr/bin/env python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.tensorboard import SummaryWriter

##############################################
# 1. Dataset for Images and Caption Embeddings
##############################################
# This dataset expects that the images are stored in one folder
# and that corresponding precomputed caption embeddings from MPNet in our case) are saved as .npy files
# in another folder. Filenames (without extension) are assumed to match.
class CLIPAdapterDataset(Dataset):
    def __init__(self, images_dir, embeddings_dir, transform=None):
        """
        Args:
            images_dir (str): Directory containing COCO images (with 12-digit filenames, e.g. 0000000139.jpg).
            embeddings_dir (str): Directory containing precomputed caption embeddings (.npy files).
            transform (callable, optional): Transformations to apply to images.
        """
        self.images_dir = images_dir
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        
        # List all .npy files in the embeddings directory
        all_embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
        
        # We'll store only those file IDs for which the corresponding .jpg file exists
        self.ids = []
        for f in all_embedding_files:
            base_id = os.path.splitext(f)[0]  # e.g., '462750'
            
            # Convert to int and zero-pad to 12 digits
            try:
                padded_id = f"{int(base_id):012d}"  # e.g., '000462750'
            except ValueError:
                # If the base_id is not purely numeric, skip or handle error
                print(f"Warning: {base_id} is not a numeric ID. Skipping.")
                continue
            
            img_filename = padded_id + ".jpg"
            img_path = os.path.join(images_dir, img_filename)
            
            if os.path.exists(img_path):
                self.ids.append(base_id)  # We'll store the original base_id for lookups
            else:
                print(f"Warning: Image file {img_path} not found. Skipping.")

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        # Grab the ID from the stored list
        base_id = self.ids[idx]  # e.g., '462750'
        
        # Convert to int and zero-pad
        padded_id = f"{int(base_id):012d}"
        
        # Construct image path
        img_path = os.path.join(self.images_dir, padded_id + ".jpg")
        
        # Load and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load the corresponding caption embedding
        emb_path = os.path.join(self.embeddings_dir, base_id + ".npy")
        caption_embed = np.load(emb_path)
        caption_embed = torch.tensor(caption_embed, dtype=torch.float)
        
        return image, caption_embed


##############################################
# 2. Define the Adapter Module
##############################################
# This module is responsible for transforming the CLIP image embeddings to the target space.
# For example, if the CLIP image embeddings are 512-dimensional and the MPNet (caption) embeddings
# are 768-dimensional, the adapter should map 512 -> 768.

class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.adapter(x)

##############################################
# 3. Composite Model: Frozen CLIP + Trainable Adapter
##############################################
##############################################
# 3. Composite Model: Frozen CLIP + Trainable Adapter
##############################################
class CLIPAdapterModel(nn.Module):
    def __init__(self, clip_model, adapter):
        """
        Args:
            clip_model (CLIPModel): A pretrained CLIP model (its weights are frozen).
            adapter (nn.Module): A trainable adapter that maps CLIP image embeddings to the target space.
        """
        super(CLIPAdapterModel, self).__init__()
        self.clip_model = clip_model  # The CLIP model, whose parameters will be frozen.
        self.adapter = adapter

    def forward(self, pixel_values):
        # We use get_image_features to get only image embeddings.
        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
        # Pass image embeddings through the adapter.
        adapted_embeds = self.adapter(image_embeds)
        return adapted_embeds

##############################################
# 4. Loss Functions
##############################################
# We use Mean Squared Error (MSE) loss to align the adapted image embeddings with the precomputed caption embeddings.
mse_loss = nn.MSELoss()

##############################################
# 5. Main Training Function
##############################################
def main():
    # Device: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories: update these paths as needed.
    images_dir = './data/coco_data/val2017'
    embeddings_dir = './data/mpnet_embeddings'
    
    # Instantiate a CLIPProcessor for image preprocessing.
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Define image transforms using the CLIP processor's expected values.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_processor.image_processor.image_mean, std=clip_processor.image_processor.image_std)
    ])

    # Create the dataset.
    dataset = CLIPAdapterDataset(images_dir, embeddings_dir, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # Split into training and validation sets.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load the pretrained CLIP model.
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Freeze all parameters of CLIP.
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model = clip_model.to(device)
    
    # Determine CLIP image embedding dimension.
    # For "openai/clip-vit-base-patch32", the image embeddings are 512-dimensional (pretty sure this is right).
    clip_dim = 512
    # Target caption embedding dimension for MPNet is typically 768.
    target_dim = 768
    
    # Instantiate the adapter.
    adapter = Adapter(input_dim=clip_dim, output_dim=target_dim).to(device)
    
    # Create the composite model.
    model = CLIPAdapterModel(clip_model, adapter).to(device)
    
    # Define the optimizer (only adapter parameters will be updated).
    optimizer = optim.Adam(model.adapter.parameters(), lr=1e-4)
    
    # Set up TensorBoard logging.
    writer = SummaryWriter('runs/clip_adapter_training')
    
    num_epochs = 10
    global_step = 0
    
    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, caption_embeds) in enumerate(train_loader):
            images = images.to(device)
            caption_embeds = caption_embeds.to(device)
            
            optimizer.zero_grad()
            # Forward pass: get adapted embeddings.
            outputs = model(images)
            loss = mse_loss(outputs, caption_embeds)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1
            
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, caption_embeds in val_loader:
                images = images.to(device)
                caption_embeds = caption_embeds.to(device)
                outputs = model(images)
                loss = mse_loss(outputs, caption_embeds)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
        writer.add_scalar('Validation/EpochLoss', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    # Save the adapter (or the whole model) for later use. I left it open for what we decide to go with.
    save_dir = './data/trained_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'clip_adapter_finetuned.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    writer.close()

if __name__ == '__main__':
    main()
