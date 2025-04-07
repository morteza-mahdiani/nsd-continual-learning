import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter

# Custom Dataset for COCO images and MPNet embeddings
class COCOEmbeddingsDataset(Dataset):
    def __init__(self, images_dir, embeddings_dir, transform=None):
        """
        Args:
            images_dir (str): Directory containing COCO images.
            embeddings_dir (str): Directory with .npy files; filenames are the COCO ids.
            transform (callable, optional): Transformations to apply on an image.
        """
        self.images_dir = images_dir
        self.embeddings_dir = embeddings_dir
        self.transform = transform
        # List of .npy embedding files
        self.embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
        # Extract image ids (filenames without extension)
        self.ids = [os.path.splitext(f)[0] for f in self.embedding_files]

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # Assume corresponding image filename is <img_id>.jpeg; adjust if needed.
        img_path = os.path.join(self.images_dir, img_id + '.jpeg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load corresponding MPNet embedding
        embedding_path = os.path.join(self.embeddings_dir, img_id + '.npy')
        embedding = np.load(embedding_path)
        embedding = torch.tensor(embedding, dtype=torch.float)
        return image, embedding

# Define image transformations for AlexNet (224x224 and normalization as per ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Directories (adjust these paths as needed)
images_dir = './data/coco_data/coco_images_subj01/downloaded_images'  # Replace with your actual images directory
embeddings_dir = './data/mpnet_embeddings'

# Create the dataset
dataset = COCOEmbeddingsDataset(images_dir, embeddings_dir, transform=transform)

# Split dataset into 80% training and 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load a pretrained AlexNet and modify its classifier to output the desired embedding dimension.
# (Assuming MPNet embeddings are 768-d; adjust if different)
embedding_dim = 768
alexnet = models.alexnet(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('./data/models/alexnet/alexnet_pretrained.pth', map_location=device)
alexnet.load_state_dict(state_dict)

# --- Add Projection Head ---
# Create a projection head to help regularize training
input_features = alexnet.classifier[-1].in_features
projection_head = nn.Sequential(
    nn.Linear(input_features, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(1024, embedding_dim)
)
# Replace the final classifier layer with our projection head
alexnet.classifier[-1] = projection_head

# Freeze initial layers (feature extractor) to avoid overfitting on the small dataset.
for param in alexnet.features.parameters():
    param.requires_grad = False

# Set device and move model to GPU if available
alexnet = alexnet.to(device)

# Define individual loss functions: cosine loss and MSE loss.
def cosine_loss(pred, target):
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    return 1 - cos_sim.mean()

mse_loss = nn.MSELoss()

# Combined loss: adjust lambda values as needed.
def combined_loss(pred, target, lambda_cosine=0.5, lambda_mse=0.5):
    loss_cos = cosine_loss(pred, target)
    loss_mse = mse_loss(pred, target)
    return lambda_cosine * loss_cos + lambda_mse * loss_mse

# Use Adam optimizer; note we filter to update only parameters that require gradients.
optimizer = optim.Adam(filter(lambda p: p.requires_grad, alexnet.parameters()), lr=1e-4)

# TensorBoard SummaryWriter for logging training and validation metrics
writer = SummaryWriter('runs/alexnet_finetune_combined')

# Training loop
num_epochs = 9
global_step = 0

for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0
    for i, (images, embeddings) in enumerate(train_loader):
        images = images.to(device)
        embeddings = embeddings.to(device)
        
        optimizer.zero_grad()
        outputs = alexnet(images)
        loss = combined_loss(outputs, embeddings, lambda_cosine=0.5, lambda_mse=0.5)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        global_step += 1
        
        # Log batch loss
        if i % 10 == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation phase
    alexnet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, embeddings in val_loader:
            images = images.to(device)
            embeddings = embeddings.to(device)
            outputs = alexnet(images)
            loss = combined_loss(outputs, embeddings, lambda_cosine=0.5, lambda_mse=0.5)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    # Log epoch-level metrics
    writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
    writer.add_scalar('Validation/EpochLoss', avg_val_loss, epoch)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save the fine-tuned model
torch.save(alexnet.state_dict(), './data/trained_models/alexnet_finetuned_combined.pth')
writer.close()
