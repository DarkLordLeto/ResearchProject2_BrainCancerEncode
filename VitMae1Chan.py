
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
import json
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        # Duplicate to 3 channels
        image = Image.merge('RGB', (image, image, image))
        
        if self.transform:
            image = self.transform(image)
        return image

class ViTMAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = ViTMAEForPreTraining(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def mask_and_restore_loss(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.loss
    
    def patch_shuffle_loss(self, pixel_values):
        
        # Get embeddings from the encoder
        encoder_outputs = self.model.vit(
            pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the last hidden state
        embeddings = encoder_outputs.last_hidden_state  # Shape: (B, N, D)
        #print(embeddings.shape)
        B, N, D = embeddings.shape
        
        # The decoder expects exactly 197 patches (14x14 + 1 cls token)
        target_num_patches = 196
        
        # Create random permutation for patch shuffling
        shuffle_indices = torch.randperm(N).to(self.device)
        restore_indices = torch.argsort(shuffle_indices)
        
        # Shuffle embeddings along patch dimension
        shuffled_embeddings = embeddings[:, shuffle_indices]
        
        # Ensure we have exactly 197 patches
        if N > target_num_patches:
            # Truncate to 197 patches
            shuffled_embeddings = shuffled_embeddings[:, :target_num_patches, :]
            restore_indices = restore_indices[:target_num_patches]
        elif N < target_num_patches:
            # Pad with zeros to reach 197 patches
            padding = torch.zeros(B, target_num_patches - N, D).to(self.device)
            shuffled_embeddings = torch.cat([shuffled_embeddings, padding], dim=1)
            padding_indices = torch.arange(N, target_num_patches).to(self.device)
            restore_indices = torch.cat([restore_indices, padding_indices])
        
        # Prepare ids_restore with exact size
        ids_restore = restore_indices.unsqueeze(0).repeat(B, 1)
        
        # Forward through decoder with exact size matching
        decoder_outputs = self.model.decoder(
            hidden_states=shuffled_embeddings[:, :target_num_patches, :],  # Ensure exactly 197 patches
            ids_restore=ids_restore[:, :target_num_patches]  # Ensure exactly 197 indices
        )
        
        # Ensure decoder outputs and original embeddings match in size for loss calculation
        #if N != target_num_patches:
            #decoder_outputs = decoder_outputs[:, :N, :]
        
        # Calculate reconstruction loss
        #print(f"Type of decoder_outputs: {type(decoder_outputs)}")
        #print(f"decoder_outputs: {decoder_outputs}")
        criterion = nn.MSELoss()
        #print(decoder_outputs.logits.shape)
        #print(embeddings.shape)
        reconstruction_loss = criterion(decoder_outputs.logits[:, :50, :], embeddings)
        #reconstruction_loss = criterion(decoder_outputs.logits, embeddings)
        
        return reconstruction_loss
    
    def contrastive_loss(self, embeddings, temperature=0.07):
        # Ensure we're using the correct part of embeddings
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]  # Get the actual tensor
            
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        
        # Calculate similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / temperature
        
        # Prepare labels (diagonal is positive pairs)
        labels = torch.arange(similarity.size(0)).to(self.device)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        return criterion(similarity, labels)
    
    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                pixel_values = batch.to(self.device)
                
                # Calculate all three losses
                mae_loss = self.mask_and_restore_loss(pixel_values)
                shuffle_loss = self.patch_shuffle_loss(pixel_values)
                
                # Get embeddings for contrastive loss
                with torch.no_grad():
                    embeddings = self.model.vit.embeddings(pixel_values)
                #print(f"Type of embeddings: {type(embeddings)}")
                #print(f"embeddings: {embeddings}")
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]  # Get the actual tensor
                contrastive_loss = self.contrastive_loss(embeddings.mean(dim=1))
                
                # Combine losses with weights
                loss = mae_loss + 0.5 * shuffle_loss + 0.1 * contrastive_loss
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def save_model(self, save_dir):
        # Create directory if it doesn't exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        
        # Save config
        self.config.save_pretrained(save_dir)
        
        print(f"Model and config saved to {save_dir}")

class GrayscaleToTriplicate(object):
    """Convert grayscale image to 3 channel by duplicating the grayscale channel"""
    def __call__(self, x):
        return torch.cat([x, x, x], dim=0)

# Define transforms globally
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure images are 224x224
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    GrayscaleToTriplicate(),  # Duplicate channel to create 3 identical channels
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def main():
    # Configuration
    num_epochs=100
    batch_size=32
    
    
    # Configuration
    config = ViTMAEConfig(
        image_size=224,  # This with patch_size=16 gives us 14x14=196 patches + 1 cls token
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        learning_rate=1e-4,
        mask_ratio=0.75,
        decoder_num_attention_heads=12,
        decoder_hidden_size=768,
        decoder_num_hidden_layers=8,
        num_patches=196  # 14x14 patches (cls token is handled separately)
    )
    
    # Dataset and loader setup remains the same
    dataset = CustomImageDataset(
        folder_path='/Users/xinyueliang/Documents/Research/RP2/ResearchProject2_BrainCancerEncode/PretrainData/',
        transform=transform_pipeline
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    trainer = ViTMAETrainer(config)
    trainer.train(train_loader, num_epochs=num_epochs)
    trainer.save_model('/Users/xinyueliang/Documents/Research/RP2/ResearchProject2_BrainCancerEncode/SavedModels/vit_mae_pretrained/')

if __name__ == "__main__":
    main()