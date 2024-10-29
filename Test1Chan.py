import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTMAEForPreTraining
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def process_single_image(image_path, model_path):
    # Load and set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model
    model = ViTMAEForPreTraining.from_pretrained(model_path).to(device)
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),  # Duplicate channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load and transform image
    image = Image.open(image_path)
    transformed_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(pixel_values=transformed_image)
        
        # Get the reconstructed image from the decoder
        reconstructed = outputs.logits  # Shape: [1, num_patches, patch_dim]
        
        # Convert the patch-wise output back to image
        patch_size = 16  # As defined in the config
        num_patches = int(np.sqrt(reconstructed.shape[1]))  # Should be 14 for 224x224 input
        hidden_size = reconstructed.shape[-1]
        
        # Reshape to [batch, height, width, channels]
        reconstructed = reconstructed.reshape(
            1, 
            num_patches, 
            num_patches, 
            patch_size, 
            patch_size, 
            3
        )
        
        # Rearrange dimensions
        reconstructed = reconstructed.permute(0, 1, 3, 2, 4, 5)
        reconstructed = reconstructed.reshape(
            1,
            num_patches * patch_size,
            num_patches * patch_size,
            3
        )
        
        # Convert to numpy and denormalize
        reconstructed = reconstructed.cpu().numpy()
        reconstructed = (reconstructed * 0.5 + 0.5).clip(0, 1)
        
        return reconstructed[0]  # Remove batch dimension

def visualize_and_save_results(original_path, reconstructed_img, save_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and resize original image
    original = Image.open(original_path)
    original = original.resize((224, 224))
    original = np.array(original) / 255.0
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(save_dir, f"comparison_{timestamp}.png")
    plt.savefig(comparison_path)
    plt.show()
    
    # Save the reconstructed image separately
    reconstructed_path = os.path.join(save_dir, f"reconstructed_{timestamp}.png")
    
    # Convert reconstructed image to PIL Image and save
    reconstructed_pil = Image.fromarray((reconstructed_img * 255).astype(np.uint8))
    reconstructed_pil = reconstructed_pil.resize((1080, 1080), Image.Resampling.LANCZOS)  # Resize back to 1080x1080
    reconstructed_pil.save(reconstructed_path)
    
    print(f"Saved comparison plot to: {comparison_path}")
    print(f"Saved reconstructed image to: {reconstructed_path}")

# Example usage
if __name__ == "__main__":
    image_path = "E:/Research/RP2/RP2TestData/testIn.png"  # Replace with your image path
    model_path = "E:/Research/RP2/ResearchProject2_BrainCancerEncode/SavedModels"  # Path from the original code
    output_dir = "E:/Research/RP2/RP2TestData/output"  # Directory where outputs will be saved
    
    reconstructed_img = process_single_image(image_path, model_path)
    visualize_and_save_results(image_path, reconstructed_img, output_dir)

