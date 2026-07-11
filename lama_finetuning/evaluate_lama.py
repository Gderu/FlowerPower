import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lama_finetuning.fine_tune_lama import get_lama_generator
from shared_utils import ImageDataset, get_large_random_mask
import glob

def evaluate_lama(checkpoint_path=None, num_images=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find latest checkpoint if none specified
    if checkpoint_path is None:
        checkpoints = glob.glob("checkpoints/128/fine_tune/lama_netG_epoch_*.pth")
        if not checkpoints:
            print("No generator checkpoints found!")
            return
        
        # Sort by epoch number (e.g. lama_netG_epoch_9.pth -> 9)
        def get_epoch(x):
            filename = os.path.basename(x)
            # Extracts the number from strings like 'lama_netG_epoch_10.pth'
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                return -1
                
        checkpoints.sort(key=get_epoch)
        checkpoint_path = checkpoints[-1]
        
    print(f"Evaluating with checkpoint: {checkpoint_path}")

    # 1. Load the model (get_lama_generator handles both original and fine-tuned checkpoints!)
    netG = get_lama_generator(device, pretrained_path=checkpoint_path)
    netG.eval()
    
    # 2. Setup Dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Make sure this points to the right dataset location
    dataset = ImageDataset(directory='data_128x128', transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    
    # 3. Get a batch of images
    real_imgs = next(iter(dataloader)).to(device)
    
    # 4. Generate masks and inputs
    masks = get_large_random_mask(num_images, 128, 128, device)
    masked_imgs = real_imgs * (1.0 - masks)
    g_in = torch.cat((masked_imgs, masks), dim=1)
    
    # 5. Generate outputs
    with torch.no_grad():
        fake_imgs = netG(g_in)
        # LaMa FFC ResNet generator outputs raw pixels directly trained for [0, 1]
        fake_imgs = torch.clamp(fake_imgs, 0, 1)
        
    comp_imgs = masked_imgs + fake_imgs * masks
    
    # Helper to un-normalize images for plotting
    def unnorm(img_tensor):
        img = img_tensor.cpu().clone()
        # No un-normalization needed, inputs are already in [0, 1]
        return img.permute(1, 2, 0).numpy() # (H, W, C)

    # 6. Plotting
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    plt.suptitle(f"LaMa Evaluation using {checkpoint_path}", fontsize=16)
    
    for i in range(num_images):
        # Real Image
        ax = axes[i, 0] if num_images > 1 else axes[0]
        ax.imshow(unnorm(real_imgs[i]))
        ax.set_title("Real Image" if i == 0 else "")
        ax.axis('off')
        
        # Masked Image
        ax = axes[i, 1] if num_images > 1 else axes[1]
        ax.imshow(unnorm(masked_imgs[i]))
        ax.set_title("Masked Image" if i == 0 else "")
        ax.axis('off')
        
        # Generated / Composite Image
        ax = axes[i, 2] if num_images > 1 else axes[2]
        ax.imshow(unnorm(comp_imgs[i]))
        ax.set_title("Inpainted Image" if i == 0 else "")
        ax.axis('off')
        
    plt.tight_layout()
    
    # Ensure evaluations directory exists
    os.makedirs("evaluations/lama_finetuning", exist_ok=True)
    name = os.path.basename(checkpoint_path)
    save_path = f"evaluations/lama_finetuning/evaluation_sample_{name.split('.')[0]}.png"
    plt.savefig(save_path)
    print(f"Saved evaluation plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_lama(r"C:\Users\orrsh\Documents\deep learning course\project\evaluations\big-lama\models\best.ckpt", num_images=5)
