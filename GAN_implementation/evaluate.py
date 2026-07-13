import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import UNetGenerator
from shared_utils import ImageDataset, get_large_random_mask
import glob

def evaluate(checkpoint_path=None, num_images=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find latest checkpoint if none specified
    if checkpoint_path is None:
        checkpoints = glob.glob("checkpoints/128/netG*.pth")
        if not checkpoints:
            print("No generator checkpoints found!")
            return
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x[len("checkpoints/128/netG"):].split('_')[0]))
        checkpoint_path = checkpoints[-1]
        
    print(f"Evaluating with checkpoint: {checkpoint_path}")

    netG = UNetGenerator(in_channels=4, out_channels=3).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(directory='data_128x128', transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    
    real_imgs = next(iter(dataloader)).to(device)
    
    masks = get_large_random_mask(num_images, 128, 128, device)
    masked_imgs = real_imgs * (1.0 - masks)
    g_in = torch.cat((masked_imgs, masks), dim=1)
    
    with torch.no_grad():
        fake_imgs = netG(g_in)
        
    comp_imgs = masked_imgs + fake_imgs * masks
    
    # Un-normalize from [-1, 1] back to [0, 1] for plotting
    def unnorm(img_tensor):
        img = img_tensor.cpu().clone()
        img = img * 0.5 + 0.5 # un-normalize back to [0, 1]
        return img.permute(1, 2, 0).numpy() # (H, W, C)

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    plt.suptitle(f"Evaluation using {checkpoint_path}", fontsize=16)
    
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
    name = checkpoint_path.split('\\')[-1]
    save_path = f"evaluations/128/evaluation_sample_{name.split('.')[0]}.png"
    plt.savefig(save_path)
    print(f"Saved evaluation plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate(num_images=5)
