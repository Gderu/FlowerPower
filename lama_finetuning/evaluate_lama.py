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
        checkpoints = glob.glob("checkpoints/128/fine_tuned_correctly/lama_netG_epoch_*.pth")
        if not checkpoints:
            print("No generator checkpoints found!")
            return
        
        # Sort by epoch number
        def get_epoch(x):
            filename = os.path.basename(x)
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                return -1
                
        checkpoints.sort(key=get_epoch)
        checkpoint_path = checkpoints[-1]
        
    print(f"Evaluating with checkpoint: {checkpoint_path}")

    netG = get_lama_generator(device, pretrained_path=checkpoint_path)
    netG.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageDataset(directory='data_128x128', transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    
    real_imgs = next(iter(dataloader)).to(device)
    
    masks = get_large_random_mask(num_images, 128, 128, device)
    masked_imgs = real_imgs * (1.0 - masks)
    g_in = torch.cat((masked_imgs, masks), dim=1)
    
    with torch.no_grad():
        fake_imgs = netG(g_in)
        fake_imgs = torch.clamp(fake_imgs, 0, 1)
        
    comp_imgs = masked_imgs + fake_imgs * masks
    
    def to_numpy(img_tensor):
        return img_tensor.cpu().clone().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    plt.suptitle(f"LaMa Evaluation using {checkpoint_path}", fontsize=16)
    
    for i in range(num_images):
        # Real Image
        ax = axes[i, 0] if num_images > 1 else axes[0]
        ax.imshow(to_numpy(real_imgs[i]))
        ax.set_title("Real Image" if i == 0 else "")
        ax.axis('off')
        
        # Masked Image
        ax = axes[i, 1] if num_images > 1 else axes[1]
        ax.imshow(to_numpy(masked_imgs[i]))
        ax.set_title("Masked Image" if i == 0 else "")
        ax.axis('off')
        
        # Generated / Composite Image
        ax = axes[i, 2] if num_images > 1 else axes[2]
        ax.imshow(to_numpy(comp_imgs[i]))
        ax.set_title("Inpainted Image" if i == 0 else "")
        ax.axis('off')
        
    plt.tight_layout()
    
    os.makedirs("evaluations/lama_finetuning", exist_ok=True)
    name = os.path.basename(checkpoint_path)
    save_path = f"evaluations/lama_finetuning/evaluation_sample_{name.split('.')[0]}.png"
    plt.savefig(save_path)
    print(f"Saved evaluation plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_lama(num_images=5)
