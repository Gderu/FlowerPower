"""Shared components used by both GAN and LaMa training pipelines:
dataset loading, discriminator, masking, and loss utilities."""

import random
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = Path(directory)
        self.transform = transform
        # Find all images
        self.image_paths = [
            p for p in self.directory.iterdir() 
            if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (3, 128, 128)
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features, 64, 64)
            
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*2, 32, 32)
            
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*4, 16, 16)
            
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*8, 8, 8)
            
            nn.Conv2d(features * 8, features * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*16, 4, 4)
            
            # Final output layer
            nn.Conv2d(features * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def get_large_random_mask(batch_size, h, w, device):
    """
    Generates a mask for large contiguous areas (quadrants, halves, or large random rectangles).
    Mask is 1 for the region to be erased (inpainted), 0 elsewhere.
    """
    masks = torch.zeros((batch_size, 1, h, w), device=device)
    for i in range(batch_size):
        strategy = random.choice(['half', 'quadrant', 'rectangle'])
        
        if strategy == 'half':
            half_choice = random.randint(0, 3)
            if half_choice == 0: masks[i, 0, :h//2, :] = 1.0 # Top half
            elif half_choice == 1: masks[i, 0, h//2:, :] = 1.0 # Bottom half
            elif half_choice == 2: masks[i, 0, :, :w//2] = 1.0 # Left half
            else: masks[i, 0, :, w//2:] = 1.0 # Right half
            
        elif strategy == 'quadrant':
            quad_choice = random.randint(0, 3)
            if quad_choice == 0: masks[i, 0, :h//2, :w//2] = 1.0
            elif quad_choice == 1: masks[i, 0, :h//2, w//2:] = 1.0
            elif quad_choice == 2: masks[i, 0, h//2:, :w//2] = 1.0
            else: masks[i, 0, h//2:, w//2:] = 1.0
            
        else: # Large random rectangle
            # Width and height between 30% and 70% of the image
            rect_h = random.randint(int(0.3 * h), int(0.7 * h))
            rect_w = random.randint(int(0.3 * w), int(0.7 * w))
            
            # Random top-left corner
            top = random.randint(0, h - rect_h)
            left = random.randint(0, w - rect_w)
            
            masks[i, 0, top:top+rect_h, left:left+rect_w] = 1.0
            
    return masks

def get_image_gradients(image):
    """Returns the x and y gradients of an image."""
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dx, dy

def compute_gradient_loss(fake, real):
    """Computes L1 loss between image gradients to penalize blurriness."""
    dx_fake, dy_fake = get_image_gradients(fake)
    dx_real, dy_real = get_image_gradients(real)
    criterion = nn.L1Loss()
    return criterion(dx_fake, dx_real) + criterion(dy_fake, dy_real)
