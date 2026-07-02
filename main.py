import os
import random
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output, display
    in_jupyter = True
except ImportError:
    in_jupyter = False


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

# --- GENERATOR (U-NET) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use ConvTranspose2d for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 size (though here we assume sizes match perfectly)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Expected input: 64x64
        self.inc = DoubleConv(in_channels, 64)       # 64x64
        self.down1 = Down(64, 128)                   # 32x32
        self.down2 = Down(128, 256)                  # 16x16
        self.down3 = Down(256, 512)                  # 8x8
        self.down4 = Down(512, 1024)                 # 4x4
        
        self.up1 = Up(1024, 512)                     # 8x8
        self.up2 = Up(512, 256)                      # 16x16
        self.up3 = Up(256, 128)                      # 32x32
        self.up4 = Up(128, 64)                       # 64x64
        
        self.outc = OutConv(64, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return self.tanh(logits)


# --- DISCRIMINATOR ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features, 32, 32)
            
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*2, 16, 16)
            
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*4, 8, 8)
            
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (features*8, 4, 4)
            
            # Final output layer
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


# --- UTILS ---
def get_quadrant_mask(batch_size, h, w, device):
    """
    Generates a mask for one of the four quadrants.
    Mask is 1 for the region to be erased (inpainted), 0 elsewhere.
    """
    masks = torch.zeros((batch_size, 1, h, w), device=device)
    for i in range(batch_size):
        quadrant = random.randint(0, 3)
        if quadrant == 0:   # Top-Left
            masks[i, 0, :h//2, :w//2] = 1.0
        elif quadrant == 1: # Top-Right
            masks[i, 0, :h//2, w//2:] = 1.0
        elif quadrant == 2: # Bottom-Left
            masks[i, 0, h//2:, :w//2] = 1.0
        else:               # Bottom-Right
            masks[i, 0, h//2:, w//2:] = 1.0
    return masks

# Weight initialization for GAN
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif hasattr(m, 'weight') and (classname.find('BatchNorm') != -1):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # --- HYPERPARAMETERS ---
    batch_size = 32
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5
    lambda_l1 = 100  # Weight for L1 reconstruction loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms (normalize to [-1, 1] for Tanh output in generator)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the dataset
    dataset = ImageDataset(directory='data_64x64', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Models
    netG = UNetGenerator(in_channels=4, out_channels=3).to(device)
    netG.apply(weights_init)
    
    netD = Discriminator(in_channels=3, features=64).to(device)
    netD.apply(weights_init)

    # Optimizers & Loss
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print("Starting Training Loop...")
    
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            real_imgs = data.to(device)
            b_size = real_imgs.size(0)
            
            # Generate random quadrant masks (1 for missing regions, 0 for kept)
            masks = get_quadrant_mask(b_size, 64, 64, device)
            
            # Mask the real images (set erased pixels to 0)
            masked_imgs = real_imgs * (1.0 - masks)
            
            # Generator Input: Concatenate masked_imgs (3 channels) with mask (1 channel)
            g_in = torch.cat((masked_imgs, masks), dim=1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            netD.zero_grad()
            
            # Format labels
            real_label = torch.ones((b_size,), dtype=torch.float, device=device)
            fake_label = torch.zeros((b_size,), dtype=torch.float, device=device)

            # Train with real images
            output_real = netD(real_imgs)
            errD_real = criterion_GAN(output_real, real_label)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Train with fake images
            fake_imgs = netG(g_in)
            
            # Composite image: original image outside the mask, generated image inside the mask
            comp_imgs = masked_imgs + fake_imgs * masks
            
            output_fake = netD(comp_imgs.detach())
            errD_fake = criterion_GAN(output_fake, fake_label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            netG.zero_grad()
            
            # GAN Loss: fool the discriminator
            output_fake_for_G = netD(comp_imgs)
            errG_GAN = criterion_GAN(output_fake_for_G, real_label)
            
            # L1 Loss: pixel-level accuracy only on the masked region
            errG_L1 = criterion_L1(fake_imgs * masks, real_imgs * masks)
            
            # Total Generator Loss
            errG = errG_GAN + lambda_l1 * errG_L1
            errG.backward()
            D_G_z2 = output_fake_for_G.mean().item()
            optimizerG.step()

            # Print output periodically
            if i % 50 == 0:                
                # Real-time plotting
                if in_jupyter and len(G_losses) > 0:
                    clear_output(wait=True)
                    fig = plt.figure(figsize=(10,5))
                    plt.title("Generator and Discriminator Loss During Training")
                    plt.plot(G_losses, label="G")
                    plt.plot(D_losses, label="D")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    display(fig)
                    plt.close(fig)
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                      f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
                
        # Optional: save output
        # vutils.save_image(comp_imgs, f"generated_epoch_{epoch}.png", normalize=True)
        
        # Checkpoint every epoch
        torch.save(netG.state_dict(), f"netG_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"netD_epoch_{epoch}.pth")
        print(f"Checkpoints saved for epoch {epoch}")

    print("Training finished.")

    # Plot the training losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()