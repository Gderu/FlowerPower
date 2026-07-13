import os
import sys
import random
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output, display
    in_jupyter = True
except ImportError:
    in_jupyter = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared_utils import ImageDataset, Discriminator, get_large_random_mask, compute_gradient_loss


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
        # Pad x1 to match x2 size if needed
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

        # Expected input: 128x128
        self.inc = DoubleConv(in_channels, 64)       # 128x128
        self.down1 = Down(64, 128)                   # 64x64
        self.down2 = Down(128, 256)                  # 32x32
        self.down3 = Down(256, 512)                  # 16x16
        self.down4 = Down(512, 1024)                 # 8x8
        self.down5 = Down(1024, 2048)                # 4x4
        
        self.up1 = Up(2048, 1024)                    # 8x8
        self.up2 = Up(1024, 512)                     # 16x16
        self.up3 = Up(512, 256)                      # 32x32
        self.up4 = Up(256, 128)                      # 64x64
        self.up5 = Up(128, 64)                       # 128x128
        
        self.outc = OutConv(64, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        
        logits = self.outc(x)
        return self.tanh(logits)



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
    lambda_edge = 50 # Weight for gradient/edge loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms (normalize to [-1, 1] for Tanh output in generator)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the dataset
    dataset = ImageDataset(directory='data_128x128', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Models
    netG = UNetGenerator(in_channels=4, out_channels=3).to(device)
    netD = Discriminator(in_channels=3, features=64).to(device)
    
    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 0
    resume_checkpoint_G = None  # e.g. "checkpoints/gan/netG_epoch_3.pth"
    resume_checkpoint_D = None  # e.g. "checkpoints/gan/netD_epoch_3.pth"
    
    if resume_checkpoint_G and resume_checkpoint_D:
        print(f"Resuming training from {resume_checkpoint_G} and {resume_checkpoint_D}")
        netG.load_state_dict(torch.load(resume_checkpoint_G, map_location=device))
        netD.load_state_dict(torch.load(resume_checkpoint_D, map_location=device))
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    # --- MULTI-GPU SUPPORT ---
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # Optimizers & Loss
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print("Starting Training Loop...")
    os.makedirs("checkpoints/gan", exist_ok=True)
    os.makedirs("evaluations/gan", exist_ok=True)
    
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    

    for epoch in range(start_epoch, num_epochs):
        for i, data in enumerate(dataloader):
            real_imgs = data.to(device)
            b_size = real_imgs.size(0)
            
            # Generate random large masks (1 for missing regions, 0 for kept)
            masks = get_large_random_mask(b_size, 128, 128, device)
            
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

            output_real = netD(real_imgs)
            errD_real = criterion_GAN(output_real, real_label)
            errD_real.backward()
            D_x = output_real.mean().item()

            fake_imgs = netG(g_in)
            
            # Composite: original pixels outside the mask, generated pixels inside
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
            
            # Edge loss on composite to penalize boundary artifacts
            errG_edge = compute_gradient_loss(comp_imgs, real_imgs)
            
            # Total Generator Loss
            errG = errG_GAN + lambda_l1 * errG_L1 + lambda_edge * errG_edge
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
                
        # Checkpoint every epoch
        state_dict_G = netG.module.state_dict() if isinstance(netG, nn.DataParallel) else netG.state_dict()
        state_dict_D = netD.module.state_dict() if isinstance(netD, nn.DataParallel) else netD.state_dict()
        torch.save(state_dict_G, f"checkpoints/gan/netG_epoch_{epoch}.pth")
        torch.save(state_dict_D, f"checkpoints/gan/netD_epoch_{epoch}.pth")
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
    plt.savefig("evaluations/gan/loss_curve.png")
    plt.show()