import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Dataset and Discriminator from shared_utils
from shared_utils import ImageDataset, Discriminator, get_large_random_mask, compute_gradient_loss

# ---------------------------------------------------------
# LAMA FFC GENERATOR SETUP
# ---------------------------------------------------------
# To fine-tune the raw PyTorch model, we import the architecture from the official repo.
# Make sure the 'lama_repo' folder is in your project directory (git cloned).
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lama_repo')))

try:
    from saicinpainting.training.modules.ffc import FFCResNetGenerator
except ImportError as e:
    print(f"WARNING: Could not import FFCResNetGenerator from lama_repo. Error: {e}")
    import traceback
    traceback.print_exc()
    print("Ensure you have cloned https://github.com/advimman/lama.git into 'lama_repo'")
    FFCResNetGenerator = None

def get_lama_generator(device, pretrained_path=None):
    if FFCResNetGenerator is None:
        raise RuntimeError("LaMa repo not found.")
        
    # Standard big-lama FFC architecture config
    netG = FFCResNetGenerator(
        input_nc=4, 
        output_nc=3, 
        ngf=64, 
        n_blocks=18, 
        add_out_act=False,
        init_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False},
        downsample_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False},
        resnet_conv_kwargs={'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}
    ).to(device)
    
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        # Handle state dict wrapping if necessary
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Strip 'generator.' prefix if it exists in the checkpoint
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('generator.'):
                clean_state_dict[k.replace('generator.', '')] = v
            else:
                clean_state_dict[k] = v
                
        netG.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded pretrained LaMa weights from {pretrained_path}")
    else:
        print("Starting LaMa from scratch (No pretrained weights found).")
        
    return netG

# ---------------------------------------------------------
# FINE-TUNING LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # HYPERPARAMETERS FOR FINE-TUNING
    # We use a very low learning rate for the generator so we don't destroy its pretrained knowledge
    batch_size = 16 # Might need to lower this if LaMa runs out of memory (FFC is heavy)
    num_epochs = 10
    lr_G = 1e-5     # Low learning rate for fine-tuning
    lr_D = 1e-4     # Discriminator learns faster
    lambda_l1 = 100 
    lambda_edge = 50

    # Dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(directory='data_128x128', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Models
    # NOTE: You will need to download big-lama weights (best.ckpt) from the official repo
    # and place it in the checkpoints folder.
    pretrained_weights_path = "checkpoints/big-lama/models/best.ckpt" 
    
    netG = get_lama_generator(device, pretrained_path=pretrained_weights_path)
    netD = Discriminator(in_channels=3, features=64).to(device)

    # Optimizers & Loss
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

    print("Starting Fine-Tuning Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            real_imgs = data.to(device)
            b_size = real_imgs.size(0)
            
            masks = get_large_random_mask(b_size, 128, 128, device)
            masked_imgs = real_imgs * (1.0 - masks)
            
            # LaMa input format
            g_in = torch.cat((masked_imgs, masks), dim=1)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            netD.zero_grad()
            real_label = torch.ones((b_size,), dtype=torch.float, device=device)
            fake_label = torch.zeros((b_size,), dtype=torch.float, device=device)

            output_real = netD(real_imgs)
            errD_real = criterion_GAN(output_real, real_label)
            errD_real.backward()

            fake_imgs = netG(g_in)
            
            # Since add_out_act=False in FFC setup, we apply Tanh here
            fake_imgs = torch.tanh(fake_imgs)

            comp_imgs = masked_imgs + fake_imgs * masks
            
            output_fake = netD(comp_imgs.detach())
            errD_fake = criterion_GAN(output_fake, fake_label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            netG.zero_grad()
            output_fake_for_G = netD(comp_imgs)
            
            errG_GAN = criterion_GAN(output_fake_for_G, real_label)
            errG_L1 = criterion_L1(fake_imgs * masks, real_imgs * masks)
            errG_edge = compute_gradient_loss(comp_imgs, real_imgs)
            
            errG = errG_GAN + lambda_l1 * errG_L1 + lambda_edge * errG_edge
            errG.backward()
            optimizerG.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
                      
        # Save checkpoints
        torch.save(netG.state_dict(), f"checkpoints/lama_netG_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"checkpoints/lama_netD_epoch_{epoch}.pth")
