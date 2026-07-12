import os
import torch
import torch.nn as nn
import pickle

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Force PyTorch to disable the global weights_only override (fixes Kaggle issues)
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Dataset and Discriminator from shared_utils
from shared_utils import ImageDataset, Discriminator, get_large_random_mask, compute_gradient_loss

# ---------------------------------------------------------
# LAMA FFC GENERATOR SETUP
# ---------------------------------------------------------
# To fine-tune the raw PyTorch model, we import the architecture from the official repo.
# Make sure the 'lama_repo' folder is in your project directory (git cloned).
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
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=False, pickle_module=pickle)
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


class VGGPerceptualLoss(nn.Module):
    """Compares images in VGG feature space rather than pixel space.
    Frozen VGG16 is used as a fixed feature extractor — no trainable params.
    This is much more sensitive to color/style inconsistencies than L1."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Extract features at three depth levels
        self.slice1 = nn.Sequential(*vgg[:4])   # relu1_2 — colors, edges
        self.slice2 = nn.Sequential(*vgg[4:9])   # relu2_2 — textures
        self.slice3 = nn.Sequential(*vgg[9:16])  # relu3_3 — patterns, structure
        # Freeze all weights — we never train VGG
        for param in self.parameters():
            param.requires_grad = False
        # VGG expects ImageNet-normalized inputs
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        fake = self.normalize(fake)
        real = self.normalize(real)
        loss = 0.0
        for layer in [self.slice1, self.slice2, self.slice3]:
            fake = layer(fake)
            real = layer(real)
            loss += nn.functional.l1_loss(fake, real)
        return loss

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
    lambda_l1_hole = 100 
    lambda_l1_valid = 20  # Enforces color consistency with the unmasked areas
    lambda_edge = 50
    lambda_perceptual = 10  # Perceptual loss weight — enforces color/style consistency in feature space

    # Dataloader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageDataset(directory='data_128x128', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize Models
    pretrained_weights_path = "checkpoints/big-lama/models/best.ckpt" 
    
    netG = get_lama_generator(device, pretrained_path=pretrained_weights_path)
    netD = Discriminator(in_channels=3, features=64).to(device)

    # --- MULTI-GPU SUPPORT ---
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # Optimizers & Loss
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device).eval()
    
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
            
            # LaMa outputs raw pixels in [0, 1], so we clamp to avoid instabilities
            fake_imgs = torch.clamp(fake_imgs, 0, 1)

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
            # L1 loss on the masked region (the hole)
            errG_L1_hole = criterion_L1(fake_imgs * masks, real_imgs * masks)
            # L1 loss on the unmasked region (valid pixels). This forces the generator to maintain 
            # color consistency with the surrounding area instead of drifting in color space.
            errG_L1_valid = criterion_L1(fake_imgs * (1 - masks), real_imgs * (1 - masks))
            
            errG_edge = compute_gradient_loss(comp_imgs, real_imgs)
            errG_perceptual = perceptual_loss_fn(comp_imgs, real_imgs)
            
            errG = errG_GAN + lambda_l1_hole * errG_L1_hole + lambda_l1_valid * errG_L1_valid + lambda_edge * errG_edge + lambda_perceptual * errG_perceptual
            errG.backward()
            optimizerG.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
                      
        # Save checkpoints
        state_dict_G = netG.module.state_dict() if isinstance(netG, nn.DataParallel) else netG.state_dict()
        state_dict_D = netD.module.state_dict() if isinstance(netD, nn.DataParallel) else netD.state_dict()
        torch.save(state_dict_G, f"checkpoints/lama_netG_epoch_{epoch}.pth")
        torch.save(state_dict_D, f"checkpoints/lama_netD_epoch_{epoch}.pth")
