"""Run the fine-tuned PIC model on flower images and compare it to the pretrained ImageNet
model. Loads whatever checkpoint currently lives in the fine-tuned folder (default
`pic_inpainting/checkpoints/finetuned/`), picking best_ > latest_ > highest epoch_N.

Usage (from the repo root):
    python pic_finetuning/run_finetuned.py
    python pic_finetuning/run_finetuned.py --num_images 6 --sample_num 4 --seed 7
    python pic_finetuning/run_finetuned.py --finetuned_dir <folder with *_net_E.pth / *_net_G.pth>

Writes side-by-side grids to evaluations/pic_finetuning/ and prints hole-L1 / PSNR.
"""

import os
import sys
import random
import argparse

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [REPO, os.path.join(REPO, "pic_inpainting"),
          os.path.join(REPO, "pic_inpainting", "pic_repo"),
          os.path.join(REPO, "pic_finetuning")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from shared_utils import ImageDataset, get_large_random_mask
from pic_inference import get_pic_inpainter, pic_inpaint, resolve_checkpoints
from evaluate_pic import make_figure
import pic_finetune_core as core


def main():
    ap = argparse.ArgumentParser(description="Run + compare the fine-tuned PIC model")
    ap.add_argument("--finetuned_dir", default=os.path.join(REPO, "pic_inpainting", "checkpoints", "finetuned"))
    ap.add_argument("--pretrained_dir", default=os.path.join(REPO, "pic_inpainting", "checkpoints", "pretrained"))
    ap.add_argument("--data_dir", default=os.path.join(REPO, "data_128x128"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "evaluations", "pic_finetuning"))
    ap.add_argument("--num_images", type=int, default=4)
    ap.add_argument("--sample_num", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)

    tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    ds = ImageDataset(directory=args.data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.num_images, shuffle=True)
    reals = next(iter(loader)).to(device)
    masks = get_large_random_mask(reals.size(0), 128, 128, device)  # 1 = hole
    maskeds = reals * (1.0 - masks)

    print("loading pretrained (ImageNet) ...")
    pre_E, pre_G = get_pic_inpainter(args.pretrained_dir, device)
    print("loading fine-tuned (flowers) ...")
    ft_e_path, ft_g_path = resolve_checkpoints(args.finetuned_dir)
    print("fine-tuned checkpoint: %s / %s"
          % (os.path.basename(ft_e_path), os.path.basename(ft_g_path)))
    ft_E, ft_G = get_pic_inpainter(args.finetuned_dir, device)

    def comps(E, G):
        fills = pic_inpaint(E, G, reals, masks, sample_num=args.sample_num, device=device)
        return torch.stack([maskeds + fills[:, k] * masks for k in range(args.sample_num)], dim=1)

    print("running completions ...")
    p1 = make_figure(reals, maskeds, comps(pre_E, pre_G), os.path.join(args.out_dir, "compare_pretrained.png"))
    p2 = make_figure(reals, maskeds, comps(ft_E, ft_G), os.path.join(args.out_dir, "compare_finetuned.png"))

    pre_l1, pre_psnr = core.evaluate_set(pre_E, pre_G, reals, masks, args.num_images, device)
    ft_l1, ft_psnr = core.evaluate_set(ft_E, ft_G, reals, masks, args.num_images, device)
    print("\n{:<24}{:>16}{:>16}".format("model", "hole-L1", "PSNR dB"))
    print("{:<24}{:>16.4f}{:>16.2f}".format("pretrained (ImageNet)", pre_l1, pre_psnr))
    print("{:<24}{:>16.4f}{:>16.2f}".format("fine-tuned (flowers)", ft_l1, ft_psnr))
    print("{:<24}{:>16.4f}{:>16.2f}".format("improvement", pre_l1 - ft_l1, ft_psnr - pre_psnr))
    print("\nfigures:\n ", p1, "\n ", p2)


if __name__ == "__main__":
    main()
