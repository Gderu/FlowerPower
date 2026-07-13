"""Run the fine-tuned PIC model on flower images and compare it to the pretrained ImageNet
model. Loads whatever checkpoint currently lives in the fine-tuned folder (default
`pic_inpainting/checkpoints/flowers_finetuned_best/`), picking best_ > latest_ > highest epoch_N.

Usage (from the repo root):
    python pic_finetuning/run_finetuned.py
    python pic_finetuning/run_finetuned.py --num_images 6 --sample_num 4 --seed 7
    python pic_finetuning/run_finetuned.py --finetuned_dir pic_inpainting/checkpoints/flowers_finetuned

Writes side-by-side grids to evaluations/pic_finetuning/ and prints hole-L1 / PSNR.
"""

import os
import sys
import glob
import re
import shutil
import tempfile
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
from pic_inference import get_pic_inpainter, pic_inpaint
from evaluate_pic import make_figure
import pic_finetune_core as core


def _pick(dirpath, which):
    """Pick the {which}=E/G checkpoint: best_ > latest_ > highest epoch_N."""
    cands = glob.glob(os.path.join(dirpath, f"*_net_{which}.pth"))
    if not cands:
        raise FileNotFoundError(f"no *_net_{which}.pth in {dirpath}")

    def rank(p):
        b = os.path.basename(p)
        if b.startswith("best_"):
            return (3, 0)
        if b.startswith("latest_"):
            return (2, 0)
        m = re.match(r"epoch_(\d+)_", b)
        return (1, int(m.group(1))) if m else (0, 0)

    return max(cands, key=rank)


def load_finetuned(dirpath, device):
    e_path, g_path = _pick(dirpath, "E"), _pick(dirpath, "G")
    print(f"fine-tuned checkpoint: {os.path.basename(e_path)} / {os.path.basename(g_path)}")
    tmp = tempfile.mkdtemp()
    shutil.copy(e_path, os.path.join(tmp, "latest_net_E.pth"))
    shutil.copy(g_path, os.path.join(tmp, "latest_net_G.pth"))
    return get_pic_inpainter(tmp, device)


def main():
    ap = argparse.ArgumentParser(description="Run + compare the fine-tuned PIC model")
    ap.add_argument("--finetuned_dir", default=os.path.join(REPO, "pic_inpainting", "checkpoints", "flowers_finetuned_best"))
    ap.add_argument("--pretrained_dir", default=os.path.join(REPO, "pic_inpainting", "checkpoints", "imagenet_random"))
    ap.add_argument("--data_dir", default=os.path.join(REPO, "GAN_implementation", "data_128x128"))
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
    ft_E, ft_G = load_finetuned(args.finetuned_dir, device)

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
