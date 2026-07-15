"""Evaluate PIC (Pluralistic Image Completion) on flower images.

Produces a qualitative comparison figure with columns:
    Real | Masked | Sample 1 | ... | Sample K
where the Sample columns are diverse completions of the SAME masked input --
the pluralistic-VAE advantage over the deterministic U-Net GAN and LaMa.

Mirrors lama_finetuning/evaluate_lama.py, but:
  * calls the feed-forward PIC wrapper (pic_inference.pic_inpaint) instead of a
    single generator, and
  * renders the grid with torchvision/PIL rather than matplotlib (matplotlib is
    currently broken in this env under numpy 2.x).

Reuses the shared FlowerPower infrastructure unchanged:
  * shared_utils.ImageDataset          -> loads images as [0,1] tensors
  * shared_utils.get_large_random_mask -> (B,1,H,W), 1 = hole
  * composition rule  comp = masked + fill * mask   (as in evaluate_lama.py)
"""

import os
import sys
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

# --- FlowerPower shared utilities (repo root) ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from shared_utils import ImageDataset, get_large_random_mask  # noqa: E402

# --- PIC wrapper (this package) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pic_inference import get_pic_inpainter, pic_inpaint  # noqa: E402


def _to_pil(img_chw):
    """(3,H,W) float [0,1] tensor -> PIL.Image."""
    arr = (img_chw.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr)


def make_figure(reals, maskeds, samples, out_path, cell=128, pad=4, header=22):
    """Build a labeled [Real | Masked | Sample 1..K] grid and save as PNG.

    reals, maskeds: (B,3,128,128); samples: (B,K,3,128,128) -- all in [0,1].
    """
    B = reals.size(0)
    K = samples.size(1)
    ncol = 2 + K
    labels = ["Real", "Masked"] + ["Sample %d" % (k + 1) for k in range(K)]

    W = ncol * cell + (ncol + 1) * pad
    H = header + B * cell + (B + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # column headers
    for c, text in enumerate(labels):
        x = pad + c * (cell + pad) + cell // 2
        draw.text((x - 3 * len(text), header // 2 - 6), text, fill=(0, 0, 0))

    # rows
    for b in range(B):
        row_imgs = [reals[b], maskeds[b]] + [samples[b, k] for k in range(K)]
        y = header + pad + b * (cell + pad)
        for c, img in enumerate(row_imgs):
            x = pad + c * (cell + pad)
            canvas.paste(_to_pil(img), (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate PIC on flower images")
    parser.add_argument("--data_dir", type=str, default="data_128x128",
                        help="folder of images (resized to 128 on load)")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "checkpoints", "pretrained"),
                        help="folder with an E/G checkpoint pair (*_net_E.pth / *_net_G.pth)")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=3)
    parser.add_argument("--out", type=str,
                        default=os.path.join(_REPO_ROOT, "evaluations", "pic_inpainting", "evaluation_sample.png"))
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible masks (z stays unseeded)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            "data dir '%s' not found. Point --data_dir at a folder of images "
            "(e.g. data_128x128, regenerated via `python prepare_data.py` from the repo root)."
            % args.data_dir
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # [0,1] tensors, resized to 128 (no-op for data_128x128; robust for other folders)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = ImageDataset(directory=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.num_images, shuffle=args.seed is None)
    reals = next(iter(loader)).to(device)

    masks = get_large_random_mask(reals.size(0), 128, 128, device)  # (B,1,128,128), 1 = hole
    maskeds = reals * (1.0 - masks)

    print("loading PIC from %s ..." % args.ckpt_dir)
    net_E, net_G = get_pic_inpainter(args.ckpt_dir, device)

    print("running %d diverse completions per image..." % args.sample_num)
    import time
    t0 = time.time()
    fills = pic_inpaint(net_E, net_G, reals, masks, sample_num=args.sample_num, device=device)  # (B,K,3,128,128)
    dt = time.time() - t0

    # compose each sample against the known region (FlowerPower rule)
    comps = torch.stack(
        [maskeds + fills[:, k] * masks for k in range(args.sample_num)], dim=1
    )  # (B,K,3,128,128)

    out_path = make_figure(reals, maskeds, comps, args.out)
    n = reals.size(0) * args.sample_num
    print("done: %d completions in %.1fs (%.2fs each) -> %s" % (n, dt, dt / max(n, 1), out_path))


if __name__ == "__main__":
    main()
