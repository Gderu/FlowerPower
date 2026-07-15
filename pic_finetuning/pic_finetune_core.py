"""Core logic for fine-tuning PIC (Pluralistic Image Completion) on the flowers
dataset with a VAE-only objective (KL + multi-scale L1, no discriminators).

The training step is a faithful port of the NON-adversarial parts of
`pic_repo/model/pluralistic_model.py` (get_distribution / get_G_inputs / forward /
backward_G) -- that module can't be imported directly because it contains a
Python-3.6 `async=True` kwarg that is a SyntaxError on modern Python. We build the
nets via the existing `pic_inference.get_pic_inpainter` wrapper and reproduce the
two-path VAE objective here.

Conventions (identical to the PIC wrapper):
  * training resolution 256x256 (flowers loaded at 128, upscaled -- matches the
    inference path in `pic_inference.pic_inpaint`);
  * PIC mask polarity is 1 = KNOWN, 0 = HOLE (inverted from FlowerPower's 1 = hole);
  * images live in [-1, 1] inside the nets (generator ends in Tanh).

Validation / visualization reuse `pic_inference.pic_inpaint` unchanged, so the
metric and the periodic previews go through the exact same forward pass we use at
inference time for both the pretrained and the fine-tuned weights.
"""

import os
import re
import glob
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# `pic_inpaint` is reused verbatim for validation + visualization. The caller is
# expected to have `pic_inpainting/` on sys.path (the notebook / build_bundle set
# this up); we import lazily inside functions to keep this module import-light.


# --------------------------------------------------------------------------------------
# Multi-scale helpers (copied from pic_repo/util/task.py so we don't pull in cv2).
# --------------------------------------------------------------------------------------
def scale_img(img, size):
    return F.interpolate(img, size=size, mode="bilinear", align_corners=True)


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    h, w = img.size(2), img.size(3)
    for i in range(1, num_scales):
        ratio = 2 ** i
        scaled_imgs.append(scale_img(img, size=[h // ratio, w // ratio]))
    scaled_imgs.reverse()  # smallest scale first, full-res last
    return scaled_imgs


# --------------------------------------------------------------------------------------
# Pre-processing: FlowerPower [0,1] 128px images + (1=hole) masks -> PIC training tensors.
# Mirrors pic_inference.pic_inpaint (128->256, invert polarity, [-1,1]) and PIC's
# set_input (img_m / img_c, multi-scale pyramids). Mask is kept 3-channel to match
# pluralistic_model exactly (its get_G_inputs does mask.chunk(3, dim=1)[0]).
# --------------------------------------------------------------------------------------
def preprocess(images128, masks_hole128, output_scale, device):
    images128 = images128.to(device)
    masks_hole128 = masks_hole128.to(device)

    img256 = F.interpolate(images128, size=(256, 256), mode="bilinear", align_corners=True)
    mask_hole256 = F.interpolate(masks_hole128, size=(256, 256), mode="nearest")
    mask_known = (1.0 - mask_hole256).repeat(1, 3, 1, 1)  # 1 = known, 3ch

    img_truth = img256 * 2.0 - 1.0            # [0,1] -> [-1,1]
    img_m = mask_known * img_truth            # known region kept, hole zeroed
    img_c = (1.0 - mask_known) * img_truth    # complement (hole content)

    scale_img_pyr = scale_pyramid(img_truth, output_scale)
    scale_mask_pyr = scale_pyramid(mask_known, output_scale)
    return {
        "img_truth": img_truth, "img_m": img_m, "img_c": img_c,
        "mask_known": mask_known,
        "scale_img_pyr": scale_img_pyr, "scale_mask_pyr": scale_mask_pyr,
    }


# --------------------------------------------------------------------------------------
# VAE distribution + KL (port of get_distribution, two-path only, non-adversarial).
# --------------------------------------------------------------------------------------
def _get_distribution(distributions, mask_known):
    sum_valid = (torch.mean(mask_known.view(mask_known.size(0), -1), dim=1) - 1e-5).view(-1, 1, 1, 1)
    m_sigma = 1.0 / (1.0 + ((sum_valid - 0.8) * 8).exp())

    p_distribution = q_distribution = None
    kl_rec, kl_g = 0.0, 0.0
    for distribution in distributions:
        p_mu, p_sigma, q_mu, q_sigma = distribution
        m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma))
        p_distribution = torch.distributions.Normal(p_mu, p_sigma)
        p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_sigma.detach())
        q_distribution = torch.distributions.Normal(q_mu, q_sigma)
        kl_rec = kl_rec + torch.distributions.kl_divergence(m_distribution, p_distribution)
        kl_g = kl_g + torch.distributions.kl_divergence(p_distribution_fix, q_distribution)  # two-path
    return p_distribution, q_distribution, kl_rec, kl_g


def _get_G_inputs(p_distribution, q_distribution, f, mask_known):
    f_m = torch.cat([f[-1].chunk(2)[0], f[-1].chunk(2)[0]], dim=0)
    f_e = torch.cat([f[2].chunk(2)[0], f[2].chunk(2)[0]], dim=0)
    scale_mask = scale_img(mask_known, size=[f_e.size(2), f_e.size(3)])
    mask = torch.cat([scale_mask.chunk(3, dim=1)[0], scale_mask.chunk(3, dim=1)[0]], dim=0)
    z_p = p_distribution.rsample()
    z_q = q_distribution.rsample()
    z = torch.cat([z_p, z_q], dim=0)
    return z, f_m, f_e, mask


def forward_train(net_E, net_G, batch):
    """Two-path VAE forward. Returns (img_rec_list, img_g_list, kl_rec, kl_g)."""
    distributions, f = net_E(batch["img_m"], batch["img_c"])
    p_dist, q_dist, kl_rec, kl_g = _get_distribution(distributions, batch["mask_known"])
    z, f_m, f_e, gmask = _get_G_inputs(p_dist, q_dist, f, batch["mask_known"])
    results, _attn = net_G(z, f_m, f_e, gmask)
    img_rec, img_g = [], []
    for result in results:
        rec, gen = result.chunk(2)
        img_rec.append(rec)
        img_g.append(gen)
    return img_rec, img_g, kl_rec, kl_g


def compute_losses(img_rec, img_g, kl_rec, kl_g, batch, lambda_kl, lambda_rec, output_scale):
    """Port of the non-adversarial terms of backward_G. Returns (total, components_dict)."""
    l1 = nn.functional.l1_loss
    loss_kl_rec = kl_rec.mean() * lambda_kl * output_scale
    loss_kl_g = kl_g.mean() * lambda_kl * output_scale

    loss_app_rec, loss_app_g = 0.0, 0.0
    for rec_i, g_i, real_i, mask_i in zip(img_rec, img_g, batch["scale_img_pyr"], batch["scale_mask_pyr"]):
        loss_app_rec = loss_app_rec + l1(rec_i, real_i)
        loss_app_g = loss_app_g + l1(g_i * mask_i, real_i * mask_i)  # two-path: known region
    loss_app_rec = loss_app_rec * lambda_rec
    loss_app_g = loss_app_g * lambda_rec

    total = loss_kl_rec + loss_kl_g + loss_app_rec + loss_app_g
    components = {
        "total": total.item(),
        "kl_rec": loss_kl_rec.item(), "kl_g": loss_kl_g.item(),
        "app_rec": loss_app_rec.item(), "app_g": loss_app_g.item(),
    }
    return total, components


def train_step(net_E, net_G, optimizer, images128, masks_hole128,
               lambda_kl, lambda_rec, output_scale, device):
    """One VAE-only optimization step (KL + multi-scale L1)."""
    batch = preprocess(images128, masks_hole128, output_scale, device)
    img_rec, img_g, kl_rec, kl_g = forward_train(net_E, net_G, batch)
    total, components = compute_losses(img_rec, img_g, kl_rec, kl_g, batch,
                                       lambda_kl, lambda_rec, output_scale)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return components


# --------------------------------------------------------------------------------------
# Validation metric: reuse pic_inpaint (test-time forward) -> composite -> hole L1 / PSNR.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_set(net_E, net_G, images128, masks_hole128, batch_size, device):
    """Mean masked (hole-region) L1 and PSNR over a fixed validation set.

    images128: (N,3,128,128) in [0,1]; masks_hole128: (N,1,128,128), 1 = hole.
    """
    from pic_inference import pic_inpaint  # reused verbatim

    was_training = net_E.training
    net_E.eval(); net_G.eval()

    n = images128.size(0)
    sum_abs, sum_sq, sum_hole = 0.0, 0.0, 0.0
    for start in range(0, n, batch_size):
        imgs = images128[start:start + batch_size].to(device)
        masks = masks_hole128[start:start + batch_size].to(device)
        fills = pic_inpaint(net_E, net_G, imgs, masks, sample_num=1, device=device)[:, 0]
        comp = imgs * (1.0 - masks) + fills * masks
        diff = (comp - imgs).abs()
        hole = masks  # (B,1,H,W); broadcast over the 3 channels
        sum_abs += (diff * hole).sum().item()
        sum_sq += ((comp - imgs) ** 2 * hole).sum().item()
        sum_hole += hole.sum().item() * 3.0  # 3 channels per masked pixel

    if was_training:
        net_E.train(); net_G.train()

    denom = sum_hole + 1e-8
    l1 = sum_abs / denom
    mse = sum_sq / denom
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return l1, psnr


# --------------------------------------------------------------------------------------
# Checkpoint I/O.
# --------------------------------------------------------------------------------------
def _state_dict(net):
    return net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()


def save_checkpoint(net_E, net_G, out_dir, prefix):
    """Save encoder+generator as {prefix}_net_E.pth / {prefix}_net_G.pth."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(_state_dict(net_E), os.path.join(out_dir, f"{prefix}_net_E.pth"))
    torch.save(_state_dict(net_G), os.path.join(out_dir, f"{prefix}_net_G.pth"))


# --------------------------------------------------------------------------------------
# Resume support: find the last saved epoch and (optionally) restore optimizer + history.
# --------------------------------------------------------------------------------------
def find_last_epoch(ckpt_dir):
    """Highest N with both epoch_N_net_E.pth and epoch_N_net_G.pth in ckpt_dir.

    Returns (last_epoch, e_path, g_path); (0, None, None) if none found.
    """
    best_n, best_paths = 0, (None, None)
    for e_path in glob.glob(os.path.join(ckpt_dir, "epoch_*_net_E.pth")):
        m = re.search(r"epoch_(\d+)_net_E\.pth$", os.path.basename(e_path))
        if not m:
            continue
        n = int(m.group(1))
        g_path = e_path[:-len("_net_E.pth")] + "_net_G.pth"
        if n > best_n and os.path.exists(g_path):
            best_n, best_paths = n, (e_path, g_path)
    return best_n, best_paths[0], best_paths[1]


def load_weights(net_E, net_G, e_path, g_path, device):
    """Load encoder/generator state_dicts (stripping a DataParallel 'module.' prefix)."""
    for net, path in [(net_E, e_path), (net_G, g_path)]:
        sd = torch.load(path, map_location=device)
        if len(sd) > 0 and all(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        net.load_state_dict(sd)


def save_train_state(path, optimizer, completed_epochs, history, best_val_l1, best_epoch):
    """Persist optimizer moments + counters + history for a faithful resume."""
    torch.save({
        "optimizer": optimizer.state_dict(),
        "completed_epochs": completed_epochs,
        "history": history,               # dict of epoch / train / val_l1 / val_psnr lists
        "best_val_l1": best_val_l1,
        "best_epoch": best_epoch,
    }, path)


def load_train_state(path, optimizer, device):
    """Restore optimizer state (in place) and return the saved bookkeeping dict."""
    st = torch.load(path, map_location=device)
    if optimizer is not None and "optimizer" in st:
        optimizer.load_state_dict(st["optimizer"])
    return st


# --------------------------------------------------------------------------------------
# Checkpoint discovery: load whatever E/G weights live in a folder (best_ > latest_ > epoch_N).
# --------------------------------------------------------------------------------------
def load_model(ckpt_dir, device):
    """Build net_E/net_G and load the discovered checkpoint. Returns
    (net_E, net_G, e_name, g_name)."""
    from pic_inference import get_pic_inpainter, resolve_checkpoints
    e_path, g_path = resolve_checkpoints(ckpt_dir)
    net_E, net_G = get_pic_inpainter(ckpt_dir, device)
    return net_E, net_G, os.path.basename(e_path), os.path.basename(g_path)
