"""In-process inference wrapper for PIC (Pluralistic Image Completion).

PIC (Zheng et al., CVPR 2019, repo `lyndonzheng/Pluralistic-Inpainting`) is a
conditional VAE: it encodes the masked image into a Gaussian latent prior, samples
`z`, and decodes -- in a single feed-forward pass -- producing *diverse* completions.
This makes it CPU-practical (a few seconds/image), unlike the autoregressive
VQ-VAE models (DSI/ICT/PUT).

This module reproduces PIC's test-time forward WITHOUT importing
`model/pluralistic_model.py` (which contains a Python-3.6-only `async=True`
kwarg that is a syntax error on Python 3.7+) and without loading the
discriminators. It builds only the encoder (`net_E`) and generator (`net_G`)
directly from `model.network`, matching the exact hyper-parameters used in
`pluralistic_model.Pluralistic.__init__` (lines 36-39 of that file).

Key contracts (verified against the vendored PIC source):
  * PIC mask polarity is INVERTED vs FlowerPower: PIC uses 1 = KNOWN, 0 = HOLE,
    whereas `shared_utils.get_large_random_mask` uses 1 = HOLE. We invert here.
  * PIC works internally in [-1, 1] (`img_truth = img*2 - 1`); the generator ends
    in Tanh. We accept/return [0, 1] to match the FlowerPower pipeline.
  * PIC operates at 256x256; we upscale 128->256 in, and downscale 256->128 out.
"""

import os
import sys
import contextlib

import torch
import torch.nn.functional as F

# --- make the vendored PIC package importable ---
_PIC_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pic_repo")
if _PIC_REPO not in sys.path:
    sys.path.insert(0, _PIC_REPO)

from model import network  # noqa: E402  (imports cleanly: torch-only + util.util)

# Exact architecture hyper-parameters from
# pic_repo/model/pluralistic_model.py __init__ (net_E line 36-37, net_G line 38-39).
_NGF = 32
_Z_NC = 128
_IMG_F = 128
_LAYERS = 5
_OUTPUT_SCALE = 4


def _strip_module_prefix(state_dict):
    """Pretrained PIC weights were saved from a (DataParallel) GPU model, so keys
    may be prefixed with 'module.'. Strip it so they load into the bare network."""
    if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_net(net, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "PIC checkpoint not found: %s\n"
            "Download the 'Imagenet_random' model from the PIC README (Google Drive) "
            "and place latest_net_E.pth / latest_net_G.pth under the checkpoints dir." % path
        )
    state_dict = torch.load(path, map_location=device)
    state_dict = _strip_module_prefix(state_dict)
    net.load_state_dict(state_dict)  # strict: architecture is reproduced exactly
    return net


def get_pic_inpainter(ckpt_dir, device=None):
    """Build net_E + net_G and load pretrained weights.

    Args:
        ckpt_dir: folder containing `latest_net_E.pth` and `latest_net_G.pth`.
        device:   torch.device; defaults to cuda-if-available-else-cpu.
    Returns:
        (net_E, net_G) in eval mode on `device`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define_e / define_g print the full network + init logs; silence them.
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        net_E = network.define_e(
            ngf=_NGF, z_nc=_Z_NC, img_f=_IMG_F, layers=_LAYERS,
            norm="none", activation="LeakyReLU", init_type="orthogonal", gpu_ids=[],
        )
        net_G = network.define_g(
            ngf=_NGF, z_nc=_Z_NC, img_f=_IMG_F, L=0, layers=_LAYERS,
            output_scale=_OUTPUT_SCALE, norm="instance", activation="LeakyReLU",
            init_type="orthogonal", gpu_ids=[],
        )

    _load_net(net_E, os.path.join(ckpt_dir, "latest_net_E.pth"), device)
    _load_net(net_G, os.path.join(ckpt_dir, "latest_net_G.pth"), device)
    net_E.to(device).eval()
    net_G.to(device).eval()
    return net_E, net_G


@torch.no_grad()
def pic_inpaint(net_E, net_G, images, masks, sample_num=1, device=None):
    """Run PIC's feed-forward pluralistic completion.

    Args:
        images: (B, 3, 128, 128) float in [0, 1]  (FlowerPower convention).
        masks:  (B, 1, 128, 128) float, 1 = HOLE / 0 = keep (FlowerPower convention).
        sample_num: number of diverse completions to draw per image.
    Returns:
        (B, sample_num, 3, 128, 128) float in [0, 1] -- the raw generator fills
        (compose against the known region on the caller side).
    """
    if device is None:
        device = next(net_G.parameters()).device
    images = images.to(device)
    masks = masks.to(device)

    # 128 -> 256, and invert mask polarity to PIC's (1 = known).
    img256 = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=True)
    mask_hole256 = F.interpolate(masks, size=(256, 256), mode="nearest")  # 1 = hole
    mask_pic = (1.0 - mask_hole256).repeat(1, 3, 1, 1)                    # 1 = known, 3ch

    img_truth = img256 * 2.0 - 1.0        # [0,1] -> [-1,1]
    img_m = mask_pic * img_truth          # masked image (hole zeroed), like PIC set_input

    # Encoder: one-path (test) -> distribution list [[mu, sigma]] and feature pyramid f.
    distribution, f = net_E(img_m)
    mu, sigma = distribution[-1]
    q_distribution = torch.distributions.Normal(mu, sigma)

    # Attention mask at the f[2] feature resolution (first channel), as in PIC test().
    scale_mask = F.interpolate(
        mask_pic, size=(f[2].size(2), f[2].size(3)), mode="bilinear", align_corners=True
    )[:, :1]

    outs = []
    for _ in range(sample_num):
        z = q_distribution.sample()
        results, _attn = net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask)
        out256 = (results[-1] + 1.0) / 2.0                                # [-1,1] -> [0,1]
        out128 = F.interpolate(out256, size=(128, 128), mode="bilinear", align_corners=True)
        outs.append(out128.clamp(0.0, 1.0))

    return torch.stack(outs, dim=1)  # (B, sample_num, 3, 128, 128)


if __name__ == "__main__":
    # Tiny self-check with random input (no checkpoint needed for shape sanity).
    import argparse

    parser = argparse.ArgumentParser(description="PIC inference shape smoke test")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="folder with latest_net_E.pth / latest_net_G.pth")
    parser.add_argument("--sample_num", type=int, default=2)
    args = parser.parse_args()

    dev = torch.device("cpu")
    if args.ckpt_dir:
        E, G = get_pic_inpainter(args.ckpt_dir, dev)
        imgs = torch.rand(1, 3, 128, 128)
        msk = torch.zeros(1, 1, 128, 128)
        msk[:, :, 32:96, 32:96] = 1.0  # center hole
        out = pic_inpaint(E, G, imgs, msk, sample_num=args.sample_num, device=dev)
        print("output shape:", tuple(out.shape), "range:", (out.min().item(), out.max().item()))
    else:
        print("Provide --ckpt_dir to run a real forward pass.")
