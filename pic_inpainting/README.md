# PIC (Pluralistic Image Completion) — VAE inpainting for FlowerPower

Adds a third inpainting paradigm alongside the U-Net GAN and fine-tuned LaMa: a
**conditional VAE** (PIC, Zheng et al., CVPR 2019, [`lyndonzheng/Pluralistic-Inpainting`](https://github.com/lyndonzheng/Pluralistic-Inpainting)).
It encodes the masked image into a Gaussian latent prior, samples `z`, and decodes
in a **single feed-forward pass**, producing **diverse** completions — the advantage
neither the deterministic GAN nor LaMa provides. Used **inference-only, no retraining**.

Feed-forward ⇒ CPU-practical: ~**0.13 s per 256×256 completion** on this CPU-only box
(the E+G nets are only ~6M params), unlike the autoregressive VQ-VAE models
(DSI/ICT/PUT), which take minutes–hours per image on CPU.

## Files
- `pic_repo/` — vendored upstream PIC (git clone, `.git` stripped), like `lama_finetuning/lama_repo/`.
- `pic_inference.py` — `get_pic_inpainter()` + `pic_inpaint()`. Builds only `net_E`/`net_G`
  directly from `model.network` (bypassing `pluralistic_model.py`, which has a Python-3.6-only
  `async=True` syntax error, and skipping the unused discriminators). CPU-safe; `map_location`
  aware; strips a `module.` prefix if the weights were saved from a DataParallel model.
- `evaluate_pic.py` — renders a `Real | Masked | Sample 1..K` figure to
  `evaluations/pic_inpainting/`. Uses torchvision/PIL (not matplotlib, which is broken here
  under numpy 2.x).
- `checkpoints/` — pretrained weights go here (git-ignored via `*.pth`).

## Get the pretrained weights (manual — Drive is rate-limiting `gdown`)
`gdown` currently fails on these files ("Cannot retrieve the public link … many accesses").
Download in a browser from the PIC README and place the files as below.

**ImageNet, random-mask** (best domain + mask match for the flower dataset's large masks):
<https://drive.google.com/open?id=1hS6D4gjOkvEOlAEOAKxCCzjhpCoddU2S>

Place at least these two files here:
```
pic_inpainting/checkpoints/imagenet_random/latest_net_E.pth
pic_inpainting/checkpoints/imagenet_random/latest_net_G.pth
```
(`latest_net_D.pth` / `latest_net_D_rec.pth` are not needed for inference.)
If the download is a folder/zip with a different name, either rename it to
`imagenet_random` or pass `--ckpt_dir <path>` to `evaluate_pic.py`.

Other options from the PIC README: `Places2_random`, `CelebA_random`, or the `*_center`
variants; the Baidu mirror; or retry `gdown <id>` after the Drive quota resets (~24 h).

## Run
```bash
# real flowers (regenerate data_128x128 via GAN_implementation/{download_data,resize_images}.py if absent)
python pic_inpainting/evaluate_pic.py --num_images 4 --sample_num 3

# quick pipeline smoke test on the bundled sample images (any folder works; images are resized to 128)
python pic_inpainting/evaluate_pic.py --data_dir pic_inpainting/pic_repo/datasets/imagenet --num_images 3 --sample_num 3
```

## Key contracts (verified against the vendored source)
- **Mask polarity is inverted vs FlowerPower.** PIC uses `1 = known, 0 = hole`; `shared_utils.get_large_random_mask` uses `1 = hole`. `pic_inpaint` inverts internally.
- **Value range.** PIC works in `[-1, 1]` (`img*2-1`, generator ends in Tanh). The wrapper takes/returns `[0, 1]` to match the FlowerPower pipeline.
- **Resolution.** PIC is 256×256 only; the wrapper upscales 128→256 in and downscales 256→128 out.
- **Composition.** `evaluate_pic.py` composes with the shared rule `comp = masked + fill * mask`, so the known region is preserved exactly (verified: `max|comp−real|` over the known region = 0).
- **Diversity.** `--sample_num K` draws K independent `z` (unseeded) → K different fills.

## Status
Integration written and **validated end-to-end on CPU** (runs, correct composition/mask polarity,
working diversity, ~0.13 s/completion) using random stand-in weights. Swap in the real ImageNet
weights above to get meaningful flower completions — no code changes needed.
