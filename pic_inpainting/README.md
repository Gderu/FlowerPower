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
- `checkpoints/` — weights go here (git-ignored via `*.pth`):
  - `pretrained/` — the original PIC ImageNet weights (`latest_net_{E,G}.pth`).
  - `finetuned/` — fine-tuned on flowers by `finetuning/` (see that folder's README).

## Get the weights
Download them from the shared weights directory — see **Model Weights** in the
[root README](../README.md#2-model-weights). The `PIN - pretrained` and `PIN - finetuned`
subfolders map to `checkpoints/pretrained/` and `checkpoints/finetuned/` respectively
("PIN" is that directory's label for PIC).

`get_pic_inpainter` resolves whichever E/G pair is in the folder (`best_` > `latest_` >
highest `epoch_N`), so the files need no renaming. Only the encoder/generator are used;
`*_net_D.pth` / `*_net_D_rec.pth` are not needed for inference.

The ImageNet, random-mask variant is the best domain + mask match for the flower dataset's
large masks. Other variants from the [upstream PIC README](https://github.com/lyndonzheng/Pluralistic-Inpainting)
(`Places2_random`, `CelebA_random`, the `*_center` variants) also load — pass `--ckpt_dir <path>`.

## Run
```bash
# pretrained, on real flowers (run `python prepare_data.py` from the repo root if data_128x128/ is absent)
python pic_inpainting/evaluate_pic.py --num_images 4 --sample_num 3

# the fine-tuned weights (or any other checkpoint folder)
python pic_inpainting/evaluate_pic.py --ckpt_dir pic_inpainting/checkpoints/finetuned

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
Integration **validated end-to-end on CPU** with the real ImageNet weights: correct
composition/mask polarity, working diversity, ~0.1 s/completion. The fine-tuned weights from
`finetuning/` load through the same path and improve hole-L1 / PSNR over the pretrained
model — see `evaluations/pic_finetuning/`.
