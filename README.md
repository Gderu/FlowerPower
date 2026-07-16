# FlowerPower Inpainting

This repository contains three approaches for image inpainting (filling in masked regions of an image) trained on the Oxford 102 Category Flower Dataset:

1. **Custom DCGAN + U-Net** (built from scratch)
2. **Fine-tuned LaMa (Large Mask Inpainting)** (utilizing Fourier convolutions)
3. **PIC (Pluralistic Image Completion)** (a conditional VAE, used both pretrained and fine-tuned on flowers)

## 1. Setup

### Clone the Repository
```bash
git clone https://github.com/Gderu/FlowerPower.git
cd FlowerPower
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download and Prepare Data
We use the Oxford 102 Flowers dataset. Run this command from the project root to download and resize the images to 128x128.
```bash
python prepare_data.py
```
This will create a `data_128x128/` folder in the project root containing the prepared images.

---

## 2. Model Weights

Pre-trained and fine-tuned weights are hosted in a shared OneDrive directory:

**<https://technionmail-my.sharepoint.com/:f:/r/personal/alon_granek_campus_technion_ac_il/Documents/FlowerPowerWeights?csf=1&web=1&e=7btdcD>**

Each subfolder there maps to a destination in this repository. Download a subfolder's `.pth`
files and place them in the matching directory (create it if it doesn't exist):

| Weights subfolder | Copy its `.pth` files to |
|---|---|
| `GAN` | `checkpoints/gan/` |
| `LaMa - finetuned` | `checkpoints/lama/` |
| `PIN - pretrained` | `pic_inpainting/checkpoints/pretrained/` |
| `PIN - finetuned`  | `pic_inpainting/checkpoints/finetuned/` |

> **Note:** the weights directory labels PIC as **PIN**; both refer to the same
> Pluralistic Image Completion model in [`pic_inpainting/`](pic_inpainting/).

Weights are not tracked in git, so this download is required before running any approach whose
weights live here. The PIC tooling loads whichever checkpoint pair it finds in a folder
(preferring `best_*`, then `latest_*`, then the highest `epoch_N_*`), so no renaming is needed.

LaMa's pretrained weights are the exception — they are fetched automatically by
`lama_finetuning/setup_lama.py` (see below).

---

## 3. Approach A: Custom GAN (DCGAN + U-Net)

This approach uses a U-Net based Generator and a custom Discriminator. It is trained from scratch.

### Train the GAN
```bash
python GAN_implementation/main.py
```
- Checkpoints will be saved to `checkpoints/gan/`.
- Training loss curves will be saved to `evaluations/gan/loss_curve.png`.

### Evaluate the GAN
```bash
python GAN_implementation/evaluate.py
```
- This will automatically load the latest checkpoint from `checkpoints/gan/`.
- To evaluate a specific checkpoint, use the `--checkpoint` flag (e.g., `python GAN_implementation/evaluate.py --checkpoint path/to/model.pth`).
- Evaluation samples will be saved to `evaluations/gan/`.

---

## 4. Approach B: Fine-Tuning LaMa

This approach fine-tunes a pretrained Big-LaMa model. It requires the official LaMa repository and its pretrained weights.

### LaMa Setup
Run the automated setup script to clone the LaMa repository, install its specific dependencies, and download the pretrained weights (~200MB):
```bash
python lama_finetuning/setup_lama.py
```

### Fine-Tune LaMa
```bash
python lama_finetuning/fine_tune_lama.py
```
- Checkpoints will be saved to `checkpoints/lama/`.

### Evaluate LaMa
```bash
python lama_finetuning/evaluate_lama.py
```
- This will automatically load the latest fine-tuned checkpoint from `checkpoints/lama/`.
- To evaluate a specific checkpoint, use the `--checkpoint` flag (e.g., `python lama_finetuning/evaluate_lama.py --checkpoint path/to/model.pth`).
- Evaluation samples will be saved to `evaluations/lama/`.

---

## 5. Approach C: PIC (Pluralistic Image Completion)

This approach uses a **conditional VAE** ([Zheng et al., CVPR 2019](https://github.com/lyndonzheng/Pluralistic-Inpainting)).
It encodes the masked image into a Gaussian latent prior, samples a latent `z`, and decodes in a
single feed-forward pass. Drawing several `z` yields **diverse** completions of the *same* masked
input — something neither the deterministic GAN nor LaMa provides. Being feed-forward keeps it
CPU-practical (~0.1s per completion).

It is used two ways: with the original **pretrained** ImageNet weights, and with weights
**fine-tuned** on the flowers dataset (VAE objective: KL + multi-scale L1, no discriminators).

Both require the weights from the [Model Weights](#2-model-weights) section above.

### Run the pretrained model
```bash
python pic_inpainting/evaluate_pic.py --num_images 4 --sample_num 3
```
- Loads `pic_inpainting/checkpoints/pretrained/`.
- Writes a `Real | Masked | Sample 1..K` grid to `evaluations/pic_inpainting/`.
- `--sample_num K` draws K independent completions per image, showing the diversity.

### Run the fine-tuned model (and compare it to the pretrained one)
```bash
python pic_inpainting/finetuning/run_finetuned.py
```
- Loads both `checkpoints/pretrained/` and `checkpoints/finetuned/` and runs them on the same
  images with the same masks.
- Writes `compare_pretrained.png` and `compare_finetuned.png` to `evaluations/pic_finetuning/`,
  and prints a hole-L1 / PSNR table.

To point the standalone evaluator at any checkpoint folder:
```bash
python pic_inpainting/evaluate_pic.py --ckpt_dir pic_inpainting/checkpoints/finetuned
```

### Fine-tune it yourself (optional)
Fine-tuning runs in Google Colab on a GPU, not locally.
```bash
python pic_inpainting/finetuning/build_bundle.py
```
- Produces `pic_inpainting/finetuning/pic_finetune_bundle.zip` (~60MB: code, pretrained weights, dataset).
- Upload it to Colab, open `pic_inpainting/finetuning/pic_finetune.ipynb`, set the runtime to **GPU**, and
  run all cells. Weights land in `checkpoints/finetuned/`; download them back into
  `pic_inpainting/checkpoints/finetuned/`.

See [`pic_inpainting/README.md`](pic_inpainting/README.md) and
[`pic_inpainting/finetuning/README.md`](pic_inpainting/finetuning/README.md) for details.

## 6. Unified Evaluation & Visualizations

If you have downloaded all model weights, you can evaluate and visualize all models simultaneously:

### Quantitative Evaluation
Calculates PSNR, SSIM, and FID for all models on the dataset:
```bash
python evaluations/evaluate_models.py
```
*(By default, this evaluates the entire dataset. You can limit it with `--num_images 100` if you edit the script, or just let it run).*

### Qualitative Visualization
Generates a side-by-side comparison grid (`Real | Masked | GAN | Base LaMa | FT LaMa | Base PIC | FT PIC`) and saves it to `evaluations/results/inpainting_examples.png`:
```bash
python evaluations/visualize_models.py
```

---

## Project Structure

```text
FlowerPower/
├── checkpoints/             # GAN + LaMa weights (generated during training)
│   ├── big-lama/            # Downloaded by setup_lama.py
│   ├── gan/                 # GAN training checkpoints
│   └── lama/                # Fine-tuned LaMa checkpoints
├── data_128x128/            # Prepared dataset (generated by prepare_data.py)
├── evaluations/             # Generated output samples and loss curves
│   ├── gan/
│   ├── lama/
│   ├── pic_inpainting/      # PIC completion grids
│   └── pic_finetuning/      # Pretrained-vs-fine-tuned comparisons
├── GAN_implementation/      # Approach A: Custom GAN code
├── lama_finetuning/         # Approach B: LaMa fine-tuning code
│   └── lama_repo/           # Official LaMa codebase (cloned by setup)
├── pic_inpainting/          # Approach C: PIC inference code
│   ├── checkpoints/         # PIC weights (downloaded, see Model Weights)
│   │   ├── pretrained/      # Original PIC weights ("PIN - pretrained")
│   │   └── finetuned/       # Fine-tuned on flowers ("PIN - finetuned")
│   ├── pic_repo/            # Official PIC codebase (vendored)
│   └── finetuning/          # PIC fine-tuning code (Colab notebook + bundle)
├── prepare_data.py          # Downloads and resizes the flowers dataset
├── shared_utils.py          # Shared dataset, discriminator, and loss logic
└── README.md
```

> **Note:** PIC's weights live under `pic_inpainting/checkpoints/`, not the top-level
> `checkpoints/` used by the GAN and LaMa.
