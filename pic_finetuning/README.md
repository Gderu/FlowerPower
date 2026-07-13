# PIC Fine-Tuning on Flowers

Fine-tune the **PIC** (Pluralistic Image Completion) inpainting model — currently used with its
downloaded **ImageNet** weights — on the Oxford-102 **flowers** dataset, and compare
*pretrained vs. fine-tuned* completions.

**Objective:** VAE-only (KL + multi-scale L1, no discriminators) — a faithful port of the
non-adversarial terms of PIC's original training loop (`pic_repo/model/pluralistic_model.py`),
using the same encoder/generator (`define_e`/`define_g`) and the same loss weights
(`lambda_kl=20, lambda_rec=20`) and Adam `betas=(0, 0.999)`. Stable for a short fine-tune;
results are softer than the pretrained model (the sharpening adversarial term is intentionally
omitted). Training runs at 256×256 to match the pretrained architecture and the inference
wrapper's forward pass.

## Files

| File | Purpose |
|------|---------|
| `pic_finetune.ipynb` | The fine-tuning notebook (Colab-first). |
| `pic_finetune_core.py` | Ported VAE training step, validation metric, checkpoint I/O. |
| `build_bundle.py` | Assembles `pic_finetune_bundle.zip` (code + weights + data). |

## Workflow

1. **Build the bundle** (from the repo root):
   ```bash
   python pic_finetuning/build_bundle.py
   ```
   Produces `pic_finetuning/pic_finetune_bundle.zip` (~80 MB: code, the pretrained
   `latest_net_{E,G}.pth`, and all 8,189 flower images).

2. **Fine-tune in Colab:** open `pic_finetune.ipynb`, set the runtime to **GPU**, upload the
   zip, and run all cells. The notebook:
   - maintains a **validation-loss curve updated every epoch**;
   - **saves weights each epoch** (`epoch_{n}_net_{E,G}.pth`) plus `latest_*`, **`best_*`**
     (lowest validation hole-region L1), and a resumable `train_state.pt`;
   - **runs inference on a fixed validation image every few epochs** and displays it;
   - ends with a **pretrained-vs-fine-tuned** grid and a metric table (hole-L1 / PSNR).

   **Train longer / resume:**
   - *Same session* — just re-run the **training-loop cell**; it continues from
     `completed_epochs` and trains `EPOCHS_TO_RUN` more (history and epoch numbering are not
     reset).
   - *New session / after a disconnect* — set `RESUME = True` (and `RESUME_DIR` to the folder
     holding your `epoch_N_net_{E,G}.pth`, default `checkpoints/flowers_finetuned`). The build
     cell loads the highest epoch and continues from N+1; if a `train_state.pt` sits alongside,
     the optimizer moments and loss history are restored too (otherwise it resumes weights-only
     with a fresh optimizer).

3. **Use the fine-tuned weights.** Download `checkpoints/flowers_finetuned/best_net_{E,G}.pth`
   from Colab. To run them through the existing evaluator, place them in a folder as
   `latest_net_{E,G}.pth` and point the evaluator at it:
   ```bash
   python pic_inpainting/evaluate_pic.py --ckpt_dir <that_folder>
   ```

## Notes

- The VAE-only objective supervises the *generation* path on the known region and the
  *reconstruction* path full-image; hole content stays diverse (the pluralistic property).
- Discriminators were **not** shipped with the ImageNet weights, so the adversarial terms are
  intentionally dropped rather than trained from scratch.
- `pic_finetune_core.py` reuses `pic_inference.pic_inpaint` verbatim for validation and all
  previews, so the metric goes through the exact inference path used for both models.
