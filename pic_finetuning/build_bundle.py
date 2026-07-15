"""Assemble `pic_finetune_bundle.zip` — everything `pic_finetune.ipynb` needs in a flat
layout: fine-tuning code, the PIC model source, the pretrained ImageNet weights, and the
full flowers dataset.

Bundle layout (what the notebook expects at its working directory):
    shared_utils.py
    pic_inference.py
    pic_finetune_core.py
    pic_repo/model/*.py, pic_repo/util/*.py   (PIC architecture; imported as `model`/`util`)
    checkpoints/pretrained/latest_net_{E,G}.pth
    data_128x128/*.jpg                         (8,189 flowers, 128x128)

Run:  python pic_finetuning/build_bundle.py
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

DATA_SRC = os.path.join(REPO, "data_128x128")
CKPT_SRC = os.path.join(REPO, "pic_inpainting", "checkpoints", "pretrained")
PIC_REPO = os.path.join(REPO, "pic_inpainting", "pic_repo")

STAGE = os.path.join(HERE, "_bundle_stage")
ZIP_BASENAME = os.path.join(HERE, "pic_finetune_bundle")  # -> pic_finetune_bundle.zip

_ignore_pyc = shutil.ignore_patterns("__pycache__", "*.pyc")


def _require(path, hint):
    if not os.path.exists(path):
        sys.exit(f"ERROR: missing {path}\n  {hint}")


def main():
    _require(DATA_SRC, "Regenerate via `python prepare_data.py` from the repo root")
    for name in ["latest_net_E.pth", "latest_net_G.pth"]:
        _require(os.path.join(CKPT_SRC, name),
                 "Download the pretrained PIC weights (see 'Model Weights' in the root README)")

    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    # --- single-file code ---
    shutil.copy(os.path.join(REPO, "shared_utils.py"), STAGE)
    shutil.copy(os.path.join(REPO, "pic_inpainting", "pic_inference.py"), STAGE)
    shutil.copy(os.path.join(HERE, "pic_finetune_core.py"), STAGE)

    # --- PIC model source (imported as top-level `model` / `util`) ---
    shutil.copytree(os.path.join(PIC_REPO, "model"),
                    os.path.join(STAGE, "pic_repo", "model"), ignore=_ignore_pyc)
    shutil.copytree(os.path.join(PIC_REPO, "util"),
                    os.path.join(STAGE, "pic_repo", "util"), ignore=_ignore_pyc)

    # --- pretrained weights ---
    dst_ckpt = os.path.join(STAGE, "checkpoints", "pretrained")
    os.makedirs(dst_ckpt)
    for name in ["latest_net_E.pth", "latest_net_G.pth"]:
        shutil.copy(os.path.join(CKPT_SRC, name), dst_ckpt)

    # --- data ---
    print("copying data (8,189 images, ~59MB)...")
    shutil.copytree(DATA_SRC, os.path.join(STAGE, "data_128x128"))

    # --- zip ---
    print("zipping...")
    if os.path.exists(ZIP_BASENAME + ".zip"):
        os.remove(ZIP_BASENAME + ".zip")
    shutil.make_archive(ZIP_BASENAME, "zip", STAGE)
    shutil.rmtree(STAGE)

    size_mb = os.path.getsize(ZIP_BASENAME + ".zip") / 1e6
    print(f"done -> {ZIP_BASENAME}.zip  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
