"""Setup script for LaMa fine-tuning dependencies.
Clones the LaMa repository and downloads pretrained big-lama weights.
"""

import os
import subprocess
import sys
import zipfile
import urllib.request

def setup_lama():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    lama_repo_dir = os.path.join(script_dir, 'lama_repo')
    weights_dir = os.path.join(project_root, 'checkpoints', 'big-lama', 'models')
    weights_path = os.path.join(weights_dir, 'best.ckpt')
    
    # Step 1: Clone LaMa repo
    if os.path.exists(os.path.join(lama_repo_dir, 'saicinpainting')):
        print("LaMa repo already exists, skipping clone.")
    else:
        print("Cloning LaMa repository...")
        subprocess.run([
            'git', 'clone', 'https://github.com/advimman/lama.git', lama_repo_dir
        ], check=True)
        print("LaMa repo cloned successfully.")
    
    # Step 2: Download pretrained big-lama weights from Hugging Face
    if os.path.exists(weights_path):
        print(f"Pretrained weights already exist at {weights_path}, skipping download.")
    else:
        zip_url = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
        zip_path = os.path.join(project_root, "big-lama.zip")
        
        print(f"Downloading pretrained big-lama weights from Hugging Face...")
        print("(This is ~200MB and may take a few minutes)")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(os.path.join(project_root, 'checkpoints'))
        
        # Clean up zip
        os.remove(zip_path)
        
        if os.path.exists(weights_path):
            print(f"Pretrained weights extracted to {weights_path}")
        else:
            print("WARNING: Extraction finished but best.ckpt not found at expected path.")
            print(f"Expected: {weights_path}")
            print("You may need to manually locate the checkpoint file.")

    print("\nLaMa setup complete! You can now run:")
    print("  python lama_finetuning/fine_tune_lama.py")

if __name__ == "__main__":
    setup_lama()
