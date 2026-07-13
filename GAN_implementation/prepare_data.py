import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from PIL import Image

def download_and_extract(project_root: str):
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    filename = os.path.join(project_root, "102flowers.tgz")
    extract_dir = os.path.join(project_root, "data")

    print(f"Downloading 102 Category Flower Dataset from {url}...")
    
    urllib.request.urlretrieve(url, filename)
    print("Download complete. Extracting files...")

    # The tarball contains a 'jpg' directory with all the images
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=project_root)
    
    # Move extracted 'jpg' folder to 'data' for consistency
    jpg_dir = os.path.join(project_root, 'jpg')
    if os.path.exists(jpg_dir):
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.rename(jpg_dir, extract_dir)
        print(f"Dataset extracted to '{extract_dir}'")

    # Clean up tarball
    if os.path.exists(filename):
        os.remove(filename)

    print("Done! Raw dataset is ready in the 'data' folder.")

def resize_images(input_dir: Path, output_dir: Path, size=(128, 128)):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    errors = 0
    
    print(f"Finding images in {input_dir}...")
    
    # Iterate through all files in the input directory
    for img_file in input_dir.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            try:
                with Image.open(img_file) as img:
                    # Convert to RGB if needed for JPEG saving
                    if img.mode != 'RGB' and img_file.suffix.lower() in ['.jpg', '.jpeg']:
                        img = img.convert('RGB')
                        
                    # Resize the image
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    
                    # Save the resized image to the output directory
                    output_file = output_dir / img_file.name
                    resized_img.save(output_file)
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} images...")
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                errors += 1

    print(f"Successfully resized {count} images.")
    if errors > 0:
        print(f"Encountered errors on {errors} images.")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    
    print("=== Step 1: Downloading and Extracting Data ===")
    download_and_extract(str(project_root))
    
    print("\n=== Step 2: Resizing Images ===")
    input_directory = project_root / "data"
    output_directory = project_root / "data_128x128"
    
    print(f"Starting image resize from {input_directory} to {output_directory}...")
    resize_images(input_directory, output_directory)
    
    print("\nData preparation complete!")
