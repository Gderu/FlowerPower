import urllib.request
import tarfile
import os
import shutil

def download_and_extract():
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
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

if __name__ == "__main__":
    download_and_extract()
