import urllib.request
import tarfile
import os
import shutil

def download_and_extract():
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    filename = "102flowers.tgz"
    extract_dir = "data"

    print(f"Downloading 102 Category Flower Dataset from {url}...")
    
    urllib.request.urlretrieve(url, filename)
    print("Download complete. Extracting files...")

    # The tarball contains a 'jpg' directory with all the images
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    
    # Move extracted 'jpg' folder to 'data' for consistency
    if os.path.exists('jpg'):
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir) # Clear old data if it exists
        os.rename('jpg', extract_dir)
        print(f"Moved images from 'jpg' to '{extract_dir}' folder.")

    # Clean up tarball
    if os.path.exists(filename):
        os.remove(filename)

    print("Success! The raw dataset is now ready in the 'data' folder.")

if __name__ == "__main__":
    download_and_extract()
