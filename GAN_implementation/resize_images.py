import os
from PIL import Image
from pathlib import Path

def resize_images(input_dir, output_dir, size=(128, 128)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    errors = 0
    
    print(f"Finding images in {input_path}...")
    
    # Iterate through all files in the input directory
    for img_file in input_path.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            try:
                with Image.open(img_file) as img:
                    # Convert to RGB if needed for JPEG saving
                    if img.mode != 'RGB' and img_file.suffix.lower() in ['.jpg', '.jpeg']:
                        img = img.convert('RGB')
                        
                    # Resize the image
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    
                    # Save the resized image to the output directory
                    output_file = output_path / img_file.name
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
    current_dir = Path(__file__).parent
    input_directory = current_dir / "data"
    output_directory = current_dir / "data_128x128"
    
    print(f"Starting image resize from {input_directory} to {output_directory}...")
    resize_images(input_directory, output_directory)
    print("Done!")
