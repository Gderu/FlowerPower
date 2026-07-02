import json

with open('main.py', 'r') as f:
    main_code = f.read()

# Split the code into sections based on existing headers
sections = {}

# Imports & Dataset
import_end = main_code.find('class ImageDataset')
dataset_end = main_code.find('# --- GENERATOR (U-NET) ---')
sections['Imports & Dataset'] = main_code[:dataset_end]

# Generator
gen_end = main_code.find('# --- DISCRIMINATOR ---')
sections['Generator'] = main_code[dataset_end:gen_end]

# Discriminator
disc_end = main_code.find('# --- UTILS ---')
sections['Discriminator'] = main_code[gen_end:disc_end]

# Utils
utils_end = main_code.find('if __name__ == "__main__":')
sections['Utilities'] = main_code[disc_end:utils_end]

# Training Setup and Loop
# We want to remove the 'if __name__ == "__main__":' and dedent the rest.
training_code = main_code[utils_end:]
lines = training_code.split('\n')
training_code_clean = ""
for line in lines[1:]: # skip 'if __name__ == "__main__":'
    if line.startswith("    "):
        training_code_clean += line[4:] + "\n"
    else:
        training_code_clean += line + "\n"
sections['Training Loop'] = training_code_clean

cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DCGAN Inpainting Training\n",
    "\n",
    "First, make sure your runtime is set to GPU (**Runtime** > **Change runtime type** > **Hardware accelerator** > **T4 GPU**).\n",
    "\n",
    "Upload the `project_colab.zip` file to the Colab environment using the files tab on the left, then run the cells below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!unzip -n -q project_colab.zip\n"
   ]
  }
]

for title, content in sections.items():
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"### {title}\n"]
    })
    
    # Split content by newline and append \n to each line for source array
    lines = content.split('\n')
    content_lines = [line + '\n' for line in lines[:-1]]
    if len(lines) > 0 and lines[-1]:
        content_lines.append(lines[-1])
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content_lines
    })

notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Colab_Training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
