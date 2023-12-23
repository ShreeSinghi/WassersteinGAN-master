from PIL import Image
import os
import shutil
import glob
import numpy as np
import gc
from tqdm import tqdm
def process_folder(folder_path, final_path):
  """
  Takes an image, extracts the largest possible center crop, resizes it to 64x64 and saves it.

  Args:
    img_path: Path to the image file.
    save_path: Optional path to save the resized image, defaults to original filename with "_cropped_64.jpg" appended.

  Returns:
    None
  """
  files = list(filter(lambda x: x.endswith(".jpg"), os.listdir(folder_path)))
  base = os.path.join(final_path, folder_path.split("/")[-1])

  for i in range(len(files)+1-n):
    dirname = f"{base}_{i:03d}"
    os.makedirs(dirname, exist_ok=True)
    for j in range(n):
      shutil.copyfile(os.path.join(folder_path,files[i+j]), os.path.join(dirname, f"{j}.jpg"))

# Loop through all images in the folder
folder_path = "C:/Users/singh/OneDrive/Documents/GitHub/dcgan/data/"
n=5
final_path = "video_5"
for folder in tqdm(os.listdir(folder_path)):
    process_folder(os.path.join(folder_path, folder), final_path)

print("Done processing all images!")