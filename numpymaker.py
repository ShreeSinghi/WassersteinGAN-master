from PIL import Image
import os
import glob
import numpy as np
import gc
from tqdm import tqdm
def process_folder(folder_path):
  """
  Takes an image, extracts the largest possible center crop, resizes it to 64x64 and saves it.

  Args:
    img_path: Path to the image file.
    save_path: Optional path to save the resized image, defaults to original filename with "_cropped_64.jpg" appended.

  Returns:
    None
  """
  files = os.listdir(folder_path)
  imgs = []
  for file in files:
    if not file.endswith(".jpg"):
      continue
    img = np.array(Image.open(os.path.join(folder_path, file)).resize((64,64))).astype(float)
    imgs.append(img)

  imgs = np.array(imgs)
  gc.collect()
  np.save(os.path.join(folder_path, "data.npy"), imgs)


# Loop through all images in the folder
folder_path = "C:/Users/singh/OneDrive/Documents/GitHub/dcgan/data/"
for folder in tqdm(os.listdir(folder_path)):
  process_folder(os.path.join(folder_path, folder))

print("Done processing all images!")