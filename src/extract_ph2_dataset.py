# src/extract_ph2_dataset.py
import os
import shutil

# === EDIT THIS PATH to where you extracted the PH2 archive ===
SOURCE_ROOT = r"C:\Users\Hemila Saravanan\Downloads\PH2Dataset\PH2 Dataset images"

# Destination inside this project (will be created if not present)
DEST_IMAGES = os.path.join("data", "PH2", "images")
DEST_MASKS  = os.path.join("data", "PH2", "masks")

os.makedirs(DEST_IMAGES, exist_ok=True)
os.makedirs(DEST_MASKS, exist_ok=True)

copied_images = 0
copied_masks = 0

# Walk recursively through all subfolders
for root, dirs, files in os.walk(SOURCE_ROOT):
    for fname in files:
        if fname.lower().endswith(".bmp"):
            src_file = os.path.join(root, fname)
            # Lesion masks end with '_lesion.bmp'
            if fname.lower().endswith("_lesion.bmp"):
                dst_file = os.path.join(DEST_MASKS, fname)
                shutil.copy2(src_file, dst_file)
                copied_masks += 1
            else:
                dst_file = os.path.join(DEST_IMAGES, fname)
                shutil.copy2(src_file, dst_file)
                copied_images += 1

print(f"Copied {copied_images} images -> {DEST_IMAGES}")
print(f"Copied {copied_masks} masks  -> {DEST_MASKS}")
