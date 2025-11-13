import os
import cv2
import numpy as np

def list_images(folder, exts=('.jpg','.png','.jpeg','.bmp','.tif')):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    return [os.path.join(folder, f) for f in files]

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image {path}')
    return img

def save_image(path, img):
    cv2.imwrite(path, img)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def save_mask(path, mask):
    # mask is boolean or 0/1
    m = (mask.astype('uint8')) * 255
    cv2.imwrite(path, m)
