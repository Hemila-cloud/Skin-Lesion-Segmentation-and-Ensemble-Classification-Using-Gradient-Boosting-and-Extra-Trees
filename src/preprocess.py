import cv2
import numpy as np
from .utils import save_image, save_mask, ensure_dir
import os

def dull_razor(img_bgr, bh_kernel_size=15, threshold_val=10):
    """
    approximate DullRazor: black-hat -> threshold -> dilate -> inpaint
    returns inpainted image (BGR) and hair_mask (uint8 0/255)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bh_kernel_size, bh_kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold_val, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3,3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(img_bgr, hair_mask, 3, cv2.INPAINT_TELEA)
    return inpainted, hair_mask

def minRGB(img_bgr):
    # returns 2D uint8 (min of B,G,R channels)
    return np.min(img_bgr, axis=2).astype('uint8')

def adaptive_init_contour(minrgb_img):
    """
    Adaptive threshold to get initial mask. Use Otsu if adaptive fails.
    """
    # apply gaussian blur then adaptive threshold
    blur = cv2.GaussianBlur(minrgb_img, (5,5), 0)
    # use adaptive mean threshold (blockSize 35 - odd)
    try:
        init = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 35, 5)
        if init.sum() == 0:  # fallback
            _, init = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        _, init = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (init > 0).astype('uint8')  # 0/1 mask

def preprocess_and_init(in_path, out_preproc_path, out_initmask_path, save_preproc=True):
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    inpainted, hair_mask = dull_razor(img)
    minrgb = minRGB(inpainted)
    init_mask = adaptive_init_contour(minrgb)
    if save_preproc:
        ensure_dir(os.path.dirname(out_preproc_path))
        save_image(out_preproc_path, minrgb)
    ensure_dir(os.path.dirname(out_initmask_path))
    save_mask(out_initmask_path, init_mask)
    return minrgb, init_mask
