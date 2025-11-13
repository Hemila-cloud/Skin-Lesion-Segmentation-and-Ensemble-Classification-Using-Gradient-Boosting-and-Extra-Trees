import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .utils import save_mask, ensure_dir
import os
from tqdm import tqdm

def compute_histogram_probs(values, nbins=64, eps=1e-8):
    hist, edges = np.histogram(values, bins=nbins, range=(0,255), density=False)
    # add small value then normalize to probability
    probs = hist.astype(np.float64) + eps
    probs /= probs.sum()
    return probs, edges

def intensity_pdf_map(img_flat, probs, edges):
    # return pdf values for flattened intensities
    bin_idx = np.searchsorted(edges, img_flat, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, len(probs)-1)
    return probs[bin_idx]

def extract_border(mask):
    # returns binary border mask (1 at boundary)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask.astype('uint8'), kernel, iterations=1)
    border = mask.astype('uint8') - eroded
    return (border>0).astype('uint8')

def kl_ls_refine(minrgb_img, init_mask, iterations=100, alpha=1.0, nbins=64, save_intermediate=False, out_folder=None):
    """
    Practical KL-LS refinement:
    - minrgb_img: 2D uint8 image
    - init_mask: binary 0/1 mask
    - iterations: number of refinement steps
    - alpha: weight of KL force (paper used alpha=1.0)
    Returns final_mask (0/1)
    """
    img = minrgb_img.copy()
    H, W = img.shape
    mask = init_mask.astype('uint8').copy()  # 0/1
    # Precompute flattened intensities
    flat = img.flatten()
    for it in range(iterations):
        # get inside/outside pixels
        inside_vals = img[mask==1]
        outside_vals = img[mask==0]

        # if either region empty, break
        if inside_vals.size == 0 or outside_vals.size == 0:
            print("Empty region reached at iter", it)
            break

        # estimate PDFs (histogram + smoothing)
        p_in, edges = compute_histogram_probs(inside_vals, nbins=nbins)
        p_out, _ = compute_histogram_probs(outside_vals, nbins=nbins)
        # smooth pdfs
        p_in = gaussian_filter(p_in, sigma=1)
        p_out = gaussian_filter(p_out, sigma=1)
        p_in = p_in / (p_in.sum())
        p_out = p_out / (p_out.sum())

        # compute per-pixel KL-like force = log(p_in / p_out) mapped to pixels
        pdf_in_map = intensity_pdf_map(flat, p_in, edges).reshape(H,W)
        pdf_out_map = intensity_pdf_map(flat, p_out, edges).reshape(H,W)
        # avoid division by zero
        ratio = (pdf_in_map + 1e-9) / (pdf_out_map + 1e-9)
        kl_force_map = np.log(ratio)  # positive -> favors inside; negative -> favors outside

        # compute border pixels
        border = extract_border(mask)
        ys, xs = np.where(border==1)
        if ys.size == 0:
            break

        # For each border pixel decide to flip according to kl_force + curvature smoothing
        # compute curvature approx via distance transform difference
        # We'll compute a small local curvature by comparing number of inside neighbors
        new_mask = mask.copy()
        for y,x in zip(ys,xs):
            # local 3x3 neighborhood
            y0, y1 = max(0,y-1), min(H-1,y+1)
            x0, x1 = max(0,x-1), min(W-1,x+1)
            neigh = mask[y0:y1+1, x0:x1+1]
            inside_neighbors = neigh.sum() - mask[y,x]  # exclude center
            # curvature term: if many inside neighbors, less likely to remove
            curvature_bias = (inside_neighbors - 4)/4.0  # approx -1..+1
            force = alpha * kl_force_map[y,x] + 0.5 * curvature_bias
            # threshold to flip
            if force > 0.2:
                new_mask[y,x] = 1
            elif force < -0.2:
                new_mask[y,x] = 0
            # else keep unchanged

        mask = new_mask

        # small morphological smoothing every few iterations for stability
        if (it % 5) == 0:
            mask = cv2.medianBlur((mask*255).astype('uint8'), 3)
            mask = (mask>127).astype('uint8')

        # optional save intermediate
        if save_intermediate and out_folder is not None and (it % 20 == 0 or it==iterations-1):
            ensure_dir(out_folder)
            save_mask(os.path.join(out_folder, f'mask_iter_{it:03d}.png'), mask)

    return mask
