import numpy as np
import cv2
from .utils import save_mask, ensure_dir
import pandas as pd
import os

def dice_coef(gt, pred):
    gtb = (gt>0).astype(int)
    pb = (pred>0).astype(int)
    inter = (gtb & pb).sum()
    denom = gtb.sum() + pb.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom

def hammoude(gt, pred):
    # HM = (|A ∪ S| - |A ∩ S|) / |A ∪ S|
    gtb = (gt>0).astype(int)
    pb = (pred>0).astype(int)
    union = (gtb | pb).sum()
    inter = (gtb & pb).sum()
    if union == 0:
        return 0.0
    return (union - inter) / union

def xor_err(gt, pred):
    gtb = (gt>0).astype(int)
    pb = (pred>0).astype(int)
    union = (gtb | pb).sum()
    xor = (gtb ^ pb).sum()
    if union == 0:
        return 0.0
    return xor / union

def evaluate_dataset(gt_paths, pred_paths, out_csv):
    records = []
    for gpath, ppath in zip(gt_paths, pred_paths):
        gt = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(ppath, cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue
        gt_bin = (gt>127).astype(int)
        pr_bin = (pred>127).astype(int)
        d = dice_coef(gt_bin, pr_bin)
        h = hammoude(gt_bin, pr_bin)
        x = xor_err(gt_bin, pr_bin)
        records.append({'gt': os.path.basename(gpath),
                        'dice': float(d),
                        'hammoude': float(h),
                        'xor': float(x)})
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    return df
