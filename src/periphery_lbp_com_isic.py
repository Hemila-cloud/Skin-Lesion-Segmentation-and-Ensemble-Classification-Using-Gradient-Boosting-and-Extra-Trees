import os, argparse, re
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# ------------------ PARAMETERS ------------------
PERIPH_SIZES = [5, 15, 25, 35, 45]
LBP_SCALES = [(8, 1.0), (8, 1.5), (8, 2.0)]
LBP_METHOD = 'uniform'
N_SPLITS = 5
RESIZE_SHAPE = (256, 256)
RANDOM_STATE = 42
# ------------------------------------------------

def read_mask_bin(path):
    """Read binary lesion mask and normalize to 0/1."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, thr = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return (thr // 255).astype(np.uint8)

def extract_inward_ring(mask, w):
    """Extract inner periphery region (ring of width w)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=w)
    ring = mask.astype(int) - eroded.astype(int)
    ring[ring < 0] = 0
    if ring.sum() == 0:
        return mask.copy()
    return ring.astype(np.uint8)

def compute_local_contrast(img_gray, P, R):
    """Compute local contrast map using circular LBP sampling."""
    coords = [(R*np.cos(2*np.pi*p/P), -R*np.sin(2*np.pi*p/P)) for p in range(P)]
    pad = int(np.ceil(R)) + 2
    padded = np.pad(img_gray.astype(np.float32), pad, mode='edge')
    yy, xx = np.indices(img_gray.shape)
    yy, xx = yy.astype(np.float32), xx.astype(np.float32)
    gc = img_gray.astype(np.float32)
    C = np.zeros_like(gc, dtype=np.float32)

    for dx, dy in coords:
        yy_p, xx_p = yy + pad + dy, xx + pad + dx
        x0, x1 = np.floor(xx_p).astype(int), np.floor(xx_p).astype(int) + 1
        y0, y1 = np.floor(yy_p).astype(int), np.floor(yy_p).astype(int) + 1
        x0, x1 = np.clip(x0, 0, padded.shape[1]-1), np.clip(x1, 0, padded.shape[1]-1)
        y0, y1 = np.clip(y0, 0, padded.shape[0]-1), np.clip(y1, 0, padded.shape[0]-1)
        Ia, Ib, Ic, Id = padded[y0, x0], padded[y1, x0], padded[y0, x1], padded[y1, x1]
        wa = (x1 - xx_p) * (y1 - yy_p)
        wb = (x1 - xx_p) * (yy_p - y0)
        wc = (xx_p - x0) * (y1 - yy_p)
        wd = (xx_p - x0) * (yy_p - y0)
        sampled = wa*Ia + wb*Ib + wc*Ic + wd*Id
        C += np.abs(sampled - gc)
    return C

def compute_lbp_hist(img_gray, P, R, mask=None):
    """Compute normalized LBP histogram."""
    lbp = local_binary_pattern(img_gray, P, R, method=LBP_METHOD)
    n_bins = P + 2 if LBP_METHOD == 'uniform' else int(np.max(lbp)) + 1
    vals = lbp[mask > 0].ravel() if mask is not None else lbp.ravel()
    if vals.size == 0:
        return np.zeros(n_bins)
    hist, _ = np.histogram(vals, bins=np.arange(0, n_bins + 1))
    return hist / np.sum(hist)

def build_features(img_gray, mask):
    """Compute LBP and LBP+Contrast histograms."""
    lbp_hists, contrast_maps = [], []
    for (P, R) in LBP_SCALES:
        lbp_hists.append(compute_lbp_hist(img_gray, P, R, mask))
        contrast_maps.append(compute_local_contrast(img_gray, P, R))
    lbp_concat = np.concatenate(lbp_hists)
    meanC = np.mean(np.stack(contrast_maps, axis=0), axis=0)
    thresh = np.median(meanC[mask > 0]) if mask.sum() > 0 else 0
    joint = []
    for (P, R) in LBP_SCALES:
        lbp_img = local_binary_pattern(img_gray, P, R, method=LBP_METHOD)
        n_bins = P + 2
        low = lbp_img[(mask > 0) & (meanC <= thresh)]
        high = lbp_img[(mask > 0) & (meanC > thresh)]
        h1, _ = np.histogram(low, bins=np.arange(0, n_bins + 1))
        h2, _ = np.histogram(high, bins=np.arange(0, n_bins + 1))
        h1 = h1 / np.sum(h1) if np.sum(h1) > 0 else np.zeros_like(h1)
        h2 = h2 / np.sum(h2) if np.sum(h2) > 0 else np.zeros_like(h2)
        joint.extend([h1, h2])
    lbp_c = np.concatenate(joint)
    return lbp_concat.astype(np.float32), lbp_c.astype(np.float32)

def overlay_periphery_visual(img, ring):
    """Overlay periphery contour in yellow."""
    overlay = img.copy()
    contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay

def build_dataset(pre_dir, mask_dir, out_dir, limit=None):
    """Build dataset of features from preprocessed + mask folders."""
    img_files = sorted(glob(os.path.join(pre_dir, "*.png")))
    if limit:
        img_files = img_files[:limit]

    feats_lbp, feats_lbp_c = {w: [] for w in PERIPH_SIZES + ['full']}, {w: [] for w in PERIPH_SIZES + ['full']}
    ids, labels = [], []

    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    for img_path in tqdm(img_files, desc="Processing images"):
        base = os.path.splitext(os.path.basename(img_path))[0].replace("_minrgb", "")
        mask_path = os.path.join(mask_dir, f"{base}_init.png")
        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = read_mask_bin(mask_path)
        if img is None or mask is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, RESIZE_SHAPE)
        mask = cv2.resize(mask, RESIZE_SHAPE)
        img = cv2.resize(img, RESIZE_SHAPE)

        # compute periphery-based features
        for w in PERIPH_SIZES:
            ring = extract_inward_ring(mask, w)
            f1, f2 = build_features(img_gray, ring)
            feats_lbp[w].append(f1)
            feats_lbp_c[w].append(f2)
            overlay = overlay_periphery_visual(img, ring)
            cv2.imwrite(os.path.join(vis_dir, f"{base}_periph{w}.png"), overlay)

        # full lesion
        f1, f2 = build_features(img_gray, mask)
        feats_lbp['full'].append(f1)
        feats_lbp_c['full'].append(f2)
        ids.append(base)

        # ---- Correct label extraction (PH2) ----
        match = re.search(r"Label(\d+)", base)
        if match:
            label_num = int(match.group(1))
        else:
            label_num = 0
        labels.append(1 if label_num == 3 else 0)

    if len(ids) == 0:
        print(" No valid image-mask pairs found.")
        return [], [], {}, {}

    for k in feats_lbp:
        feats_lbp[k] = np.vstack(feats_lbp[k]) if len(feats_lbp[k]) > 0 else np.zeros((1, 10))
        feats_lbp_c[k] = np.vstack(feats_lbp_c[k]) if len(feats_lbp_c[k]) > 0 else np.zeros((1, 10))

    return np.array(ids), np.array(labels), feats_lbp, feats_lbp_c

def classify_and_plot(labels, feats_lbp, feats_lbp_c, out_dir):
    """Perform classification + plot + export results with ROC."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    clf1 = KNeighborsClassifier(1)
    clf2 = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
    results = []

    for w in PERIPH_SIZES + ['full']:
        for feat_name, Xset in [('LBP', feats_lbp[w]), ('LBP_C', feats_lbp_c[w])]:
            for clf_name, clf in [('1NN', clf1), ('SVM', clf2)]:
                accs, aucs = [], []
                for tr, te in skf.split(Xset, labels):
                    if len(np.unique(labels[tr])) < 2:
                        continue
                    clf.fit(Xset[tr], labels[tr])
                    pred = clf.predict(Xset[te])
                    accs.append(accuracy_score(labels[te], pred))
                    # ROC computation if possible
                    try:
                        probs = clf.predict_proba(Xset[te])[:, 1]
                        aucs.append(roc_auc_score(labels[te], probs))
                    except Exception:
                        aucs.append(np.nan)
                if len(accs) > 0:
                    mean_acc, std_acc = np.mean(accs), np.std(accs)
                    mean_auc = np.nanmean(aucs) * 100
                    print(f"[{w}][{feat_name}][{clf_name}] Acc={mean_acc*100:.2f}% ROC={mean_auc:.2f}%")
                    results.append([w, feat_name, clf_name, mean_acc*100, mean_auc])

    df = pd.DataFrame(results, columns=['periphery', 'feature', 'classifier', 'Acc_%', 'ROC_%'])
    df.to_csv(os.path.join(out_dir, "classification_results_summary.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    for feat in ['LBP', 'LBP_C']:
        for clf in ['1NN', 'SVM']:
            sub = df[(df.feature == feat) & (df.classifier == clf)]
            if sub.empty:
                continue
            xvals = [float(x) if str(x).isdigit() else 50 for x in sub['periphery']]
            plt.plot(xvals, sub['Acc_%'], marker='o', label=f"{feat}+{clf}")
    plt.xlabel("Periphery (pixels)")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy vs Periphery Size")
    plt.xticks([5, 15, 25, 35, 45, 50], ['5', '15', '25', '35', '45', 'full'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_periphery.png"), dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PH2", "ISIC"])
    parser.add_argument("--use-split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.dataset == "PH2":
        pre_dir = "outputs/PH2_results/preprocessed"
        mask_dir = "outputs/PH2_results/init_masks"
        out_dir = "outputs/PH2_feature_results"
    else:
        pre_dir = f"outputs/ISIC_{args.use_split}_results/preprocessed"
        mask_dir = f"outputs/ISIC_{args.use_split}_results/init_masks"
        out_dir = f"outputs/ISIC_{args.use_split}_feature_results"

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n Dataset: {args.dataset} ({args.use_split})")
    print(f" Preprocessed path: {pre_dir}")
    print(f" Masks path: {mask_dir}\n")

    ids, labels, feats_lbp, feats_lbp_c = build_dataset(pre_dir, mask_dir, out_dir, args.limit)
    if len(ids) == 0:
        print(" No data processed. Please check your folders.")
        return
    classify_and_plot(labels, feats_lbp, feats_lbp_c, out_dir)
    print(f"\n Done! Results saved in: {out_dir}")

if __name__ == "__main__":
    main()