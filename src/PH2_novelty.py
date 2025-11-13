import os, re, argparse
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from scipy.stats import skew, kurtosis, entropy
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ---------------- #
PERIPH_SIZES = [5, 15, 25, 35, 45]
LBP_METHOD = 'uniform'
N_SPLITS = 5
RESIZE_SHAPE = (256, 256)
RANDOM_STATE = 42
FEATURE_LENGTH = 200  # consistent length for feature vectors
# --------------------------------------------- #


def read_mask_bin(path):
    """Read mask and binarize"""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, thr = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return (thr // 255).astype(np.uint8)


def extract_inward_ring(mask, w, prev_mask=None):
    """Progressively isolate thinner, non-overlapping periphery rings."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if prev_mask is None:
        inner = (dist <= w).astype(np.uint8)
    else:
        inner = (dist <= w).astype(np.uint8)
        prev_inner = (dist <= (w - 10)).astype(np.uint8)
        inner = cv2.subtract(inner, prev_inner)
    if np.sum(inner) == 0:
        inner = mask.copy()
    return inner


def compute_local_contrast(img_gray):
    """Compute local contrast using Laplacian variance"""
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    contrast = np.abs(lap)
    return contrast


def build_features(img_gray, mask):
    """Enhanced: multi-scale LBP + adaptive contrast + intensity ratios."""
    feats = []

    # Multi-scale LBP
    for (P, R) in [(8, 1.0), (8, 2.0), (16, 3.0)]:
        lbp = local_binary_pattern(img_gray, P, R, method=LBP_METHOD)
        vals = lbp[mask > 0]
        if vals.size == 0:
            feats.extend(np.zeros(P + 2))
        else:
            hist, _ = np.histogram(vals, bins=np.arange(0, P + 3), density=True)
            feats.extend(hist.tolist())

    # Contrast-based stats (entropy-weighted)
    contrast = cv2.Laplacian(img_gray, cv2.CV_64F)
    vals = contrast[mask > 0]
    if vals.size > 0:
        ent = entropy(np.histogram(vals, bins=32, density=True)[0] + 1e-6)
        feats.extend([np.mean(vals), np.std(vals), np.max(vals), np.min(vals), ent])
    else:
        feats.extend(np.zeros(5))

    # Intensity ratio
    masked_vals = img_gray[mask > 0]
    if masked_vals.size > 0:
        ratio = np.mean(masked_vals) / (np.std(masked_vals) + 1e-6)
        feats.append(ratio)
    else:
        feats.append(0)

    # Final normalization
    feats = np.nan_to_num(feats)
    if len(feats) < FEATURE_LENGTH:
        feats = np.pad(feats, (0, FEATURE_LENGTH - len(feats)))
    else:
        feats = feats[:FEATURE_LENGTH]
    return feats


def generate_visuals(base, img, ring, lbp_img, out_dir, w):
    """Save distinct visualizations for each periphery"""
    overlay = img.copy()
    contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")

    axs[1].imshow(overlay)
    axs[1].set_title(f"Periphery width={w}px")

    axs[2].imshow(lbp_img, cmap='inferno')
    axs[2].set_title("LBP Map")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{base}_periph{w}.png"))
    plt.close()


def build_dataset(pre_dir, mask_dir, out_dir, limit=None):
    img_files = sorted(glob(os.path.join(pre_dir, "*.png")))
    if limit:
        img_files = img_files[:limit]

    feats = {w: [] for w in PERIPH_SIZES}
    ids, labels = [], []
    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    for img_path in tqdm(img_files, desc="Processing PH2"):
        base = os.path.splitext(os.path.basename(img_path))[0].replace("_minrgb", "")
        mask_path = os.path.join(mask_dir, f"{base}_init.png")
        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = read_mask_bin(mask_path)
        if img is None or mask is None:
            continue

        img_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), RESIZE_SHAPE)
        mask = cv2.resize(mask, RESIZE_SHAPE)
        img = cv2.resize(img, RESIZE_SHAPE)

        prev_ring = None
        for w in PERIPH_SIZES:
            ring = extract_inward_ring(mask, w, prev_ring)
            prev_ring = ring.copy()
            feat = build_features(img_gray, ring)
            lbp_map = local_binary_pattern(img_gray, 8, 1.5, method=LBP_METHOD)
            feats[w].append(feat)
            generate_visuals(base, img, ring, lbp_map, vis_dir, w)

        match = re.search(r"Label(\d+)", base)
        label_num = int(match.group(1)) if match else 0
        labels.append(1 if label_num == 3 else 0)
        ids.append(base)

    for k in feats:
        feats[k] = np.vstack(feats[k])
    return np.array(ids), np.array(labels), feats


def compute_accuracy_table(labels, feats_dict, out_csv, out_dir):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, max_depth=8, random_state=RANDOM_STATE)
    }

    records = []
    for model_name, model in models.items():
        row = {'Classifier': model_name}
        for w in PERIPH_SIZES:
            X = np.nan_to_num(feats_dict[w])
            accs, aucs = [], []
            for tr, te in skf.split(X, labels):
                model.fit(X[tr], labels[tr])
                pred = model.predict(X[te])
                accs.append(accuracy_score(labels[te], pred))
                try:
                    probs = model.predict_proba(X[te])[:, 1]
                    aucs.append(roc_auc_score(labels[te], probs))
                except:
                    aucs.append(np.nan)
            row[f"{w}px_Acc%"] = np.mean(accs) * 100
            row[f"{w}px_ROC%"] = np.nanmean(aucs) * 100
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"✅ Accuracy table saved: {out_csv}")

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    for i, model_name in enumerate(df["Classifier"]):
        plt.plot(PERIPH_SIZES, df.iloc[i, 1::2], marker='o', label=model_name)
    plt.xlabel("Periphery Width (px)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Periphery Width")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "PH2_accuracy_trend.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    pre_dir = "outputs/PH2_results/preprocessed"
    mask_dir = "outputs/PH2_results/init_masks"
    out_dir = "outputs/PH2_feature_final_results_NOVELTY"
    os.makedirs(out_dir, exist_ok=True)

    ids, labels, feats_dict = build_dataset(pre_dir, mask_dir, out_dir, args.limit)
    compute_accuracy_table(labels, feats_dict, os.path.join(out_dir, "PH2_accuracy_table_final.csv"), out_dir)


if __name__ == "__main__":
    main()