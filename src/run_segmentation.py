import os
from .utils import list_images, ensure_dir, save_mask
from .preprocess import preprocess_and_init, minRGB
from .segmentation import kl_ls_refine
from .evaluate import evaluate_dataset
import argparse
import cv2

def process_dataset(img_folder, gt_folder, out_preproc_folder, out_init_folder, out_masks_folder, out_metrics_csv,
                    iterations=100):
    ensure_dir(out_preproc_folder)
    ensure_dir(out_init_folder)
    ensure_dir(out_masks_folder)

    img_paths = list_images(img_folder)
    pred_mask_paths = []
    gt_paths = []
    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        preproc_path = os.path.join(out_preproc_folder, f'{name}_minrgb.png')
        init_path = os.path.join(out_init_folder, f'{name}_init.png')
        mask_out_path = os.path.join(out_masks_folder, f'{name}_mask.png')

        print("Processing:", img_path)
        minrgb_img, init_mask = preprocess_and_init(img_path, preproc_path, init_path, save_preproc=True)
        # refine with KL-LS
        refined = kl_ls_refine(minrgb_img, init_mask, iterations=iterations, alpha=1.0,
                               save_intermediate=False, out_folder=None)
        save_mask(mask_out_path, refined)
        pred_mask_paths.append(mask_out_path)

        # find gt path (same base name) inside gt_folder - try common extensions
        for ext in ['.png','.bmp','.jpg','.tif','.jpeg']:
            cand = os.path.join(gt_folder, name + ext)
            if os.path.exists(cand):
                gt_paths.append(cand)
                break
        else:
            # if not found, append a placeholder None
            gt_paths.append(None)

    # filter pairs where gt exists
    paired_gt = []
    paired_pred = []
    for g,p in zip(gt_paths, pred_mask_paths):
        if g is not None:
            paired_gt.append(g)
            paired_pred.append(p)

    if len(paired_gt) > 0:
        print("Evaluating on", len(paired_gt), "images")
        df = evaluate_dataset(paired_gt, paired_pred, out_metrics_csv)
        print("Saved metrics to", out_metrics_csv)
    else:
        print("No ground truth found to evaluate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True)
    parser.add_argument('--gt_folder', required=True)
    parser.add_argument('--out_folder', required=True)
    parser.add_argument('--iter', type=int, default=120)
    args = parser.parse_args()

    out_preproc = os.path.join(args.out_folder, 'preprocessed')
    out_init = os.path.join(args.out_folder, 'init_masks')
    out_masks = os.path.join(args.out_folder, 'masks')
    out_metrics = os.path.join(args.out_folder, 'metrics.csv')

    process_dataset(args.img_folder, args.gt_folder, out_preproc, out_init, out_masks, out_metrics, iterations=args.iter)
