import os
from src.utils import list_images, ensure_dir, save_mask
from src.preprocess import preprocess_and_init
from src.segmentation import kl_ls_refine
from src.evaluate import evaluate_dataset
import cv2

if __name__ == "__main__":
    img_dir = "data/PH2/images"
    gt_dir = "data/PH2/masks"

    out_preproc_dir = "output/preprocessed"
    out_init_dir = "output/init_masks"
    out_refined_dir = "output/refined_masks"
    os.makedirs(out_preproc_dir, exist_ok=True)
    os.makedirs(out_init_dir, exist_ok=True)
    os.makedirs(out_refined_dir, exist_ok=True)

    img_paths = list_images(img_dir)
    gt_paths = list_images(gt_dir)

    print(f"Found {len(img_paths)} images for processing.")

    # Process each image
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        preproc_path = os.path.join(out_preproc_dir, base + "_minRGB.png")
        init_path = os.path.join(out_init_dir, base + "_init.png")
        refined_path = os.path.join(out_refined_dir, base + "_refined.png")

        # Step 1: Preprocess & get initial mask
        minrgb, init_mask = preprocess_and_init(img_path, preproc_path, init_path)

        # Step 2: Refine mask using KL-LS
        final_mask = kl_ls_refine(minrgb, init_mask, iterations=100, alpha=1.0)
        save_mask(refined_path, final_mask)

        print(f"Processed {base}")

    # Step 3: Evaluate final results
    pred_paths = list_images(out_refined_dir)
    eval_csv = "output/ph2_results.csv"
    df = evaluate_dataset(gt_paths, pred_paths, eval_csv)
    print("\nEvaluation complete! Results saved to:", eval_csv)
    print(df.head())
