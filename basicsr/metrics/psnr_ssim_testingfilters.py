import os
import cv2
import numpy as np
from tqdm import tqdm

# =============================
# BasicSR metric functions
# =============================
from psnr_ssim import calculate_psnr, calculate_ssim


# =============================
# Bilateral filter variants
# =============================
def bilateral_ultra_light(img):
    # Almost identity (sanity check)
    return cv2.bilateralFilter(img, d=3, sigmaColor=10, sigmaSpace=10)


def bilateral_light(img):
    return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)


def bilateral_medium(img):
    return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)


def bilateral_strong(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def bilateral_very_strong(img):
    return cv2.bilateralFilter(img, d=11, sigmaColor=100, sigmaSpace=100)


# =============================
# Methods registry
# =============================
METHODS = {
    "Baseline": None,
    "Bilateral_UltraLight": bilateral_ultra_light,
    "Bilateral_Light": bilateral_light,
    "Bilateral_Medium": bilateral_medium,
    "Bilateral_Strong": bilateral_strong,
    "Bilateral_VeryStrong": bilateral_very_strong,
}


# =============================
# Main
# =============================
def main():
    root = input("Enter root folder path: ").strip().strip('"').strip("'")

    results = {k: [] for k in METHODS.keys()}

    total_frames = 0
    total_clips = 0

    # Each subfolder = one clip
    clip_dirs = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]

    pbar = tqdm(clip_dirs, desc="Evaluating clips", ncols=90)

    for clip_dir in pbar:
        files = os.listdir(clip_dir)
        gt_files = sorted([f for f in files if f.endswith("_gt.png")])

        if len(gt_files) == 0:
            continue

        total_clips += 1
        clip_metrics = {k: [] for k in METHODS.keys()}

        for gt_name in gt_files:
            base = gt_name.replace("_gt.png", "")
            inp_name = base + "_derained.png"

            gt_path = os.path.join(clip_dir, gt_name)
            inp_path = os.path.join(clip_dir, inp_name)

            if not os.path.exists(inp_path):
                continue

            gt = cv2.imread(gt_path)
            inp = cv2.imread(inp_path)

            if gt is None or inp is None:
                continue

            if gt.shape != inp.shape:
                inp = cv2.resize(inp, (gt.shape[1], gt.shape[0]))

            total_frames += 1

            for name, fn in METHODS.items():
                out = inp if fn is None else fn(inp)

                psnr = calculate_psnr(
                    out, gt,
                    crop_border=0,
                    test_y_channel=True
                )
                ssim = calculate_ssim(
                    out, gt,
                    crop_border=0,
                    test_y_channel=True
                )

                clip_metrics[name].append((psnr, ssim))

        # ---- Average over frames → clip ----
        for name in clip_metrics:
            if len(clip_metrics[name]) == 0:
                continue
            avg_psnr = np.mean([x[0] for x in clip_metrics[name]])
            avg_ssim = np.mean([x[1] for x in clip_metrics[name]])
            results[name].append((avg_psnr, avg_ssim))

        pbar.set_postfix(clips=total_clips, frames=total_frames)

    # =============================
    # Final summary (BasicSR-style)
    # =============================
    print("\n========== FINAL SUMMARY (Bilateral Levels) ==========\n")
    print(f"Total clips evaluated : {total_clips}")
    print(f"Total frames evaluated: {total_frames}\n")

    for name, vals in results.items():
        avg_psnr = np.mean([v[0] for v in vals])
        avg_ssim = np.mean([v[1] for v in vals])
        print(f"{name:22s} PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")

    print("\n=====================================================\n")


if __name__ == "__main__":
    main()
