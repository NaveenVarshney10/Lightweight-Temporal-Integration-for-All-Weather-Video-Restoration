import os
import cv2
import numpy as np

# import Restormer’s functions
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

def evaluate_restormer_metrics(root_dir):
    total_psnr = []
    total_ssim = []
    total_pairs = 0

    for current_path, dirs, files in os.walk(root_dir):
        # pick PNG files
        images = [f for f in files if f.lower().endswith('.png')]

        # group by prefix before "_"
        prefix_map = {}
        for img in images:
            prefix = img.split('_')[0]
            prefix_map.setdefault(prefix, []).append(img)

        for prefix, flist in prefix_map.items():
            gt_path = None
            der_path = None

            for f in flist:
                if "_gt" in f:
                    gt_path = os.path.join(current_path, f)
                elif "_derained" in f:
                    der_path = os.path.join(current_path, f)

            if gt_path and der_path:
                # Read as RGB
                gt = cv2.imread(gt_path)
                der = cv2.imread(der_path)

                if gt is None or der is None:
                    print("Failed to read:", gt_path, der_path)
                    continue

                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32)
                der = cv2.cvtColor(der, cv2.COLOR_BGR2RGB).astype(np.float32)

                # Restormer-style PSNR
                psnr_val = calculate_psnr(
                    gt, 
                    der, 
                    crop_border=0, 
                    input_order='HWC', 
                    test_y_channel=True
                )

                # Restormer-style SSIM
                ssim_val = calculate_ssim(
                    gt, 
                    der, 
                    crop_border=0, 
                    input_order='HWC', 
                    test_y_channel=True
                )

                total_psnr.append(psnr_val)
                total_ssim.append(ssim_val)
                total_pairs += 1

                print(f"[OK] {prefix}  PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    if total_pairs == 0:
        return 0, 0, 0

    avg_psnr = sum(total_psnr) / total_pairs
    avg_ssim = sum(total_ssim) / total_pairs

    return total_pairs, avg_psnr, avg_ssim


# Usage
root_path = r"/content/drive/MyDrive/Restormer/results/Test_Restormer_Clips/visualization/Rain100_Test"
count, avg_psnr, avg_ssim = evaluate_restormer_metrics(root_path)

print("\n----------------------")
print("Total matched pairs:", count)
print("Average PSNR:", avg_psnr)
print("Average SSIM:", avg_ssim)
print("----------------------")
