import re
import os

def find_best_iteration():
    # 1. Ask the user for the file path
    file_path = input("Please enter the path to the .txt or .log file: ").strip()
    
    # Remove surrounding quotes if the user dragged and dropped the file
    if (file_path.startswith('"') and file_path.endswith('"')) or \
       (file_path.startswith("'") and file_path.endswith("'")):
        file_path = file_path[1:-1]

    if not os.path.exists(file_path):
        print("Error: File not found. Please check the path.")
        return

    max_ssim = -1.0
    best_iter_ssim = None
    psnr_at_max_ssim = None
    
    max_psnr = -1.0
    best_iter_psnr = None
    ssim_at_max_psnr = None
    
    current_iter = None

    # Patterns to match the iteration line and the metrics line
    iter_pattern = re.compile(r"Multi-Dataset Validation \(iter (\d+)\)")
    
    # Modified pattern to capture both PSNR and SSIM
    metrics_pattern = re.compile(r"Validation AVERAGE \(3 datasets\).*# psnr:\s*([\d\.]+).*# ssim:\s*([\d\.]+)")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Check if the line indicates the start of a validation block
                iter_match = iter_pattern.search(line)
                if iter_match:
                    current_iter = int(iter_match.group(1))
                    continue

                # If we are inside a validation block, check for metrics
                if current_iter is not None:
                    metrics_match = metrics_pattern.search(line)
                    if metrics_match:
                        current_psnr = float(metrics_match.group(1))
                        current_ssim = float(metrics_match.group(2))
                        
                        # Check if this is the best SSIM
                        if current_ssim > max_ssim:
                            max_ssim = current_ssim
                            best_iter_ssim = current_iter
                            psnr_at_max_ssim = current_psnr
                        
                        # Check if this is the best PSNR
                        if current_psnr > max_psnr:
                            max_psnr = current_psnr
                            best_iter_psnr = current_iter
                            ssim_at_max_psnr = current_ssim
                        
                        # Reset current_iter until the next validation block appears
                        current_iter = None

        if best_iter_ssim is not None or best_iter_psnr is not None:
            print("=" * 50)
            
            if best_iter_ssim is not None:
                print("BEST SSIM:")
                print(f"  Maximum SSIM: {max_ssim:.4f}")
                print(f"  Corresponding PSNR: {psnr_at_max_ssim:.4f}")
                print(f"  Iteration: {best_iter_ssim}")
            
            print("-" * 50)
            
            if best_iter_psnr is not None:
                print("BEST PSNR:")
                print(f"  Maximum PSNR: {max_psnr:.4f}")
                print(f"  Corresponding SSIM: {ssim_at_max_psnr:.4f}")
                print(f"  Iteration: {best_iter_psnr}")
            
            print("=" * 50)
            
            return {
                'best_ssim': {'iter': best_iter_ssim, 'ssim': max_ssim, 'psnr': psnr_at_max_ssim},
                'best_psnr': {'iter': best_iter_psnr, 'psnr': max_psnr, 'ssim': ssim_at_max_psnr}
            }
        else:
            print("No validation data found in the file.")
            return None

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    find_best_iteration()
