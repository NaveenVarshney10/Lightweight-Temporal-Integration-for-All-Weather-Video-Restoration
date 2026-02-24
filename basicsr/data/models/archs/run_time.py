import torch
import time
import numpy as np
from restormer_arch import Restormer

# Try to import thop for FLOP counting
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' not installed. Install with: pip install thop")

# ==========================================
# 1. CONFIGURATION (Matched to your YAML)
# ==========================================
NUM_FRAMES = 10         # From dataset config
HEIGHT = 256            # From dataset config
WIDTH = 256
BATCH_SIZE = 1

# Model Hyperparameters
MODEL_CONFIG = {
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 12,                  # Lightweight setting
    'num_blocks': [2,3,3,4],
    'num_refinement_blocks': 4,
    'heads': [1,1,1,1],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False,
    'num_frames': NUM_FRAMES
}

def calculate_flops(model, input_shape, device):
    """Calculate GFLOPs using thop library"""
    if not THOP_AVAILABLE:
        return None, None
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Clone model to avoid modifying original
    model_copy = type(model)(**MODEL_CONFIG).to(device)
    model_copy.load_state_dict(model.state_dict())
    model_copy.eval()
    
    # Calculate FLOPs
    flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
    
    # Convert to GFLOPs (Giga = 10^9)
    gflops = flops / 1e9
    
    # Get formatted strings
    flops_str, params_str = clever_format([flops, params], "%.3f")
    
    return gflops, flops_str

def main():
    # -------------------------------------------------------
    # 2. SETUP & PARAMETER COUNTING
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"   Model Analysis: Restormer (dim={MODEL_CONFIG['dim']})")
    print(f"{'='*60}")

    # Check Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Initialize Model
    model = Restormer(**MODEL_CONFIG).to(device)
    model.eval()

    # Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[1] PARAMETER COUNT")
    print(f"    Total Parameters:     {total_params / 1e6:.2f} M ({total_params:,})")
    print(f"    Trainable Parameters: {trainable_params / 1e6:.2f} M ({trainable_params:,})")

    # -------------------------------------------------------
    # 3. FLOPS CALCULATION
    # -------------------------------------------------------
    print(f"\n[2] COMPUTATIONAL COMPLEXITY (GFLOPs)")
    input_shape = (BATCH_SIZE * NUM_FRAMES, 3, HEIGHT, WIDTH)
    
    if THOP_AVAILABLE:
        print(f"    Input Shape: {input_shape}")
        print(f"    Calculating FLOPs...")
        
        gflops, flops_str = calculate_flops(model, input_shape, device)
        
        if gflops is not None:
            gflops_per_frame = gflops / NUM_FRAMES
            print(f"    Total GFLOPs (per clip):  {gflops:.3f} G")
            print(f"    GFLOPs (per frame):       {gflops_per_frame:.3f} G")
            print(f"    FLOPs (formatted):        {flops_str}")
    else:
        print(f"    [SKIPPED] Install 'thop' to calculate FLOPs")
        print(f"    Run: pip install thop")
        gflops = None

    # -------------------------------------------------------
    # 4. INFERENCE SPEED TEST
    # -------------------------------------------------------
    print(f"\n[3] INFERENCE SPEED TEST")
    print(f"    Input Shape: ({BATCH_SIZE} clips, {NUM_FRAMES} frames, {HEIGHT}x{WIDTH})")
    
    # Create Dummy Input
    dummy_input = torch.randn(input_shape).to(device)

    # Warm-up
    print("    Warming up GPU (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measurement Loop
    print("    Running measurement (100 iterations)...")
    timings = []
    with torch.no_grad():
        for _ in range(100):
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.time()
            
            _ = model(dummy_input)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)

    # -------------------------------------------------------
    # 5. REPORT RESULTS
    # -------------------------------------------------------
    avg_time_per_clip = np.mean(timings)
    std_time = np.std(timings)
    avg_time_per_frame = avg_time_per_clip / NUM_FRAMES
    fps = 1.0 / avg_time_per_clip

    print(f"\n{'='*60}")
    print(f" FINAL RESULTS")
    print(f"{'='*60}")
    print(f" Parameters:              {total_params / 1e6:.2f} M")
    
    if gflops is not None:
        print(f" GFLOPs (per clip):       {gflops:.3f} G")
        print(f" GFLOPs (per frame):      {gflops / NUM_FRAMES:.3f} G")
    
    print(f" Inference (per clip):    {avg_time_per_clip*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f" Inference (per frame):   {avg_time_per_frame*1000:.2f} ms")
    print(f" Throughput:              {fps:.2f} clips/sec")
    print(f" Frame Rate:              {fps * NUM_FRAMES:.2f} frames/sec")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()