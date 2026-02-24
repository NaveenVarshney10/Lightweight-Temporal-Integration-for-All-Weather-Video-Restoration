import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def plot_training_losses(log_file_path):
    """
    Parse training log file and plot each loss component separately.
    Mark max and min points with their values.
    
    Parameters:
    -----------
    log_file_path : str
        Path to the training log file
    """
    
    # Dictionary to store loss values
    losses = defaultdict(list)
    iterations = []
    
    # Regular expression to parse log lines
    pattern = r'iter:\s+(\d+,?\d+),.*?l_pix:\s+([\d.e+-]+)\s+l_ssim:\s+([\d.e+-]+)\s+l_percep:\s+([\d.e+-]+)\s+l_histogram:\s+([\d.e+-]+)\s+l_contrastive:\s+([\d.e+-]+)\s+l_total:\s+([\d.e+-]+)'
    
    # Read and parse the log file
    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                # Extract iteration number (remove comma if present)
                iter_num = int(match.group(1).replace(',', ''))
                iterations.append(iter_num)
                
                # Extract loss values
                losses['l_pix'].append(float(match.group(2)))
                losses['l_ssim'].append(float(match.group(3)))
                losses['l_percep'].append(float(match.group(4)))
                losses['l_histogram'].append(float(match.group(5)))
                losses['l_contrastive'].append(float(match.group(6)))
                losses['l_total'].append(float(match.group(7)))
    
    # Create subplots for each loss
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle('Training Losses over Iterations', fontsize=18, fontweight='bold', y=0.995)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Loss names and colors
    loss_names = ['l_pix', 'l_ssim', 'l_percep', 'l_histogram', 'l_contrastive', 'l_total']
    loss_labels = ['Pixel Loss', 'SSIM Loss', 'Perceptual Loss', 'Histogram Loss', 'Contrastive Loss', 'Total Loss']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot each loss
    for idx, (loss_name, label, color) in enumerate(zip(loss_names, loss_labels, colors)):
        ax = axes[idx]
        
        loss_values = losses[loss_name]
        
        # Plot the curve
        ax.plot(iterations, loss_values, color=color, linewidth=1.2, alpha=0.8)
        
        # Find max and min values
        max_idx = np.argmax(loss_values)
        min_idx = np.argmin(loss_values)
        
        max_iter = iterations[max_idx]
        max_val = loss_values[max_idx]
        min_iter = iterations[min_idx]
        min_val = loss_values[min_idx]
        
        # Plot max point (red dot)
        ax.plot(max_iter, max_val, 'ro', markersize=10, zorder=5, 
                markeredgecolor='darkred', markeredgewidth=2)
        
        # Plot min point (green dot)
        ax.plot(min_iter, min_val, 'go', markersize=10, zorder=5,
                markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Annotate max point
        ax.annotate(f'MAX\nIter: {max_iter}\nValue: {max_val:.6f}',
                   xy=(max_iter, max_val),
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', 
                            edgecolor='darkred', linewidth=2, alpha=0.9),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                 color='darkred', linewidth=2),
                   fontsize=9, fontweight='bold', zorder=6)
        
        # Annotate min point
        ax.annotate(f'MIN\nIter: {min_iter}\nValue: {min_val:.6f}',
                   xy=(min_iter, min_val),
                   xytext=(20, -30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#ccffcc',
                            edgecolor='darkgreen', linewidth=2, alpha=0.9),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                                 color='darkgreen', linewidth=2),
                   fontsize=9, fontweight='bold', zorder=6)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format y-axis
        ax.ticklabel_format(style='plain', axis='y')
        
        # Add background color
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('training_losses_with_minmax.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"Successfully plotted {len(iterations)} iterations")
    print(f"Iteration range: {iterations[0]} to {iterations[-1]}")
    print(f"Plot saved as 'training_losses_with_minmax.png'")
    print(f"{'='*60}")
    
    # Print loss statistics
    print("\nLoss Statistics:")
    print(f"{'Loss Type':<20} {'Min Value':<20} {'Max Value':<20}")
    print("-" * 60)
    for loss_name, label in zip(loss_names, loss_labels):
        loss_vals = losses[loss_name]
        min_val = min(loss_vals)
        max_val = max(loss_vals)
        min_iter = iterations[loss_vals.index(min_val)]
        max_iter = iterations[loss_vals.index(max_val)]
        print(f"{label:<20} {min_val:.6f} @ iter {min_iter:<8} {max_val:.6f} @ iter {max_iter:<8}")


# Example usage:
if __name__ == "__main__":
    print("=" * 60)
    print("Training Loss Plotter")
    print("=" * 60)
    
    # Ask user for the log file path
    log_file_path = input("\nEnter the path to your training log file: ").strip()
    
    # Remove quotes if user copied path with quotes
    log_file_path = log_file_path.strip('"').strip("'")
    
    # Check if file exists
    import os
    if not os.path.exists(log_file_path):
        print(f"\nError: File not found at '{log_file_path}'")
        print("Please check the path and try again.")
    else:
        print(f"\nProcessing log file: {log_file_path}")
        print("Generating plots...\n")
        plot_training_losses(log_file_path)