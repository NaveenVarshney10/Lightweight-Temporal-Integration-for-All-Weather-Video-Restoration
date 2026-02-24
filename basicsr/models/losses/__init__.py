# Find the imports section and add:
from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss, 
                     SSIMLoss, PerceptualLoss, HistogramLoss, VGGContrastiveLoss)  # Add HistogramLoss

# Find the __all__ list and add:
__all__ = [
    'L1Loss', 
    'MSELoss', 
    'PSNRLoss',
    'CharbonnierLoss',
    'SSIMLoss',
    'VGGContrastiveLoss'
    'PerceptualLoss',
    'HistogramLoss'  # Add this
]