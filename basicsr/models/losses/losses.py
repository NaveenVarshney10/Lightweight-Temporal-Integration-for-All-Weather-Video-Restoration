import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.

        return self.loss_weight * self.scale * torch.log(
            ((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return self.loss_weight * loss


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ⭐ NEW LOSS 1: SSIMLoss (Y-Channel SSIM)
# ----------------------------------------------------------------------
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, window_size=11):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.window = None

    def rgb2y(self, x):
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(x.device)
        return (x * weights).sum(dim=1, keepdim=True)

    def gaussian_window(self, window_size, channel):
        sigma = 1.5
        gauss = torch.Tensor(
            [np.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)]
        )
        gauss = gauss / gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, x, y, window):
        mu_x = F.conv2d(x, window, padding=self.window_size//2, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=self.window_size//2, groups=y.size(1))

        sigma_x = F.conv2d(x*x, window, padding=self.window_size//2, groups=x.size(1)) - mu_x**2
        sigma_y = F.conv2d(y*y, window, padding=self.window_size//2, groups=y.size(1)) - mu_y**2
        sigma_xy = F.conv2d(x*y, window, padding=self.window_size//2, groups=x.size(1)) - mu_x*mu_y

        ssim_map = ((2 * mu_x * mu_y + self.C1) *
                    (2 * sigma_xy + self.C2)) / (
                    (mu_x**2 + mu_y**2 + self.C1) *
                    (sigma_x + sigma_y + self.C2) + 1e-8)

        return ssim_map.mean()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)    

        pred_y = self.rgb2y(pred)
        target_y = self.rgb2y(target)

        if self.window is None or self.window.device != pred.device:
            self.window = self.gaussian_window(self.window_size, 1).to(pred.device)

        ssim_val = self.ssim(pred_y, target_y, self.window)
        return self.loss_weight * (1 - ssim_val)


# ----------------------------------------------------------------------
# ⭐ UPDATED: Flexible PerceptualLoss (Single/Multi-layer Support)
# ----------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """
    Flexible Perceptual Loss supporting single or multiple VGG19 layers.
    
    Args:
        loss_weight: Single float or comma-separated string (e.g., "0.01, 0.02, 0.02")
        layer: Single layer name or comma-separated string (e.g., "relu1_2, relu2_2, relu3_3")
        use_input_norm: Whether to normalize inputs with ImageNet mean/std
    
    Examples:
        # Single layer (backward compatible)
        PerceptualLoss(loss_weight=0.05, layer='relu3_3')
        
        # Multiple layers with individual weights
        PerceptualLoss(loss_weight="0.01, 0.02, 0.02", 
                      layer="relu1_2, relu2_2, relu3_3")
    """
    def __init__(self, loss_weight=1.0, layer='relu3_3', use_input_norm=True):
        super(PerceptualLoss, self).__init__()
        self.use_input_norm = use_input_norm
        
        # Parse layers
        if isinstance(layer, str) and ',' in layer:
            self.layers = [l.strip() for l in layer.split(',')]
        else:
            self.layers = [layer]
        
        # Parse loss weights
        if isinstance(loss_weight, str) and ',' in loss_weight:
            self.loss_weights = [float(w.strip()) for w in loss_weight.split(',')]
        elif isinstance(loss_weight, (list, tuple)):
            self.loss_weights = list(loss_weight)
        else:
            self.loss_weights = [float(loss_weight)]
        
        # Validate weights match layers
        if len(self.loss_weights) == 1 and len(self.layers) > 1:
            # If single weight for multiple layers, use same weight for all
            self.loss_weights = self.loss_weights * len(self.layers)
        
        assert len(self.loss_weights) == len(self.layers), \
            f"Number of loss_weights ({len(self.loss_weights)}) must match number of layers ({len(self.layers)})"
        
        # Load VGG19
        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except (ImportError, TypeError):
            vgg = models.vgg19(pretrained=True).features
        
        # Layer name to index mapping
        layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        # Create VGG slices for each layer
        self.vgg_slices = nn.ModuleList()
        for layer_name in self.layers:
            if layer_name not in layer_map:
                raise ValueError(f"Unknown layer: {layer_name}. Choose from {list(layer_map.keys())}")
            
            layer_idx = layer_map[layer_name]
            vgg_slice = nn.Sequential(*[vgg[i] for i in range(layer_idx + 1)])
            vgg_slice.eval()
            for p in vgg_slice.parameters():
                p.requires_grad = False
            self.vgg_slices.append(vgg_slice)
        
        # ImageNet normalization
        if self.use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def norm(self, x):
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        if self.use_input_norm:
            pred = self.norm(pred)
            target = self.norm(target)
        
        total_loss = 0.0
        
        # Compute loss for each layer (Option A: direct contribution to total loss)
        for i, vgg_slice in enumerate(self.vgg_slices):
            feat_pred = vgg_slice(pred)
            feat_target = vgg_slice(target)
            
            # L1 loss for this layer
            layer_loss = F.l1_loss(feat_pred, feat_target)
            
            # Add weighted loss (each weight directly affects total loss)
            total_loss += self.loss_weights[i] * layer_loss
        
        return total_loss


# ----------------------------------------------------------------------
# ⭐ NEW LOSS 2: VGGContrastiveLoss
# ----------------------------------------------------------------------
class VGGContrastiveLoss(nn.Module):
    """
    VGG-level Classical Contrastive Loss for Video Restoration
    
    Uses distance-based contrastive learning (not InfoNCE/probabilistic).
    Computes element-wise distances in VGG feature space (like perceptual loss).
    
    Args:
        loss_weight: Single float or comma-separated string (e.g., "0.01, 0.02")
        layer: Single layer name or comma-separated string (e.g., "relu2_2, relu3_3")
        loss_type: 'ratio' or 'margin' (default: 'ratio')
            - 'ratio': L = ||F(GT) - F(pred)|| / ||F(degraded) - F(pred)||
            - 'margin': L = max(0, D(pred, GT) - D(pred, degraded) + margin)
        margin: Margin for triplet loss (only used if loss_type='margin', default: 1.0)
        distance_norm: Distance norm ('l1' or 'l2', default: 'l1')
        use_input_norm: Whether to normalize inputs with ImageNet mean/std
        reduction: How to reduce per-sample losses ('mean', 'sum', default: 'mean')
        epsilon: Small value to avoid division by zero in ratio loss (default: 1e-8)
    
    Examples:
        # Ratio-based (element-wise L1 distance in VGG space)
        VGGContrastiveLoss(loss_weight=0.1, layer='relu3_3', loss_type='ratio', distance_norm='l1')
        
        # Margin-based triplet loss (L2 distance)
        VGGContrastiveLoss(loss_weight=0.1, layer='relu3_3', loss_type='margin',
                          margin=1.0, distance_norm='l2')
        
        # Multiple layers with individual weights
        VGGContrastiveLoss(loss_weight="0.05, 0.1", 
                          layer="relu2_2, relu3_3",
                          loss_type='ratio')
    
    Forward:
        loss(anchor=restored, positive=ground_truth, negative=degraded_input)
    """
    def __init__(self, loss_weight=1.0, layer='relu3_3', loss_type='ratio',
                 margin=1.0, distance_norm='l1', use_input_norm=True, 
                 reduction='mean', epsilon=1e-8):
        super(VGGContrastiveLoss, self).__init__()
        self.loss_type = loss_type
        self.margin = margin
        self.distance_norm = distance_norm
        self.use_input_norm = use_input_norm
        self.reduction = reduction
        self.epsilon = epsilon
        
        # Parse layers
        if isinstance(layer, str) and ',' in layer:
            self.layers = [l.strip() for l in layer.split(',')]
        else:
            self.layers = [layer]
        
        # Parse loss weights
        if isinstance(loss_weight, str) and ',' in loss_weight:
            self.loss_weights = [float(w.strip()) for w in loss_weight.split(',')]
        elif isinstance(loss_weight, (list, tuple)):
            self.loss_weights = list(loss_weight)
        else:
            self.loss_weights = [float(loss_weight)]
        
        # Validate weights match layers
        if len(self.loss_weights) == 1 and len(self.layers) > 1:
            self.loss_weights = self.loss_weights * len(self.layers)
        
        assert len(self.loss_weights) == len(self.layers), \
            f"Number of loss_weights ({len(self.loss_weights)}) must match number of layers ({len(self.layers)})"
        
        # Load VGG19
        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except (ImportError, TypeError):
            vgg = models.vgg19(pretrained=True).features
        
        # Layer name to index mapping
        layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        # Create VGG slices for each layer
        self.vgg_slices = nn.ModuleList()
        for layer_name in self.layers:
            if layer_name not in layer_map:
                raise ValueError(f"Unknown layer: {layer_name}. Choose from {list(layer_map.keys())}")
            
            layer_idx = layer_map[layer_name]
            vgg_slice = nn.Sequential(*[vgg[i] for i in range(layer_idx + 1)])
            vgg_slice.eval()
            for p in vgg_slice.parameters():
                p.requires_grad = False
            self.vgg_slices.append(vgg_slice)
        
        # ImageNet normalization
        if self.use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def norm(self, x):
        return (x - self.mean) / self.std
    

    def contrastive_loss_single_layer(self, anchor_feat, positive_feat, negative_feat):
        """
        Classical contrastive loss for a single layer.
        Computes element-wise distance in VGG feature space (like perceptual loss).
        
        Two formulations:
        
        1. 'ratio': 
           L = ||F(GT) - F(pred)|| / ||F(degraded) - F(pred)||
           Element-wise distance, then reduce per sample
        
        2. 'margin':
           L = max(0, D(pred, GT) - D(pred, degraded) + margin)
        """
        B, C, H, W = anchor_feat.shape
        
        # Compute element-wise distances (like perceptual loss)
        if self.distance_norm == 'l1':
            # L1 distance: sum of absolute differences
            dist_pos = torch.abs(anchor_feat - positive_feat)  # [B, C, H, W]
            dist_neg = torch.abs(anchor_feat - negative_feat)  # [B, C, H, W]
        else:  # l2
            # L2 distance: square root of sum of squared differences
            dist_pos = (anchor_feat - positive_feat) ** 2  # [B, C, H, W]
            dist_neg = (anchor_feat - negative_feat) ** 2  # [B, C, H, W]
        
        # Reduce to per-sample distances [B]
        # Sum over C, H, W dimensions (all feature elements)
        dist_pos_sample = dist_pos.view(B, -1).sum(dim=1)  # [B]
        dist_neg_sample = dist_neg.view(B, -1).sum(dim=1)  # [B]
        
        # Apply sqrt for L2 distance
        if self.distance_norm == 'l2':
            dist_pos_sample = torch.sqrt(dist_pos_sample + self.epsilon)
            dist_neg_sample = torch.sqrt(dist_neg_sample + self.epsilon)
        
        # Compute loss based on formulation
        if self.loss_type == 'ratio':
            # Ratio formulation: minimize dist_pos / dist_neg
            loss = dist_pos_sample / (dist_neg_sample + self.epsilon)
            
        elif self.loss_type == 'margin':
            # Margin-based triplet loss
            loss = F.relu(dist_pos_sample - dist_neg_sample + self.margin)
        
        # Reduce across batch
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Restored/generated image [B, 3, H, W]
            positive: Ground truth clean image [B, 3, H, W]
            negative: Degraded input image [B, 3, H, W]
        
        Returns:
            Weighted sum of contrastive losses across specified layers
        """
        if self.use_input_norm:
            anchor = self.norm(anchor)
            positive = self.norm(positive)
            negative = self.norm(negative)
        
        total_loss = 0.0
        
        # Compute contrastive loss for each layer
        for i, vgg_slice in enumerate(self.vgg_slices):
            # Extract features
            feat_anchor = vgg_slice(anchor)
            feat_positive = vgg_slice(positive)
            feat_negative = vgg_slice(negative)
            
            # Compute contrastive loss
            layer_loss = self.contrastive_loss_single_layer(
                feat_anchor, feat_positive, feat_negative
            )
            
            # Add weighted loss
            total_loss += self.loss_weights[i] * layer_loss
        
        return total_loss


# ----------------------------------------------------------------------
# ⭐ NEW LOSS 3: HistogramLoss
# ----------------------------------------------------------------------
class HistogramLoss(nn.Module):
    """
    Differentiable Color Histogram Matching Loss
    
    Uses soft binning (Gaussian kernels) to make histogram computation differentiable.
    """
    def __init__(self, loss_weight=1.0, num_bins=64, bandwidth=0.01, per_channel=True):
        super(HistogramLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_bins = num_bins
        self.bandwidth = bandwidth
        self.per_channel = per_channel
        self.eps = 1e-8
        
        # Create bin centers [0, 1]
        self.register_buffer('bin_centers', torch.linspace(0, 1, num_bins))
        
    def soft_histogram(self, x):
        """
        Compute differentiable soft histogram using Gaussian kernels
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        x_expanded = x_flat.unsqueeze(-1)  # [B, C, H*W, 1]
        bins_expanded = self.bin_centers.view(1, 1, 1, -1)  # [1, 1, 1, num_bins]
        
        # Compute Gaussian weights
        diff = x_expanded - bins_expanded
        weights = torch.exp(-(diff ** 2) / (2 * self.bandwidth ** 2))
        
        # Sum contributions to get histogram
        hist = weights.sum(dim=2)  # [B, C, num_bins]
        
        # Normalize
        hist = hist / (hist.sum(dim=-1, keepdim=True) + self.eps)
        
        return hist
    
    def emd_loss(self, hist1, hist2):
        """Earth Mover's Distance"""
        cdf1 = torch.cumsum(hist1, dim=-1)
        cdf2 = torch.cumsum(hist2, dim=-1)
        return torch.abs(cdf1 - cdf2).mean()
    
    def forward(self, pred, target):
        """Compute histogram matching loss"""
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        if self.per_channel:
            # Per RGB channel
            B, C, H, W = pred.shape
            total_loss = 0.0
            
            for c in range(C):
                pred_c = pred[:, c:c+1, :, :]
                target_c = target[:, c:c+1, :, :]
                
                pred_hist = self.soft_histogram(pred_c)
                target_hist = self.soft_histogram(target_c)
                
                loss = self.emd_loss(pred_hist.squeeze(1), target_hist.squeeze(1))
                total_loss += loss
            
            total_loss = total_loss / C
        else:
            # Luminance only
            weights = torch.tensor([0.299, 0.587, 0.114], device=pred.device).view(1, 3, 1, 1)
            pred_y = (pred * weights).sum(dim=1, keepdim=True)
            target_y = (target * weights).sum(dim=1, keepdim=True)
            
            pred_hist = self.soft_histogram(pred_y)
            target_hist = self.soft_histogram(target_y)
            
            total_loss = self.emd_loss(pred_hist.squeeze(1), target_hist.squeeze(1))
        
        return self.loss_weight * total_loss