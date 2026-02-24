import torch
from torch.utils import data as data
from os import path as osp
import glob

# NOTE: paired_random_crop is no longer used but left here for context
#from basicsr.data.data_util import paired_random_crop 
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir


class RainVideoDataset(data.Dataset):
    """
    Custom Dataset for Video Deraining (Rain13K clips).
    Loads all frames from a subfolder at full size (256x256).
    No cropping, no augmentation, no meta file.
    """

    def __init__(self, opt):
        super(RainVideoDataset, self).__init__()
        self.opt = opt
        self.gt_root = opt['dataroot_gt']
        self.lq_root = opt['dataroot_lq']
        
        # --- CUSTOM LOGIC: Find all subfolders (clips) directly ---
        logger = get_root_logger()
        logger.info(f'Scanning for video clips in {self.lq_root}...')
        
        self.keys = []
        import os
        
        # Use os.scandir to efficiently list entries and check if they are directories
        for entry in os.scandir(self.lq_root):
            # Check if the entry is a directory
            if entry.is_dir():
                # Append the folder name (key)
                self.keys.append(entry.name)
        
        self.keys = sorted(self.keys) # Sort the final list of clip folder names
        
        if not self.keys:
            # Raise the specific error if no folders are found
            raise FileNotFoundError(
                f"No clip subfolders found in: {self.lq_root}. "
                "Ensure your video clips are inside subfolders like '00001_01'."
            )
            
        logger.info(f'Found {len(self.keys)} video clips.')
        # -----------------------------------------------------------


    def __getitem__(self, index):
        key = self.keys[index]

        # --- 1. Get Frame Paths ---
        lq_clip_path = osp.join(self.lq_root, key)
        gt_clip_path = osp.join(self.gt_root, key)
        
        # Scan inside clip folders to get all frame paths
        lq_paths = sorted(glob.glob(osp.join(lq_clip_path, '*')))
        gt_paths = sorted(glob.glob(osp.join(gt_clip_path, '*')))
        
        assert len(lq_paths) == len(gt_paths), "LQ and GT clips have different frame counts."

        # --- 2. Load ALL Frames (Full Resolution) ---
        # Read all images as NumPy arrays
        img_lqs = [imfrombytes(FileClient('disk').get(p), float32=True) for p in lq_paths]
        img_gts = [imfrombytes(FileClient('disk').get(p), float32=True) for p in gt_paths]

        # --- 3. Final Conversion and Stacking ---
        
        # Combine lists for mass conversion
        img_gts_and_lqs = img_lqs + img_gts
        
        # Convert all to PyTorch tensors (and handle HWC to CHW conversion)
        img_results = img2tensor(img_gts_and_lqs)
        
        # Stack inputs (lq) and outputs (gt) separately: (T, C, H, W)
        num_lq = len(img_lqs)
        img_lqs = torch.stack(img_results[:num_lq], dim=0) 
        img_gts = torch.stack(img_results[num_lq:], dim=0)

        # Return the 10 frames for LQ and 10 frames for GT
        # ADDED 'lq_path': Use the clip key as the path placeholder
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'lq_path': key}

    def __len__(self):
        return len(self.keys)