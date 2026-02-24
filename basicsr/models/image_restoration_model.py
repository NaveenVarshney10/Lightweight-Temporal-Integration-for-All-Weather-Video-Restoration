import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta])
        )
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Deblur/Deweather model supporting image and clip inputs.

    Validation uses original Restormer behavior: metrics computed on images
    (numpy uint8 via tensor2img), not raw tensors. A tqdm progress bar shows
    clip-wise validation progress. All frames in each clip are saved when
    save_img=True, preserving your prior behavior.
    
    Now supports multi-dataset validation with average metric computation.
    """

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # ---- Mixup config ----
        train_opt_path = self.opt.get('train')
        if train_opt_path:
            self.mixing_flag = train_opt_path.get('mixing_augs', {}).get('mixup', False)
            if self.mixing_flag:
                mixup_beta = train_opt_path['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = train_opt_path['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        else:
            self.mixing_flag = False

        # ---- Network ----
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # ---- Load pretrained ----
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params'),
            )

        if self.is_train:
            self.init_training_settings()

    # ==========================
    # TRAIN INITIALIZATION
    # ==========================
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # EMA
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Using EMA: decay = {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)

            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt['path'].get('strict_load_g', True),
                    'params_ema',
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # ⭐ UPDATED: Multiple Loss Functions
        logger = get_root_logger()
        
        # Pixel loss (required)
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
            logger.info(f'Pixel loss [{pixel_type}] initialized.')
        else:
            raise ValueError('Pixel loss must be defined.')

        # ⭐ SSIM loss (optional)
        if train_opt.get('ssim_opt'):
            ssim_type = train_opt['ssim_opt'].pop('type')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['ssim_opt']).to(self.device)
            logger.info(f'SSIM loss [{ssim_type}] initialized with weight {self.cri_ssim.loss_weight}.')
        else:
            self.cri_ssim = None

        # ⭐ Perceptual loss (optional)
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_percep_cls = getattr(loss_module, percep_type)
            self.cri_percep = cri_percep_cls(**train_opt['perceptual_opt']).to(self.device)
            if len(self.cri_percep.layers) == 1:
                logger.info(f'Perceptual loss [{percep_type}] initialized: '
                       f'layer={self.cri_percep.layers[0]}, weight={self.cri_percep.loss_weights[0]}')
            else:
                logger.info(f'Perceptual loss [{percep_type}] initialized (multi-layer):')
                for layer, weight in zip(self.cri_percep.layers, self.cri_percep.loss_weights):
                    logger.info(f'  - {layer}: weight={weight}')    
        else:
            self.cri_percep = None

        # Histogram loss (optional)
        if train_opt.get('histogram_opt'):
            histogram_opt = deepcopy(train_opt['histogram_opt'])
            histogram_type = histogram_opt.pop('type')
            cri_histogram_cls = getattr(loss_module, histogram_type)
            self.cri_histogram = cri_histogram_cls(**histogram_opt).to(self.device)
            logger.info(f'Histogram loss [{histogram_type}] initialized with weight {self.cri_histogram.loss_weight}.')
        else:
            self.cri_histogram = None

        # ⭐ NEW: VGG Contrastive Loss (optional)
        if train_opt.get('contrastive_opt'):
            contrastive_opt = deepcopy(train_opt['contrastive_opt'])
            contrastive_type = contrastive_opt.pop('type')
            cri_contrastive_cls = getattr(loss_module, contrastive_type)
            self.cri_contrastive = cri_contrastive_cls(**contrastive_opt).to(self.device)
            if len(self.cri_contrastive.layers) == 1:
                logger.info(f'Contrastive loss [{contrastive_type}] initialized: '
                       f'layer={self.cri_contrastive.layers[0]}, weight={self.cri_contrastive.loss_weights[0]}, '
                       f'type={self.cri_contrastive.loss_type}, norm={self.cri_contrastive.distance_norm}')
            else:
                logger.info(f'Contrastive loss [{contrastive_type}] initialized (multi-layer):')
                for layer, weight in zip(self.cri_contrastive.layers, self.cri_contrastive.loss_weights):
                    logger.info(f'  - {layer}: weight={weight}')
                logger.info(f'  loss_type={self.cri_contrastive.loss_type}, distance_norm={self.cri_contrastive.distance_norm}')
        else:
            self.cri_contrastive = None            

        # Optimizers    
        self.setup_optimizers()
        self.setup_schedulers()

    # ==========================
    # OPTIMIZER SETUP
    # ==========================
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                get_root_logger().warning(f'Params {k} not optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'Unsupported optimizer: {optim_type}')
        self.optimizers.append(self.optimizer_g)

    # ==========================
    # FLATTEN 5D CLIP -> 4D IMAGES
    # ==========================
    @staticmethod
    def _flatten_if_clip(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # [B,T,C,H,W] → [B*T,C,H,W]
            b, t, c, h, w = x.size()
            return x.view(b * t, c, h, w)
        elif x.dim() == 4:
            return x
        else:
            raise ValueError(f'Unexpected ndim {x.dim()}')

    def feed_train_data(self, data):
        self.lq = self._flatten_if_clip(data['lq'].to(self.device))
        if 'gt' in data:
            self.gt = self._flatten_if_clip(data['gt'].to(self.device))

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = self._flatten_if_clip(data['lq'].to(self.device))
        if 'gt' in data:
            self.gt = self._flatten_if_clip(data['gt'].to(self.device))

    # ==========================
    # ⭐ UPDATED: TRAIN STEP WITH MULTIPLE LOSSES
    # ==========================
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        preds = preds if isinstance(preds, list) else [preds]

        self.output = preds[-1]
     
        # ⭐ Calculate all losses
        l_total = 0
        loss_dict = OrderedDict()
        
        # Pixel loss (applied to all predictions if multi-scale)
        l_pix = sum(self.cri_pix(pred, self.gt) for pred in preds)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix
        
        # ⭐ SSIM loss (only on final output)
        if self.cri_ssim is not None:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim
        
        # ⭐ Perceptual loss (only on final output)
        if self.cri_percep is not None:
            l_percep = self.cri_percep(self.output, self.gt)
            l_total += l_percep
            loss_dict['l_percep'] = l_percep

        # Histogram loss
        if self.cri_histogram is not None:
            l_histogram = self.cri_histogram(self.output, self.gt)
            l_total += l_histogram
            loss_dict['l_histogram'] = l_histogram

        # ⭐ VGG Contrastive loss (only on final output)
        # anchor=restored, positive=GT, negative=degraded_input
        if self.cri_contrastive is not None:
            l_contrastive = self.cri_contrastive(
                anchor=self.output,     # Restored/predicted image
                positive=self.gt,       # Ground truth
                negative=self.lq        # Degraded input
            )
            l_total += l_contrastive
            loss_dict['l_contrastive'] = l_contrastive
        
        # Total loss
        loss_dict['l_total'] = l_total

        # Backward and optimize
        l_total.backward()
        if self.opt['train'].get('use_grad_clip', False):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(self.ema_decay)

    # ==========================
    # INFERENCE
    # ==========================
    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size

        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)

        self.output = self.output[:, :, :h, :w]

    def nonpad_test(self, img=None):
        img = img if img is not None else self.lq

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()

        with torch.no_grad():
            pred = net(img)
        pred = pred[-1] if isinstance(pred, list) else pred

        self.output = pred
        if net is self.net_g:
            self.net_g.train()

    # ==========================
    # ⭐ UPDATED: VALIDATION WITH MULTI-DATASET SUPPORT
    # ==========================
    def dist_validation(self, dataloader_or_dict, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        """
        Validation supporting both single dataloader and multi-dataset dict.
        
        Args:
            dataloader_or_dict: Either a single DataLoader or dict of {dataset_name: DataLoader}
        """
        if os.environ.get('LOCAL_RANK', '0') == '0':
            return self.nondist_validation(dataloader_or_dict, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        return 0.0

    def nondist_validation(self, dataloader_or_dict, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        """
        Main validation method with multi-dataset support.
        
        Args:
            dataloader_or_dict: Either:
                - Single DataLoader (backward compatible)
                - Dict of {dataset_name: DataLoader} for multi-dataset validation
        """
        logger = get_root_logger()
        
        # ⭐ Detect if multi-dataset or single dataset
        if isinstance(dataloader_or_dict, dict):
            # Multi-dataset validation
            return self._validate_multiple_datasets(
                dataloader_or_dict, current_iter, tb_logger, save_img, rgb2bgr, use_image
            )
        else:
            # Single dataset validation (backward compatible)
            return self._validate_single_dataset(
                dataloader_or_dict, current_iter, tb_logger, save_img, rgb2bgr, use_image
            )

    def _validate_multiple_datasets(self, dataloaders_dict, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        """
        Validate on multiple datasets and compute average metrics.
        
        Returns:
            Average metric across all datasets
        """
        logger = get_root_logger()
        logger.info(f'\n{"="*60}')
        logger.info(f'Multi-Dataset Validation (iter {current_iter})')
        logger.info(f'{"="*60}')
        
        # Store metrics for each dataset
        all_dataset_metrics = {}
        
        # Validate each dataset
        for dataset_name, dataloader in dataloaders_dict.items():
            logger.info(f'\n--- Validating {dataset_name} ---')
            
            # Run validation on this dataset
            metric = self._validate_single_dataset(
                dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
            )
            
            # Store results
            all_dataset_metrics[dataset_name] = self.metric_results.copy()
        
        # ⭐ Calculate average metrics across all datasets
        logger.info(f'\n{"="*60}')
        logger.info('Computing Average Metrics Across All Datasets')
        logger.info(f'{"="*60}')
        
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics and len(all_dataset_metrics) > 0:
            # Initialize average metrics
            metric_names = list(next(iter(all_dataset_metrics.values())).keys())
            avg_metrics = {name: 0.0 for name in metric_names}
            
            # Sum metrics from all datasets
            for dataset_name, metrics in all_dataset_metrics.items():
                for metric_name, value in metrics.items():
                    avg_metrics[metric_name] += value
            
            # Calculate average
            num_datasets = len(all_dataset_metrics)
            for metric_name in avg_metrics:
                avg_metrics[metric_name] /= num_datasets
            
            # Log average metrics
            log = f"Validation AVERAGE ({num_datasets} datasets), "
            for name, value in avg_metrics.items():
                log += f"\t# {name}: {value:.4f}"
            logger.info(log)
            
            # Log to tensorboard with special tag
            if tb_logger:
                for name, value in avg_metrics.items():
                    tb_logger.add_scalar(f"metrics/AVERAGE_{name}", value, current_iter)
            
            logger.info(f'{"="*60}\n')
            
            # Return first metric as representative (for backward compatibility)
            return list(avg_metrics.values())[0] if avg_metrics else 0.0
        
        return 0.0

    def _validate_single_dataset(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        """
        Validate on a single dataset.
        
        Returns:
            First metric value (for backward compatibility)
        """
        logger = get_root_logger()
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0.0 for metric in self.opt['val']['metrics'].keys()}

        window_size = self.opt['val'].get('window_size', 0)
        test_fn = partial(self.pad_test, window_size) if window_size else self.nonpad_test

        cnt = 0

        # ---- tqdm progress over clips ----
        pbar = tqdm(total=len(dataloader), desc=f'Validating {dataset_name}', ncols=100)

        for idx, val_data in enumerate(dataloader):
            # Expect list like ['clip_001']; keep as plain name for logging/paths
            clip_folder = val_data['lq_path'][0] if isinstance(val_data['lq_path'], (list, tuple)) else str(val_data['lq_path'])

            self.feed_data(val_data)
            test_fn()

            visuals = self.get_current_visuals()
            sr_frames = visuals['result']                 # [N, C, H, W]
            gt_frames = visuals.get('gt', None)           # [N, C, H, W] or None
            num_frames = sr_frames.size(0)

            # ------------------------------
            # SAVE IMAGES PER FRAME (unchanged)
            # ------------------------------
            if save_img:
                for i in range(num_frames):
                    sr_img = tensor2img([sr_frames[i]], rgb2bgr=rgb2bgr)
                    gt_img = tensor2img([gt_frames[i]], rgb2bgr=rgb2bgr) if gt_frames is not None else None

                    if self.opt['is_train']:
                        save_dir = osp.join(self.opt['path']['visualization'], clip_folder, f"{current_iter}")
                    else:
                        save_dir = osp.join(self.opt['path']['visualization'], dataset_name, clip_folder)
                    os.makedirs(save_dir, exist_ok=True)

                    frame_id = f"{i:04d}"
                    imwrite(sr_img, osp.join(save_dir, f"{frame_id}_derained.png"))
                    if gt_img is not None:
                        imwrite(gt_img, osp.join(save_dir, f"{frame_id}_gt.png"))

            # ------------------------------
            # METRICS: FRAME → CLIP → DATASET (original Restormer: on images)
            # ------------------------------
            if with_metrics and gt_frames is not None:
                clip_metrics = {name: 0.0 for name in self.opt['val']['metrics'].keys()}

                for i in range(num_frames):
                    # Convert tensors to uint8 images via tensor2img (HWC), as in Restormer
                    sr_img = tensor2img([sr_frames[i]], rgb2bgr=rgb2bgr)
                    gt_img = tensor2img([gt_frames[i]], rgb2bgr=rgb2bgr)

                    metric_opt_frame = deepcopy(self.opt['val']['metrics'])
                    for name, opt_ in metric_opt_frame.items():
                        metric_type = opt_.pop('type')
                        val = getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                        clip_metrics[name] += val

                # Average per-clip over frames and accumulate to dataset metric
                for name in clip_metrics:
                    clip_metrics[name] /= num_frames
                    self.metric_results[name] += clip_metrics[name]

            # cleanup
            del self.lq, self.output
            if gt_frames is not None:
                del self.gt
            torch.cuda.empty_cache()

            cnt += 1
            pbar.update(1)

        pbar.close()

        # ------------------------------
        # FINAL DATASET METRICS
        # ------------------------------
        current_metric = 0.0
        if with_metrics and cnt > 0:
            for name in self.metric_results:
                self.metric_results[name] /= cnt
                current_metric = self.metric_results[name]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        return current_metric

    # ==========================
    # LOGGING
    # ==========================
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log = f"Validation {dataset_name}, "
        for name, value in self.metric_results.items():
            log += f"\t# {name}: {value:.4f}"
        logger = get_root_logger()
        logger.info(log)

        if tb_logger:
            for name, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{dataset_name}_{name}", value, current_iter)

    # ==========================
    # VISUALS
    # ==========================
    def get_current_visuals(self):
        out = OrderedDict()
        out['lq'] = self.lq.detach().cpu()
        out['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out['gt'] = self.gt.detach().cpu()
        return out

    # ==========================
    # SAVE
    # ==========================
    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)