"""Single-file BUSI segmentation experiments. Run --help for usage.

BG-Net operators follow Li Yu et al., JBHI 2024:
https://github.com/LiYu51/BG-Net/blob/main/gradconv.py (verified 2026-09-29).
This U-Net adaptation is not a reproduction of the full BG-Net architecture.
HD95 is measured in resized-image pixels, not physical millimetres.
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================
SEED = 42
MODEL_CHOICES = ['unet', 'sequential-gradient', 'parallel-gradient',
                 'adaptive-gradient', 'reliability-gradient', 'denoising-gradient', 'proposed']
MODEL_FILES = {'unet': 'unet', 'sequential-gradient': 'sequential_gradient',
               'proposed': 'proposed_model'}
CORRUPTIONS = {
    'gaussian': [0.03, 0.07, 0.12],  # additive standard deviation, [0,1] units
    'speckle': [0.10, 0.25, 0.40],  # multiplicative standard deviation
    'blur': [0.7, 1.3, 2.0],        # Gaussian sigma, resized-image pixels
    'contrast': [0.8, 0.5, 0.25],   # intensity scaling around image mean
}
SEVERITIES = ['mild', 'moderate', 'severe']
PACKAGES = ['torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 'opencv-python',
            'Pillow', 'scikit-learn', 'scipy', 'albumentations', 'tqdm']


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


# ============================================================
# 3. SYSTEM AND GPU CHECK
# ============================================================
def system_check(run, allow_cpu=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        smi = str(exc)
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'MISSING'
    info = dict(python_executable=sys.executable, python=sys.version, os=platform.platform(),
                pytorch=torch.__version__, cuda_available=torch.cuda.is_available(),
                cuda_version=torch.version.cuda, packages=versions,
                gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Unavailable',
                gpu_memory_gib=(torch.cuda.get_device_properties(0).total_memory / 2**30
                                if device.type == 'cuda' else 0))
    report = json.dumps(info, indent=2) + '\n\n' + smi
    print(report)
    (run / 'system_info.txt').write_text(report, encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (run / 'requirements_used.txt').write_text(freeze.stdout, encoding='utf-8')
    if device.type != 'cuda' and not allow_cpu:
        raise RuntimeError('CUDA required. NVIDIA detection is recorded above. '
                           f'PyTorch CUDA build: {torch.version.cuda}. A None value means CPU-only '
                           'PyTorch; use the existing CUDA environment. No CPU training started.')
    return device


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 5. DATASET VALIDATION
# ============================================================
def read_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L')).copy()


def read_mask(paths, shape, external=False):
    mask = np.zeros(shape, dtype=bool)
    for path in paths:
        with Image.open(path) as im:
            raw = np.asarray(im.convert('RGB') if external else im.convert('L'))
        # BUS-UCLM: red/green lesion; black background (official V3 description).
        part = np.any(raw > 0, axis=-1) if raw.ndim == 3 else raw > 0
        if part.shape != shape:
            raise ValueError(f'Mask/image dimension mismatch: {path}: {part.shape} != {shape}')
        mask |= part
    return mask.astype(np.float32)


def discover_dataset(root, run, include_normal=False, external=False):
    root = Path(root).resolve()
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f'No images under {root}. Extract BUSI benign/malignant/normal '
                                'folders into data/BUSI or pass --data-root to their parent.')
    def is_mask(p):
        return bool(re.search(r'_mask(?:_\d+)?$', p.stem, re.I)) or any(
            s.lower() in {'mask', 'masks', 'gt', 'ground_truth', 'groundtruth', 'segmentations'}
            for s in p.relative_to(root).parts[:-1])
    images, masks = [p for p in files if not is_mask(p)], [p for p in files if is_mask(p)]
    mask_index = defaultdict(list)
    for p in masks:
        key = re.sub(r'_mask(?:_\d+)?$', '', p.stem, flags=re.I).lower()
        mask_index[key].append(p)
    summary = dict(root=str(root), total_images=len(images), total_masks=len(masks),
                   unmatched_images=[], unmatched_masks=[], corrupted_files=[],
                   dimension_mismatches=[], excluded=[], duplicate_filenames={},
                   duplicate_contents=[], classes={}, dimensions={}, mask_dimensions={})
    name_counts = Counter(p.name for p in files)
    summary['duplicate_filenames'] = {k: v for k, v in name_counts.items() if v > 1}
    records, used, hashes = [], set(), defaultdict(list)
    dimensions, mask_dimensions, classes = Counter(), Counter(), Counter()
    for p in masks:
        try:
            with Image.open(p) as im:
                im.load()
                mask_dimensions[str(im.size)] += 1
        except Exception as exc:
            summary['corrupted_files'].append(f'{p}: {exc}')
    for p in images:
        label = next((s.lower() for s in p.parts if s.lower() in ['benign', 'malignant', 'normal']),
                     re.split(r'[ (_]', p.stem.lower())[0])
        candidates = mask_index[p.stem.lower()]
        local = [m for m in candidates if m.parent == p.parent]
        if local:
            candidates = local
        elif len(candidates) > 1:
            # Match repeated stems using their class or mirrored relative directories.
            filtered = [m for m in candidates if label in [s.lower() for s in m.parts]]
            if filtered:
                candidates = filtered
            if len({m.parent for m in candidates}) > 1:
                raise ValueError(f'Ambiguous mask mapping for {p}; use unique names or paired folders.')
        classes[label] += 1
        if not candidates:
            summary['unmatched_images'].append(str(p))
        used.update(candidates)
        try:
            x = read_gray(p)
            dimensions[str(x.shape)] += 1
            digest = hashlib.sha256(str(x.shape).encode() + x.tobytes()).hexdigest()
            hashes[digest].append(str(p))
            y = read_mask(candidates, x.shape, external)
            if not candidates and not (include_normal and label == 'normal'):
                summary['excluded'].append(f'{p}: no mask')
                continue
            if (label == 'normal' or not y.any()) and not include_normal:
                summary['excluded'].append(f'{p}: normal/empty mask')
                continue
            if not y.any() and label != 'normal':
                summary['excluded'].append(f'{p}: empty lesion annotation')
                continue
            records.append(dict(image=str(p), masks=[str(m) for m in candidates],
                                mask_hash=hashlib.sha256(y.tobytes()).hexdigest(),
                                label=label, image_hash=digest, height=x.shape[0], width=x.shape[1]))
        except ValueError as exc:
            summary['dimension_mismatches'].append(str(exc))
        except Exception as exc:
            summary['corrupted_files'].append(f'{p}: {exc}')
    summary['unmatched_masks'] = [str(p) for p in masks if p not in used]
    summary['duplicate_contents'] = [v for v in hashes.values() if len(v) > 1]
    summary.update(classes=dict(classes), dimensions=dict(dimensions),
                   mask_dimensions=dict(mask_dimensions), eligible_images=len(records))
    # Keep exact duplicates together during splitting; do not discard annotation differences.
    summary['split_policy'] = ('Patient identifiers unavailable; reproducible stratified image-level '
                               'split used. Exact duplicate image contents are grouped to prevent leakage.')
    name = 'external_dataset_summary.txt' if external else 'dataset_summary.txt'
    (run / name).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Dataset {root}: {len(images)} images, {len(masks)} masks, {len(records)} eligible; '
          f'{len(summary["duplicate_contents"])} duplicate-content groups; '
          f'{len(summary["corrupted_files"])} corrupt, {len(summary["dimension_mismatches"])} mismatches.')
    print('Classes:', dict(classes), '\nAudit saved:', run / name)
    print('Unmatched images:', len(summary['unmatched_images']),
          'unmatched masks:', len(summary['unmatched_masks']),
          'duplicate filenames:', len(summary['duplicate_filenames']))
    if summary['corrupted_files'] or summary['dimension_mismatches'] or summary['unmatched_masks']:
        raise ValueError('Dataset audit failed; inspect the saved report before training.')
    if not records:
        raise ValueError('No eligible images with valid lesion masks.')
    return records


def make_splits(records, seed, run):
    groups = defaultdict(list)
    for r in records:
        groups[r['image_hash']].append(r)
    keys = sorted(groups)
    labels = [Counter(r['label'] for r in groups[k]).most_common(1)[0][0] for k in keys]
    train, rest = train_test_split(keys, test_size=0.30, stratify=labels, random_state=seed)
    rest_labels = [Counter(r['label'] for r in groups[k]).most_common(1)[0][0] for k in rest]
    val, test = train_test_split(rest, test_size=0.5, stratify=rest_labels, random_state=seed)
    splits = [[r for k in part for r in groups[k]] for part in (train, val, test)]
    print('Patient identifiers unavailable; reproducible stratified image-level split used.')
    print('Exact duplicate content grouped across splits; ratios may differ slightly from 70/15/15.')
    for name, part in zip(('train', 'val', 'test'), splits):
        rows = [dict(r, masks=json.dumps(r['masks'])) for r in part]
        pd.DataFrame(rows).to_csv(run / f'{name}_split.csv', index=False)
        print(name, len(part), dict(Counter(r['label'] for r in part)))
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return splits


# ============================================================
# 6. BUSI DATASET LOADER
# ============================================================
class UltrasoundDataset(Dataset):
    def __init__(self, records, size, augment=False, external=False):
        self.records, self.size, self.augment, self.external = records, size, augment, external

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        x = read_gray(r['image']).astype(np.float32) / 255.0
        y = read_mask(r['masks'], x.shape, self.external)
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(y, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            x, y = augment_pair(x, y)
        return (torch.from_numpy(np.ascontiguousarray(x[None])).float(),
                torch.from_numpy(np.ascontiguousarray(y[None] > 0)).float(), index)


def make_loader(dataset, batch, workers, device, seed, shuffle=False):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), persistent_workers=False)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================
def augment_pair(x, y):
    if random.random() < 0.5:
        x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
    h, w = x.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-10, 10), random.uniform(.95, 1.05))
    matrix[:, 2] += np.array([random.uniform(-.03, .03) * w, random.uniform(-.03, .03) * h])
    x = cv2.warpAffine(x, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    y = cv2.warpAffine(y, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    x = np.clip(x * random.uniform(.9, 1.1) + random.uniform(-.04, .04), 0, 1)
    return x, y


# ============================================================
# 8. SYNTHETIC DEGRADATION AND RELIABILITY MAP
# ============================================================
def corrupt(x, kind, level, spatial, generator=None, values=None):
    values = CORRUPTIONS if values is None else values
    amount = values[kind][level]
    if kind in ('gaussian', 'speckle'):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        degraded = x + amount * noise * (x if kind == 'speckle' else 1)
    elif kind == 'contrast':
        mean = x.mean(dim=(-2, -1), keepdim=True)
        degraded = mean + amount * (x - mean)
    elif kind == 'blur':
        radius = math.ceil(3 * amount)
        grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-grid.square() / (2 * amount**2))
        kernel = kernel / kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        degraded = F.conv2d(F.pad(x, (radius,) * 4, mode='reflect'), kernel)
    else:
        raise ValueError(kind)
    if spatial:
        coarse = torch.rand((x.shape[0], 1, 4, 4), device=x.device, generator=generator)
        strength = F.interpolate(coarse, size=x.shape[-2:], mode='bicubic', align_corners=False)
        low = strength.amin(dim=(-2, -1), keepdim=True)
        high = strength.amax(dim=(-2, -1), keepdim=True)
        strength = ((strength - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    else:
        strength = torch.ones_like(x)
    return ((1 - strength) * x + strength * degraded.clamp(0, 1)).clamp(0, 1), 1 - strength


def training_corruption(x, args):
    if random.random() > args.corruption_probability:
        return x, torch.ones_like(x)
    return corrupt(x, random.choice(list(args.corruptions)), random.randrange(3),
                   random.random() < .5, values=args.corruptions)


# ============================================================
# 9. BASIC CNN BLOCKS
# ============================================================
def conv_norm_relu(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(math.gcd(8, cout), cout), nn.ReLU(inplace=False))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(conv_norm_relu(cin, cout), conv_norm_relu(cout, cout))


# ============================================================
# 10. U-NET ENCODER
# ============================================================
# ============================================================
# U-NET ENCODER
# ============================================================
class Encoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.blocks = nn.ModuleList([DoubleConv(1, base)] +
                                   [DoubleConv(base * 2**i, base * 2**(i+1)) for i in range(3)])
        # ============================================================
        # U-NET BOTTLENECK
        # ============================================================
        self.bottleneck = DoubleConv(base * 8, base * 16)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = F.max_pool2d(x, 2)
        return features, self.bottleneck(x)


# ============================================================
# 11. U-NET DECODER
# ============================================================
# ============================================================
# U-NET DECODER
# ============================================================
class Decoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.blocks = nn.ModuleList([DoubleConv(base * 2**i * 3, base * 2**i)
                                     for i in reversed(range(4))])
        # ============================================================
        # U-NET OUTPUT LAYER
        # ============================================================
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, skips, x):
        for skip, block in zip(reversed(skips), self.blocks):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([skip, x], dim=1))
        return self.output(x)


# ============================================================
# 12. D-GCONV
# ============================================================
class GradientConv(nn.Module):
    """Exact BG-Net gd/cygd weight transforms, depthwise like GradConvBlock.

    Attribution: Li Yu, Wenwen Min, Shunfang Wang, BG-Net gradconv.py.
    gd: learned W multiplied by directional sign matrices, then vector magnitude.
    cygd: two cyclic permutations minus W, then vector magnitude.
    Unlike the original device literal, buffers follow the input device.
    Magnitude is computed in float32 under AMP to prevent overflow/underflow.
    """
    def __init__(self, channels, kind):
        super().__init__()
        self.kind, self.channels = kind, channels
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer('direction_x', torch.tensor([[-1., 0, 1]] * 3)[None, None])
        self.register_buffer('direction_y', torch.tensor([[-1., -1, -1], [0, 0, 0], [1, 1, 1]])[None, None])

    def forward(self, x):
        if self.kind == 'gd':
            wx, wy = self.weight * self.direction_x, self.weight * self.direction_y
        else:
            # ============================================================
            # 13. C-GCONV
            # ============================================================
            w = self.weight.flatten(2)
            wx = (w[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]] - w).reshape_as(self.weight)
            wy = (w[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]] - w).reshape_as(self.weight)
        gx = F.conv2d(x, wx, padding=1, groups=self.channels)
        gy = F.conv2d(x, wy, padding=1, groups=self.channels)
        return (gx.float().square() + gy.float().square() + 1e-7).sqrt().to(x.dtype)


# ============================================================
# 14. RELIABILITY ESTIMATOR
# ============================================================
class ReliabilityEstimator(nn.Sequential):
    def __init__(self, channels):
        super().__init__(conv_norm_relu(3 * channels, 16, 1),
                         nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())


# ============================================================
# 15. SPATIAL ADAPTIVE GRADIENT GATE
# ============================================================
class SpatialGate(nn.Sequential):
    def __init__(self, channels):
        super().__init__(conv_norm_relu(2 * channels + 1, 16, 1),
                         nn.Conv2d(16, 2, 3, padding=1), nn.Softmax(dim=1))


# ============================================================
# 16. GRADIENT DENOISING AUTOENCODER
# ============================================================
class GradientDenoisingAE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 2)
        # ============================================================
        # GRADIENT DENOISING AUTOENCODER - ENCODER
        # ============================================================
        self.encoder = conv_norm_relu(channels, hidden, 1)
        # ============================================================
        # GRADIENT DENOISING AUTOENCODER - LATENT SPACE
        # ============================================================
        self.latent = conv_norm_relu(hidden, hidden)
        # ============================================================
        # GRADIENT DENOISING AUTOENCODER - DECODER
        # ============================================================
        self.decoder = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        return x + self.decoder(self.latent(self.encoder(x)))


class GradientModule(nn.Module):
    def __init__(self, channels, mode):
        super().__init__()
        self.mode = mode
        self.use_reliability = mode in ('reliability-gradient', 'proposed')
        self.use_denoising = mode in ('denoising-gradient', 'proposed')
        self.use_gate = mode in ('adaptive-gradient', 'reliability-gradient', 'denoising-gradient', 'proposed')
        self.d = GradientConv(channels, 'gd')
        self.c = GradientConv(channels, 'cygd')
        self.reliability = ReliabilityEstimator(channels) if self.use_reliability else None
        self.gate = SpatialGate(channels) if self.use_gate else None
        self.ae = GradientDenoisingAE(channels) if self.use_denoising else nn.Identity()
        self.fuse = conv_norm_relu(2 * channels, channels, 1)

    def forward(self, f):
        gd = self.d(f)
        # Sequential baseline is explicitly different from the parallel proposal.
        gc_ = self.c(gd if self.mode == 'sequential-gradient' else f)
        r = (self.reliability(torch.cat([f, gd, gc_], 1)) if self.use_reliability
             else torch.ones_like(f[:, :1]))
        a = (self.gate(torch.cat([gd, gc_, r], 1)) if self.use_gate
             else torch.full_like(f[:, :2], .5))
        ga = gc_ if self.mode == 'sequential-gradient' else a[:, :1] * gd + a[:, 1:] * gc_
        clean = self.ae(ga)
        debug = dict(G_D=gd, G_C=gc_, R=r, A_D=a[:, :1], A_C=a[:, 1:],
                     G_adaptive=ga, G_denoised=clean)
        return self.fuse(torch.cat([f, clean], 1)), debug


# ============================================================
# 17. SEQUENTIAL GRADIENT BASELINE
# ============================================================
# ============================================================
# 18. PROPOSED MODEL
# ============================================================
class SegmentationUNet(nn.Module):
    def __init__(self, mode='unet', base=32):
        super().__init__()
        self.mode = mode
        self.encoder = Encoder(base)
        self.decoder = Decoder(base)
        # Three scales (F1/F2/F3), depthwise operators, memory-conscious 6-GB default.
        self.gradient_modules = nn.ModuleList(
            [GradientModule(base * 2**i, mode) for i in range(3)] if mode != 'unet' else [])

    def forward(self, x, return_debug=False, clean_targets=None, reliability_target=None, reliability_loss_type='l1'):
        skips, bottleneck = self.encoder(x)
        debug, denoise_losses, reliability_losses = [], [], []
        for i, module in enumerate(self.gradient_modules):
            skips[i], maps = module(skips[i])
            if clean_targets is not None and module.use_denoising:
                denoise_losses.append(F.l1_loss(maps['G_denoised'].float(), clean_targets[i].float()))
            if reliability_target is not None and module.use_reliability:
                target = F.interpolate(reliability_target, maps['R'].shape[-2:], mode='area')
                with torch.amp.autocast(device_type=x.device.type, enabled=False):
                    loss_fn = F.l1_loss if reliability_loss_type == 'l1' else F.binary_cross_entropy
                    reliability_losses.append(loss_fn(maps['R'].float(), target.float()))
            if return_debug:
                debug.append(maps)
        prediction = self.decoder(skips, bottleneck)
        zero = prediction.new_zeros(())
        if return_debug or clean_targets is not None or reliability_target is not None:
            return dict(prediction=prediction, scales=debug,
                        denoise_loss=torch.stack(denoise_losses).mean() if denoise_losses else zero,
                        reliability_loss=torch.stack(reliability_losses).mean() if reliability_losses else zero)
        return prediction

    @torch.no_grad()
    def clean_targets(self, x):
        # GroupNorm has no running statistics; teacher uses clean semantic features.
        features, _ = self.encoder(x)
        targets = []
        for f, module in zip(features, self.gradient_modules):
            _, maps = module(f)
            targets.append(maps['G_adaptive'].detach())
        return targets


class SequentialGradientUNet(SegmentationUNet):
    def __init__(self, base=32):
        super().__init__('sequential-gradient', base)


# ============================================================
# 19. LOSS FUNCTIONS
# ============================================================
def soft_dice(prob, target):
    dims = (1, 2, 3)
    return ((2 * (prob * target).sum(dims) + 1e-6) /
            (prob.sum(dims) + target.sum(dims) + 1e-6)).mean()


def boundary(x):
    return F.max_pool2d(x, 3, 1, 1) + F.max_pool2d(-x, 3, 1, 1)


def losses(output, y, args):
    logits = output['prediction'].float()
    p = logits.sigmoid()
    seg = F.binary_cross_entropy_with_logits(logits, y.float()) + 1 - soft_dice(p, y)
    edge = F.l1_loss(boundary(p), boundary(y.float()))
    denoise, reliability = output['denoise_loss'], output['reliability_loss']
    total = seg + args.lambda_boundary * edge + args.lambda_denoise * denoise + args.lambda_reliability * reliability
    return dict(total_loss=total, seg_loss=seg, boundary_loss=edge,
                denoise_loss=denoise, reliability_loss=reliability)


# ============================================================
# 20. EVALUATION METRICS
# ============================================================
def image_metrics(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    def ratio(a, b, empty=1.):
        return float(a / b) if b else float(empty)
    result = dict(dice=ratio(2 * tp, 2 * tp + fp + fn), iou=ratio(tp, tp + fp + fn),
                  precision=ratio(tp, tp + fp, float(not gt.any())), recall=ratio(tp, tp + fn),
                  specificity=ratio(tn, tn + fp), accuracy=ratio(tp + tn, pred.size))
    if not pred.any() and not gt.any():
        hd, bf = 0., 1.
    elif not pred.any() or not gt.any():
        hd, bf = float(np.hypot(*gt.shape)), 0.
    else:
        ep = pred ^ ndimage.binary_erosion(pred, border_value=0)
        eg = gt ^ ndimage.binary_erosion(gt, border_value=0)
        dp, dg = ndimage.distance_transform_edt(~ep), ndimage.distance_transform_edt(~eg)
        # Symmetric surface-distance percentile; 2-pixel Boundary F1 tolerance.
        hd = float(np.percentile(np.concatenate([dg[ep], dp[eg]]), 95))
        precision, recall = float((dg[ep] <= 2).mean()), float((dp[eg] <= 2).mean())
        bf = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    result.update(hd95=hd, boundary_f1=bf)
    return result


def summarize(rows, seed=42):
    df = pd.DataFrame(rows)
    metrics = ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy', 'hd95', 'boundary_f1']
    result = {}
    rng = np.random.default_rng(seed)
    for key in metrics:
        x = df[key].to_numpy(float)
        result[key] = float(x.mean())
        result[key + '_std'] = float(x.std(ddof=1)) if len(x) > 1 else 0.
        means = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
        result[key + '_ci95_low'], result[key + '_ci95_high'] = np.percentile(means, [2.5, 97.5]).tolist()
    return result

# ============================================================
# 21. VISUALIZATION FUNCTIONS
# ============================================================
def finish_figure(fig, path, show):
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print('Figure:', path.resolve())
    if show and plt.get_backend().lower() != 'agg':
        plt.show(block=False)
        plt.pause(1)
    elif show:
        print('Interactive display unavailable; see saved figure.')
    plt.close(fig)


def plot_grid(items, path, show=False, columns=4):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(math.ceil(len(items) / columns), columns,
                             figsize=(4 * columns, 3.7 * math.ceil(len(items) / columns)), squeeze=False)
    for ax, (title, data) in zip(axes.flat, items):
        if torch.is_tensor(data):
            data = data.detach().float().cpu().numpy()
        data = np.asarray(data).squeeze()
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    for ax in list(axes.flat)[len(items):]:
        ax.axis('off')
    finish_figure(fig, path, show)


def data_visuals(records, args, run):
    originals, processed = [], []
    chosen = []
    for label in sorted({r['label'] for r in records}):
        chosen.append(next(r for r in records if r['label'] == label))
    chosen += [r for r in records if r not in chosen]
    chosen = chosen[:3]
    ds = UltrasoundDataset(chosen, args.image_size, augment=True)
    state = random.getstate()
    random.seed(args.seed)
    for i, r in enumerate(chosen):
        x = read_gray(r['image'])
        y = read_mask(r['masks'], x.shape)
        originals.extend([(Path(r['image']).name, x), ('Ground truth', y)])
        px, py, _ = ds[i]
        processed.extend([('Original', x), ('Processed / augmented', px), ('Processed mask', py)])
    random.setstate(state)
    plot_grid(originals, run / 'plots/data_checks/input_masks.png', args.show_plots, 2)
    plot_grid(processed, run / 'plots/data_checks/augmented.png', args.show_plots, 3)


def magnitude(tensor):
    x = tensor[0].detach().float().square().mean(0).sqrt().cpu().numpy()
    return (x - x.min()) / (np.ptp(x) + 1e-8)


def overlay(x, mask):
    out = np.repeat(x[..., None], 3, axis=-1)
    out[mask] = .5 * out[mask] + .5 * np.array([1., 0, 0])
    return out


@torch.no_grad()
def prediction_visuals(model, dataset, device, args, run, tag):
    model.eval()
    for index in range(min(3, len(dataset))):
        x, y, _ = dataset[index]
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
            output = model(x[None].to(device), return_debug=True)
        pred = output['prediction'].sigmoid()[0, 0].cpu().numpy() >= .5
        image = x[0].numpy()
        items = [('Input', image), ('Ground truth', y[0])]
        if output['scales']:
            maps = output['scales'][0]
            for key in ['G_D', 'G_C', 'R', 'A_D', 'A_C', 'G_adaptive', 'G_denoised']:
                items.append((key, maps[key][0, 0] if key in ('R', 'A_D', 'A_C') else magnitude(maps[key])))
        items += [('Prediction', pred), ('Overlay', overlay(image, pred)),
                  ('Error map', pred != y[0].numpy().astype(bool))]
        plot_grid(items, run / f'plots/{tag}/sample_{index}.png', args.show_plots)
        dest = run / 'predictions' / tag
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred * 255).astype('uint8')).save(dest / f'sample_{index}.png')
        if model.mode == 'proposed':
            for scale, maps in enumerate(output['scales']):
                plot_grid([(key, maps[key][0, 0] if key in ('R', 'A_D', 'A_C') else magnitude(maps[key]))
                           for key in maps], run / f'plots/gradient_debug/{tag}_{index}_scale{scale}.png',
                          args.show_plots)


def history_plots(history, args, run, mode):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    for name, cols in [('training_loss', ['train_total_loss', 'val_total_loss']),
                       ('validation_dice', ['val_dice']), ('validation_iou', ['val_iou']),
                       ('learning_rate', ['learning_rate'])]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in cols:
            ax.plot(df.epoch, df[col], label=col)
        ax.set_xlabel('Epoch')
        ax.legend()
        finish_figure(fig, run / f'plots/training_curves/{mode}/{name}.png', args.show_plots)


def gpu_memory(device):
    return f'{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated' if device.type == 'cuda' else 'CPU'


def model_info(model, args, run, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = (f'Model: {model.mode}\nTotal parameters: {total:,}\nTrainable parameters: {trainable:,}\n'
              f'FP32 parameter size: {total * 4 / 2**20:.2f} MiB\n')
    (run / 'logs' / f'architecture_{model.mode}.txt').write_text(report + str(model), encoding='utf-8')
    print(report)
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
        x = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
        out = model(x, return_debug=True)
    print('Input shape:', tuple(x.shape), 'Output shape:', tuple(out['prediction'].shape))
    for scale, maps in enumerate(out['scales']):
        print('Scale', scale, {k: tuple(v.shape) for k, v in maps.items()})


# ============================================================
# 22. TRAINING FUNCTION
# ============================================================
def train_epoch(model, loader, optimizer, scaler, args, device, epoch):
    model.train()
    totals, seen = defaultdict(float), 0
    bar = tqdm(loader, desc=f'Train {model.mode} {epoch}/{args.epochs}', file=sys.stdout)
    for clean, y, _ in bar:
        clean, y = clean.to(device, non_blocking=True), y.to(device, non_blocking=True)
        noisy, target = training_corruption(clean, args)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            teacher = model.clean_targets(clean) if model.mode in ('denoising-gradient', 'proposed') else None
            output = model(noisy, clean_targets=teacher, reliability_target=target,
                           reliability_loss_type=args.reliability_loss)
            terms = losses(output, y, args)
        if not torch.isfinite(terms['total_loss']):
            raise FloatingPointError(f'Non-finite loss: {model.mode}, epoch {epoch}')
        scaler.scale(terms['total_loss']).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=not scaler.is_enabled())
        if not torch.isfinite(norm):
            print('AMP gradient overflow: skipping update and reducing loss scale.')
        scaler.step(optimizer)
        scaler.update()
        batch = len(clean)
        for key, val in terms.items():
            totals[key] += val.item() * batch
        with torch.no_grad():
            p = output['prediction'].sigmoid() >= .5
            tp = (p * y).sum((1, 2, 3))
            denom = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
            dice = ((2 * tp + 1e-6) / (denom + 1e-6)).mean().item()
            iou = ((tp + 1e-6) / (denom - tp + 1e-6)).mean().item()
        totals['dice'] += dice * batch
        totals['iou'] += iou * batch
        seen += batch
        bar.set_postfix(loss=f'{terms["total_loss"].item():.4f}', dice=f'{dice:.3f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.2g}', gpu=gpu_memory(device))
        del teacher, output, terms
    return {k: v / seen for k, v in totals.items()}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best, counter, history, args, splits):
    return dict(epoch=epoch, model_name=model.mode, model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                amp_scaler_state_dict=scaler.state_dict(), best_validation_dice=best,
                configuration=vars(args), seed=args.seed, early_stopping_counter=counter,
                history=history, rng_state=rng_state(), splits=splits)


def atomic_checkpoint(path, payload):
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_model(mode, splits, args, run, device):
    seed_everything(args.seed)
    model = SegmentationUNet(mode, args.base_channels).to(device)
    model_info(model, args, run, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=.5)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == 'cuda')
    start, best, counter, history = 1, -1., 0, []
    stem = MODEL_FILES.get(mode, mode.replace('-', '_'))
    best_path, last_path = [run / 'checkpoints' / f'{prefix}_{stem}.pth' for prefix in ['best', 'last']]
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['model_name'] != mode:
            raise ValueError('--resume model does not match --model')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['amp_scaler_state_dict'])
        start, best = state['epoch'] + 1, state['best_validation_dice']
        counter, history = state['early_stopping_counter'], state['history']
        restore_rng(state['rng_state'])
        previous_best = Path(args.resume).parent / f'best_{stem}.pth'
        if not previous_best.exists():
            raise FileNotFoundError('Resume requires the best checkpoint alongside the last checkpoint.')
        import shutil
        shutil.copy2(previous_best, best_path)
        del state
    train_ds = UltrasoundDataset(splits[0], args.image_size, augment=True)
    val_ds = UltrasoundDataset(splits[1], args.image_size)
    batch = int(history[-1]['batch_size']) if history else args.batch_size
    epoch = start
    if history:
        print('Resuming with recorded effective batch size:', batch)
    while epoch <= args.epochs:
        began = time.perf_counter()
        recovery_path = run / 'checkpoints' / f'recovery_{stem}.pth'
        atomic_checkpoint(recovery_path, checkpoint_payload(model, optimizer, scheduler, scaler,
                          epoch-1, best, counter, history, args, splits))
        oom = False
        try:
            train_loader = make_loader(train_ds, batch, args.num_workers, device, args.seed + epoch, True)
            val_loader = make_loader(val_ds, batch, args.num_workers, device, args.seed)
            lr = optimizer.param_groups[0]['lr']
            training = train_epoch(model, train_loader, optimizer, scaler, args, device, epoch)
            validation, _ = evaluate(model, val_loader, args, device)
        except torch.cuda.OutOfMemoryError:
            print(f'CUDA OOM during {mode} epoch {epoch}; batch size {batch}. Restoring epoch start.')
            oom = True
        # Leave exception scope before releasing traceback-held activation graphs.
        if oom:
            if batch == 1:
                raise RuntimeError('OOM at batch size 1; explicitly choose smaller --base-channels or --image-size.')
            batch = max(1, batch // 2)
            args.final_batch_sizes[mode] = batch
            save_json(run / 'config.json', vars(args))
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            state = torch.load(recovery_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == 'cuda')
            scaler.load_state_dict(state['amp_scaler_state_dict'])
            restore_rng(state['rng_state'])
            del state
            print('Retry batch size:', batch)
            continue
        improved = validation['dice'] > best
        best, counter = (validation['dice'], 0) if improved else (best, counter + 1)
        scheduler.step(validation['dice'])
        row = dict(model=mode, epoch=epoch, learning_rate=lr, batch_size=batch,
                   **{'train_' + k: v for k, v in training.items()},
                   **{'val_' + k: v for k, v in validation.items()}, epoch_time=time.perf_counter()-began)
        history.append(row)
        pd.DataFrame(history).to_csv(run / 'metrics' / f'training_history_{stem}.csv', index=False)
        histories = [pd.read_csv(p) for p in (run / 'metrics').glob('training_history_*.csv')]
        pd.concat(histories, ignore_index=True).to_csv(run / 'metrics/training_history.csv', index=False)
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best, counter, history, args, splits)
        atomic_checkpoint(last_path, payload)
        if improved:
            atomic_checkpoint(best_path, payload)
        print('\n' + '=' * 60 + f'\nEpoch {epoch}/{args.epochs}: {mode}')
        print(f'Learning Rate: {lr:.6f}\nGPU: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}'
              f'\nGPU Memory Used: {gpu_memory(device)}')
        for key, val in training.items():
            print(f'Train {key}: {val:.6f}')
        for key, val in validation.items():
            print(f'Validation {key}: {val:.6f}')
        print(f'Best Validation Dice: {best:.6f}\nEarly Stopping Counter: {counter}/{args.patience}'
              f'\nEpoch Time: {row["epoch_time"]:.1f}s\n' + '=' * 60)
        del payload
        if counter >= args.patience:
            break
        epoch += 1
    args.final_batch_sizes[mode] = batch
    save_json(run / 'config.json', vars(args))
    history_plots(history, args, run, mode)
    state = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    return model, best, best_path


# ============================================================
# 23. VALIDATION FUNCTION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, args, device, corruption=None, spatial=False, save_predictions=None):
    model.eval()
    rows, totals, seen = [], defaultdict(float), 0
    for clean, y, indices in tqdm(loader, desc='Evaluate', file=sys.stdout):
        clean, y = clean.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if corruption:
            noisy, reliability = [], []
            for x, index in zip(clean, indices):
                gen = torch.Generator(device=device).manual_seed(args.seed + int(index) * 101 + corruption[1])
                cx, cr = corrupt(x[None], *corruption, spatial, generator=gen, values=args.corruptions)
                noisy.append(cx)
                reliability.append(cr)
            inputs, target = torch.cat(noisy), torch.cat(reliability)
        else:
            inputs, target = clean, torch.ones_like(clean)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
            teacher = model.clean_targets(clean) if model.mode in ('denoising-gradient', 'proposed') else None
            output = model(inputs, clean_targets=teacher, reliability_target=target,
                           reliability_loss_type=args.reliability_loss)
            terms = losses(output, y, args)
        if not torch.isfinite(output['prediction']).all():
            raise FloatingPointError('Non-finite evaluation predictions')
        for key, val in terms.items():
            totals[key] += val.item() * len(clean)
        seen += len(clean)
        predictions = output['prediction'].sigmoid().cpu().numpy()[:, 0] >= .5
        truth = y.cpu().numpy()[:, 0] > 0
        for pred, gt, index in zip(predictions, truth, indices):
            record = loader.dataset.records[int(index)]
            rows.append(dict(image=record['image'], label=record['label'], **image_metrics(pred, gt)))
            if save_predictions:
                save_predictions.mkdir(parents=True, exist_ok=True)
                Image.fromarray((pred * 255).astype('uint8')).save(save_predictions / f'{int(index):04d}.png')
    summary = {k: float(np.mean([r[k] for r in rows])) for k in image_metrics(np.zeros((2, 2)), np.zeros((2, 2)))}
    summary.update({k: v / seen for k, v in totals.items()})
    return summary, rows


# ============================================================
# 24. TEST FUNCTION
# ============================================================
def test_model(model, records, args, run, device):
    ds = UltrasoundDataset(records, args.image_size)
    loader = make_loader(ds, args.final_batch_sizes.get(model.mode, args.batch_size), args.num_workers, device, args.seed)
    _, rows = evaluate(model, loader, args, device, save_predictions=run / 'predictions' / model.mode / 'test')
    for row in rows:
        row['model'] = model.mode
    pd.DataFrame(rows).to_csv(run / 'metrics' / f'per_image_{model.mode}.csv', index=False)
    prediction_visuals(model, ds, device, args, run, MODEL_FILES.get(model.mode, model.mode) + '_predictions')
    return dict(model=model.mode, **summarize(rows, args.seed)), rows


# ============================================================
# 25. ROBUSTNESS EVALUATION
# ============================================================
def robustness_model(model, records, args, run, device):
    ds = UltrasoundDataset(records, args.image_size)
    loader = make_loader(ds, args.final_batch_sizes.get(model.mode, args.batch_size), args.num_workers, device, args.seed)
    rows = []
    for spatial in ([False, True] if args.robustness_scope == 'both' else [args.robustness_scope == 'spatial']):
        for kind in args.corruptions:
            for level, severity in enumerate(SEVERITIES):
                print(f'Robustness: {model.mode}, {kind}, {severity}, spatial={spatial}')
                _, images = evaluate(model, loader, args, device, (kind, level), spatial)
                row = dict(model=model.mode, corruption=kind, severity=severity, level=level + 1,
                           strength=args.corruptions[kind][level], scope='spatial' if spatial else 'global',
                           **summarize(images, args.seed))
                rows.append(row)
                pd.DataFrame(images).to_csv(run / 'metrics' /
                    f'robustness_per_image_{model.mode}_{kind}_{severity}_{row["scope"]}.csv', index=False)
    return rows


def robustness_plots(rows, args, run):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    for metric in ['dice', 'iou', 'hd95']:
        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        for ax, kind in zip(axes, args.corruptions):
            for (model, scope), part in df[df.corruption.isin([kind, 'clean'])].groupby(['model', 'scope']):
                part = part.sort_values('level')
                ax.plot(part.level, part[metric], marker='o', label=f'{model}/{scope}')
            ax.set_title(kind)
            ax.set_xticks([0, 1, 2, 3], ['clean', 'mild', 'moderate', 'severe'])
            ax.set_ylabel(metric + (' (pixels)' if metric == 'hd95' else ''))
        axes[-1].legend(fontsize=6)
        finish_figure(fig, run / f'plots/robustness/{metric}_vs_severity.png', args.show_plots)


@torch.no_grad()
def robustness_visuals(checkpoints, records, args, run, device):
    ds = UltrasoundDataset(records[:3], args.image_size)
    for kind in args.corruptions:
        for index in range(min(3, len(ds))):
            x, _, _ = ds[index]
            x = x[None].to(device)
            inputs = [x]
            for level in range(3):
                gen = torch.Generator(device=device).manual_seed(args.seed + index * 101 + level)
                inputs.append(corrupt(x, kind, level, False, gen, args.corruptions)[0])
            items = [(name, t[0, 0]) for name, t in zip(['Clean'] + SEVERITIES, inputs)]
            for mode, path in checkpoints.items():
                model = SegmentationUNet(mode, args.base_channels).to(device)
                state = torch.load(path, map_location='cpu', weights_only=False)
                model.load_state_dict(state['model_state_dict'])
                model.eval()
                for name, inp in zip(['Clean'] + SEVERITIES, inputs):
                    with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                        pred = model(inp).sigmoid()[0, 0] >= .5
                    items.append((f'{mode}: {name}', pred))
                del model, state
            plot_grid(items, run / f'plots/robustness/{kind}_sample_{index}.png', args.show_plots, 4)


# ============================================================
# 26. EXTERNAL VALIDATION
# ============================================================
def external_validation(model, records, args, run, device):
    ds = UltrasoundDataset(records, args.image_size, external=True)
    loader = make_loader(ds, args.final_batch_sizes.get(model.mode, args.batch_size), args.num_workers, device, args.seed)
    _, rows = evaluate(model, loader, args, device)
    pd.DataFrame(rows).to_csv(run / 'metrics' / f'external_per_image_{model.mode}.csv', index=False)
    prediction_visuals(model, ds, device, args, run, f'external_validation/{model.mode}')
    return dict(model=model.mode, dataset='BUS-UCLM', **summarize(rows, args.seed))


def self_test(device):
    """Contract tests: real operators, metric edge cases and backpropagation."""
    import torch
    import numpy as np
    empty = np.zeros((32, 32), dtype=bool)
    full = ~empty
    assert image_metrics(empty, empty)['dice'] == 1.0
    assert image_metrics(empty, full)['dice'] == 0.0
    assert np.isfinite(image_metrics(empty, full)['hd95'])
    x = torch.randn(2, 4, 32, 32, device=device, requires_grad=True)
    for kind in ('gd', 'cygd'):
        op = GradientConv(4, kind).to(device)
        y = op(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.mean().backward()
        assert op.weight.grad is not None and torch.isfinite(op.weight.grad).all()
    for kind in MODEL_CHOICES:
        model = SegmentationUNet(kind, base=8).to(device)
        out = model(torch.rand(2, 1, 32, 32, device=device), return_debug=True)
        assert out['prediction'].shape == (2, 1, 32, 32)
        for scale in out['scales']:
            assert torch.allclose(scale['A_D'] + scale['A_C'], torch.ones_like(scale['A_D']), atol=1e-6)
            assert scale['R'].min() >= 0 and scale['R'].max() <= 1
        out['prediction'].mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('SELF TEST PASSED: metrics, operators, all model shapes, gates and backward')


def integration_check(splits, args, run, device):
    """Small real-data I/O checks; outputs are explicitly not research results."""
    args = copy.deepcopy(args)
    args.epochs, args.resume, args.show_plots = 1, None, False
    args.corruption_probability = 1.0
    args.experiment_purpose = 'SOFTWARE INTEGRATION CHECK ONLY: 8 train / 2 val / 2 test images'
    check = run / 'logs' / 'integration_check'
    for directory in ['checkpoints', 'metrics', 'plots', 'predictions', 'logs']:
        (check / directory).mkdir(parents=True, exist_ok=True)
    save_json(check / 'config.json', vars(args))
    small = [part[:n] for part, n in zip(splits, [8, 2, 2])]
    checks, checkpoints = [], {}
    for mode in ['unet', 'sequential-gradient', 'proposed']:
        model, _, path = train_model(mode, small, args, check, device)
        checkpoints[mode] = path
        _, rows = test_model(model, small[2], args, check, device)
        assert len(rows) == 2 and all(np.isfinite(r['hd95']) for r in rows)
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert {'optimizer_state_dict', 'scheduler_state_dict', 'amp_scaler_state_dict',
                'rng_state', 'splits', 'seed'}.issubset(state)
        checks.append({'model': mode, 'checkpoint_and_evaluation': 'PASS'})
        del state, model
    # Verify optimizer/scaler/history restoration by actually continuing one epoch.
    args.resume = str(check / 'checkpoints/last_proposed_model.pth')
    args.model, args.epochs = 'proposed', 2
    resumed = check / 'resumed'
    for directory in ['checkpoints', 'metrics', 'plots', 'predictions', 'logs']:
        (resumed / directory).mkdir(parents=True, exist_ok=True)
    model, _, _ = train_model('proposed', small, args, resumed, device)
    history = pd.read_csv(resumed / 'metrics/training_history.csv')
    assert history.epoch.tolist() == [1, 2]
    robust = robustness_model(model, small[2], args, resumed, device)
    assert len(robust) == (24 if args.robustness_scope == 'both' else 12)
    assert all(np.isfinite(r['dice']) and np.isfinite(r['hd95']) for r in robust)
    pd.DataFrame(robust).to_csv(resumed / 'metrics/robustness_results.csv', index=False)
    robustness_plots(robust, args, resumed)
    checks.append({'resume': 'PASS', 'robustness_conditions': len(robust),
                   'purpose': args.experiment_purpose})
    save_json(check / 'checks_passed.json', checks)
    print('INTEGRATION CHECK PASSED: training, validation, test, checkpoints, resume, robustness and plots.')
    return checks


def sanity_pipeline(splits, args, run, device):
    """Real BUSI forward/backward tests followed by tiny-training-set memorization."""
    self_test(device)
    ds = UltrasoundDataset(splits[0][:4], args.image_size)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    y = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    print('Real BUSI input:', tuple(x.shape), 'mask:', tuple(y.shape), 'device:', x.device)
    results = []
    for mode in ['unet', 'sequential-gradient', 'proposed']:
        seed_everything(args.seed)
        model = SegmentationUNet(mode, args.base_channels).to(device)
        model_info(model, args, run, device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == 'cuda', init_scale=256.)
        # One corrupted batch verifies all auxiliary targets, gates and gradients.
        noisy, reliability = corrupt(x[:1], 'speckle', 1, True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            teacher = model.clean_targets(x[:1]) if mode == 'proposed' else None
            output = model(noisy, return_debug=True, clean_targets=teacher, reliability_target=reliability,
                           reliability_loss_type=args.reliability_loss)
            terms = losses(output, y[:1], args)
        print(mode, 'losses:', {k: float(v.detach()) for k, v in terms.items()})
        for i, maps in enumerate(output['scales']):
            for key, value in maps.items():
                assert torch.isfinite(value).all(), (mode, key)
                print(f'Scale {i}: {key} {tuple(value.shape)}, range '
                      f'[{value.min().item():.4f}, {value.max().item():.4f}]')
            error = (maps['A_D'].float() + maps['A_C'].float() - 1).abs().max().item()
            assert error < .002, error
            print('Gate sum maximum error:', error)
        scaler.scale(terms['total_loss']).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        print('Backward PASS;', len(grads), 'parameter gradients;', gpu_memory(device))
        del output, terms, teacher, grads, noisy, reliability
        optimizer.zero_grad(set_to_none=True)
        # Diagnostic unscale was not an optimizer step: start a fresh scaling cycle.
        scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == 'cuda', init_scale=256.)
        history, achieved = [], False
        began = time.perf_counter()
        for step in tqdm(range(1, args.overfit_steps + 1), desc=f'Tiny-set overfit: {mode}', file=sys.stdout):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            # Microbatches keep this sanity check safe on a 6-GB GPU.
            loss_value = 0.
            for i in range(len(x)):
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    out = model(x[i:i+1], reliability_target=torch.ones_like(x[i:i+1]),
                                reliability_loss_type=args.reliability_loss)
                    terms = losses(out, y[i:i+1], args)
                    loss = terms['total_loss'] / len(x)
                scaler.scale(loss).backward()
                loss_value += loss.item()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=not scaler.is_enabled())
            if not torch.isfinite(norm):
                print('AMP overflow in overfit: skipped update; lowering scale.')
            scaler.step(optimizer)
            scaler.update()
            if step % 10 == 0 or step == 1 or step == args.overfit_steps:
                model.eval()
                values = []
                with torch.no_grad():
                    for i in range(len(x)):
                        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                            p = model(x[i:i+1]).sigmoid()[0, 0].cpu().numpy() >= .5
                        values.append(image_metrics(p, y[i, 0].cpu().numpy())['dice'])
                dice = float(np.mean(values))
                history.append(dict(model=mode, step=step, loss=loss_value, dice=dice))
                print(f'\nTiny-set {mode}: step={step}, loss={loss_value:.5f}, Dice={dice:.5f}')
                if dice >= args.overfit_dice:
                    achieved = True
                    break
        pd.DataFrame(history).to_csv(run / 'metrics' / f'sanity_overfit_{mode}.csv', index=False)
        prediction_visuals(model, ds, device, args, run, f'sanity_overfit_{mode}')
        torch.save(dict(model_state_dict=model.state_dict(), configuration=vars(args),
                        model_name=mode, purpose='TINY TRAINING SET SANITY ONLY, NOT RESEARCH RESULTS'),
                   run / 'checkpoints' / f'sanity_only_{mode}.pth')
        results.append(dict(model=mode, passed=achieved, final_training_dice=dice,
                            steps=step, seconds=time.perf_counter()-began))
        save_json(run / 'sanity_results.json', results)
        del model, optimizer, scaler, out, terms, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if not achieved:
            raise RuntimeError(f'{mode} tiny-set Dice {dice:.4f} did not reach {args.overfit_dice}; '
                               'full training blocked. Inspect sanity plots and adjust/debug before proceeding.')
    print('SANITY PASSED: real data, shapes, losses, backpropagation and tiny-set overfit for all three models.')
    return results


# ============================================================
# 27. MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--external-data-root', default=None)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--mode', choices=['sanity', 'full', 'self-test', 'audit'], default='sanity')
    parser.add_argument('--model', choices=MODEL_CHOICES + ['all'], default='all')
    parser.add_argument('--ablation', choices=list('ABCDEFG'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lambda-boundary', type=float, default=.2)
    parser.add_argument('--lambda-denoise', type=float, default=.1)
    parser.add_argument('--lambda-reliability', type=float, default=.1)
    parser.add_argument('--reliability-loss', choices=['l1', 'bce'], default='l1')
    parser.add_argument('--corruption-probability', type=float, default=.5)
    parser.add_argument('--corruption-config', help='JSON file overriding all four three-level corruption arrays')
    parser.add_argument('--resume', help='Trusted local last checkpoint; matching best checkpoint must be alongside it')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--run-robustness', action='store_true')
    parser.add_argument('--robustness-scope', choices=['global', 'spatial', 'both'], default='both')
    parser.add_argument('--run-external-validation', action='store_true')
    parser.add_argument('--include-normal-empty-masks', action='store_true')
    parser.add_argument('--overfit-steps', type=int, default=200)
    parser.add_argument('--overfit-dice', type=float, default=.90)
    parser.add_argument('--sanity-report', help='Previously passed sanity_results.json to avoid repeating overfit in full mode')
    parser.add_argument('--integration-check', action='store_true', help='Also exercise checkpoint resume and robustness on a small BUSI subset')
    parser.add_argument('--allow-cpu-self-test', action='store_true', help='Only allows CPU in self-test mode')
    args = parser.parse_args()
    if args.ablation:
        args.model = dict(zip('ABCDEFG', MODEL_CHOICES))[args.ablation]
    if args.image_size < 32 or args.image_size % 16:
        parser.error('--image-size must be >=32 and divisible by 16')
    if min(args.batch_size, args.epochs, args.overfit_steps, args.base_channels) < 1:
        parser.error('Batch size, epochs, overfit steps and base channels must be positive')
    if args.base_channels < 8 or not 0 <= args.corruption_probability <= 1:
        parser.error('base channels >=8 and corruption probability in [0,1] required')
    if not 0 < args.overfit_dice <= 1 or args.num_workers < 0 or args.patience < 1:
        parser.error('Invalid overfit threshold, worker count or early-stopping patience')
    if args.lr <= 0 or min(args.weight_decay, args.lambda_boundary, args.lambda_denoise, args.lambda_reliability) < 0:
        parser.error('Learning rate must be positive; loss weights and weight decay must be nonnegative')
    if args.resume and args.model == 'all':
        parser.error('--resume requires a single --model')
    args.corruptions = json.loads(Path(args.corruption_config).read_text()) if args.corruption_config else copy.deepcopy(CORRUPTIONS)
    if set(args.corruptions) != set(CORRUPTIONS) or any(len(v) != 3 for v in args.corruptions.values()):
        parser.error('Corruption config must specify gaussian, speckle, blur, contrast with three values each')
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for values in args.corruptions.values() for x in values):
        parser.error('Corruption values must be finite positive numbers')
    args.final_batch_sizes = {}
    return args


def locate_busi(data_root):
    if data_root:
        return Path(data_root).resolve()
    for name in ['data', 'dataset', 'datasets', 'BUSI', 'Dataset_BUSI_with_GT']:
        root = Path(name)
        if root.exists() and any(root.rglob('*_mask.png')):
            return root.resolve()
    raise FileNotFoundError('BUSI unavailable. Place extracted BUSI under data/BUSI/benign, malignant, normal. '
                            'Official source: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset . '
                            'No substitute dataset will be used.')


def main():
    args = parse_args()
    import matplotlib
    if not args.show_plots:
        matplotlib.use('Agg')
    run = Path(args.output_dir).resolve() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    for folder in ['checkpoints', 'logs', 'metrics', 'plots', 'predictions',
                   'plots/data_checks', 'plots/training_curves', 'plots/unet_predictions',
                   'plots/sequential_gradient_predictions', 'plots/gradient_debug',
                   'plots/proposed_predictions', 'plots/robustness', 'plots/external_validation']:
        (run / folder).mkdir(parents=True, exist_ok=True)
    log = (run / 'logs/terminal.log').open('w', encoding='utf-8')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    started = time.perf_counter()
    try:
        print('Run directory:', run)
        args.operator_source = 'https://github.com/LiYu51/BG-Net/blob/main/gradconv.py; verified 2026-09-29'
        args.gradient_scales = [1, 2, 3]
        args.synthetic_reliability = dict(map_grid=[4, 4], interpolation='bicubic then min-max to [0,1]',
                                          spatial_probability=.5, clean_target=1,
                                          global_corruption_target=0, target_definition='1-M')
        args.metric_conventions = 'HD95: symmetric concatenated surface distances, resized pixels; empty mismatch=image diagonal; Boundary F1 tolerance=2 pixels; CI=1000 image bootstrap means'
        args.augmentation = dict(horizontal_flip=.5, rotation_degrees=10, scale=[.95, 1.05],
                                 translation_fraction=.03, brightness=[-.04, .04], contrast=[.9, 1.1])
        args.script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save_json(run / 'config.json', vars(args))
        device = system_check(run, args.allow_cpu_self_test and args.mode == 'self-test')
        seed_everything(args.seed)
        if args.mode == 'self-test':
            self_test(device)
            return
        args.data_root = str(locate_busi(args.data_root))
        records = discover_dataset(args.data_root, run, args.include_normal_empty_masks)
        splits = make_splits(records, args.seed, run)
        if args.resume:
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            if saved['splits'] != splits:
                raise ValueError('Dataset/splits differ from checkpoint. Restore the original dataset, seed and paths.')
            for key in ['image_size', 'base_channels', 'seed', 'corruptions', 'lambda_boundary', 'lambda_denoise', 'lambda_reliability', 'corruption_probability', 'reliability_loss']:
                if saved['configuration'][key] != getattr(args, key):
                    raise ValueError(f'Resume configuration mismatch: {key}')
            del saved
        save_json(run / 'config.json', vars(args))
        data_visuals(splits[0], args, run)
        if args.mode == 'audit':
            print('DATASET AUDIT COMPLETE:', run)
            return
        if args.sanity_report and args.mode == 'full':
            report_path = Path(args.sanity_report)
            report = json.loads(report_path.read_text())
            old = json.loads((report_path.parent / 'config.json').read_text())
            if not all(r['passed'] for r in report) or {r['model'] for r in report} != {'unet', 'sequential-gradient', 'proposed'}:
                raise ValueError('Sanity report is incomplete or failed')
            for key in ['image_size', 'base_channels', 'data_root', 'seed', 'script_sha256']:
                if old[key] != getattr(args, key):
                    raise ValueError(f'Sanity report does not match current {key}')
            print('Using passed sanity report:', report_path)
        else:
            sanity_pipeline(splits, args, run, device)
        if args.mode == 'sanity':
            if args.integration_check:
                integration_check(splits, args, run, device)
            print('Sanity experiment completed. Research training NOT RUN. Results:', run)
            return
        modes = ['unet', 'sequential-gradient', 'proposed'] if args.model == 'all' else [args.model]
        external = None
        if args.run_external_validation:
            ext = Path(args.external_data_root) if args.external_data_root else None
            if ext is None:
                candidates = [p for root in ['data', 'dataset', 'datasets', '.']
                              if Path(root).exists() for p in Path(root).glob('*')
                              if p.is_dir() and 'uclm' in p.name.lower()]
                ext = candidates[0] if candidates else None
            if ext and ext.is_dir():
                args.external_data_root = str(ext.resolve())
                save_json(run / 'config.json', vars(args))
                external = discover_dataset(ext, run, args.include_normal_empty_masks, external=True)
            else:
                print('External BUS-UCLM skipped: dataset unavailable; no retraining or substitution.')
        comparisons, image_rows, robust_rows, external_rows, checkpoints, bests = [], [], [], [], {}, {}
        for mode in modes:
            model, bests[mode], checkpoints[mode] = train_model(mode, splits, args, run, device)
            if mode == 'proposed':
                prediction_visuals(model, UltrasoundDataset(splits[1], args.image_size), device,
                                   args, run, 'proposed_validation')
            result, rows = test_model(model, splits[2], args, run, device)
            comparisons.append(result)
            image_rows.extend(rows)
            pd.DataFrame(comparisons).to_csv(run / 'metrics/model_comparison.csv', index=False)
            pd.DataFrame(comparisons).to_csv(run / 'metrics/clean_test_results.csv', index=False)
            pd.DataFrame(image_rows).to_csv(run / 'metrics/per_image_metrics.csv', index=False)
            if mode == 'unet' and bests[mode] < .1:
                raise RuntimeError('U-Net validation Dice below 0.1: inspect masks before continuing to gradient models.')
            if args.run_robustness:
                for scope in (['global', 'spatial'] if args.robustness_scope == 'both' else [args.robustness_scope]):
                    robust_rows.append(dict(result, corruption='clean', severity='clean', level=0, strength=0, scope=scope))
                robust_rows.extend(robustness_model(model, splits[2], args, run, device))
                pd.DataFrame(robust_rows).to_csv(run / 'metrics/robustness_results.csv', index=False)
            if external:
                external_rows.append(external_validation(model, external, args, run, device))
                pd.DataFrame(external_rows).to_csv(run / 'metrics/external_validation_BUS_UCLM.csv', index=False)
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if robust_rows:
            robustness_plots(robust_rows, args, run)
            robustness_visuals(checkpoints, splits[2], args, run, device)
        print('\nExperiment completed.\nGPU used:', torch.cuda.get_device_name(0))
        print('Dataset:', args.data_root, '\nTraining/validation/test images:', [len(s) for s in splits])
        for mode in ['unet', 'sequential-gradient', 'proposed']:
            print('Best validation Dice', mode, ':', bests.get(mode, 'NOT RUN'))
        print(pd.DataFrame(comparisons)[['model', 'dice', 'iou', 'precision', 'recall', 'specificity', 'hd95', 'boundary_f1']].to_string(index=False))
        print('Best models:', checkpoints, '\nPlots:', run / 'plots')
        print('Robustness:', run / 'metrics/robustness_results.csv' if robust_rows else 'NOT RUN')
        print('External BUS-UCLM:', 'Completed' if external_rows else 'Skipped')
        save_json(run / 'completion.json', dict(status='completed', best_validation=bests, checkpoints=checkpoints))
    except Exception:
        traceback.print_exc()
        (run / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        print(f'Total runtime: {time.perf_counter()-started:.1f}s\nResults: {run}')
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()


if __name__ == '__main__':
    main()
