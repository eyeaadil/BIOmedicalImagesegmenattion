"""Single-file BUSI segmentation experiments. Run --help for usage.

BG-Net operators follow Li Yu et al., JBHI 2024:
https://github.com/LiYu51/BG-Net/blob/main/gradconv.py (verified 2026-09-29).
This U-Net adaptation is not a reproduction of the full BG-Net architecture.
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
from torch.utils.data import Dataset, DataLoader, Sampler
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
def seed_everything(seed, fast_mode=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = fast_mode
    torch.backends.cudnn.deterministic = not fast_mode
    torch.use_deterministic_algorithms(not fast_mode, warn_only=True)
    # RTX Ada GPUs can accelerate convolutions/matmuls with TF32 in fast mode.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = fast_mode
        torch.backends.cudnn.allow_tf32 = fast_mode


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
        epoch = 0
        if isinstance(index, tuple):
            epoch, index = index
        r = self.records[index]
        x = read_gray(r['image']).astype(np.float32) / 255.0
        y = read_mask(r['masks'], x.shape, self.external)
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        y = cv2.resize(y, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            state = random.getstate()
            random.seed(getattr(self, 'seed', SEED) + epoch * 1000003 + index)
            x, y = augment_pair(x, y)
            random.setstate(state)
        return (torch.from_numpy(np.ascontiguousarray(x[None])).float(),
                torch.from_numpy(np.ascontiguousarray(y[None] > 0)).float(), index)


class EpochSampler(Sampler):
    """Epoch-tagged indices make persistent-worker augmentation reproducible on resume."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter((self.epoch, i) for i in torch.randperm(len(self), generator=gen).tolist())


def make_loader(dataset, batch, workers, device, seed, shuffle=False):
    dataset.seed = seed
    options = dict(batch_size=batch, num_workers=workers, pin_memory=device.type == 'cuda',
                   worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    if shuffle:
        options['sampler'] = EpochSampler(dataset, seed)
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


def close_loader(loader):
    # Release persistent Windows workers when a model finishes or batch size changes.
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


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
    severity_score = (level + 1) / 3.0
    reliability = (1 - severity_score * strength).clamp(0, 1)
    return ((1 - strength) * x + strength * degraded.clamp(0, 1)).clamp(0, 1), reliability


def training_corruption(x, args):
    """Return (training_input, reliability_target, was_corrupted)."""
    if random.random() > args.corruption_probability:
        return x, torch.ones_like(x), False
    noisy, reliability = corrupt(x, random.choice(list(args.corruptions)), random.randrange(3),
                                 random.random() < .5, values=args.corruptions)
    return noisy, reliability, True


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

    def selected_features(self, x, count):
        features = []
        for i, block in enumerate(self.blocks[:count]):
            x = block(x)
            features.append(x)
            if i + 1 < count:
                x = F.max_pool2d(x, 2)
        return features


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
        if kind not in ('gd', 'cygd'):
            raise ValueError(f'Unsupported gradient kind: {kind}')
        self.kind, self.channels = kind, channels
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer('direction_x', torch.tensor([[-1., 0, 1]] * 3)[None, None])
        self.register_buffer('direction_y', torch.tensor([[-1., -1., -1.], [0, 0, 0], [1., 1., 1.]])[None, None])

    def transformed_weights(self):
        """Return the exact x/y transformed kernels used by the BG-Net operator."""
        if self.kind == 'gd':
            wx = self.weight * self.direction_x
            wy = self.weight * self.direction_y
        else:
            w = self.weight.flatten(2)
            wx = (w[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]] - w).reshape_as(self.weight)
            wy = (w[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]] - w).reshape_as(self.weight)
        return wx, wy

    @staticmethod
    def magnitude(gx, gy, dtype):
        return (gx.float().square() + gy.float().square() + 1e-7).sqrt().to(dtype)

    def forward(self, x):
        wx, wy = self.transformed_weights()
        gx = F.conv2d(x, wx, padding=1, groups=self.channels)
        gy = F.conv2d(x, wy, padding=1, groups=self.channels)
        return self.magnitude(gx, gy, x.dtype)


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
                         nn.Conv2d(16, 2, 3, padding=1))

    def forward(self, x):
        logits = self[1](self[0](x))
        # Softmax itself is evaluated in FP32 for stability, then returned to the
        # feature dtype so AMP does not force the following feature fusion to FP32.
        return F.softmax(logits.float(), dim=1).to(logits.dtype)


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
    def __init__(self, channels, mode, fused_gradient=True):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.fused_gradient = bool(fused_gradient and mode != 'sequential-gradient')
        self.use_reliability = mode in ('reliability-gradient', 'proposed')
        self.use_denoising = mode in ('denoising-gradient', 'proposed')
        self.use_gate = mode in ('adaptive-gradient', 'reliability-gradient', 'denoising-gradient', 'proposed')
        self.d = GradientConv(channels, 'gd')
        self.c = GradientConv(channels, 'cygd')
        self.reliability = ReliabilityEstimator(channels) if self.use_reliability else None
        self.gate = SpatialGate(channels) if self.use_gate else None
        self.ae = GradientDenoisingAE(channels) if self.use_denoising else nn.Identity()
        self.fuse = conv_norm_relu(2 * channels, channels, 1)

    def parallel_gradients_reference(self, f):
        """Verified reference path: four grouped conv calls (two per operator)."""
        return self.d(f), self.c(f)

    def parallel_gradients_fused(self, f):
        """Mathematically identical D/C gradient evaluation using one grouped conv.

        Each input channel is one group and emits four maps in this exact order:
        D_x, D_y, C_x, C_y. This only reduces kernel-launch overhead; the learned
        D-GConv and C-GConv parameters remain separate in the state_dict.
        """
        d_x, d_y = self.d.transformed_weights()
        c_x, c_y = self.c.transformed_weights()
        # [C,4,1,3,3] -> [4C,1,3,3]. With groups=C, every input channel owns
        # its four output kernels, which is required for correct grouped-conv mapping.
        weight = torch.stack((d_x, d_y, c_x, c_y), dim=1).reshape(4 * self.channels, 1, 3, 3)
        maps = F.conv2d(f, weight, padding=1, groups=self.channels)
        b, _, h, w = maps.shape
        maps = maps.reshape(b, self.channels, 4, h, w)
        dx, dy, cx, cy = maps.unbind(dim=2)
        gd = GradientConv.magnitude(dx, dy, f.dtype)
        gc_ = GradientConv.magnitude(cx, cy, f.dtype)
        return gd, gc_

    def adaptive_features(self, f, return_debug=False):
        if self.mode == 'sequential-gradient':
            gd = self.d(f)
            gc_ = self.c(gd)
        elif self.fused_gradient:
            gd, gc_ = self.parallel_gradients_fused(f)
        else:
            gd, gc_ = self.parallel_gradients_reference(f)

        r = (self.reliability(torch.cat([f, gd, gc_], 1)) if self.use_reliability
             else torch.ones_like(f[:, :1]))
        a = (self.gate(torch.cat([gd, gc_, r], 1)) if self.use_gate
             else torch.full_like(f[:, :2], .5))
        ga = gc_ if self.mode == 'sequential-gradient' else a[:, :1] * gd + a[:, 1:] * gc_
        debug = (dict(G_D=gd, G_C=gc_, R=r, A_D=a[:, :1], A_C=a[:, 1:], G_adaptive=ga)
                 if return_debug else None)
        return ga, r, debug

    def forward(self, f, return_debug=False):
        ga, r, debug = self.adaptive_features(f, return_debug)
        clean = self.ae(ga)
        if return_debug:
            debug['G_denoised'] = clean
        return self.fuse(torch.cat([f, clean], 1)), r, clean, debug


# ============================================================
# 17. SEQUENTIAL GRADIENT BASELINE
# ============================================================
# ============================================================
# 18. PROPOSED MODEL
# ============================================================
class SegmentationUNet(nn.Module):
    def __init__(self, mode='unet', base=32, fused_gradient=True):
        super().__init__()
        self.mode = mode
        self.fused_gradient = fused_gradient
        self.encoder = Encoder(base)
        self.decoder = Decoder(base)
        # Three scales (F1/F2/F3), depthwise operators, memory-conscious 6-GB default.
        self.gradient_modules = nn.ModuleList(
            [GradientModule(base * 2**i, mode, fused_gradient=fused_gradient) for i in range(3)] if mode != 'unet' else [])

    def forward(self, x, return_debug=False, clean_targets=None, reliability_target=None, reliability_loss_type='l1'):
        skips, bottleneck = self.encoder(x)
        debug, denoise_losses, reliability_losses = [], [], []
        for i, module in enumerate(self.gradient_modules):
            skips[i], reliability, denoised, maps = module(skips[i], return_debug)
            if clean_targets is not None and module.use_denoising:
                denoise_losses.append(F.l1_loss(denoised.float(), clean_targets[i].float()))
            if reliability_target is not None and module.use_reliability:
                target = F.interpolate(reliability_target, reliability.shape[-2:], mode='area')
                with torch.amp.autocast(device_type=x.device.type, enabled=False):
                    loss_fn = F.l1_loss if reliability_loss_type == 'l1' else F.binary_cross_entropy
                    reliability_losses.append(loss_fn(reliability.float(), target.float()))
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
    def compute_clean_gradient_targets(self, x):
        # Stop at selected encoder scales: no bottleneck, AE, fusion or decoder.
        features = self.encoder.selected_features(x, len(self.gradient_modules))
        return [module.adaptive_features(f)[0].detach()
                for f, module in zip(features, self.gradient_modules)]

    def clean_targets(self, x):
        return self.compute_clean_gradient_targets(x)


class SequentialGradientUNet(SegmentationUNet):
    """This is a sequential gradient ablation inspired by BG-Net operators;
    it is not a reproduction of the full BG-Net architecture.
    """
    def __init__(self, base=32):
        super().__init__('sequential-gradient', base, fused_gradient=False)


# ============================================================
# 19. LOSS FUNCTIONS
# ============================================================
def soft_dice(prob, target):
    dims = (1, 2, 3)
    return ((2 * (prob * target).sum(dims) + 1e-6) /
            (prob.sum(dims) + target.sum(dims) + 1e-6)).mean()


def boundary(x, kernel_size=3):
    """Differentiable morphological boundary magnitude: local max - local min."""
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size, 1, pad) + F.max_pool2d(-x, kernel_size, 1, pad)


def boundary_dice_loss(prob, target, kernel_size=3):
    """Soft Dice loss computed on differentiable boundary maps."""
    bp = boundary(prob, kernel_size)
    bg = boundary(target.float(), kernel_size)
    dims = (1, 2, 3)
    score = ((2.0 * (bp * bg).sum(dims) + 1e-6) /
             (bp.sum(dims) + bg.sum(dims) + 1e-6)).mean()
    return 1.0 - score


def losses(output, y, args):
    logits = output['prediction'].float()
    p = logits.sigmoid()
    seg = F.binary_cross_entropy_with_logits(logits, y.float()) + 1 - soft_dice(p, y)
    edge_l1 = F.l1_loss(boundary(p, args.boundary_kernel),
                        boundary(y.float(), args.boundary_kernel))
    edge_dice = boundary_dice_loss(p, y, args.boundary_kernel)
    if args.boundary_loss == 'l1':
        edge = edge_l1
    elif args.boundary_loss == 'dice':
        edge = edge_dice
    else:  # hybrid
        edge = args.boundary_l1_mix * edge_l1 + (1.0 - args.boundary_l1_mix) * edge_dice
    denoise, reliability = output['denoise_loss'], output['reliability_loss']
    total = seg + args.lambda_boundary * edge + args.lambda_denoise * denoise + args.lambda_reliability * reliability
    return dict(total_loss=total, seg_loss=seg, boundary_loss=edge,
                boundary_l1_loss=edge_l1, boundary_dice_loss=edge_dice,
                denoise_loss=denoise, reliability_loss=reliability)


# ============================================================
# FAST METRICS
# ============================================================
@torch.no_grad()
def fast_metrics(logits, target):
    pred, gt = logits.float() >= 0, target > .5
    dims = (1, 2, 3)
    tp = (pred & gt).sum(dims).float()
    fp = (pred & ~gt).sum(dims).float()
    fn = (~pred & gt).sum(dims).float()
    tn = (~pred & ~gt).sum(dims).float()
    def divide(n, d, fallback=1.):
        return torch.where(d > 0, n / d.clamp_min(1), torch.as_tensor(fallback, device=n.device))
    return dict(dice=divide(2 * tp, 2 * tp + fp + fn), iou=divide(tp, tp + fp + fn),
                precision=divide(tp, tp + fp, (tp + fn == 0).float()),
                recall=divide(tp, tp + fn), specificity=divide(tn, tn + fp),
                accuracy=divide(tp + tn, tp + tn + fp + fn))


def segmentation_loss(logits, target):
    logits = logits.float()
    return F.binary_cross_entropy_with_logits(logits, target.float()) + 1 - soft_dice(logits.sigmoid(), target)


# ============================================================
# FINAL EVALUATION METRICS
# ============================================================
def image_metrics(pred, gt, include_boundary=True):
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
    if not include_boundary:
        return result
    if not pred.any() and not gt.any():
        bf = 1.
    elif not pred.any() or not gt.any():
        bf = 0.
    else:
        ep = pred ^ ndimage.binary_erosion(pred, border_value=0)
        eg = gt ^ ndimage.binary_erosion(gt, border_value=0)
        # A radius-two disk gives the same pixel tolerance without distance transforms.
        yy, xx = np.ogrid[-2:3, -2:3]
        disk = xx * xx + yy * yy <= 4
        precision = float(ndimage.binary_dilation(eg, structure=disk)[ep].mean())
        recall = float(ndimage.binary_dilation(ep, structure=disk)[eg].mean())
        bf = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    result['boundary_f1'] = bf
    return result


def summarize(rows, seed=42):
    df = pd.DataFrame(rows)
    metrics = ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy', 'boundary_f1']
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
        fixed = dict(vmin=0, vmax=1) if title in ('R', 'A_D', 'A_C', 'R_target', 'R_predicted') else {}
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None, **fixed)
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
            output = model(prepare_image_tensor(x[None], device, args.channels_last), return_debug=True)
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


@torch.no_grad()
def reliability_visuals(model, dataset, args, run, device, tag):
    if model.mode not in ('reliability-gradient', 'proposed'):
        return
    model.eval()
    for index in range(min(3, len(dataset))):
        clean, _, _ = dataset[index]
        clean = prepare_image_tensor(clean[None], device, args.channels_last)
        gen = torch.Generator(device=device).manual_seed(args.seed + index)
        corrupted, target = corrupt(clean, 'speckle', index, True, gen, args.corruptions)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
            out = model(corrupted, return_debug=True)
        plot_grid([('Clean image', clean), (SEVERITIES[index] + ' spatial speckle', corrupted),
                   ('R_target', target), ('R_predicted', out['scales'][0]['R'])],
                  run / f'plots/gradient_debug/{tag}_reliability_{index}.png', args.show_plots)


def history_plots(history, args, run, mode):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    for name, cols in [('training_loss', ['train_total_loss', 'val_seg_loss']),
                       ('validation_dice', ['val_dice']), ('validation_iou', ['val_iou']),
                       ('learning_rate', ['learning_rate'])]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for col in cols:
            ax.plot(df.epoch, df[col], label=col)
        ax.set_xlabel('Epoch')
        ax.legend()
        finish_figure(fig, run / f'plots/training_curves/{mode}/{name}.png', args.show_plots)


def gpu_memory(device):
    if device.type != 'cuda':
        return 'CPU'
    allocated = torch.cuda.memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    return f'alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB'


def prepare_image_tensor(x, device, channels_last=False):
    x = x.to(device, non_blocking=True)
    if channels_last and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def model_info(model, args, run, device, debug=False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = (f'Model: {model.mode}\nTotal parameters: {total:,}\nTrainable parameters: {trainable:,}\n'
              f'FP32 parameter size: {total * 4 / 2**20:.2f} MiB\n')
    (run / 'logs' / f'architecture_{model.mode}.txt').write_text(report + str(model), encoding='utf-8')
    print(report)
    if not debug:
        return
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
        x = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
        if args.channels_last and device.type == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)
        out = model(x, return_debug=True)
    print('Input shape:', tuple(x.shape), 'Output shape:', tuple(out['prediction'].shape))
    for scale, maps in enumerate(out['scales']):
        print('Scale', scale, {k: tuple(v.shape) for k, v in maps.items()})


# ============================================================
# 22. TRAINING FUNCTION
# ============================================================
def train_epoch(model, loader, optimizer, scaler, args, device, epoch, run=None):
    model.train()
    totals, seen = defaultdict(float), 0
    profile_rows = []
    bar = tqdm(loader, desc=f'Train {model.mode} {epoch}/{args.epochs}', file=sys.stdout)
    for batch_index, (clean_cpu, y_cpu, _) in enumerate(bar):
        profile_this = device.type == 'cuda' and batch_index < args.profile_batches

        if profile_this:
            torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            stage_started = batch_started

        clean = prepare_image_tensor(clean_cpu, device, args.channels_last)
        y = y_cpu.to(device, non_blocking=True)

        if profile_this:
            torch.cuda.synchronize(device)
            transfer_time = time.perf_counter() - stage_started
            stage_started = time.perf_counter()

        noisy, target, was_corrupted = training_corruption(clean, args)
        if args.channels_last and noisy.ndim == 4:
            noisy = noisy.contiguous(memory_format=torch.channels_last)

        if profile_this:
            torch.cuda.synchronize(device)
            corruption_time = time.perf_counter() - stage_started
            stage_started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        need_teacher = model.mode in ('denoising-gradient', 'proposed')
        if args.denoise_corrupted_only and not was_corrupted:
            need_teacher = False

        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            teacher = model.compute_clean_gradient_targets(clean) if need_teacher else None

        if profile_this:
            torch.cuda.synchronize(device)
            teacher_time = time.perf_counter() - stage_started
            stage_started = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            output = model(noisy, clean_targets=teacher, reliability_target=target,
                           reliability_loss_type=args.reliability_loss)

        if profile_this:
            torch.cuda.synchronize(device)
            forward_time = time.perf_counter() - stage_started
            stage_started = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            terms = losses(output, y, args)

        if profile_this:
            torch.cuda.synchronize(device)
            loss_time = time.perf_counter() - stage_started
            stage_started = time.perf_counter()

        if not torch.isfinite(terms['total_loss']):
            raise FloatingPointError(f'Non-finite loss: {model.mode}, epoch {epoch}')
        scaler.scale(terms['total_loss']).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=not scaler.is_enabled())

        if profile_this:
            torch.cuda.synchronize(device)
            backward_time = time.perf_counter() - stage_started
            stage_started = time.perf_counter()

        if not torch.isfinite(norm):
            print('AMP gradient overflow: skipping update and reducing loss scale.')
        scaler.step(optimizer)
        scaler.update()

        if profile_this:
            torch.cuda.synchronize(device)
            optimizer_time = time.perf_counter() - stage_started
            total_time = time.perf_counter() - batch_started
            profile_rows.append(dict(
                epoch=epoch, batch=batch_index, batch_size=len(clean),
                transfer_s=transfer_time, corruption_s=corruption_time,
                teacher_s=teacher_time, forward_s=forward_time, loss_s=loss_time,
                backward_s=backward_time, optimizer_s=optimizer_time, total_s=total_time,
                images_per_second=(len(clean) / total_time if total_time > 0 else 0),
                corrupted=bool(was_corrupted)))

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

    if profile_rows:
        profile_df = pd.DataFrame(profile_rows)
        print('\n' + '=' * 72)
        print(f'GPU PROFILE SUMMARY — {model.mode} epoch {epoch} ({len(profile_rows)} batches)')
        print('=' * 72)
        labels = [
            ('transfer_s', 'Average Data Transfer'),
            ('corruption_s', 'Average Corruption'),
            ('teacher_s', 'Average Clean Teacher'),
            ('forward_s', 'Average Forward'),
            ('loss_s', 'Average Loss'),
            ('backward_s', 'Average Backward'),
            ('optimizer_s', 'Average Optimizer'),
            ('total_s', 'Average Total'),
            ('images_per_second', 'Average Images/Second'),
        ]
        for key, label in labels:
            suffix = '' if key == 'images_per_second' else ' s'
            print(f'{label}: {profile_df[key].mean():.4f}{suffix}')
        print('GPU Memory:', gpu_memory(device))
        print('=' * 72)
        if run is not None:
            path = Path(run) / 'logs' / f'profile_{model.mode}_epoch{epoch}.csv'
            profile_df.to_csv(path, index=False)
            print('Profile CSV:', path)

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
    began_training = time.perf_counter()
    seed_everything(args.seed, args.fast_mode)
    model = SegmentationUNet(mode, args.base_channels, fused_gradient=args.fused_gradient).to(device)
    if args.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
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
    elif args.finetune_checkpoint:
        state = torch.load(args.finetune_checkpoint, map_location='cpu', weights_only=False)
        if state.get('model_name') != mode:
            raise ValueError('--finetune-checkpoint model does not match --model')
        model.load_state_dict(state['model_state_dict'], strict=True)
        source_epoch = int(state.get('epoch', -1))
        source_best = float(state.get('best_validation_dice', float('nan')))
        print(f'BOUNDARY FINE-TUNE: loaded {mode} weights from source epoch {source_epoch} '
              f'(source best val Dice={source_best:.6f}).')
        print(f'Fresh AdamW optimizer/scheduler/scaler; fine-tune epochs 1-{args.epochs}; lr={args.lr:g}.')
        # Deliberately DO NOT restore optimizer, scheduler, scaler, early-stopping counter or history.
        # This is a new objective/experiment initialized from the pretrained segmentation model.
        del state
    train_ds = UltrasoundDataset(splits[0], args.image_size, augment=True)
    val_ds = UltrasoundDataset(splits[1], args.image_size)
    batch = int(history[-1]['batch_size']) if history else args.batch_size
    epoch = start
    if history:
        print('Resuming with recorded effective batch size:', batch)
    train_loader = make_loader(train_ds, batch, args.num_workers, device, args.seed, True)
    val_loader = make_loader(val_ds, batch, args.num_workers, device, args.seed)
    print(f'DataLoader: workers={args.num_workers}, pin_memory={device.type == "cuda"}, '
          f'persistent_workers={args.num_workers > 0}, prefetch_factor={2 if args.num_workers else None}')
    args.final_batch_sizes[mode] = batch
    save_json(run / 'config.json', vars(args))
    while epoch <= args.epochs:
        if counter >= args.patience:
            print('Checkpoint already reached early-stopping patience; no further epochs.')
            break
        began = time.perf_counter()
        recovery_path = run / 'checkpoints' / f'recovery_{stem}.pth'
        atomic_checkpoint(recovery_path, checkpoint_payload(model, optimizer, scheduler, scaler,
                          epoch-1, best, counter, history, args, splits))
        oom = False
        try:
            train_loader.sampler.epoch = epoch
            lr = optimizer.param_groups[0]['lr']
            training = train_epoch(model, train_loader, optimizer, scaler, args, device, epoch, run)
            validation, _ = evaluate(model, val_loader, args, device, final=False)
        except torch.cuda.OutOfMemoryError:
            print(f'CUDA OOM during {mode} epoch {epoch}; batch size {batch}. Restoring epoch start.')
            oom = True
        # Leave exception scope before releasing traceback-held activation graphs.
        if oom:
            if batch == 1:
                raise RuntimeError('OOM at batch size 1; explicitly choose smaller --base-channels or --image-size.')
            close_loader(train_loader)
            close_loader(val_loader)
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
            batch = max(1, batch // 2)
            args.final_batch_sizes[mode] = batch
            save_json(run / 'config.json', vars(args))
            train_loader = make_loader(train_ds, batch, args.num_workers, device, args.seed, True)
            val_loader = make_loader(val_ds, batch, args.num_workers, device, args.seed)
            print('Retry batch size:', batch)
            continue
        improved = validation['dice'] > best
        best, counter = (validation['dice'], 0) if improved else (best, counter + 1)
        scheduler.step(validation['dice'])
        row = dict(model=mode, epoch=epoch, learning_rate=lr, batch_size=batch,
                   image_size=args.image_size, base_channels=args.base_channels,
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
        print('\n' + '=' * 60 + f'\nEpoch {epoch}/{args.epochs}\nModel: {mode}')
        print(f'Execution mode: {"FAST" if args.fast_mode else "DETERMINISTIC"}'
              f'\nEffective Batch Size: {batch}\nImage Size: {args.image_size}\nBase Channels: {args.base_channels}')
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
    args.total_training_time[mode] = time.perf_counter() - began_training
    save_json(run / 'config.json', vars(args))
    history_plots(history, args, run, mode)
    state = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    close_loader(train_loader)
    close_loader(val_loader)
    return model, best, best_path


# ============================================================
# 23. VALIDATION FUNCTION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, args, device, corruption=None, spatial=False, save_predictions=None, final=True):
    model.eval()
    rows, totals, seen = [], {}, 0
    for clean, y, indices in tqdm(loader, desc='Evaluate', file=sys.stdout):
        clean, y = prepare_image_tensor(clean, device, args.channels_last), y.to(device, non_blocking=True)
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
            if args.debug_aux_validation:
                teacher = model.compute_clean_gradient_targets(clean) if model.mode in ('denoising-gradient', 'proposed') else None
                output = model(inputs, clean_targets=teacher, reliability_target=target,
                               reliability_loss_type=args.reliability_loss)
                logits = output['prediction']
                auxiliary = losses(output, y, args)
            else:
                logits = model(inputs)
                auxiliary = {}
            seg = segmentation_loss(logits, y)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite evaluation predictions')
        batch_metrics = fast_metrics(logits, y)
        for key, val in batch_metrics.items():
            totals[key] = totals.get(key, 0) + val.sum()
        totals['seg_loss'] = totals.get('seg_loss', 0) + seg.detach() * len(clean)
        for key in ['denoise_loss', 'reliability_loss']:
            if key in auxiliary:
                totals[key] = totals.get(key, 0) + auxiliary[key].detach() * len(clean)
        seen += len(clean)
        if not final:
            continue
        predictions = logits.float().cpu().numpy()[:, 0] >= 0
        truth = y.cpu().numpy()[:, 0] > 0
        for pred, gt, index in zip(predictions, truth, indices):
            record = loader.dataset.records[int(index)]
            rows.append(dict(image=record['image'], label=record['label'], **image_metrics(pred, gt)))
            if save_predictions:
                save_predictions.mkdir(parents=True, exist_ok=True)
                Image.fromarray((pred * 255).astype('uint8')).save(save_predictions / f'{int(index):04d}.png')
    summary = {k: float(v.item() / seen) for k, v in totals.items()}
    if final:
        summary['boundary_f1'] = float(np.mean([r['boundary_f1'] for r in rows]))
    return summary, rows


# ============================================================
# 24. TEST FUNCTION
# ============================================================
def test_model(model, records, args, run, device):
    ds = UltrasoundDataset(records, args.image_size)
    loader = make_loader(ds, args.final_batch_sizes.get(model.mode, args.batch_size), args.num_workers, device, args.seed)
    _, rows = evaluate(model, loader, args, device, save_predictions=run / 'predictions' / model.mode / 'test')
    close_loader(loader)
    for row in rows:
        row['model'] = model.mode
    pd.DataFrame(rows).to_csv(run / 'metrics' / f'per_image_{model.mode}.csv', index=False)
    prediction_visuals(model, ds, device, args, run, MODEL_FILES.get(model.mode, model.mode) + '_predictions')
    reliability_visuals(model, ds, args, run, device, model.mode)
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
    close_loader(loader)
    return rows


def robustness_plots(rows, args, run):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    for metric in ['dice', 'iou', 'boundary_f1']:
        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        for ax, kind in zip(axes, args.corruptions):
            for (model, scope), part in df[df.corruption.isin([kind, 'clean'])].groupby(['model', 'scope']):
                part = part.sort_values('level')
                ax.plot(part.level, part[metric], marker='o', label=f'{model}/{scope}')
            ax.set_title(kind)
            ax.set_xticks([0, 1, 2, 3], ['clean', 'mild', 'moderate', 'severe'])
            ax.set_ylabel(metric)
        axes[-1].legend(fontsize=6)
        finish_figure(fig, run / f'plots/robustness/{metric}_vs_severity.png', args.show_plots)


@torch.no_grad()
def robustness_visuals(checkpoints, records, args, run, device):
    ds = UltrasoundDataset(records[:3], args.image_size)
    for kind in args.corruptions:
        for index in range(min(3, len(ds))):
            x, _, _ = ds[index]
            x = prepare_image_tensor(x[None], device, args.channels_last)
            inputs = [x]
            for level in range(3):
                gen = torch.Generator(device=device).manual_seed(args.seed + index * 101 + level)
                inputs.append(corrupt(x, kind, level, False, gen, args.corruptions)[0])
            items = [(name, t[0, 0]) for name, t in zip(['Clean'] + SEVERITIES, inputs)]
            for mode, path in checkpoints.items():
                model = SegmentationUNet(mode, args.base_channels, fused_gradient=args.fused_gradient).to(device)
                if args.channels_last and device.type == 'cuda':
                    model = model.to(memory_format=torch.channels_last)
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
    close_loader(loader)
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
    assert image_metrics(empty, full)['boundary_f1'] == 0.0
    x = torch.randn(2, 4, 32, 32, device=device, requires_grad=True)
    for kind in ('gd', 'cygd'):
        op = GradientConv(4, kind).to(device)
        y = op(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.mean().backward()
        assert op.weight.grad is not None and torch.isfinite(op.weight.grad).all()
    # Fused parallel D/C path must be numerically and gradient equivalent to the
    # verified four-convolution reference path in FP32.
    module = GradientModule(4, 'proposed', fused_gradient=True).to(device)
    xr = torch.randn(2, 4, 32, 32, device=device, requires_grad=True)
    gd_ref, gc_ref = module.parallel_gradients_reference(xr)
    ref_grads = torch.autograd.grad((gd_ref.mean() + gc_ref.mean()),
                                    (xr, module.d.weight, module.c.weight), retain_graph=False)
    xf = xr.detach().clone().requires_grad_(True)
    gd_fused, gc_fused = module.parallel_gradients_fused(xf)
    fused_grads = torch.autograd.grad((gd_fused.mean() + gc_fused.mean()),
                                      (xf, module.d.weight, module.c.weight), retain_graph=False)
    assert torch.allclose(gd_ref.detach(), gd_fused.detach(), rtol=2e-5, atol=2e-6)
    assert torch.allclose(gc_ref.detach(), gc_fused.detach(), rtol=2e-5, atol=2e-6)
    for a, b in zip(ref_grads, fused_grads):
        assert torch.allclose(a, b, rtol=5e-5, atol=5e-6), 'Fused/reference gradient mismatch'
    if device.type == 'cuda':
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True):
            xa = torch.randn(2, 4, 32, 32, device=device)
            gd_ref_amp, gc_ref_amp = module.parallel_gradients_reference(xa)
            gd_fused_amp, gc_fused_amp = module.parallel_gradients_fused(xa)
        assert torch.allclose(gd_ref_amp.float(), gd_fused_amp.float(), rtol=5e-3, atol=5e-3)
        assert torch.allclose(gc_ref_amp.float(), gc_fused_amp.float(), rtol=5e-3, atol=5e-3)
    print('FUSED GRADIENT TEST PASSED: forward + gradients match reference')
    for kind in MODEL_CHOICES:
        model = SegmentationUNet(kind, base=8).to(device)
        out = model(torch.rand(2, 1, 32, 32, device=device), return_debug=True)
        assert out['prediction'].shape == (2, 1, 32, 32)
        for scale in out['scales']:
            assert scale['A_D'].shape == scale['A_C'].shape == (2, 1, *scale['G_D'].shape[-2:])
            assert torch.allclose(scale['A_D'] + scale['A_C'], torch.ones_like(scale['A_D']), atol=1e-6)
            assert scale['R'].min() >= 0 and scale['R'].max() <= 1
        out['prediction'].mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    fast = parse_args(['--preset', 'fast'])
    final = parse_args(['--preset', 'final'])
    assert (fast.epochs, fast.image_size, fast.base_channels, fast.num_workers, fast.patience) == (25, 192, 24, 4, 8)
    assert (final.epochs, final.batch_size, final.image_size, final.base_channels, final.patience) == (50, 16, 256, 32, 15)
    explicit = parse_args(['--preset', 'fast', '--batch-size', '8', '--no-fast-mode', '--run-robustness'])
    assert explicit.batch_size == 8 and not explicit.fast_mode and explicit.run_robustness
    for level in range(3):
        clean = torch.rand(2, 1, 32, 32, device=device)
        _, reliability = corrupt(clean, 'speckle', level, False)
        assert torch.allclose(reliability, torch.full_like(reliability, 1 - (level + 1) / 3), atol=1e-6)
        _, reliability = corrupt(clean, 'speckle', level, True)
        assert abs(reliability.min().item() - (1 - (level + 1) / 3)) < 1e-6
        assert abs(reliability.max().item() - 1) < 1e-6
    model = SegmentationUNet('proposed', 8).to(device)
    calls = []
    forbidden = [model.encoder.bottleneck, model.decoder] + [m for b in model.gradient_modules for m in (b.ae, b.fuse)]
    hooks = [m.register_forward_hook(lambda *unused: calls.append(True)) for m in forbidden]
    targets = model.compute_clean_gradient_targets(clean)
    assert not calls and all(not t.requires_grad for t in targets)
    for hook in hooks:
        hook.remove()
    debug = model(clean, return_debug=True)
    assert all(torch.allclose(t, d['G_adaptive'], atol=1e-6) for t, d in zip(targets, debug['scales']))
    print('SELF TEST PASSED: presets/overrides, severity-aware reliability, teacher excludes AE/fusion/decoder')
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
    modes = ['unet', 'sequential-gradient', 'proposed'] if args.model == 'all' else [args.model]
    for mode in modes:
        model, _, path = train_model(mode, small, args, check, device)
        checkpoints[mode] = path
        _, rows = test_model(model, small[2], args, check, device)
        assert len(rows) == 2 and all(np.isfinite(r['boundary_f1']) for r in rows)
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert {'optimizer_state_dict', 'scheduler_state_dict', 'amp_scaler_state_dict',
                'rng_state', 'splits', 'seed'}.issubset(state)
        checks.append({'model': mode, 'checkpoint_and_evaluation': 'PASS'})
        del state, model
    # Verify optimizer/scaler/history restoration by actually continuing one epoch.
    resume_mode = modes[-1]
    args.resume = str(check / 'checkpoints' / f'last_{MODEL_FILES.get(resume_mode, resume_mode.replace("-", "_"))}.pth')
    args.model, args.epochs = resume_mode, 2
    resumed = check / 'resumed'
    for directory in ['checkpoints', 'metrics', 'plots', 'predictions', 'logs']:
        (resumed / directory).mkdir(parents=True, exist_ok=True)
    model, _, _ = train_model(resume_mode, small, args, resumed, device)
    history = pd.read_csv(resumed / 'metrics/training_history.csv')
    assert history.epoch.tolist() == [1, 2]
    # A failing teacher hook proves ordinary validation never uses its clean target.
    original_teacher = model.compute_clean_gradient_targets
    def unexpected_teacher(*unused):
        raise AssertionError('Normal validation called the clean teacher')
    model.compute_clean_gradient_targets = unexpected_teacher
    loader = make_loader(UltrasoundDataset(small[1], args.image_size), 2, args.num_workers, device, args.seed)
    validation, rows = evaluate(model, loader, args, device, final=False)
    close_loader(loader)
    assert not rows and set(validation) == {'seg_loss', 'dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy'}
    model.compute_clean_gradient_targets = original_teacher
    robust = robustness_model(model, small[2], args, resumed, device)
    assert len(robust) == (24 if args.robustness_scope == 'both' else 12)
    assert all(np.isfinite(r['dice']) and np.isfinite(r['boundary_f1']) for r in robust)
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
    modes = ['unet', 'sequential-gradient', 'proposed'] if args.model == 'all' else [args.model]
    for mode in modes:
        seed_everything(args.seed, args.fast_mode)
        model = SegmentationUNet(mode, args.base_channels, fused_gradient=args.fused_gradient).to(device)
        if args.channels_last and device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        model_info(model, args, run, device, debug=True)
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
            assert torch.allclose(maps['A_D'] + maps['A_C'], torch.ones_like(maps['A_D']), atol=1e-5), error
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
                        values.append(image_metrics(p, y[i, 0].cpu().numpy(), include_boundary=False)['dice'])
                dice = float(np.mean(values))
                history.append(dict(model=mode, step=step, loss=loss_value, dice=dice))
                print(f'\nTiny-set {mode}: step={step}, loss={loss_value:.5f}, Dice={dice:.5f}')
                if dice >= args.overfit_dice:
                    achieved = True
                    break
        pd.DataFrame(history).to_csv(run / 'metrics' / f'sanity_overfit_{mode}.csv', index=False)
        prediction_visuals(model, ds, device, args, run, f'sanity_overfit_{mode}')
        reliability_visuals(model, ds, args, run, device, f'sanity_{mode}')
        torch.save(dict(model_state_dict=model.state_dict(), configuration=vars(args),
                        model_name=mode, purpose='TINY TRAINING SET SANITY ONLY, NOT RESEARCH RESULTS'),
                   run / 'checkpoints' / f'sanity_only_{mode}.pth')
        # A short development probe checks learning, not exhaustive memorization.
        learning_observed = history[-1]['loss'] < .85 * history[0]['loss'] and dice >= .5
        short_probe = (args.preset == 'fast' and args.overfit_steps <= 50
                       and '--overfit-dice' not in args.explicit_options)
        passed = achieved or (short_probe and learning_observed)
        if passed and not achieved:
            print(f'FAST learning probe passed: Dice={dice:.4f}, loss decreased. '
                  f'Full overfit threshold {args.overfit_dice} not yet reached; '
                  'use --preset final or more --tiny-overfit-steps for an exhaustive check.')
        results.append(dict(model=mode, passed=passed, overfit_converged=achieved,
                            criterion='full overfit' if achieved else 'fast learning probe', final_training_dice=dice,
                            steps=step, seconds=time.perf_counter()-began))
        save_json(run / 'sanity_results.json', results)
        del model, optimizer, scaler, out, terms, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if not passed:
            raise RuntimeError(f'{mode} tiny-set Dice {dice:.4f} did not reach {args.overfit_dice}; '
                               'full training blocked. Inspect sanity plots and adjust/debug before proceeding.')
    print('SANITY / SOFTWARE TEST PASSED: real data, shapes, losses, backpropagation and tiny-set learning:', modes)
    return results


# ============================================================
# 27. MAIN
# ============================================================
def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument('--preset', choices=['fast', 'final'], default='final')
    selected, _ = preset_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--external-data-root', default=None)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--preset', choices=['fast', 'final'], default=selected.preset)
    parser.add_argument('--mode', choices=['sanity', 'full', 'evaluate', 'self-test', 'audit'], default='sanity')
    parser.add_argument('--model', choices=MODEL_CHOICES + ['all'], default='proposed')
    parser.add_argument('--ablation', choices=list('ABCDEFG'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lambda-boundary', type=float, default=.2)
    parser.add_argument('--boundary-loss', choices=['l1', 'dice', 'hybrid'], default='hybrid',
                        help='Boundary objective. hybrid mixes boundary L1 and soft boundary Dice.')
    parser.add_argument('--boundary-l1-mix', type=float, default=.5,
                        help='For hybrid boundary loss: weight of L1; (1-weight) is boundary Dice.')
    parser.add_argument('--boundary-kernel', type=int, choices=[3, 5, 7], default=3,
                        help='Kernel used to extract differentiable morphological boundary maps.')
    parser.add_argument('--lambda-denoise', type=float, default=.1)
    parser.add_argument('--lambda-reliability', type=float, default=.1)
    parser.add_argument('--reliability-loss', choices=['l1', 'bce'], default='l1')
    parser.add_argument('--corruption-probability', type=float, default=.5)
    parser.add_argument('--corruption-config', help='JSON file overriding all four three-level corruption arrays')
    parser.add_argument('--resume', help='Trusted local last checkpoint; matching best checkpoint must be alongside it')
    parser.add_argument('--finetune-checkpoint',
                        help='Load model weights and original splits from a trusted checkpoint, but start a NEW optimizer/scheduler. Use this when changing the loss objective.')
    parser.add_argument('--additional-epochs', type=int, default=None,
                        help='When resuming, train this many MORE epochs from the saved epoch. Example: checkpoint epoch 15 + --additional-epochs 10 trains epochs 16-25.')
    parser.add_argument('--checkpoint', help='Trusted best/last checkpoint for pure evaluation without training')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fast-mode', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fused-gradient', action=argparse.BooleanOptionalAction, default=True,
                        help='Fuse parallel D/C gradient kernel execution after numerical-equivalence tests')
    parser.add_argument('--channels-last', action=argparse.BooleanOptionalAction, default=False,
                        help='Use NHWC/channels-last memory format on CUDA when beneficial')
    parser.add_argument('--denoise-corrupted-only', action=argparse.BooleanOptionalAction, default=False,
                        help='Compute denoising teacher only for batches that were synthetically corrupted')
    parser.add_argument('--profile-batches', type=int, default=0,
                        help='Profile the first N training batches; 0 disables synchronization/timing overhead')
    parser.add_argument('--debug-aux-validation', action='store_true', help='Opt in to auxiliary validation teacher losses')
    parser.add_argument('--run-robustness', action='store_true')
    parser.add_argument('--robustness-scope', choices=['global', 'spatial', 'both'], default='both')
    parser.add_argument('--run-external-validation', action='store_true')
    parser.add_argument('--include-normal-empty-masks', action='store_true')
    parser.add_argument('--tiny-overfit-steps', '--overfit-steps', dest='overfit_steps', type=int, default=200)
    parser.add_argument('--overfit-dice', type=float, default=.90)
    parser.add_argument('--sanity-report', help='Previously passed sanity_results.json to avoid repeating overfit in full mode')
    parser.add_argument('--integration-check', action='store_true', help='Also exercise checkpoint resume and robustness on a small BUSI subset')
    parser.add_argument('--allow-cpu-self-test', action='store_true', help='Only allows CPU in self-test mode')
    if selected.preset == 'fast':
        parser.set_defaults(epochs=25, batch_size=16, image_size=192, base_channels=24,
                            num_workers=4, patience=8, amp=True, fast_mode=True, overfit_steps=50,
                            run_robustness=False, run_external_validation=False, model='proposed')
    args = parser.parse_args(argv)
    args.explicit_options = [v.split('=')[0] for v in argv if v.startswith('--')]
    if args.ablation:
        args.model = dict(zip('ABCDEFG', MODEL_CHOICES))[args.ablation]
    if args.image_size < 32 or args.image_size % 16:
        parser.error('--image-size must be >=32 and divisible by 16')
    if min(args.batch_size, args.epochs, args.overfit_steps, args.base_channels) < 1:
        parser.error('Batch size, epochs, overfit steps and base channels must be positive')
    if args.base_channels < 8 or not 0 <= args.corruption_probability <= 1:
        parser.error('base channels >=8 and corruption probability in [0,1] required')
    if not 0 < args.overfit_dice <= 1 or args.num_workers < 0 or args.patience < 1 or args.profile_batches < 0:
        parser.error('Invalid overfit threshold, worker count, early-stopping patience or profile-batches')
    if not (0.0 <= args.boundary_l1_mix <= 1.0):
        parser.error('--boundary-l1-mix must be in [0,1]')
    if args.lr <= 0 or min(args.weight_decay, args.lambda_boundary, args.lambda_denoise, args.lambda_reliability) < 0:
        parser.error('Learning rate must be positive; loss weights and weight decay must be nonnegative')
    if args.resume and args.model == 'all':
        parser.error('--resume requires a single --model')
    if args.resume and args.mode != 'full':
        parser.error('--resume is for --mode full chunk training')
    if args.resume and args.finetune_checkpoint:
        parser.error('--resume and --finetune-checkpoint are mutually exclusive')
    if args.additional_epochs is not None and not args.resume:
        parser.error('--additional-epochs requires --resume')
    if args.additional_epochs is not None and args.additional_epochs < 1:
        parser.error('--additional-epochs must be >= 1')
    if args.mode == 'evaluate' and not args.checkpoint:
        parser.error('--mode evaluate requires --checkpoint')
    if args.mode == 'evaluate' and (args.resume or args.model == 'all'):
        parser.error('Evaluate one checkpoint at a time; do not combine with --resume or --model all')
    if args.checkpoint and args.mode != 'evaluate':
        parser.error('--checkpoint is for --mode evaluate; use --resume to continue training')
    args.corruptions = json.loads(Path(args.corruption_config).read_text()) if args.corruption_config else copy.deepcopy(CORRUPTIONS)
    if set(args.corruptions) != set(CORRUPTIONS) or any(len(v) != 3 for v in args.corruptions.values()):
        parser.error('Corruption config must specify gaussian, speckle, blur, contrast with three values each')
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for values in args.corruptions.values() for x in values):
        parser.error('Corruption values must be finite positive numbers')
    args.final_batch_sizes = {}
    args.total_training_time = {}
    args.batch_size_requested = args.batch_size
    args.deterministic = not args.fast_mode
    args.protocol_version = 3
    args.tiny_overfit_steps = args.overfit_steps
    args.learning_rate = args.lr
    args.experiment_purpose = 'SANITY / SOFTWARE TEST' if args.mode in ('sanity', 'self-test') else 'research experiment'
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


def evaluation_configuration(args):
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'splits' not in state:
        raise ValueError('Checkpoint lacks held-out split records; use a full-training best/last checkpoint.')
    saved = state['configuration']
    for key in ['amp', 'fast_mode']:
        positive = '--' + key.replace('_', '-')
        negative = '--no-' + key.replace('_', '-')
        if positive not in args.explicit_options and negative not in args.explicit_options:
            setattr(args, key, saved.get(key, getattr(args, key)))
    args.deterministic = not args.fast_mode
    for key in ['image_size', 'base_channels', 'seed', 'include_normal_empty_masks']:
        flag = '--' + key.replace('_', '-')
        if flag in args.explicit_options and getattr(args, key) != saved[key]:
            raise ValueError(f'{flag} conflicts with the trained checkpoint')
        setattr(args, key, saved[key])
    if '--model' in args.explicit_options and args.model != state['model_name']:
        raise ValueError('--model conflicts with checkpoint architecture')
    args.model = state['model_name']
    if not args.data_root:
        args.data_root = saved.get('data_root')
    if not args.corruption_config:
        args.corruptions = saved.get('corruptions', args.corruptions)
    if '--batch-size' not in args.explicit_options:
        args.batch_size = state.get('history', [{}])[-1].get('batch_size', args.batch_size)
    args.source_checkpoint = str(Path(args.checkpoint).resolve())
    args.checkpoint_purpose = saved.get('experiment_purpose', 'research training')
    args.experiment_purpose = 'evaluation of ' + args.checkpoint_purpose
    args.final_batch_sizes[args.model] = args.batch_size
    print('Pure evaluation: checkpoint architecture and exact held-out split restored. No training.')
    print('Checkpoint purpose:', args.checkpoint_purpose)
    return state


def verify_saved_splits(records, splits):
    index = {r['image']: r for r in records}
    groups = []
    for part in splits:
        hashes = set()
        for record in part:
            current = index.get(record['image'])
            if current is None or any(current.get(k) != record.get(k) for k in ['image_hash', 'mask_hash']):
                raise ValueError(f'Dataset differs from checkpoint: {record["image"]}')
            hashes.add(record['image_hash'])
        groups.append(hashes)
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Checkpoint split contains duplicate-content leakage')


def reuse_sanity_report(args, splits):
    path = Path(args.sanity_report)
    report = json.loads(path.read_text())
    old = json.loads((path.parent / 'config.json').read_text())
    requested = ['unet', 'sequential-gradient', 'proposed'] if args.model == 'all' else [args.model]
    passed = {r['model'] for r in report if r['passed']}
    if not set(requested).issubset(passed):
        raise ValueError('Sanity report has no passing result for all requested models')
    for key in ['image_size', 'base_channels', 'data_root', 'seed', 'fast_mode', 'amp', 'fused_gradient', 'channels_last', 'denoise_corrupted_only', 'protocol_version']:
        if old.get(key) != getattr(args, key):
            raise ValueError(f'Sanity report incompatible with {key}; rerun --mode sanity with these settings')
    for name, expected in zip(['train', 'val', 'test'], splits):
        df = pd.read_csv(path.parent / f'{name}_split.csv')
        df['masks'] = df['masks'].map(json.loads)
        if df.to_dict('records') != expected:
            raise ValueError('Sanity report dataset/splits have changed')
    print('Reusing passed sanity report:', path)


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
        saved_checkpoint = evaluation_configuration(args) if args.mode == 'evaluate' else None
        args.operator_source = 'https://github.com/LiYu51/BG-Net/blob/main/gradconv.py; verified 2026-09-29'
        args.gradient_scales = [1, 2, 3]
        args.synthetic_reliability = dict(map_grid=[4, 4], interpolation='bicubic then min-max to [0,1]',
                                          spatial_probability=.5, clean_target=1,
                                          severity_scores=[1/3, 2/3, 1.0],
                                          target_definition='1-severity_score*M; global M=1')
        args.metric_conventions = 'Boundary F1 disk tolerance=2 pixels, final evaluation only; CI=1000 image bootstrap means'
        args.augmentation = dict(horizontal_flip=.5, rotation_degrees=10, scale=[.95, 1.05],
                                 translation_fraction=.03, brightness=[-.04, .04], contrast=[.9, 1.1])
        args.script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save_json(run / 'config.json', vars(args))
        device = system_check(run, args.allow_cpu_self_test and args.mode == 'self-test')
        seed_everything(args.seed, args.fast_mode)
        print('CUDA execution mode:', 'FAST' if args.fast_mode else 'DETERMINISTIC')
        print('Parallel gradient execution:', 'FUSED' if args.fused_gradient else 'REFERENCE')
        print('Channels-last:', args.channels_last)
        print('Denoise teacher only on corrupted batches:', args.denoise_corrupted_only)
        print('Profile batches:', args.profile_batches)
        if args.mode == 'self-test':
            self_test(device)
            return
        args.data_root = str(locate_busi(args.data_root))
        records = discover_dataset(args.data_root, run, args.include_normal_empty_masks)
        resume_state = None
        if saved_checkpoint is not None:
            splits = saved_checkpoint['splits']
            verify_saved_splits(records, splits)
            for name, part in zip(['train', 'val', 'test'], splits):
                pd.DataFrame([dict(r, masks=json.dumps(r['masks'])) for r in part]).to_csv(run / f'{name}_split.csv', index=False)
        elif args.resume or args.finetune_checkpoint:
            # Resume/fine-tune must restore the ORIGINAL train/val/test split from the source checkpoint.
            # This avoids silently creating a different split after a reboot/new run directory.
            source_checkpoint = args.resume if args.resume else args.finetune_checkpoint
            resume_state = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
            splits = resume_state['splits']
            verify_saved_splits(records, splits)
            for name, part in zip(['train', 'val', 'test'], splits):
                pd.DataFrame([dict(r, masks=json.dumps(r['masks'])) for r in part]).to_csv(run / f'{name}_split.csv', index=False)

            # Architecture/data compatibility is mandatory for both exact resume and loss fine-tuning.
            compatibility_keys = ['image_size', 'base_channels', 'seed', 'corruptions',
                                  'corruption_probability', 'reliability_loss', 'fast_mode',
                                  'protocol_version', 'amp', 'fused_gradient', 'channels_last',
                                  'denoise_corrupted_only']
            for key in compatibility_keys:
                if resume_state['configuration'].get(key) != getattr(args, key):
                    raise ValueError(f'Checkpoint configuration mismatch: {key}')

            saved_epoch = int(resume_state['epoch'])
            if args.resume:
                # Exact resume also requires the objective to be unchanged.
                for key in ['lambda_boundary', 'lambda_denoise', 'lambda_reliability']:
                    if resume_state['configuration'].get(key) != getattr(args, key):
                        raise ValueError(f'Resume configuration mismatch: {key}')
                if args.additional_epochs is not None:
                    args.epochs = saved_epoch + args.additional_epochs
                elif args.epochs <= saved_epoch:
                    raise ValueError(
                        f'Checkpoint already completed epoch {saved_epoch}, but --epochs={args.epochs}. '
                        f'Use --additional-epochs N (recommended) or set --epochs above {saved_epoch}.')
                print(f'CHUNK RESUME: checkpoint epoch={saved_epoch}; next epoch={saved_epoch + 1}; '
                      f'target epoch={args.epochs}.')
                print('Optimizer, scheduler, AMP scaler, RNG state, early-stopping counter, history and splits will be restored.')
            else:
                print(f'BOUNDARY FINE-TUNE SETUP: source checkpoint epoch={saved_epoch}; original splits restored.')
                print('Loss settings may differ; optimizer/scheduler/history will start fresh.')
        else:
            splits = make_splits(records, args.seed, run)
        save_json(run / 'config.json', vars(args))
        if args.mode != 'evaluate':
            data_visuals(splits[0], args, run)
        if args.mode == 'audit':
            print('DATASET AUDIT COMPLETE:', run)
            return
        if args.resume or args.finetune_checkpoint:
            print('Checkpoint supplied: skipping repeated sanity/overfit pipeline and proceeding directly to training.')
        elif args.sanity_report and args.mode in ('full', 'sanity'):
            reuse_sanity_report(args, splits)
        elif args.mode != 'evaluate':
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
            if saved_checkpoint is not None:
                model = SegmentationUNet(mode, args.base_channels, fused_gradient=args.fused_gradient).to(device)
                if args.channels_last and device.type == 'cuda':
                    model = model.to(memory_format=torch.channels_last)
                model.load_state_dict(saved_checkpoint['model_state_dict'])
                model_info(model, args, run, device)
                bests[mode] = saved_checkpoint['best_validation_dice']
                checkpoints[mode] = Path(args.checkpoint).resolve()
            else:
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
            if args.mode != 'evaluate' and mode == 'unet' and bests[mode] < .1:
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
        print(pd.DataFrame(comparisons)[['model', 'dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy', 'boundary_f1']].to_string(index=False))
        print('Best models:', checkpoints, '\nPlots:', run / 'plots')
        print('Robustness:', run / 'metrics/robustness_results.csv' if robust_rows else 'NOT RUN')
        print('External BUS-UCLM:', 'Completed' if external_rows else 'Skipped')
        print('Total training time:', sum(args.total_training_time.values()), 'seconds')
        save_json(run / 'completion.json', dict(status='completed', best_validation=bests, checkpoints=checkpoints,
                  total_training_time=sum(args.total_training_time.values()), execution_mode=args.mode))
    except Exception:
        if args.num_workers:
            print('If Windows worker startup failed, retry with --num-workers 0 or --num-workers 2.')
        traceback.print_exc()
        (run / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        print(f'Total runtime: {time.perf_counter()-started:.1f}s\nResults: {run}')
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()


if __name__ == '__main__':
    main()
