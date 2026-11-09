"""Semantic-layer and perceptual metrics.

Mask IoU and edge F1 are placeholders for a future lossy semantic codec (for
the lossless v1 codec they are trivially 1.0). VMAF and LPIPS are perceptual
metrics for the high-fidelity layer and are best-effort: they return ``None``
when their optional dependencies are unavailable.
"""

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from PIL import Image

_VGG_RELU_INDICES = {
    'relu1_2': 3,
    'relu2_2': 8,
    'relu3_3': 15,
    'relu4_3': 22,
    'relu5_3': 29,
}
_VGG_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_vgg_cache = {}


def _to_binary(arr):
    return (np.asarray(arr) > 0).astype(np.float32)


def mask_iou(pred, gt):
    pred = _to_binary(pred)
    gt = _to_binary(gt)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def edge_f1(pred, gt):
    pred = _to_binary(pred)
    gt = _to_binary(gt)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, 1 - gt).sum()
    fn = np.logical_and(1 - pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _save_png(arr, path):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[..., None]
    img = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(path)


_ffmpeg_libvmaf = None


def _ffmpeg_libvmaf_available():
    """Return True once we know ffmpeg has the libvmaf filter."""
    global _ffmpeg_libvmaf
    if _ffmpeg_libvmaf is None:
        try:
            r = subprocess.run(['ffmpeg', '-hide_banner', '-filters'],
                               capture_output=True, text=True, timeout=15)
            _ffmpeg_libvmaf = 'libvmaf' in r.stdout
        except Exception:
            _ffmpeg_libvmaf = False
    return _ffmpeg_libvmaf


def _vmaf_ffmpeg(ref, dist, tmp):
    ref_path = os.path.join(tmp, 'ref.png')
    dist_path = os.path.join(tmp, 'dist.png')
    log_path = os.path.join(tmp, 'vmaf.json')
    _save_png(ref, ref_path)
    _save_png(dist, dist_path)
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', dist_path, '-i', ref_path,
        '-lavfi', 'libvmaf=log_fmt=json:log_path=%s' % log_path,
        '-f', 'null', '-',
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with open(log_path) as fh:
        data = json.load(fh)
    return float(data['pooled_metrics']['vmaf']['mean'])


def _vmaf_cli(ref, dist, tmp):
    ref_png = os.path.join(tmp, 'ref.png')
    dist_png = os.path.join(tmp, 'dist.png')
    _save_png(ref, ref_png)
    _save_png(dist, dist_png)
    ref_y4m = os.path.join(tmp, 'ref.y4m')
    dist_y4m = os.path.join(tmp, 'dist.y4m')
    for png, y4m in ((ref_png, ref_y4m), (dist_png, dist_y4m)):
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                        '-i', png, '-pix_fmt', 'yuv420p',
                        '-f', 'yuv4mpegpipe', y4m],
                       check=True, capture_output=True)
    log_path = os.path.join(tmp, 'vmaf.json')
    subprocess.run(['vmaf', '--reference', ref_y4m, '--distorted', dist_y4m,
                    '--output', log_path], check=True, capture_output=True)
    with open(log_path) as fh:
        data = json.load(fh)
    return float(data['pooled_metrics']['vmaf']['mean'])


def vmaf(ref, dist):
    """Best-effort VMAF. Returns None if no VMAF backend is available.

    Prefers the ffmpeg ``libvmaf`` filter, then falls back to the ``vmaf``
    command-line tool (Netflix's pip package) via a y4m intermediate.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if _ffmpeg_libvmaf_available():
                return _vmaf_ffmpeg(ref, dist, tmp)
            if shutil.which('vmaf'):
                return _vmaf_cli(ref, dist, tmp)
    except Exception:
        pass
    return None


def _get_vgg_features(device):
    """Return the (cached) VGG16 feature extractor on ``device``."""
    key = str(device)
    feats = _vgg_cache.get(key)
    if feats is not None:
        return feats
    import torchvision
    try:
        weights = torchvision.models.VGG16_Weights.DEFAULT
    except AttributeError:
        weights = None
    try:
        model = torchvision.models.vgg16(weights=weights)
    except Exception:
        model = torchvision.models.vgg16(weights=None)
    feats = model.features.eval().to(device)
    for p in feats.parameters():
        p.requires_grad_(False)
    _vgg_cache[key] = feats
    return feats


def _resolve_device(device):
    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


def lpips(ref, dist, device=None):
    """Perceptual distance in VGG feature space (unweighted). None if no torch.

    Runs on ``device`` (the current CUDA device by default). Each image is
    processed on its own under ``torch.no_grad()``, so peak memory stays small
    during the in-training evaluation.
    """
    if torch is None:
        return None
    device = _resolve_device(device)
    with torch.no_grad():
        ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
        dist = torch.as_tensor(dist, dtype=torch.float32, device=device)
        if ref.ndim == 3:
            ref = ref.unsqueeze(0)
            dist = dist.unsqueeze(0)
        if ref.shape[1] == 1:
            ref = ref.repeat(1, 3, 1, 1)
            dist = dist.repeat(1, 3, 1, 1)
        feats = _get_vgg_features(device)
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        ref = (ref - mean) / std
        dist = (dist - mean) / std

        def acts(x):
            out = {}
            for idx, module in enumerate(feats):
                x = module(x)
                if isinstance(module, torch.nn.ReLU):
                    for name, relu_idx in _VGG_RELU_INDICES.items():
                        if idx == relu_idx:
                            out[name] = x
            return out

        a_ref = acts(ref)
        a_dist = acts(dist)
        total = 0.0
        for layer in _VGG_LAYERS:
            d = (a_ref[layer] - a_dist[layer]).norm(2, dim=1).mean()
            total = total + float(d)
    return total / len(_VGG_LAYERS)
