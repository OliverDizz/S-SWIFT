"""Semantic-layer and perceptual metrics.

Mask IoU and edge F1 are placeholders for a future lossy semantic codec (for
the lossless v1 codec they are trivially 1.0). VMAF and LPIPS are perceptual
metrics for the high-fidelity layer and are best-effort: they return ``None``
when their optional dependencies are unavailable.

LPIPS uses the official ``lpips`` package (AlexNet, learned linear weights) for
both the full-frame scalar and the mask-weighted ROI variant. A non-standard
VGG-L2 distance is retained only behind ``--use-vgg-l2`` as a diagnostic.
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

from metric import msssim


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


_lpips_cache = {}


def _get_lpips_model(device, net='alex'):
    """Return the (cached) official LPIPS model on ``device``."""
    key = (str(device), net)
    model = _lpips_cache.get(key)
    if model is not None:
        return model
    import lpips as lpips_pkg
    model = lpips_pkg.LPIPS(net=net, spatial=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    _lpips_cache[key] = model
    return model


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


def _lpips_tensor(img, device):
    """Return an RGB ``(1, 3, H, W)`` tensor scaled from [0, 1] to [-1, 1]."""
    t = torch.as_tensor(img, dtype=torch.float32, device=device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    return (t - 0.5) * 2.0


def lpips(ref, dist, device=None, net='alex'):
    """Official AlexNet LPIPS scalar. None if torch or lpips is unavailable.

    ``ref`` and ``dist`` are RGB images in [0, 1] of shape ``(C, H, W)`` or
    ``(1, C, H, W)``.
    """
    if torch is None:
        return None
    device = _resolve_device(device)
    with torch.no_grad():
        ref_t = _lpips_tensor(ref, device)
        dist_t = _lpips_tensor(dist, device)
        try:
            model = _get_lpips_model(device, net)
        except Exception:
            return None
        out = model(ref_t, dist_t)
    return float(out.mean())


def lpips_map(ref, dist, device=None, net='alex'):
    """Return the spatial official-LPIPS map ``(1, 1, H, W)`` for [0, 1] images.

    None when torch or lpips is unavailable. Callers derive full/ROI/BG/edge
    scalars from this single map.
    """
    if torch is None:
        return None
    device = _resolve_device(device)
    with torch.no_grad():
        ref_t = _lpips_tensor(ref, device)
        dist_t = _lpips_tensor(dist, device)
        try:
            model = _get_lpips_model(device, net)
        except Exception:
            return None
        return model(ref_t, dist_t)


def lpips_map_batch(ref, dists, device=None, net='alex'):
    """Return spatial LPIPS maps for one reference and several distortions.

    ``ref`` is an RGB image in [0, 1]. ``dists`` is a sequence of RGB images in
    [0, 1]. Returns a ``(N, 1, H, W)`` map, or None when torch or lpips is
    unavailable. This is one batched forward pass, so progressive evaluation
    does not launch a separate LPIPS forward for every step.
    """
    if torch is None or not dists:
        return None
    device = _resolve_device(device)
    with torch.no_grad():
        ref_t = _lpips_tensor(ref, device)
        dist_t = torch.cat([_lpips_tensor(d, device) for d in dists], dim=0)
        if ref_t.shape[0] == 1 and dist_t.shape[0] > 1:
            ref_t = ref_t.expand(dist_t.shape[0], -1, -1, -1)
        try:
            model = _get_lpips_model(device, net)
        except Exception:
            return None
        return model(ref_t, dist_t)


# Optional VGG-L2 diagnostic (kept behind --use-vgg-l2; distinct cache name to
# avoid the historical _vgg name collision with the training loss module).
_VGG_L2_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
_VGG_L2_RELU_INDICES = {
    'relu1_2': 3,
    'relu2_2': 8,
    'relu3_3': 15,
    'relu4_3': 22,
    'relu5_3': 29,
}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_vgg_l2_cache = {}


def _get_vgg_l2_features(device):
    key = str(device)
    feats = _vgg_l2_cache.get(key)
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
    _vgg_l2_cache[key] = feats
    return feats


def _vgg_l2_activations(feats, x):
    out = {}
    for idx, module in enumerate(feats):
        x = module(x)
        if isinstance(module, torch.nn.ReLU):
            for name, relu_idx in _VGG_L2_RELU_INDICES.items():
                if idx == relu_idx:
                    out[name] = x
    return out


def vgg_l2(ref, dist, device=None):
    """VGG16 feature-space L2 distance (diagnostic only).

    Best-effort: returns None when torch, torchvision, or the VGG16 weights are
    unavailable, or when the forward pass fails.
    """
    if torch is None:
        return None
    device = _resolve_device(device)
    try:
        with torch.no_grad():
            ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)
            dist_t = torch.as_tensor(dist, dtype=torch.float32, device=device)
            if ref_t.ndim == 3:
                ref_t = ref_t.unsqueeze(0)
                dist_t = dist_t.unsqueeze(0)
            if ref_t.shape[1] == 1:
                ref_t = ref_t.repeat(1, 3, 1, 1)
                dist_t = dist_t.repeat(1, 3, 1, 1)
            mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
            feats = _get_vgg_l2_features(device)
            a_ref = _vgg_l2_activations(feats, (ref_t - mean) / std)
            a_dist = _vgg_l2_activations(feats, (dist_t - mean) / std)
            total = 0.0
            for layer in _VGG_L2_LAYERS:
                d = (a_ref[layer] - a_dist[layer]).norm(2, dim=1).mean()
                total = total + float(d)
        return total / len(_VGG_L2_LAYERS)
    except Exception:
        return None


def psnr_masked(ref, dist, mask):
    """PSNR computed only over the pixels where ``mask`` is nonzero.

    ``ref`` and ``dist`` are ``(H, W, C)`` images (uint8 or float in [0, 255]).
    ``mask`` is a ``(H, W)`` or ``(H, W, 1)`` binary map. Returns ``99.99`` when
    the mask is empty or the masked MSE is zero.
    """
    ref = np.asarray(ref).astype(np.float64)
    dist = np.asarray(dist).astype(np.float64)
    mask = (np.asarray(mask) > 0).astype(np.float64)
    if mask.ndim == 2:
        mask = mask[..., None]
    denom = mask.sum() * ref.shape[2]
    if denom == 0:
        return 99.99
    mse = float(np.sum(((ref - dist) ** 2) * mask) / denom)
    if mse == 0.0:
        return 99.99
    return float(np.clip(10.0 * np.log10(255.0 ** 2 / mse), 0.0, 99.99))


def bbox_from_mask(mask):
    """Return the inclusive ``(rmin, rmax, cmin, cmax)`` bounding box, or None."""
    mask = np.asarray(mask) > 0
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin = int(np.argmax(rows))
    rmax = int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin = int(np.argmax(cols))
    cmax = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (rmin, rmax, cmin, cmax)


def msssim_bbox(ref, dist, mask):
    """MS-SSIM over the mask bounding box; full-frame fallback for small crops.

    MS-SSIM window statistics are ill-defined on scattered pixels, so this
    crops to the mask bounding box. If either crop side is below 32 px it falls
    back to full-frame MS-SSIM. Returns None when the mask is empty.
    """
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    rmin, rmax, cmin, cmax = bbox
    ref = np.asarray(ref)
    dist = np.asarray(dist)
    crop_ref = ref[rmin:rmax + 1, cmin:cmax + 1]
    crop_dist = dist[rmin:rmax + 1, cmin:cmax + 1]
    if crop_ref.shape[0] < 32 or crop_ref.shape[1] < 32:
        return float(msssim(ref, dist))
    return float(msssim(crop_ref, crop_dist))
