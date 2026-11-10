"""Semantic weighting for the Swift visual loss.

The mask ``S`` and edge ``E`` are binary side-information channels at full
frame resolution (0/1). They are converted into a spatially-varying weight map

    W = 1 + alpha_sem * S + alpha_edge * E

which is normalized to mean 1 per sample so that the loss magnitude matches the
unweighted baseline and ``alpha_*`` controls only the ROI-vs-background
emphasis. All-zero masks degrade gracefully to the baseline (W == 1).
"""

import torch
import torch.nn.functional as F

VGG_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_vgg_cache = {}


def build_weight_map(S, E, alpha_sem=1.0, alpha_edge=0.5):
    """Return ``W`` of shape ``(B, 1, H, W)`` normalized to mean 1 per sample.

    ``S`` and ``E`` are tensors of shape ``(B, 1, H, W)`` (either may be None).
    """
    device = None
    shape = None
    if S is not None:
        device = S.device
        shape = S.shape
    elif E is not None:
        device = E.device
        shape = E.shape

    if shape is None:
        raise ValueError('build_weight_map needs at least one of S or E.')

    W = torch.ones(shape, device=device, dtype=torch.float32)
    if S is not None:
        W = W + alpha_sem * S
    if E is not None:
        W = W + alpha_edge * E

    mean = W.mean(dim=(1, 2, 3), keepdim=True)
    mean = mean.clamp(min=1e-8)
    return W / mean


def weighted_l1(x, y, W):
    """Mean absolute error weighted by ``W``.

    ``W`` is broadcast over the channel dimension of ``x``/``y``.
    """
    diff = (x - y).abs()
    if W is not None:
        diff = diff * W
    return diff.mean()


def masked_l1(x, y, mask, eps=1e-8):
    """Masked mean absolute error over the batch.

    ``mask`` is a ``(B, 1, H, W)`` weight map broadcast over the channel
    dimension of ``x``/``y`` (which have ``C`` channels). Returns a scalar:
    ``sum(|x - y| * mask) / (sum(mask) * C + eps)``.
    """
    diff = (x - y).abs()
    num = (diff * mask).sum()
    den = mask.sum() * x.size(1) + eps
    return num / den


def build_roi_map(S, boundary_band=0, boundary_weight=0.5):
    """Return a boundary-aware ROI map from the binary mask ``S``.

    With ``boundary_band <= 0`` this is just ``S``. Otherwise the ROI interior
    keeps weight 1 and a max-pool dilated ring around the ROI (the boundary
    band) gets an extra ``boundary_weight``. The dilation is a max pool with
    kernel ``2 * band + 1``, stride 1, and same padding, so no SciPy dependency
    is needed.
    """
    if S is None:
        return None
    if boundary_band <= 0:
        return S
    kernel = 2 * boundary_band + 1
    padding = boundary_band
    dilated = F.max_pool2d(S, kernel_size=kernel, stride=1, padding=padding)
    dilated = (dilated > 0.0).to(S.dtype)
    band = torch.clamp(dilated - S, min=0.0)
    return S + boundary_weight * band


def ramp_multipliers(step_idx, num_steps, roi_ramp='none', roi_ramp_start=0.5):
    """Return ``(roi_mult, bg_mult)`` for a progressive ROI-emphasis ramp.

    With ``roi_ramp == 'linear'`` the ROI weight scales from
    ``roi_ramp_start`` at step 1 to ``1.0`` at step ``num_steps``, and the
    background weight scales inversely. ``none`` keeps both constant.
    """
    if roi_ramp != 'linear' or num_steps <= 1:
        return 1.0, 1.0
    t = float(step_idx) / float(num_steps - 1)
    roi_mult = roi_ramp_start + (1.0 - roi_ramp_start) * t
    bg_mult = 1.0 + (1.0 - roi_ramp_start) * (1.0 - t)
    return roi_mult, bg_mult


def region_l1(recon, target, S, E, roi_map,
              roi_weight=2.0, bg_weight=0.5, edge_weight=0.5,
              roi_mult=1.0, bg_mult=1.0, eps=1e-8):
    """Per-sample region loss with a legacy fallback for empty masks.

    Returns ``(scalar_loss, roi, bg, edge, fallback_count)``. ``scalar_loss`` is
    a differentiable tensor. ``roi``/``bg``/``edge`` are detached scalar tensors
    and ``fallback_count`` is a tensor; callers accumulate them and convert to
    Python numbers only at logging time, so the training loop does not force a
    GPU sync on every progressive step.
    """
    diff = (recon - target).abs()
    channels = recon.size(1)

    def masked_mean(mask):
        num = (diff * mask).sum(dim=(1, 2, 3))
        den = mask.sum(dim=(1, 2, 3)) * channels + eps
        return num / den

    legacy = diff.mean(dim=(1, 2, 3))

    if S is None:
        return legacy.mean(), torch.zeros(()), torch.zeros(()), torch.zeros(()), torch.zeros((), dtype=torch.long)

    roi = masked_mean(roi_map) if roi_map is not None else masked_mean(S)
    bg = masked_mean(1.0 - S)
    if E is not None:
        edge = masked_mean(E)
    else:
        edge = torch.zeros_like(roi)

    region = (roi_weight * roi_mult * roi
              + bg_weight * bg_mult * bg
              + edge_weight * edge)

    has_mask = S.sum(dim=(1, 2, 3)) > 0
    combined = torch.where(has_mask, region, legacy)

    return (combined.mean(),
            roi.detach().mean(),
            bg.detach().mean(),
            edge.detach().mean(),
            (~has_mask).sum().detach())


def _vgg_features(device=None):
    """Lazily build a (cached) VGG16 feature extractor for the perceptual term."""
    key = str(device) if device is not None else 'cpu'
    feats = _vgg_cache.get(key)
    if feats is not None:
        return feats

    import torchvision

    try:
        weights = torchvision.models.VGG16_Weights.DEFAULT
    except AttributeError:
        weights = None
    try:
        vgg = torchvision.models.vgg16(weights=weights)
    except Exception:
        vgg = torchvision.models.vgg16(weights=None)

    feats = vgg.features
    feats.eval()
    for p in feats.parameters():
        p.requires_grad_(False)
    if device is not None:
        feats = feats.to(device)
    _vgg_cache[key] = feats
    return feats


# VGG16 feature indices of the ReLU layers used for the perceptual distance.
_VGG_RELU_INDICES = {
    'relu1_2': 3,
    'relu2_2': 8,
    'relu3_3': 15,
    'relu4_3': 22,
    'relu5_3': 29,
}


def _vgg_layer_activations(feats, x, layers):
    acts = {}
    for idx, module in enumerate(feats):
        x = module(x)
        if isinstance(module, torch.nn.ReLU):
            for name, relu_idx in _VGG_RELU_INDICES.items():
                if name in layers and idx == relu_idx and name not in acts:
                    acts[name] = x
    return acts


def weighted_lpips(x, y, W, weight=1.0):
    """Spatially-weighted perceptual loss in VGG feature space.

    ``x`` and ``y`` are RGB images in [0, 1] of shape ``(B, 3, H, W)``. ``W``
    is a weight map of shape ``(B, 1, H, W)`` (or None). Feature maps are
    extracted at ``VGG_LAYERS`` and weighted by ``W`` average-pooled to each
    feature scale. Returns a scalar.
    """
    if weight == 0.0:
        return torch.zeros((), device=x.device)

    feats = _vgg_features(x.device)

    # ImageNet normalization expected by the pretrained VGG16.
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = (y - mean) / std

    ax = _vgg_layer_activations(feats, x, set(VGG_LAYERS))
    ay = _vgg_layer_activations(feats, y, set(VGG_LAYERS))

    total = torch.zeros((), device=x.device)
    for layer in VGG_LAYERS:
        if layer not in ax or layer not in ay:
            continue
        fx, fy = ax[layer], ay[layer]
        diff = (fx - fy).norm(2, dim=1, keepdim=True)
        if W is not None:
            w = F.adaptive_avg_pool2d(W, (fx.size(2), fx.size(3)))
            diff = diff * w
        total = total + diff.mean()

    return weight * total / len(VGG_LAYERS)
