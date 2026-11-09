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
