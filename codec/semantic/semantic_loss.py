"""Semantic weighting for the Swift visual loss.

The mask ``S`` and edge ``E`` are binary side-information channels at full
frame resolution (0/1). Two loss paths exist:

- ``legacy`` builds a spatially-varying weight map ``W = 1 + alpha_sem*S +
  alpha_edge*E`` normalized to mean 1 per sample.
- ``region`` builds a mean-1 weighted L1 term from per-phase ROI/edge/background
  weights over the feathered masks ``S_f``/``E_f``.

The perceptual term uses the official ``lpips`` package (AlexNet, spatial map)
instead of the non-standard VGG-L2 distance.
"""

import torch
import torch.nn.functional as F


CODE_BLOCK_MASK_THRESHOLD = 0.0


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


def _as_int_list(values):
    if isinstance(values, str):
        return [int(x) for x in values.split(',') if x.strip() != '']
    return [int(x) for x in values]


def _as_float_list(values):
    if isinstance(values, str):
        return [float(x) for x in values.split(',') if x.strip() != '']
    return [float(x) for x in values]


def phase_of_step(step_idx, phase_schedule):
    """Return the 0-based phase index for a progressive step.

    ``phase_schedule`` is a list (or comma-separated string) of length
    ``num_steps`` mapping each step to its phase. Steps past the end reuse the
    last phase.
    """
    schedule = _as_int_list(phase_schedule)
    if not schedule:
        return 0
    if step_idx < len(schedule):
        return schedule[step_idx]
    return schedule[-1]


def phase_multipliers(phase, w_roi, w_edge, w_bg):
    """Return ``(w_roi, w_edge, w_bg)`` for the given phase index.

    ``w_roi``/``w_edge``/``w_bg`` are per-phase weight lists (or comma-separated
    strings). A phase past the end reuses the last weight.
    """
    roi = _as_float_list(w_roi)
    edge = _as_float_list(w_edge)
    bg = _as_float_list(w_bg)

    def pick(lst):
        return lst[phase] if phase < len(lst) else lst[-1]

    return pick(roi), pick(edge), pick(bg)


def gate_for_phase(phase, S_f, E_f, gate_mode='mask'):
    """Return the residual gate ``(B, 1, H, W)`` for a phase, or None.

    - G0 (phase 0) and G3 (phase 3) gate on the full frame (gate = 1).
    - G1 (phase 1) gates on the feathered ROI mask ``S_f``.
    - G2 (phase 2) gates on the feathered edge map ``E_f``.

    ``gate_mode == 'none'`` or missing masks degrade to ``None``, which callers
    treat as gate = 1 (no gating).
    """
    if gate_mode == 'none' or (S_f is None and E_f is None):
        return None
    if phase == 1 and S_f is not None:
        return S_f
    if phase == 2 and E_f is not None:
        return E_f
    return torch.ones_like(S_f if S_f is not None else E_f)


def code_block_mask(phase, S, E, h, w, threshold=CODE_BLOCK_MASK_THRESHOLD):
    """Return a binary code-domain block mask ``(B, 1, h, w)``, or None.

    The mask downsamples the *raw* side information (not the feathered masks) to
    code resolution ``h x w`` via adaptive average pooling and thresholds each
    block at ``> threshold``, so a block is transmitted iff any object (phase 1)
    or edge (phase 2) coverage exists in it. Phases 0 and 3 return ``None``
    (full code map). Using the raw ``S``/``E`` keeps the mask a deterministic
    pure function of the lossless side information, so the decoder can recompute
    it identically.
    """
    if phase == 1 and S is not None:
        raw = S
    elif phase == 2 and E is not None:
        raw = E
    else:
        return None
    pooled = F.adaptive_avg_pool2d(raw, (h, w))
    return (pooled > threshold).to(raw.dtype)


def region_l1(recon, target, S_f, E_f,
              w_roi=1.0, w_edge=1.0, w_bg=1.0, norm='mean1', eps=1e-8,
              partition='overlap'):
    """Per-sample region loss with per-phase weights.

    ``S_f`` and ``E_f`` are the feathered masks ``(B, 1, H, W)`` (either may be
    None). The combined weight map is

        w = w_roi * S_f + w_edge * E_f + w_bg * (1 - S_f)        (overlap)

    With ``partition == 'disjoint'`` the regions are mutually exclusive, so
    "background" strictly means pixels that are neither object nor edge:

        w = w_roi * S_f + w_edge * E_f * (1 - S_f)
            + w_bg * (1 - S_f) * (1 - E_f)

    With ``norm == 'mean1'`` the loss is the mean-1 weighted mean
    ``sum(|x - y| * w) / (sum(w) * C + eps)``. With ``norm == 'sum'`` it uses
    the sum of the ROI/background/edge masked means, and those terms stay
    connected to the graph.

    A crop uses the region loss when it has an object mask *or* an edge map. A
    missing ``S_f`` with a present ``E_f`` is replaced by a zero tensor so the
    edge/background weights still apply. Plain L1 is used only when both maps
    are empty for that sample.

    Returns ``(scalar_loss, roi, bg, edge, fallback_count)`` where the middle
    entries are detached tensors for logging. ``fallback_count`` counts samples
    with neither object nor edge coverage.
    """
    diff = (recon - target).abs()
    channels = recon.size(1)

    def masked_mean(mask, detach=False):
        num = (diff * mask).sum(dim=(1, 2, 3))
        den = mask.sum(dim=(1, 2, 3)) * channels + eps
        value = num / den
        return value.detach() if detach else value

    legacy = diff.mean(dim=(1, 2, 3))

    if S_f is None and E_f is None:
        return (legacy.mean(),
                torch.zeros(()),
                torch.zeros(()),
                torch.zeros(()),
                torch.full((), legacy.numel(), dtype=torch.long))

    if S_f is None:
        S_f = torch.zeros_like(E_f)

    w = w_roi * S_f
    if E_f is not None:
        if partition == 'disjoint':
            w = w + w_edge * E_f * (1.0 - S_f)
            w = w + w_bg * (1.0 - S_f) * (1.0 - E_f)
        else:
            w = w + w_edge * E_f
            w = w + w_bg * (1.0 - S_f)
    else:
        w = w + w_bg * (1.0 - S_f)

    roi_term = masked_mean(S_f)
    bg_term = masked_mean(1.0 - S_f)
    if E_f is not None:
        edge_term = masked_mean(E_f)
    else:
        edge_term = torch.zeros_like(roi_term)

    roi = roi_term.detach()
    bg = bg_term.detach()
    edge = edge_term.detach()

    if norm == 'sum':
        region = w_roi * roi_term + w_edge * edge_term + w_bg * bg_term
    else:
        num = (diff * w).sum(dim=(1, 2, 3))
        den = w.sum(dim=(1, 2, 3)) * channels + eps
        region = num / den

    has_region = S_f.sum(dim=(1, 2, 3)) > 0
    if E_f is not None:
        has_region = has_region | (E_f.sum(dim=(1, 2, 3)) > 0)
    combined = torch.where(has_region, region, legacy)

    return (combined.mean(),
            roi.mean(),
            bg.mean(),
            edge.mean(),
            (~has_region).sum().detach())


_lpips_cache = {}


def _get_lpips(device, net='alex'):
    """Lazily build (and cache) the official LPIPS model on ``device``."""
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


def weighted_lpips(x, y, W, weight=1.0, net='alex'):
    """Spatially-weighted perceptual loss from the official LPIPS package.

    ``x`` and ``y`` are RGB images centered in ``[-0.5, 0.5]`` of shape
    ``(B, 3, H, W)``. The official model expects ``[-1, 1]``, so the inputs are
    scaled by 2 internally. ``W`` is a weight map ``(B, 1, H, W)`` (or None);
    the spatial LPIPS map is weighted by ``W`` and reduced to a mask-weighted
    mean. Returns a scalar.
    """
    if weight == 0.0:
        return torch.zeros((), device=x.device)

    model = _get_lpips(x.device, net)
    d = model(x * 2.0, y * 2.0)  # (B, 1, H, W)

    if W is not None:
        w_sum = W.sum() + 1e-8
        d = (d * W).sum() / w_sum
    else:
        d = d.mean()

    return weight * d
