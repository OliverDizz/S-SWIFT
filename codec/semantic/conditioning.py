"""Semantic conditioning modules for the Swift codec.

Both modules take the concatenated binary side-information ``sem`` (shape
``(B, 2, H, W)``, channels = [mask, edge]) and produce an additive or
multiplicative modulation of an existing feature map. They do not change the
input channels of any existing conv/ConvLSTM layer, so checkpoints from the
unconditioned baseline still load and only these new heads are randomly
initialized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAdd(nn.Module):
    """Additive 1x1 branch from ``(S, E)`` into an encoder feature map."""

    def __init__(self, in_ch, out_ch, init='random'):
        super(MaskAdd, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1,
                              padding=0, bias=False)
        if init == 'zero':
            nn.init.zeros_(self.conv.weight)

    def forward(self, sem, h, w):
        sem = F.adaptive_avg_pool2d(sem, (h, w))
        return self.conv(sem)


class FiLM(nn.Module):
    """Feature-wise linear modulation (per-channel scale and shift)."""

    def __init__(self, in_ch, out_ch):
        super(FiLM, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=1, stride=1,
                              padding=0, bias=True)

    def forward(self, x, sem):
        s = F.adaptive_avg_pool2d(sem, (x.size(2), x.size(3)))
        s = F.adaptive_avg_pool2d(s, (1, 1))
        gamma, beta = self.conv(s).chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


class SpatialFiLM(nn.Module):
    """Spatially varying FiLM: per-channel, per-position scale and shift.

    The semantic map is average-pooled to the feature resolution (not to
    ``(1, 1)``), so the modulation varies across space. With ``init='zero'``
    the 1x1 conv is zero-initialized and the module is the identity, so a
    warm-started model starts exactly from its source checkpoint.
    """

    def __init__(self, in_ch, out_ch, init='random'):
        super(SpatialFiLM, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=1, stride=1,
                              padding=0, bias=True)
        if init == 'zero':
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, sem):
        sem_pool = F.adaptive_avg_pool2d(sem, (x.size(2), x.size(3)))
        gamma, beta = self.conv(sem_pool).chunk(2, dim=1)
        return x * (1.0 + gamma) + beta
