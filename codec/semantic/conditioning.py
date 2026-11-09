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

    def __init__(self, in_ch, out_ch):
        super(MaskAdd, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1,
                              padding=0, bias=False)

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
