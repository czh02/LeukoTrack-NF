# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
"C2fRefine",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "DualPathAttention",
    "DecoupledDPA",
    "DecoupledDPA_V21",
    "DecoupledDPA_V31",
    "DPABottleneck",
    "C2fDPA",
    "DPAStandalone",
    "DPA_v3_LDMAstyle",
    "DPABottleneck_V21",
    "C2fDPA_V21",
    "C2fDPARefine_V21",
    "DPABottleneck_V31",
    "C2fDPA_V31",
    "C2fDPARefine_V31",
    "C2f_LDMA",
    "LDMA_Bottleneck",
    "LDMA",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """
        Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """
        Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: Tuple[int, ...] = (5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: Tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: List[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: List[int], k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: List[int]):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k2DPA(C2f):
    """C3k2 with DPABottleneck blocks (c3k=False) or standard C3k blocks (c3k=True)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        auxiliary_path: str = "",
        use_deep_aux: bool = False,
        use_dynamic_heads: bool = False,
        use_gated_fusion: bool = False,
        use_refine: bool = False,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        if c3k:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) for _ in range(n))
        else:
            self.m = nn.ModuleList(
                DPABottleneck(
                    self.c,
                    self.c,
                    auxiliary_path=auxiliary_path,
                    shortcut=shortcut,
                    g=g,
                    k=((3, 3), (3, 3)),
                    e=1.0,
                    use_deep_aux=use_deep_aux,
                    use_dynamic_heads=use_dynamic_heads,
                    use_gated_fusion=use_gated_fusion,
                    use_refine=use_refine,
                )
                for _ in range(n)
            )
        self.set_auxiliary_path(auxiliary_path)

    def set_image_names(self, image_names):
        for module in self.m:
            if hasattr(module, "set_image_names"):
                module.set_image_names(image_names)

    def set_auxiliary_path(self, path):
        for module in self.m:
            if hasattr(module, "set_auxiliary_path"):
                module.set_auxiliary_path(path)


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """
        Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """
        Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: List[int], c3: int, embed: int):
        """
        Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (List[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: List[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


# ========================= DPA v1 主实现区 =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from threading import Lock
from PIL import Image


# 假设当前环境已有 LOGGER 和 Conv，如果没有请自行引入
from ultralytics.utils import LOGGER
from ultralytics.nn.modules.conv import Conv

class AuxiliaryImageCache:
    """Cache auxiliary images to avoid repeated disk IO for DPA modules."""

    _instances = {}
    _lock = Lock()
    _supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, root: Path):
        self.root = root
        self.cache = {}
        self.available = self._scan_available()
        self.keys = tuple(self.available.keys())

    @classmethod
    def get(cls, root: str):
        """Create or retrieve a shared cache instance for the given directory."""
        if not root:
            return None
        try:
            cache_root = Path(root).expanduser()
        except Exception:
            cache_root = Path(root)
        cache_root = cache_root.resolve() if cache_root.exists() else cache_root
        key = str(cache_root)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(cache_root)
        return cls._instances[key]

    def _scan_available(self):
        """Index auxiliary files for quick lookup."""
        if not self.root.exists():
            LOGGER.warning(f"[DPA] Auxiliary path '{self.root}' does not exist. Falling back to random tensors.")
            return {}
        available = {}
        for ext in self._supported_exts:
            for file in self.root.glob(f"*{ext}"):
                available[file.stem.lower()] = file
        if not available:
            LOGGER.warning(f"[DPA] No auxiliary images found under '{self.root}'.")
        return available

    def _load_tensor(self, path: Path):
        """
        加载图像并转换为归一化的 CHW Tensor。
        [新增] 增加对 16-bit (uint16) 图像的支持，防止全白问题。
        """
        try:
            im = Image.open(path)

            # === 1. 处理 16位 深度图像 (I;16) ===
            if im.mode == 'I' or im.mode == 'I;16':
                # 16位图的值是 0-65535
                arr = np.array(im).astype(np.float32)
                # 归一化到 0-1 之间
                tensor = torch.from_numpy(arr).unsqueeze(0).float().div_(65535.0)
                return tensor

            # === 2. 处理 8位 图像 (原有逻辑) ===
            # Detect image mode and preserve original channels
            if im.mode == 'L':  # Grayscale (8-bit)
                arr = np.asarray(im, dtype=np.uint8).copy()
                tensor = torch.from_numpy(arr).unsqueeze(0).float().div_(255.0)  # (1, H, W)
            elif im.mode == 'RGB':  # RGB
                arr = np.asarray(im, dtype=np.uint8).copy()
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)  # (3, H, W)
            elif im.mode in ('RGBA', 'LA'):  # RGBA or LA (with alpha)
                if 'A' in im.mode:
                    im = im.convert('RGB' if 'RGB' in im.mode else 'L')
                arr = np.asarray(im, dtype=np.uint8).copy()
                if len(arr.shape) == 2:
                    tensor = torch.from_numpy(arr).unsqueeze(0).float().div_(255.0)
                else:
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            else:
                # Fallback
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.uint8).copy()
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            return tensor
        except Exception as e:
            LOGGER.warning(f"[DPA] Failed to load auxiliary image '{path}': {e}")
            return None

    def get_by_name(self, name: str):
        """Fetch tensor by base filename."""
        if not name:
            return None
        key = Path(name).stem.lower()
        if key in self.cache:
            return self.cache[key]
        path = self.available.get(key)
        if path is None or not path.exists():
            return None
        tensor = self._load_tensor(path)
        if tensor is not None:
            self.cache[key] = tensor
        return tensor

    def get_by_index(self, index: int):
        """Fallback: return tensor by deterministic index."""
        if not self.keys:
            return None
        key = self.keys[index % len(self.keys)]
        return self.get_by_name(key)

# 辅助图像特征提取分支（必须定义为模块级类，以便pickle序列化）
class AuxConvBranch(nn.Module):
    """辅助图像特征提取分支模块"""
    def __init__(self, in_ch, out_ch, n_layers_inner, use_residual_inner):
        super().__init__()
        self.stem = Conv(in_ch, out_ch, 3, 1, 1)
        self.layers = nn.ModuleList(Conv(out_ch, out_ch, 3, 1, 1) for _ in range(n_layers_inner))
        self.use_residual = use_residual_inner

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            y = layer(x)
            x = x + y if self.use_residual else y
        return x

class DualPathAttention(nn.Module):
    """
    Dual Path Attention (DPA) module.
    包含 4 个消融实验开关，同时保留了防止爆显存的优化逻辑。
    """

    def __init__(self, c1, c2, auxiliary_path="", reduction_ratio=16, num_heads=2,
                 # === [消融实验控制开关 (默认全开)] ===
                 use_deep_aux=True,  # 验证: 多层残差分支 vs 单层卷积
                 use_dynamic_heads=True,  # 验证: 动态头数 vs 固定头数
                 use_gated_fusion=True,  # 验证: 可学习权重融合 vs 直接拼接
                 use_refine=True  # 验证: CBAM精修 vs 无精修 (NEW!)
                 # ==================================
                 ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.auxiliary_path = ""
        self.image_cache = None
        self._warned_missing_aux = False

        # 保存配置
        self.use_deep_aux = use_deep_aux
        self.use_dynamic_heads = use_dynamic_heads
        self.use_gated_fusion = use_gated_fusion
        self.use_refine = use_refine  # <--- 新增

        self.reduction_ratio = reduction_ratio

        # [消融点 2: 动态头数 vs 固定单头]
        if self.use_dynamic_heads:
            # 动态模式：算法会自动根据通道数算出最优头数 (比如 2, 4, 8)
            self.num_heads = self._auto_heads(c1, requested=num_heads)
        else:
            # 固定模式 (Baseline)：
            # 强制设为 1。
            # 意义：模拟最原始的注意力机制（没有多头机制），作为弱基准。
            # 这样对比起来，"动态多头" 的优势就体现出来了。
            self.num_heads = 1

        self._detected_aux_channels = None

        # [消融点 3: 融合策略]
        if self.use_gated_fusion:
            self.cross_fusion_conv1 = Conv(c1, c1 // 2, 1)
            self.cross_fusion_conv2 = Conv(c1, c1 // 2, 1)
            self.alpha = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            self.simple_concat_reduce = Conv(c1 * 2, c1, 1)

        self.fusion_conv = Conv(c1, c2, 1)

        # [消融点 4: 精修模块 (CBAM)]
        if self.use_refine:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c1 // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1 // reduction_ratio, c1, 1),
                nn.Sigmoid()
            )
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3),
                nn.Sigmoid()
            )
        else:
            # 如果不使用精修，就不初始化这些层，节省显存
            self.channel_attention = None
            self.spatial_attention = None

        # Attention 核心
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=c1,
            num_heads=self.num_heads,
            batch_first=True,
        )

        # [消融点 1: 辅助分支]
        # ============================================================
        # 【修改后】强制统一使用 3 通道分支
        # 逻辑：灰度图会在 load_auxiliary_image 中被 repeat 成 3 通道
        # 这样模型结构就固定了，不会因为图片格式不同而报错
        # ============================================================
        self.aux_conv = self._build_aux_branch_adaptive(3, c1)

        # 保持不变
        self.align_conv = Conv(c1 * 2, c1, 1)
        self.current_image_names = None
        self.set_auxiliary_path(auxiliary_path)

    def _build_aux_branch_adaptive(self, in_channels, out_channels):
        if self.use_deep_aux:
            return AuxConvBranch(in_channels, out_channels, n_layers_inner=2, use_residual_inner=True)
        else:
            return Conv(in_channels, out_channels, 3, 1)

    def _auto_heads(self, c1, requested=2, base=64, min_heads=2, max_heads=16):
        est = max(min_heads, c1 // base) if c1 >= base else min_heads
        if requested is not None and requested > 0:
            est = min(est, int(requested))
        est = max(min_heads, min(max_heads, est))
        for h in range(est, min_heads - 1, -1):
            if c1 % h == 0:
                return h
        return 1

    def _resolve_auxiliary_path(self, path):
        if not path: return ""
        try:
            aux_path = Path(str(path)).expanduser()
            if not aux_path.is_absolute():
                aux_path = (Path.cwd() / aux_path).resolve()
            return str(aux_path)
        except Exception:
            return str(path)

    def set_auxiliary_path(self, path):
        normalized = self._resolve_auxiliary_path(path)
        if normalized == self.auxiliary_path and self.image_cache is not None:
            return
        self.auxiliary_path = normalized
        self.image_cache = AuxiliaryImageCache.get(self.auxiliary_path) if self.auxiliary_path else None
        self._warned_missing_aux = False

    def set_image_names(self, image_names):
        if image_names is None:
            self.current_image_names = None
        elif isinstance(image_names, (list, tuple)):
            self.current_image_names = list(image_names)
        else:
            self.current_image_names = [image_names]

    def load_auxiliary_image(self, main_image_shape, device, image_names=None):
        """
        加载辅助图像。
        核心修改：
        1. 强制将所有输入转换为 3 通道 (RGB)，解决通道不匹配报错。
        2. [新增] 缩放使用 Letterbox 逻辑，等比例缩放+居中黑边填充，与YOLO保持绝对空间对齐。
        """
        B, _, H, W = main_image_shape
        cache_available = self.image_cache is not None
        names = image_names or self.current_image_names or []
        if isinstance(names, tuple): names = list(names)

        if not hasattr(self, '_detected_aux_channels'): self._detected_aux_channels = None
        if not hasattr(self, '_logged_aux_success'): self._logged_aux_success = False

        if not hasattr(self, 'aux_conv'):
            if hasattr(self, 'aux_conv_3ch'):
                self.aux_conv = self.aux_conv_3ch
            else:
                self.aux_conv = self._build_aux_branch_adaptive(3, self.c1).to(device)

        aux_images = []

        for i in range(B):
            aux_tensor = None
            if cache_available:
                name = names[i] if i < len(names) else None
                aux_tensor = self.image_cache.get_by_name(name)
                if aux_tensor is None:
                    aux_tensor = self.image_cache.get_by_index(i)
                    if aux_tensor is None and not self._warned_missing_aux:
                        LOGGER.warning(f"[DPA] Missing auxiliary image for '{name}', using random noise.")
                        self._warned_missing_aux = True

            if aux_tensor is None:
                aux_tensor = torch.randn(3, H, W, dtype=torch.float32)
            else:
                if not self._logged_aux_success:
                    LOGGER.info(f"[DPA] ✓ Auxiliary image loaded: '{name}' | aux_path={self.auxiliary_path} | shape={aux_tensor.shape}")
                    self._logged_aux_success = True
                C = aux_tensor.shape[0]
                if C == 1:
                    aux_tensor = aux_tensor.repeat(3, 1, 1)
                elif C == 3:
                    pass
                elif C > 3:
                    aux_tensor = aux_tensor[:3, :, :]

                # ========================================================
                # [修改点] 完美贴合 YOLO 的 Letterbox 等比例缩放 + 黑边填充
                # ========================================================
                aux_tensor = aux_tensor.to(torch.float32).unsqueeze(0)
                _, _, orig_H, orig_W = aux_tensor.shape

                if (orig_H, orig_W) != (H, W):
                    # 1. 计算长边对齐的缩放比例 r
                    r = min(H / orig_H, W / orig_W)
                    new_h, new_w = int(round(orig_H * r)), int(round(orig_W * r))

                    # 2. 等比例缩放
                    aux_tensor = F.interpolate(aux_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

                    # 3. 计算需要 Padding 的像素量 (上下左右居中均分)
                    pad_h = H - new_h
                    pad_w = W - new_w

                    top = pad_h // 2
                    bottom = pad_h - top
                    left = pad_w // 2
                    right = pad_w - left

                    # 4. 执行 Padding (F.pad 的顺序是：左, 右, 上, 下)，填充值为0.0即纯黑
                    aux_tensor = F.pad(aux_tensor, (left, right, top, bottom), value=0.0)

                aux_tensor = aux_tensor.squeeze(0)

            aux_images.append(aux_tensor)

        aux_batch = torch.stack(aux_images).to(device)

        if hasattr(self, 'aux_conv_3ch') and self.aux_conv != self.aux_conv_3ch:
            self.aux_conv = self.aux_conv_3ch

        first_param = next(self.aux_conv.parameters(), None)
        dtype = first_param.dtype if first_param is not None else torch.float32
        if aux_batch.dtype != dtype:
            aux_batch = aux_batch.to(dtype)

        return self.aux_conv(aux_batch)

    def channel_cross_fusion(self, x1, x2):
        if self.use_gated_fusion:
            x1_half1 = self.cross_fusion_conv1(x1)
            x2_half2 = self.cross_fusion_conv2(x2)
            weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
            x1_weighted = x1_half1 * weights[0]
            x2_weighted = x2_half2 * weights[1]
            return torch.cat([x1_weighted, x2_weighted], dim=1)
        else:
            cat_feat = torch.cat([x1, x2], dim=1)
            return self.simple_concat_reduce(cat_feat)

    def apply_attention(self, x):
        """[消融点 4: 精修]
        """
        if not self.use_refine:
            return x  # 直接返回，不进行 CBAM 处理

        ca = self.channel_attention(x)
        x = x * ca
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        return x

    def multihead_attention_fusion(self, main_features, aux_features):
        """保留显存优化逻辑"""
        B, C, H, W = main_features.shape
        max_size = 48  # 从32提升到48，减少注意力图上采样失真，改善定位精度（80会OOM）
        if H > max_size or W > max_size:
            scale_factor = min(max_size / H, max_size / W)
            main_small = torch.nn.functional.interpolate(main_features, scale_factor=scale_factor, mode='bilinear',
                                                         align_corners=False)
            aux_small = torch.nn.functional.interpolate(aux_features, scale_factor=scale_factor, mode='bilinear',
                                                        align_corners=False)
            new_h, new_w = main_small.shape[-2:]
            main_flat = main_small.view(B, C, new_h * new_w).transpose(1, 2)
            aux_flat = aux_small.view(B, C, new_h * new_w).transpose(1, 2)
            attn_output, _ = self.multihead_attn(query=main_flat, key=aux_flat, value=aux_flat)
            attn_output = attn_output.transpose(1, 2).view(B, C, new_h, new_w)
            attn_output = torch.nn.functional.interpolate(attn_output, size=(H, W), mode='bilinear',
                                                          align_corners=False)
        else:
            main_flat = main_features.view(B, C, H * W).transpose(1, 2)
            aux_flat = aux_features.view(B, C, H * W).transpose(1, 2)
            attn_output, _ = self.multihead_attn(query=main_flat, key=aux_flat, value=aux_flat)
            attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        return attn_output

    def forward(self, x):
        aux_features = self.load_auxiliary_image(x.shape, x.device, self.current_image_names)
        attn_fused = self.multihead_attention_fusion(x, aux_features)
        cross_fused = self.channel_cross_fusion(x, attn_fused)
        attended_features = self.apply_attention(cross_fused)
        output = self.fusion_conv(attended_features)
        return output
#
#
class DPABottleneck(nn.Module):
    # 增加 use_refine
    def __init__(self, c1, c2, auxiliary_path="", shortcut=True, g=1, k=(3, 3), e=0.5,
                 use_deep_aux=False, use_dynamic_heads=False, use_gated_fusion=False, use_refine=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # 传入所有开关
        self.dpa = DualPathAttention(c_, c_, auxiliary_path,
                                     use_deep_aux=use_deep_aux,
                                     use_dynamic_heads=use_dynamic_heads,
                                     use_gated_fusion=use_gated_fusion,
                                     use_refine=use_refine)  # <---
        self.add = shortcut and c1 == c2

    def set_image_names(self, image_names): self.dpa.set_image_names(image_names)

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        x = self.dpa(x)
        x = self.cv2(x)
        return x + identity if self.add else x

    def set_auxiliary_path(self, path): self.dpa.set_auxiliary_path(path)


class C2fDPA(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, auxiliary_path="", g=1, e=0.5,
                 use_deep_aux=False, use_dynamic_heads=False, use_gated_fusion=False, use_refine=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DPABottleneck(self.c, self.c, auxiliary_path=auxiliary_path, shortcut=shortcut, g=g, k=((3, 3), (3, 3)),
                          e=1.0,
                          use_deep_aux=use_deep_aux,
                          use_dynamic_heads=use_dynamic_heads,
                          use_gated_fusion=use_gated_fusion,
                          use_refine=use_refine)  # <---
            for _ in range(n)
        )
        self.set_auxiliary_path(auxiliary_path)

    def set_image_names(self, image_names):
        for module in self.m: module.set_image_names(image_names)

    def set_auxiliary_path(self, path):
        for module in self.m: module.set_auxiliary_path(path)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================
# DPA v1 独立版本（DPAStandalone）
# 直接作用于完整特征图，不再嵌入 C2f 内部，避免辅路信号被稀释
# 内置残差门控：output = x + sigmoid(gate) * dpa(x)
# ============================================================
class DPAStandalone(nn.Module):
    """独立 DPA 模块，作用于完整特征图，带可学习残差门控。"""

    def __init__(self, c1, c2, auxiliary_path="",
                 use_deep_aux=True, use_dynamic_heads=True,
                 use_gated_fusion=True, use_refine=True):
        super().__init__()
        self.dpa = DualPathAttention(
            c1, c2, auxiliary_path,
            use_deep_aux=use_deep_aux,
            use_dynamic_heads=use_dynamic_heads,
            use_gated_fusion=use_gated_fusion,
            use_refine=use_refine,
        )
        self.add = (c1 == c2)
        self.gate = nn.Parameter(torch.zeros(1))  # 初始 sigmoid(0)=0.5，模型自学权重

    def set_image_names(self, image_names):
        self.dpa.set_image_names(image_names)

    def set_auxiliary_path(self, path):
        self.dpa.set_auxiliary_path(path)

    def forward(self, x):
        y = self.dpa(x)
        if self.add:
            return x + torch.sigmoid(self.gate) * y
        return y


# ============================================================
# DPA v3 — LDMA 风格的双路注意力
# 设计要点：
#   1. 强残差保护：output = x + gamma * (attn * aux)，gamma 初始为 0
#   2. 全程 4D 张量：深度卷积保留空间结构，不展平
#   3. 像素级 sigmoid 门控（不是标量权重）
#   4. 极简结构：无 MultiheadAttention，无 CBAM
# ============================================================
class DPA_v3_LDMAstyle(nn.Module):
    """LDMA 风格的 DPA v3：强残差 + 像素级空间门控 + 简化结构。"""

    def __init__(self, c1, c2, auxiliary_path=""):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.add = (c1 == c2)

        self.auxiliary_path = ""
        self.image_cache = None
        self.current_image_names = None
        self._warned_missing_aux = False
        self._logged_aux_success = False

        # 辅路 IMD 3 通道 → c1 通道投影
        self.aux_proj = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # 主路 / 辅路深度卷积
        self.q_conv = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.k_conv = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)

        # 像素级门控：(B, 2c1, H, W) → (B, c1, H, W) 的 sigmoid
        mid = max(c1 // 2, 1)
        self.attn_gate = nn.Sequential(
            nn.Conv2d(c1 * 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, c1, 1, bias=False),
            nn.Sigmoid(),
        )

        # 残差缩放，初始为 0（等价恒等映射，模型自学是否启用辅路）
        self.gamma = nn.Parameter(torch.zeros(1))

        # 通道适配（c1 ≠ c2 时使用 1×1 卷积调整）
        if not self.add:
            self.out_proj = Conv(c1, c2, 1)

        self.set_auxiliary_path(auxiliary_path)

    def _resolve_auxiliary_path(self, path):
        if not path:
            return ""
        try:
            aux_path = Path(str(path)).expanduser()
            if not aux_path.is_absolute():
                aux_path = (Path.cwd() / aux_path).resolve()
            return str(aux_path)
        except Exception:
            return str(path)

    def set_auxiliary_path(self, path):
        normalized = self._resolve_auxiliary_path(path)
        if normalized == self.auxiliary_path and self.image_cache is not None:
            return
        self.auxiliary_path = normalized
        self.image_cache = AuxiliaryImageCache.get(self.auxiliary_path) if self.auxiliary_path else None
        self._warned_missing_aux = False

    def set_image_names(self, image_names):
        if image_names is None:
            self.current_image_names = None
        elif isinstance(image_names, (list, tuple)):
            self.current_image_names = list(image_names)
        else:
            self.current_image_names = [image_names]

    def _load_aux_letterbox(self, main_shape, device):
        """加载 IMD 辅助图并 letterbox 对齐到主路尺寸，返回 (B, 3, H, W) 张量。"""
        B, _, H, W = main_shape
        names = self.current_image_names or []
        aux_images = []
        for i in range(B):
            aux_tensor = None
            name = names[i] if i < len(names) else None
            if self.image_cache is not None and name is not None:
                aux_tensor = self.image_cache.get_by_name(name)

            if aux_tensor is None:
                aux_tensor = torch.zeros(3, H, W, dtype=torch.float32)
                if not self._warned_missing_aux:
                    LOGGER.warning(f"[DPA_v3] Missing auxiliary image for '{name}'. Using zeros.")
                    self._warned_missing_aux = True
            else:
                if not self._logged_aux_success:
                    LOGGER.info(f"[DPA_v3] ✓ Auxiliary image loaded: '{name}' | shape={aux_tensor.shape}")
                    self._logged_aux_success = True
                C = aux_tensor.shape[0]
                if C == 1:
                    aux_tensor = aux_tensor.repeat(3, 1, 1)
                elif C > 3:
                    aux_tensor = aux_tensor[:3]

                aux_tensor = aux_tensor.to(torch.float32).unsqueeze(0)
                _, _, orig_H, orig_W = aux_tensor.shape
                if (orig_H, orig_W) != (H, W):
                    r = min(H / orig_H, W / orig_W)
                    new_h, new_w = int(round(orig_H * r)), int(round(orig_W * r))
                    aux_tensor = F.interpolate(aux_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    pad_h, pad_w = H - new_h, W - new_w
                    top, left = pad_h // 2, pad_w // 2
                    aux_tensor = F.pad(aux_tensor, (left, pad_w - left, top, pad_h - top), value=0.0)
                aux_tensor = aux_tensor.squeeze(0)

            aux_images.append(aux_tensor)

        return torch.stack(aux_images).to(device)

    def forward(self, x):
        aux_raw = self._load_aux_letterbox(x.shape, x.device)
        aux_raw = aux_raw.to(dtype=x.dtype)  # 兼容 AMP 混合精度
        aux = self.aux_proj(aux_raw)
        q = self.q_conv(x)
        k = self.k_conv(aux)
        attn = self.attn_gate(torch.cat([q, k], dim=1))
        out = x + self.gamma * (attn * aux)
        if not self.add:
            out = self.out_proj(out)
        return out


class DPA_v3_MH(nn.Module):
    """LDMA 风格 + 多头门控的 DPA：保留 4D 全程、像素级 sigmoid 与强残差，
    通过 grouped 1×1 conv 把 c1 通道拆成 num_heads 组独立的"主辅匹配子空间"，
    每个头各自打分，对小目标精细定位（mAP50-95）更友好。

    与 DPA_v3_LDMAstyle 的差异：
      1. attn_gate 内部 1×1 conv 改为 grouped conv（groups=num_heads），头间隔离不互相稀释
      2. gamma 改为 (num_heads,1,1) 形状，让模型对每个头独立学习辅路使用强度
    """

    def __init__(self, c1, c2, num_heads=4, auxiliary_path=""):
        super().__init__()
        if c1 % num_heads != 0:
            raise ValueError(f"DPA_v3_MH: c1={c1} 不能被 num_heads={num_heads} 整除")
        self.c1 = c1
        self.c2 = c2
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.add = (c1 == c2)

        self.auxiliary_path = ""
        self.image_cache = None
        self.current_image_names = None
        self._warned_missing_aux = False
        self._logged_aux_success = False

        self.aux_proj = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        self.q_conv = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.k_conv = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)

        # grouped 1×1：每个头一个独立的小 MLP；输入 (q, k) 拼接后必须按头交错，
        # 因此外部需要在 forward 里 reshape，使每组的 2*head_dim 通道连续。
        mid = max(c1 // 2, num_heads)
        if mid % num_heads != 0:
            mid = num_heads * max(self.head_dim // 2, 1)
        self.attn_gate = nn.Sequential(
            nn.Conv2d(c1 * 2, mid, 1, groups=num_heads, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, c1, 1, groups=num_heads, bias=False),
            nn.Sigmoid(),
        )

        # 每头独立 gamma，初始为 0（残差恒等，模型自学是否启用每个头）
        self.gamma = nn.Parameter(torch.zeros(num_heads, 1, 1))

        if not self.add:
            self.out_proj = Conv(c1, c2, 1)

        self.set_auxiliary_path(auxiliary_path)

    def _resolve_auxiliary_path(self, path):
        if not path:
            return ""
        try:
            aux_path = Path(str(path)).expanduser()
            if not aux_path.is_absolute():
                aux_path = (Path.cwd() / aux_path).resolve()
            return str(aux_path)
        except Exception:
            return str(path)

    def set_auxiliary_path(self, path):
        normalized = self._resolve_auxiliary_path(path)
        if normalized == self.auxiliary_path and self.image_cache is not None:
            return
        self.auxiliary_path = normalized
        self.image_cache = AuxiliaryImageCache.get(self.auxiliary_path) if self.auxiliary_path else None
        self._warned_missing_aux = False

    def set_image_names(self, image_names):
        if image_names is None:
            self.current_image_names = None
        elif isinstance(image_names, (list, tuple)):
            self.current_image_names = list(image_names)
        else:
            self.current_image_names = [image_names]

    def _load_aux_letterbox(self, main_shape, device):
        B, _, H, W = main_shape
        names = self.current_image_names or []
        aux_images = []
        for i in range(B):
            aux_tensor = None
            name = names[i] if i < len(names) else None
            if self.image_cache is not None and name is not None:
                aux_tensor = self.image_cache.get_by_name(name)

            if aux_tensor is None:
                aux_tensor = torch.zeros(3, H, W, dtype=torch.float32)
                if not self._warned_missing_aux:
                    LOGGER.warning(f"[DPA_v3_MH] Missing auxiliary image for '{name}'. Using zeros.")
                    self._warned_missing_aux = True
            else:
                if not self._logged_aux_success:
                    LOGGER.info(f"[DPA_v3_MH] ✓ Auxiliary image loaded: '{name}' | shape={aux_tensor.shape}")
                    self._logged_aux_success = True
                C = aux_tensor.shape[0]
                if C == 1:
                    aux_tensor = aux_tensor.repeat(3, 1, 1)
                elif C > 3:
                    aux_tensor = aux_tensor[:3]

                aux_tensor = aux_tensor.to(torch.float32).unsqueeze(0)
                _, _, orig_H, orig_W = aux_tensor.shape
                if (orig_H, orig_W) != (H, W):
                    r = min(H / orig_H, W / orig_W)
                    new_h, new_w = int(round(orig_H * r)), int(round(orig_W * r))
                    aux_tensor = F.interpolate(aux_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    pad_h, pad_w = H - new_h, W - new_w
                    top, left = pad_h // 2, pad_w // 2
                    aux_tensor = F.pad(aux_tensor, (left, pad_w - left, top, pad_h - top), value=0.0)
                aux_tensor = aux_tensor.squeeze(0)

            aux_images.append(aux_tensor)

        return torch.stack(aux_images).to(device)

    def _interleave_qk(self, q, k):
        """把 q、k 按头交错拼接，保证每个头的 2*head_dim 通道在 grouped conv 中连续。

        输入：q,k: (B, c1, H, W) 其中 c1 = num_heads * head_dim
        输出：    (B, 2*c1, H, W) 排布为 [q_h0, k_h0, q_h1, k_h1, ...]
        """
        B, C, H, W = q.shape
        q4 = q.view(B, self.num_heads, self.head_dim, H, W)
        k4 = k.view(B, self.num_heads, self.head_dim, H, W)
        qk = torch.cat([q4, k4], dim=2)               # (B, num_heads, 2*head_dim, H, W)
        return qk.view(B, 2 * C, H, W)

    def forward(self, x):
        aux_raw = self._load_aux_letterbox(x.shape, x.device)
        aux_raw = aux_raw.to(dtype=x.dtype)
        aux = self.aux_proj(aux_raw)

        q = self.q_conv(x)
        k = self.k_conv(aux)

        qk = self._interleave_qk(q, k)
        attn = self.attn_gate(qk)                     # (B, c1, H, W)，按头分组的 sigmoid 门控

        B, C, H, W = x.shape
        gated = (attn * aux).view(B, self.num_heads, self.head_dim, H, W)
        gated = gated * self.gamma.view(1, self.num_heads, 1, 1, 1)
        gated = gated.view(B, C, H, W)

        out = x + gated
        if not self.add:
            out = self.out_proj(out)
        return out


class C2fRefine(C2f):
    """C2f with Channel and Spatial Attention"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction_ratio=16):
        # 调用父类初始化，只传递标准参数
        super().__init__(c1, c2, n, shortcut, g, e)

        # 添加注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // reduction_ratio, c2, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先执行标准 C2f 前向
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x = self.cv2(torch.cat(y, 1))

        # 应用通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 应用空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x





# 历史注释版 DPA 草稿已整理为可并存类：
# DecoupledDPA_V21 / C2fDPA_V21 / C2fDPA_V31 等。


# ============================================================================
# Innovation Module: LDMA (Latent Denoising Memory Attention)
# 灵感来源: NeurIPS 2024 <DeTrack> (特征去噪) & arXiv 2026 <Attention Residuals> (跨层检索)
# ============================================================================



# ##########################################################################################
class LDMA(nn.Module):
    def __init__(self, c, use_hist_smooth=False):
        super().__init__()
        self.use_hist_smooth = use_hist_smooth

        self.q_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.k_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        self.attn_gate = nn.Sequential(
            nn.Conv2d(c * 2, c // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.SiLU(),
            nn.Conv2d(c // 2, c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_curr, x_hist):
        if self.use_hist_smooth:
            x_hist_used = F.avg_pool2d(x_hist, kernel_size=3, stride=1, padding=1)
        else:
            x_hist_used = x_hist

        q = self.q_conv(x_curr)
        k = self.k_conv(x_hist_used)

        attn_weight = self.attn_gate(torch.cat([q, k], dim=1))
        refined_feat = x_curr + self.gamma * (attn_weight * x_hist_used)

        return refined_feat

class LDMA_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, use_hist_smooth=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.ldma = LDMA(c_, use_hist_smooth=use_hist_smooth)

class C2f_LDMA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, use_hist_smooth=False):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            LDMA_Bottleneck(self.c, self.c, shortcut, g, e=1.0, use_hist_smooth=use_hist_smooth)
            for _ in range(n)
        )

#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 如果你环境中已经有 Conv，保留你自己的导入
# # from ultralytics.nn.modules.conv import Conv
#
#
# 历史注释中的 Prior/Suppress Head 已整理为正式类定义。
# def masked_bce_with_logits(logits, targets, valid_mask, eps=1e-6):
#     loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
#     denom = valid_mask.sum().clamp(min=eps)
#     return (loss * valid_mask).sum() / denom
#
#
# def masked_dice_with_logits(logits, targets, valid_mask, eps=1e-6):
#     probs = torch.sigmoid(logits)
#     probs = probs * valid_mask
#     targets = targets * valid_mask
#     num = 2 * (probs * targets).sum(dim=(1, 2, 3))
#     den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
#     loss = 1 - (num + eps) / den
#     return loss.mean()
#
#

class PriorHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class SuppressHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)


def masked_bce_with_logits(logits, targets, valid_mask, eps=1e-6):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    denom = valid_mask.sum().clamp(min=eps)
    return (loss * valid_mask).sum() / denom


def masked_dice_with_logits(logits, targets, valid_mask, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs * valid_mask
    targets = targets * valid_mask
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    loss = 1 - (num + eps) / den
    return loss.mean()


# ========================= DPA v2 主实现区 =========================
class DecoupledDPA(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        reduction_ratio=16,
        use_refine=False,
        prior_thr=0.15,
        prior_loss_weight=1.0,
        bg_loss_weight=0.2,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.use_refine = use_refine
        self.prior_thr = float(prior_thr)
        self.prior_loss_weight = float(prior_loss_weight)
        self.bg_loss_weight = float(bg_loss_weight)

        self.conv_main_fg = Conv(c1, c1, 1)
        self.conv_main_bg = Conv(c1, c1, 1)

        self.prior_head = PriorHead(c1)
        self.bg_head = SuppressHead(c1)

        self.fuse_logits = nn.Parameter(torch.zeros(2))
        self.fusion_conv = Conv(c1, c2, 1)

        if self.use_refine:
            hidden = max(c1 // reduction_ratio, 8)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, c1, 1, bias=True),
                nn.Sigmoid(),
            )
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.channel_attention = None
            self.spatial_attention = None

        self.decouple_loss = torch.tensor(0.0)

        self.current_aux = None
        self.current_prior_fallback = None
        self.current_prior_valid = None

    def set_aux_batch(self, aux_batch):
        self.current_aux = aux_batch

    def set_prior_targets(self, prior_fallback=None, prior_valid=None):
        self.current_prior_fallback = prior_fallback
        self.current_prior_valid = prior_valid

    def get_aligned_aux(self, main_image_shape, device):
        B, _, H, W = main_image_shape
        if self.current_aux is None:
            return torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)

        aux = self.current_aux.to(device=device, dtype=torch.float32)

        if aux.shape[-2:] != (H, W):
            src_h, src_w = aux.shape[-2:]
            if H <= src_h and W <= src_w:
                aux = F.adaptive_max_pool2d(aux, output_size=(H, W))
            else:
                aux = F.interpolate(aux, size=(H, W), mode='nearest')

        return aux.clamp_(0.0, 1.0)

    def apply_attention(self, x):
        if not self.use_refine:
            return x
        ca = self.channel_attention(x)
        x = x * ca
        avg_p = torch.mean(x, dim=1, keepdim=True)
        max_p, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_p, max_p], dim=1))
        return x * sa

    def forward(self, x):
        # 1. 获取对齐好的先验掩码 (aux_batch)
        prior_raw = self.get_aligned_aux(x.shape, x.device)

        # 归一化/截断，确保 prior_soft 在 0~1 之间，prior_pos 是二值掩码
        prior_soft = torch.clamp(
            (prior_raw - self.prior_thr) / (1.0 - self.prior_thr + 1e-6),
            min=0.0,
            max=1.0,
        )
        prior_pos = (prior_raw > self.prior_thr).float()

        # 2. 主特征分裂：前景分支与背景分支
        f_fg = self.conv_main_fg(x)
        f_bg = self.conv_main_bg(x)

        # 3. 训练阶段：计算解耦损失 (Decouple Loss)
        if self.training:
            prior_logits = self.prior_head(f_fg)
            bg_logits = self.bg_head(f_bg)
            bg_probs = torch.sigmoid(bg_logits)

            # --- A. 构造鲁棒监督目标 (Target) ---
            if self.current_prior_fallback is not None:
                # 确保 fallback 和特征图尺寸一致
                fallback = F.interpolate(
                    self.current_prior_fallback.to(x.device, dtype=torch.float32),
                    size=prior_soft.shape[-2:],
                    mode='nearest',
                )
            else:
                fallback = torch.zeros_like(prior_soft)

            # 正样本目标：IMD 提取到的白点，或者 GT 兜底的白点，取最大值
            target_pos = torch.maximum(prior_soft, fallback)

            # --- B. 构造有效掩码 (Valid Mask) - v3.1 核心修复 ---
            # 必须全局监督！整张图都是有效的，包括背景(0)也必须被惩罚！
            # 否则网络会作弊把整张图全预测成1。
            valid_mask = torch.ones_like(target_pos)

            # --- C. 计算前景色损失 (Prior Loss = BCE + 0.5 * Dice) ---
            if valid_mask.sum().item() > 0:
                loss_prior_bce = masked_bce_with_logits(prior_logits, target_pos, valid_mask)
                loss_prior_dice = masked_dice_with_logits(
                    prior_logits, (target_pos > 0).float(), valid_mask
                )
                loss_prior = loss_prior_bce + 0.5 * loss_prior_dice
            else:
                loss_prior = x.new_zeros(())

            # --- D. 计算背景抑制损失 (Background Loss) ---
            # 背景分支只在“确实有白细胞(prior_pos=1)”的地方被施加负监督，压制背景响应
            if prior_pos.sum().item() > 0:
                denom = prior_pos.sum().clamp(min=1.0)
                loss_bg = (bg_probs * prior_pos).sum() / denom
            else:
                loss_bg = x.new_zeros(())

            # 汇总总解耦损失
            self.decouple_loss = (
                    self.prior_loss_weight * loss_prior
                    + self.bg_loss_weight * loss_bg
            ).to(dtype=x.dtype)
        else:
            # 推理/验证阶段：无需计算损失
            self.decouple_loss = x.new_zeros(())

        # 4. 特征融合与精修 (推理/测试时仅执行此部分)
        # Softmax 保证权重之和为 1，进行加权融合
        weights = torch.softmax(self.fuse_logits, dim=0).to(dtype=f_fg.dtype)
        fused = weights[0] * f_fg + weights[1] * f_bg

        # CBAM 精修 (如果开启)
        fused = self.apply_attention(fused)

        # 最终输出映射
        return self.fusion_conv(fused)


class DecoupledDPA_V21(nn.Module):
    """
    DPA-v2.1（从历史注释版整理为可并存实现）:
    - prior supervision only
    - incomplete IMD robust
    - IMD blank != negative
    """

    def __init__(
        self,
        c1,
        c2,
        auxiliary_path="",
        reduction_ratio=16,
        use_refine=False,
        prior_thr=0.15,
        prior_loss_weight=1.0,
        bg_loss_weight=0.2,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.auxiliary_path = ""
        self.image_cache = None
        self._warned_missing_aux = False
        self.current_image_names = None
        self.use_refine = use_refine
        self.prior_thr = float(prior_thr)
        self.prior_loss_weight = float(prior_loss_weight)
        self.bg_loss_weight = float(bg_loss_weight)

        self.conv_main_fg = Conv(c1, c1, 1)
        self.conv_main_bg = Conv(c1, c1, 1)
        self.prior_head = PriorHead(c1)
        self.bg_head = SuppressHead(c1)
        self.fuse_logits = nn.Parameter(torch.zeros(2))
        self.fusion_conv = Conv(c1, c2, 1)

        if self.use_refine:
            hidden = max(c1 // reduction_ratio, 8)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, c1, 1, bias=True),
                nn.Sigmoid(),
            )
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.channel_attention = None
            self.spatial_attention = None

        self.decouple_loss = torch.tensor(0.0)
        self.current_prior_fallback = None
        self.current_prior_valid = None
        self.set_auxiliary_path(auxiliary_path)

    def set_prior_targets(self, prior_fallback=None, prior_valid=None):
        self.current_prior_fallback = prior_fallback
        self.current_prior_valid = prior_valid

    def _resolve_auxiliary_path(self, path):
        if not path:
            return ""
        try:
            aux_path = Path(str(path)).expanduser()
            if not aux_path.is_absolute():
                aux_path = (Path.cwd() / aux_path).resolve()
            return str(aux_path)
        except Exception:
            return str(path)

    def set_auxiliary_path(self, path):
        normalized = self._resolve_auxiliary_path(path)
        if normalized == self.auxiliary_path and self.image_cache is not None:
            return
        self.auxiliary_path = normalized
        self.image_cache = AuxiliaryImageCache.get(self.auxiliary_path) if self.auxiliary_path else None
        self._warned_missing_aux = False

    def set_image_names(self, image_names):
        if image_names is None:
            self.current_image_names = None
        elif isinstance(image_names, (list, tuple)):
            self.current_image_names = list(image_names)
        else:
            self.current_image_names = [image_names]

    def _resize_aux_to_feature(self, aux_tensor, h, w):
        aux_tensor = aux_tensor.to(torch.float32).unsqueeze(0)
        _, _, orig_h, orig_w = aux_tensor.shape
        if (orig_h, orig_w) != (h, w):
            r = min(h / orig_h, w / orig_w)
            new_h = max(1, int(round(orig_h * r)))
            new_w = max(1, int(round(orig_w * r)))
            if new_h <= orig_h and new_w <= orig_w:
                aux_tensor = F.adaptive_max_pool2d(aux_tensor, output_size=(new_h, new_w))
            else:
                aux_tensor = F.interpolate(aux_tensor, size=(new_h, new_w), mode="nearest")
            pad_h = h - new_h
            pad_w = w - new_w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            aux_tensor = F.pad(aux_tensor, (left, right, top, bottom), value=0.0)
        return aux_tensor.squeeze(0).clamp_(0.0, 1.0)

    def load_auxiliary_image(self, main_image_shape, device, image_names=None):
        b, _, h, w = main_image_shape
        names = image_names or self.current_image_names or []
        if isinstance(names, tuple):
            names = list(names)
        aux_images = []
        for i in range(b):
            aux_tensor = None
            name = names[i] if i < len(names) else None
            if self.image_cache is not None and name is not None:
                aux_tensor = self.image_cache.get_by_name(name)
            if aux_tensor is None:
                aux_tensor = torch.zeros(1, h, w, dtype=torch.float32)
                if not self._warned_missing_aux:
                    LOGGER.warning(f"[DPA_V21] Missing auxiliary image for '{name}'. Using zero prior.")
                    self._warned_missing_aux = True
            else:
                aux_tensor = self._resize_aux_to_feature(aux_tensor, h, w)
            aux_images.append(aux_tensor)
        return torch.stack(aux_images, dim=0).to(device=device, dtype=torch.float32)

    def apply_attention(self, x):
        if not self.use_refine:
            return x
        ca = self.channel_attention(x)
        x = x * ca
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        return x * sa

    def forward(self, x):
        prior_raw = self.load_auxiliary_image(x.shape, x.device, self.current_image_names)
        prior_soft = torch.clamp((prior_raw - self.prior_thr) / (1.0 - self.prior_thr + 1e-6), min=0.0, max=1.0)
        prior_pos = (prior_raw > self.prior_thr).float()
        f_fg = self.conv_main_fg(x)
        f_bg = self.conv_main_bg(x)

        if self.training:
            prior_logits = self.prior_head(f_fg)
            bg_logits = self.bg_head(f_bg)
            bg_probs = torch.sigmoid(bg_logits)
            if self.current_prior_fallback is not None:
                fallback = F.interpolate(
                    self.current_prior_fallback.to(x.device, dtype=torch.float32),
                    size=prior_soft.shape[-2:],
                    mode="nearest",
                )
            else:
                fallback = torch.zeros_like(prior_soft)
            target_pos = torch.maximum(prior_soft, fallback)
            if self.current_prior_valid is not None:
                valid_mask = F.interpolate(
                    self.current_prior_valid.to(x.device, dtype=torch.float32),
                    size=prior_soft.shape[-2:],
                    mode="nearest",
                )
            else:
                valid_mask = (target_pos > 0).float()
            valid_mask = torch.maximum(valid_mask, (target_pos > 0).float())
            if valid_mask.sum().item() > 0:
                loss_prior_bce = masked_bce_with_logits(prior_logits, target_pos, valid_mask)
                loss_prior_dice = masked_dice_with_logits(prior_logits, (target_pos > 0).float(), valid_mask)
                loss_prior = loss_prior_bce + 0.5 * loss_prior_dice
            else:
                loss_prior = x.new_zeros(())
            if prior_pos.sum().item() > 0:
                denom = prior_pos.sum().clamp(min=1.0)
                loss_bg = (bg_probs * prior_pos).sum() / denom
            else:
                loss_bg = x.new_zeros(())
            self.decouple_loss = (self.prior_loss_weight * loss_prior + self.bg_loss_weight * loss_bg).to(dtype=x.dtype)
        else:
            self.decouple_loss = x.new_zeros(())

        weights = torch.softmax(self.fuse_logits, dim=0).to(dtype=f_fg.dtype)
        fused = weights[0] * f_fg + weights[1] * f_bg
        fused = self.apply_attention(fused)
        return self.fusion_conv(fused)


class DPABottleneck_V21(nn.Module):
    def __init__(self, c1, c2, auxiliary_path="", shortcut=True, g=1, k=(3, 3), e=0.5, use_refine=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.dpa = DecoupledDPA_V21(c_, c_, auxiliary_path=auxiliary_path, use_refine=use_refine)
        self.add = shortcut and c1 == c2

    def set_image_names(self, image_names):
        self.dpa.set_image_names(image_names)

    def set_auxiliary_path(self, path):
        self.dpa.set_auxiliary_path(path)

    def set_prior_targets(self, prior_fallback=None, prior_valid=None):
        self.dpa.set_prior_targets(prior_fallback, prior_valid)

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        x = self.dpa(x)
        x = self.cv2(x)
        return x + identity if self.add else x


class C2fDPA_V21(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, auxiliary_path="", g=1, e=0.5, use_refine=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DPABottleneck_V21(
                self.c, self.c, auxiliary_path=auxiliary_path, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0, use_refine=use_refine
            )
            for _ in range(n)
        )
        self.set_auxiliary_path(auxiliary_path)

    def set_image_names(self, image_names):
        for module in self.m:
            module.set_image_names(image_names)

    def set_auxiliary_path(self, path):
        for module in self.m:
            module.set_auxiliary_path(path)

    def set_prior_targets(self, prior_fallback=None, prior_valid=None):
        for module in self.m:
            module.set_prior_targets(prior_fallback, prior_valid)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fDPARefine_V21(C2fDPA_V21):
    def __init__(self, c1, c2, n=1, shortcut=False, auxiliary_path="", g=1, e=0.5):
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, auxiliary_path=auxiliary_path, g=g, e=e, use_refine=True)


DecoupledDPA_V31 = DecoupledDPA


class DPABottleneck_V31(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, use_refine=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.dpa = DecoupledDPA_V31(c_, c_, use_refine=use_refine)
        self.add = shortcut and c1 == c2

    def set_aux_batch(self, aux_tensor):
        self.dpa.set_aux_batch(aux_tensor)

    def set_prior_targets(self, prior_fallback=None, prior_valid=None):
        self.dpa.set_prior_targets(prior_fallback, prior_valid)

    def forward(self, x):
        y = self.cv2(self.dpa(self.cv1(x)))
        return x + y if self.add else y


class C2fDPA_V31(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, use_refine=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DPABottleneck_V31(self.c, self.c, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=1.0, use_refine=use_refine)
            for _ in range(n)
        )

    def set_aux_batch(self, aux_tensor):
        for module in self.m:
            module.set_aux_batch(aux_tensor)

    def set_prior_targets(self, prior_fallback=None, prior_valid=None):
        for module in self.m:
            module.set_prior_targets(prior_fallback, prior_valid)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fDPARefine_V31(C2fDPA_V31):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e, use_refine=True)
