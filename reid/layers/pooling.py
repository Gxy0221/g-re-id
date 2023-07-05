import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    'Identity',
    'Flatten',
    'GlobalAvgPool',
    'GlobalMaxPool',
    'GeneralizedMeanPooling',
    'GeneralizedMeanPoolingP',
    'FastGlobalAvgPool',
    'AdaptiveAvgMaxPool',
    'ClipGlobalAvgPool',
]

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input

class Flatten(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)

class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)

class GlobalMaxPool(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)

class ClipGlobalAvgPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.avgpool = FastGlobalAvgPool()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.clamp(x, min=0., max=1.)
        return x

class AdaptiveAvgMaxPool(nn.Module):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__()
        self.gap = FastGlobalAvgPool()
        self.gmp = GlobalMaxPool(output_size)

    def forward(self, x):
        avg_feat = self.gap(x)
        max_feat = self.gmp(x)
        feat = avg_feat + max_feat
        return feat

class FastGlobalAvgPool(nn.Module):
    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


