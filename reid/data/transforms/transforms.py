__all__ = ['ToTensor', 'RandomPatch', 'AugMix', ]
import torch
import math
import random
from collections import deque
import numpy as np
from .functional import to_tensor, augmentations


class ToTensor(object):
    def __call__(self, pic):
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomPatch(object):
    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch

        return img


class AugMix(object):
    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        self.prob           = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width  = mixture_width
        self.mixture_depth  = mixture_depth
        self.aug_severity   = aug_severity
        self.augmentations  = augmentations

    def __call__(self, image):
        if random.random() > self.prob:
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros([image.size[1], image.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)
