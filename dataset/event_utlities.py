#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import random

def remove_hot_pixels_historgram(x):
    mask = x[x > 0.1]
    if mask.size(0) > 0:
        mask = torch.quantile(mask,0.75) / 10
    else:
        return torch.zeros_like(x.shape) * 0
    x = torch.clip(x,max=mask*10)
    return x


def event_historgram(event_tensor, resolution = [260,346], remove_hot_pixel = False):
    H, W = resolution
    event_tensor = event_tensor.half()
    pos = event_tensor[event_tensor[:, 3] == 1]
    neg = event_tensor[event_tensor[:, 3] < 0.5]

    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)

    result = torch.stack([pos_count, neg_count], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()

    if remove_hot_pixel:
        result = remove_hot_pixels_historgram(result)
    
    return (result / (result.amax([1,2],True) + 1)).float()

def reshape_event_no_sample(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)
    return event

def slice_event(event, length = 15000, slice_augment_width=2000):
    if slice_augment_width !=0:
        length = random.randint(length - slice_augment_width, length + slice_augment_width)
    if len(event) > length:
        start = random.choice(range(len(event) - length + 1))
        event = event[start: start + length]
    return event