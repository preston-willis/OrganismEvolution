"""Local-frame conv1: bucket-offset gather + einsum + relu."""

import torch

_RING_CIJ = [(1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]
_coord_cache = None


def conv1_forward(input_channels, weight, bias, rotation_matrix, bucket_offsets):
    global _coord_cache
    bucket = (torch.round(rotation_matrix / (torch.pi / 4)) % 8).long()
    C, H, W = input_channels.shape
    cache_key = (H, W, input_channels.device)
    if _coord_cache is None or _coord_cache[0] != cache_key:
        y_coords = torch.arange(H, device=input_channels.device).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, device=input_channels.device).view(1, W).expand(H, W)
        _coord_cache = (cache_key, y_coords, x_coords)
    y_coords, x_coords = _coord_cache[1], _coord_cache[2]

    offsets = bucket_offsets[bucket]
    ring_y = (y_coords.unsqueeze(0) + offsets[..., 0].permute(2, 0, 1)) % H
    ring_x = (x_coords.unsqueeze(0) + offsets[..., 1].permute(2, 0, 1)) % W
    ring_samples = input_channels[:, ring_y, ring_x]

    patches = torch.empty(
        C, 3, 3, H, W,
        device=input_channels.device,
        dtype=input_channels.dtype,
    )
    patches[:, 1, 1, :, :] = input_channels
    for l_local, (ci_l, cj_l) in enumerate(_RING_CIJ):
        patches[:, ci_l, cj_l, :, :] = ring_samples[:, l_local]
    x = torch.einsum("ocxy,cxyhw->ohw", weight, patches) + bias.view(-1, 1, 1)
    return torch.nn.functional.relu(x)
