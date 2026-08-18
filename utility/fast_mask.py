import math

import torch
import torch.nn.functional as F


def _pool_extrema(mask, radius, dilate, tapered_corners):
    """Apply the morphology used by GrowMaskWithBlur to a BHW tensor."""
    if radius <= 0:
        return mask

    work = mask.unsqueeze(1)
    if not tapered_corners:
        pool = F.max_pool2d
        if not dilate:
            work = -work
        work = pool(work, kernel_size=(1, radius * 2 + 1), stride=1, padding=(0, radius))
        work = pool(work, kernel_size=(radius * 2 + 1, 1), stride=1, padding=(radius, 0))
        if not dilate:
            work = -work
        return work.squeeze(1)

    # The tapered kernel is a 3x3 cross. Processing the whole batch together
    # avoids the original per-frame kernel launch loop while retaining its shape.
    pad_value = -float("inf") if dilate else float("inf")
    reducer = torch.maximum if dilate else torch.minimum
    for _ in range(radius):
        padded = F.pad(work, (1, 1, 1, 1), mode="constant", value=pad_value)
        work = padded[:, :, 1:-1, 1:-1]
        work = reducer(work, padded[:, :, :-2, 1:-1])
        work = reducer(work, padded[:, :, 2:, 1:-1])
        work = reducer(work, padded[:, :, 1:-1, :-2])
        work = reducer(work, padded[:, :, 1:-1, 2:])
    return work.squeeze(1)


def expand_mask_batch(mask, expand_amounts, tapered_corners):
    """Expand/contract a BHW mask batch, grouping frames with equal amounts."""
    if len(expand_amounts) != mask.shape[0]:
        raise ValueError("expand_amounts must contain one value per mask")

    if len(set(expand_amounts)) == 1:
        amount = expand_amounts[0]
        return _pool_extrema(
            mask,
            radius=abs(amount),
            dilate=amount >= 0,
            tapered_corners=tapered_corners,
        )

    # Avoid index_copy_, which is unavailable on some MPS/PyTorch versions.
    result_frames = [None] * mask.shape[0]
    for amount in sorted(set(expand_amounts)):
        indices = [index for index, value in enumerate(expand_amounts) if value == amount]
        index_tensor = torch.tensor(indices, device=mask.device, dtype=torch.long)
        selected = mask.index_select(0, index_tensor)
        processed = _pool_extrema(
            selected,
            radius=abs(amount),
            dilate=amount >= 0,
            tapered_corners=tapered_corners,
        )
        for processed_index, original_index in enumerate(indices):
            result_frames[original_index] = processed[processed_index]
    return torch.stack(result_frames, dim=0)


def _pillow_gaussian_box_radius(radius, passes=3):
    """Port Pillow's _gaussian_blur_radius calculation."""
    sigma_squared = radius * radius / passes
    length = math.sqrt(12.0 * sigma_squared + 1.0)
    integer_radius = math.floor((length - 1.0) / 2.0)
    numerator = (2 * integer_radius + 1) * (
        integer_radius * (integer_radius + 1) - 3 * sigma_squared
    )
    denominator = 6 * (sigma_squared - (integer_radius + 1) ** 2)
    return integer_radius + numerator / denominator


def _horizontal_extended_box_blur(image, radius):
    """GPU-friendly equivalent of Pillow's fractional horizontal box blur."""
    integer_radius = int(radius)
    scale = 1 << 24
    window_weight = math.floor(scale / (radius * 2 + 1))
    far_weight = (scale - (integer_radius * 2 + 1) * window_weight) // 2

    width = image.shape[-1]
    padded = F.pad(
        image,
        (integer_radius + 1, integer_radius + 1, 0, 0),
        mode="replicate",
    )
    cumulative = F.pad(torch.cumsum(padded, dim=-1), (1, 0, 0, 0))
    central = (
        cumulative[..., 2 * integer_radius + 2 : 2 * integer_radius + 2 + width]
        - cumulative[..., 1 : 1 + width]
    )
    far_left = padded[..., :width]
    far_right = padded[..., 2 * integer_radius + 2 : 2 * integer_radius + 2 + width]
    return (central * window_weight + (far_left + far_right) * far_weight) / scale


def gaussian_blur_like_pillow(mask, radius, passes=3):
    """Approximate Pillow GaussianBlur on-device without PIL/CPU round-trips.

    Pillow implements GaussianBlur as three fractional extended box passes in
    each direction. Quantizing between passes preserves its 8-bit mask behavior.
    """
    if radius <= 0:
        return mask

    box_radius = _pillow_gaussian_box_radius(radius, passes)
    # tensor2pil truncates the original float mask to an 8-bit L image.
    work = torch.floor(mask.clamp(0.0, 1.0) * 255.0) / 255.0
    work = work.unsqueeze(1)

    for _ in range(passes):
        work = _horizontal_extended_box_blur(work, box_radius)
        work = torch.round(work.clamp(0.0, 1.0) * 255.0) / 255.0
    work = work.transpose(-2, -1)
    for _ in range(passes):
        work = _horizontal_extended_box_blur(work, box_radius)
        work = torch.round(work.clamp(0.0, 1.0) * 255.0) / 255.0

    return work.transpose(-2, -1).squeeze(1)
