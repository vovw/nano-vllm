"""Custom vision preprocessing for Qwen3-VL that works with our implementation.

This preprocessor outputs raw pixel tensors that can be fed directly to our
PatchEmbed layer, unlike HF's processor which pre-embeds patches.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    import math
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    import math
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """
    Rescale image to have dimensions divisible by factor while respecting min/max pixels.
    Based on Qwen3-VL's preprocessing code.

    Args:
        height: Original height
        width: Original width
        factor: Dimensions must be divisible by this (default 28 = patch_size * spatial_merge_size)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        (new_height, new_width): Resized dimensions
    """
    import math

    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def preprocess_image(
    image: Union[Image.Image, str],
    conv_patch_size: int = 16,  # The actual Conv3D patch size
    spatial_merge_size: int = 2,
    min_pixels: int = None,
    max_pixels: int = None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    Preprocess image for Qwen3-VL vision encoder.

    NOTE: This matches the HF image processor behavior but outputs (B, C, T, H, W) format
    instead of pre-embedded patches.

    Args:
        image: PIL Image or path to image
        conv_patch_size: Conv3D patch size (default: 16)
        spatial_merge_size: Spatial merge size (default: 2)
        min_pixels: Minimum total pixels (default: 4 * 32 * 32)
        max_pixels: Maximum total pixels (default: 16384 * 32 * 32)

    Returns:
        pixel_values: Tensor of shape (1, 3, 2, H, W) with values in [-1, 1]
        grid_thw: (temporal, height_patches, width_patches) tuple AFTER patch embedding
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image or path, got {type(image)}")

    # Constants
    IMAGE_MIN_TOKEN_NUM = 4
    IMAGE_MAX_TOKEN_NUM = 16384
    # Use correct factor: conv_patch_size * spatial_merge_size = 16 * 2 = 32
    patch_factor = conv_patch_size * spatial_merge_size

    if min_pixels is None:
        min_pixels = IMAGE_MIN_TOKEN_NUM * (patch_factor ** 2)
    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * (patch_factor ** 2)

    # Get smart resize dimensions
    orig_width, orig_height = image.size
    new_height, new_width = smart_resize(
        orig_height,
        orig_width,
        factor=patch_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Resize image using BICUBIC
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # Convert to tensor and rescale to [0, 1]
    pixels = np.array(image).astype(np.float32) / 255.0

    # Normalize with mean and std of [0.5, 0.5, 0.5] (per Qwen3-VL)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    pixels = (pixels - mean) / std

    # Convert to (C, H, W)
    pixels = torch.from_numpy(pixels).permute(2, 0, 1)

    # Add batch and temporal dimensions: (C, H, W) -> (1, C, 1, H, W)
    pixels = pixels.unsqueeze(0).unsqueeze(2)

    # Calculate grid dimensions AFTER Conv3D patch embedding
    # Our Conv3D uses kernel=(temporal_patch_size=2, patch_size=16, patch_size=16)
    # For a single image with T=1, the temporal dimension will become 0 with kernel=2
    # We need to PAD the temporal dimension to T=2 to get T=1 output
    # Actually, let's check if we need padding...

    # With input (1, 3, 1, H, W) and Conv3D(kernel=(2, 16, 16), stride=(2, 16, 16))
    # Output T = (1 - 2) / 2 + 1 = 0 (invalid!)
    # We need to pad temporal to T=2: (1, 3, 2, H, W)
    pixels = torch.nn.functional.pad(pixels, (0, 0, 0, 0, 0, 1), mode='replicate')  # Pad T dimension

    t = 1  # After Conv3D with T=2 input, kernel=2, stride=2 → output T=1
    h = new_height // conv_patch_size
    w = new_width // conv_patch_size

    # Ensure h and w are divisible by spatial_merge_size (2)
    # If not, we need to adjust the image size
    if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
        # This shouldn't happen if smart_resize used correct factor, but let's handle it
        import warnings
        warnings.warn(f"Grid ({h}, {w}) not divisible by merge_size {spatial_merge_size}. This indicates a bug in smart_resize.")
        # Pad the grid dimensions
        h_padded = ((h + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size
        w_padded = ((w + spatial_merge_size - 1) // spatial_merge_size) * spatial_merge_size
        # Actually, we can't easily pad the image at this point. The issue is smart_resize needs to be fixed.
        # For now, let's just warn and use the values we got
        pass

    grid_thw = (t, h, w)

    return pixels, grid_thw


def create_prompt_with_image(
    question: str,
    grid_thw: tuple[int, int, int],
    spatial_merge_size: int = 2,
) -> tuple[list[int], int, int, int]:
    """
    Create prompt tokens for image + text input.

    Args:
        question: Question text
        grid_thw: (T, H, W) grid dimensions from preprocessing
        spatial_merge_size: Spatial merge size (default 2)

    Returns:
        token_ids: List of token IDs
        vision_start_pos: Position of vision_start token
        vision_end_pos: Position of vision_end token
        num_image_tokens: Number of image tokens between start/end
    """
    from transformers import AutoTokenizer
    import os

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser("~/huggingface/Qwen3-VL-2B"),
        trust_remote_code=True
    )

    # Special token IDs
    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    user_token = tokenizer.encode("user\n", add_special_tokens=False)
    assistant_token = tokenizer.encode("assistant\n", add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)

    vision_start_token_id = 151652
    vision_end_token_id = 151653
    image_token_id = 151655

    # Calculate number of image tokens after spatial merge
    t, h, w = grid_thw
    num_image_tokens = (t * h * w) // (spatial_merge_size ** 2)

    # Build token sequence
    vision_start = [vision_start_token_id]
    vision_end = [vision_end_token_id]
    image_tokens = [image_token_id] * num_image_tokens
    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    # Format: <|im_start|>user\n<|vision_start|><image_tokens><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n
    token_ids = (
        im_start + user_token + vision_start + image_tokens + vision_end +
        question_tokens + im_end + newline + im_start + assistant_token
    )

    # Calculate positions
    vision_start_pos = len(im_start + user_token)
    vision_end_pos = vision_start_pos + 1 + num_image_tokens

    return token_ids, vision_start_pos, vision_end_pos, num_image_tokens


if __name__ == "__main__":
    # Test preprocessing
    pixels, grid = preprocess_image("simple_cat.jpg")
    print(f"Pixel values shape: {pixels.shape}")
    print(f"Pixel values range: [{pixels.min():.3f}, {pixels.max():.3f}]")
    print(f"Grid (T, H, W): {grid}")
    print(f"Number of patches: {grid[0] * grid[1] * grid[2]}")
    print(f"After 2x2 merge: {grid[0] * grid[1] * grid[2] // 4}")

    # Test prompt creation
    tokens, start_pos, end_pos, num_img = create_prompt_with_image(
        "What color is the cat?",
        grid,
    )
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Vision start at: {start_pos}")
    print(f"Vision end at: {end_pos}")
    print(f"Image tokens: {num_img}")
