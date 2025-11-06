import torch
from torch import nn


class Qwen3VLVisionPatchEmbed(nn.Module):
    """
    3D Convolutional patch embedding for images/videos.

    Converts pixels to patches using 3D convolution to support both
    images (T=1) and videos (T>1).

    Args:
        patch_size: Spatial patch size (default: 16)
        temporal_patch_size: Temporal patch size (default: 2)
        in_channels: Input channels (default: 3 for RGB)
        embed_dim: Output embedding dimension (default: 1152)
        spatial_merge_size: Merge size for reordering patches (default: 2)

    Input shape: (batch, channels, time, height, width)
    Output shape: (batch, num_patches, embed_dim) - patches reordered for spatial merge
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        self.spatial_merge_size = spatial_merge_size

        # 3D convolution: (C, T, H, W) -> (embed_dim, T/tp, H/p, W/p)
        # Use temporal_patch_size for videos, 1 for images during forward
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """
        Args:
            x: (batch, channels, time, height, width)

        Returns:
            embeddings: (batch, num_patches, embed_dim) - reordered for spatial merge
            grid_thw: (time_patches, height_patches, width_patches)
        """
        # Pad temporal dimension if needed for images (T=1 -> T=2)
        if x.size(2) < self.temporal_patch_size:
            pad_size = self.temporal_patch_size - x.size(2)
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_size))

        # Apply 3D convolution
        x = self.proj(x)  # (B, embed_dim, T', H', W')

        # Get grid dimensions
        B, _, T, H, W = x.shape
        grid_thw = (T, H, W)

        # Transpose to (B, T, H, W, embed_dim)
        x = x.permute(0, 2, 3, 4, 1)

        # Reorder patches to match spatial merge pattern
        # This matches what fast_pos_embed_interpolate does
        m = self.spatial_merge_size
        x = (
            x.view(B, T, H // m, m, W // m, m, -1)
            .permute(0, 1, 2, 4, 3, 5, 6)  # (B, T, H//m, W//m, m, m, embed_dim)
            .flatten(1, 5)  # (B, T*H//m*W//m*m*m, embed_dim)
        )

        return x, grid_thw
