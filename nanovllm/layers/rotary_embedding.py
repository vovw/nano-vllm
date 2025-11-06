from functools import lru_cache
import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimension (a, b) -> (-b, a)."""
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings supporting both half- and full-dim cos/sin inputs."""

    def _broadcast(t: torch.Tensor) -> torch.Tensor:
        while t.ndim < x.ndim:
            t = t.unsqueeze(1)
        return t

    cos = _broadcast(cos)
    sin = _broadcast(sin)

    head_dim = x.size(-1)
    last_dim = cos.size(-1)

    if last_dim == head_dim // 2:
        x_float = x.float()
        cos_float = cos.to(dtype=x_float.dtype)
        sin_float = sin.to(dtype=x_float.dtype)
        x1, x2 = x_float[..., :last_dim], x_float[..., last_dim:]
        y1 = x1 * cos_float - x2 * sin_float
        y2 = x2 * cos_float + x1 * sin_float
        output = torch.cat((y1, y2), dim=-1)
    elif last_dim == head_dim:
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        output = (x * cos) + (rotate_half(x) * sin)
    else:
        raise ValueError(
            f"Unsupported rotary dimensions: cos/sin last dim {last_dim}, head dim {head_dim}"
        )

    return output.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class VisionRotaryEmbedding(nn.Module):
    """2D rotary position embeddings for the vision encoder."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        grid_thw: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings for a (T, H, W) grid."""
        t, h, w = grid_thw

        device = self.inv_freq.device
        dtype = self.inv_freq.dtype

        pos_h = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w).repeat(t)
        pos_w = torch.arange(w, device=device, dtype=dtype).repeat(h * t)

        freqs_h = torch.outer(pos_h, self.inv_freq)
        freqs_w = torch.outer(pos_w, self.inv_freq)

        freqs = torch.cat([freqs_h, freqs_w], dim=-1)

        return freqs.cos(), freqs.sin()


class MultiModalRotaryEmbedding(nn.Module):
    """
    Multi-dimensional RoPE for Qwen3-VL text decoder.

    Handles 3D position IDs: [temporal, height, width] for vision tokens
    and standard 1D positions for text tokens.
    """

    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int,
        base: float = 5000000.0,
        mrope_section: list[int] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        if mrope_section is None:
            # Default from HuggingFace Qwen3-VL implementation
            mrope_section = [24, 20, 20]

        self.mrope_section = mrope_section

        # Total RoPE dimension is the sum of sections
        self.rope_dim = sum(mrope_section)

        # Compute inverse frequencies - same for ALL dimensions (like HF implementation)
        # All 3 dimensions use the same inv_freq, just with different position values
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rope_dim, dtype=torch.float) / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_interleaved_mrope(self, freqs: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked to interleaved [THTHW...THTHW...TT].
        Matches HuggingFace implementation exactly.

        Args:
            freqs: List of 3 tensors, each (seq_len, rope_dim)
                   All computed with the same inv_freq but different positions

        Returns:
            Interleaved frequencies (seq_len, rope_dim)
        """
        # Start with T frequencies (will be overwritten in certain positions)
        freqs_t = freqs[0].clone()

        # Interleave H and W frequencies into specific positions
        # H goes to positions: 1, 4, 7, ..., (mrope_section[1] * 3 - 2)
        # W goes to positions: 2, 5, 8, ..., (mrope_section[2] * 3 - 1)
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[:, idx] = freqs[dim][:, idx]

        return freqs_t

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: Position IDs tensor
                - If 1D (seq_len,): Standard 1D RoPE (use only T dimension)
                - If 3D (3, batch, seq_len): Multi-dimensional RoPE
            query: (batch * seq_len, num_heads, head_dim)
            key: (batch * seq_len, num_kv_heads, head_dim)

        Returns:
            query, key with RoPE applied
        """
        seq_len = query.size(0)

        # Handle both 1D and 3D position IDs
        if positions.ndim == 1:
            # Standard 1D RoPE: expand to 3D with H=W=0
            positions_3d = torch.zeros(3, positions.size(0), dtype=positions.dtype, device=positions.device)
            positions_3d[0] = positions  # T dimension
            positions = positions_3d

        # positions is now (3, seq_len) or (3, batch, seq_len)
        if positions.ndim == 3:
            # Flatten batch dimension: (3, batch * seq_len)
            positions = positions.reshape(3, -1)

        # Ensure we have the right sequence length
        if positions.size(1) != seq_len:
            # Repeat or truncate as needed
            if positions.size(1) < seq_len:
                positions = positions.repeat(1, (seq_len + positions.size(1) - 1) // positions.size(1))
            positions = positions[:, :seq_len]

        # Compute frequencies for each dimension using the SAME inv_freq
        # (matching HuggingFace implementation)
        freqs = []
        for dim_idx in range(3):
            pos = positions[dim_idx].float()  # (seq_len,)
            # All dimensions use the same inv_freq, just different position values
            freq = torch.outer(pos, self.inv_freq)  # (seq_len, rope_dim)
            freqs.append(freq)

        # Apply interleaved MRoPE: reorganize from [TTT...HHH...WWW] to [THTHW...THTHW...]
        freqs_interleaved = self._apply_interleaved_mrope(freqs)

        # Pad to head_dim/2 if needed (the non-RoPE part stays unchanged)
        target_size = self.head_size // 2
        if freqs_interleaved.size(1) < target_size:
            pad_size = target_size - freqs_interleaved.size(1)
            freqs_interleaved = torch.cat([freqs_interleaved, torch.zeros(seq_len, pad_size, device=freqs_interleaved.device)], dim=-1)

        # Return cos/sin with shape (seq_len, head_dim/2)
        # These will be broadcast across heads when applied
        cos = freqs_interleaved.cos().unsqueeze(1)  # (seq_len, 1, head_dim/2)
        sin = freqs_interleaved.sin().unsqueeze(1)  # (seq_len, 1, head_dim/2)

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
