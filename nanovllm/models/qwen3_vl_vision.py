import torch
from torch import nn

from nanovllm.layers.vision_embed import Qwen3VLVisionPatchEmbed
from nanovllm.layers.rotary_embedding import VisionRotaryEmbedding, apply_rotary_emb
from nanovllm.layers.layernorm import RMSNorm


class Qwen3VLVisionAttention(nn.Module):
    """
    Multi-head attention for vision encoder with 2D RoPE.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_tokens, hidden_size)
            position_embeddings: Tuple of cosine/sine tensors (broadcastable to q/k)
            cu_seqlens: (batch_size + 1,) cumulative sequence lengths for FlashAttention

        Returns:
            output: (batch * num_patches, hidden_size)
        """
        seq_len = hidden_states.size(0)

        qkv = self.qkv(hidden_states)
        qkv = qkv.view(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)

        cos, sin = position_embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        lengths = (
            (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            if cu_seqlens is not None
            else [seq_len]
        )

        def _eager_attention(lengths_list: list[int]) -> torch.Tensor:
            outputs = []
            start = 0
            for length in lengths_list:
                if length == 0:
                    continue
                end = start + length
                q_chunk = q[start:end].transpose(0, 1)
                k_chunk = k[start:end].transpose(0, 1)
                v_chunk = v[start:end].transpose(0, 1)

                attn_scores = torch.matmul(q_chunk * self.scaling, k_chunk.transpose(-1, -2))
                attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_chunk.dtype)
                attn_output = torch.matmul(attn_weights, v_chunk)
                outputs.append(attn_output.transpose(0, 1))
                start = end

            return torch.cat(outputs, dim=0) if outputs else torch.zeros_like(q)

        output: torch.Tensor
        if cu_seqlens is not None:
            try:
                from flash_attn import flash_attn_varlen_func

                max_seqlen = max(lengths) if lengths else 0
                output = flash_attn_varlen_func(
                    q * self.scaling,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    dropout_p=0.0,
                    softmax_scale=None,
                    causal=False,
                )
            except (ImportError, RuntimeError):
                output = _eager_attention(lengths)
        else:
            output = _eager_attention(lengths)

        output = output.view(seq_len, self.hidden_size)
        output = self.proj(output)

        return output


class Qwen3VLVisionMLP(nn.Module):
    """
    MLP for vision encoder with GELU activation.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc1(x)
        x = self.act(x)
        x = self.linear_fc2(x)
        return x


class Qwen3VLVisionBlock(nn.Module):
    """
    Vision Transformer block with pre-norm architecture.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
        intermediate_size: int = 4304,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3VLVisionPatchMerger(nn.Module):
    """
    Merges 2x2 spatial patches and projects to text hidden size.

    This reduces the sequence length by 4x while projecting to the
    text decoder's hidden dimension.
    """

    def __init__(
        self,
        vision_hidden_size: int = 1152,
        text_hidden_size: int = 3584,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-6,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.vision_hidden_size = vision_hidden_size

        mlp_input_size = vision_hidden_size * (spatial_merge_size ** 2)

        if not use_postshuffle_norm:
            self.norm = nn.LayerNorm(vision_hidden_size, eps=norm_eps)
        else:
            self.norm = nn.LayerNorm(mlp_input_size, eps=norm_eps)

        self.linear_fc1 = nn.Linear(mlp_input_size, mlp_input_size, bias=True)
        self.act = nn.GELU()
        self.linear_fc2 = nn.Linear(mlp_input_size, text_hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch * T*H*W, vision_hidden_size)
            grid_thw: (num_images, 3) grid dimensions

        Returns:
            merged: (batch * T*H/m*W/m, text_hidden_size)
        """
        if isinstance(grid_thw, tuple):
            grid_thw = torch.tensor([grid_thw], device=hidden_states.device, dtype=torch.long)
        elif isinstance(grid_thw, list):
            grid_thw = torch.tensor(grid_thw, device=hidden_states.device, dtype=torch.long)
        else:
            grid_thw = grid_thw.to(hidden_states.device)

        tokens_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        splits = hidden_states.split(tokens_per_image, dim=0)

        merged_outputs = []
        for states, dims in zip(splits, grid_thw.tolist()):
            t, h, w = map(int, dims)
            merged_outputs.append(self._merge_single(states, t, h, w))

        return torch.cat(merged_outputs, dim=0)

    def _merge_single(self, states: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        m = self.spatial_merge_size

        states = states.contiguous()

        if not self.use_postshuffle_norm:
            states = self.norm(states)

        states = states.view(t, h // m, m, w // m, m, self.vision_hidden_size)
        states = states.permute(0, 1, 3, 2, 4, 5).contiguous()
        states = states.view(t * (h // m) * (w // m), -1)

        if self.use_postshuffle_norm:
            states = self.norm(states)

        states = self.linear_fc1(states)
        states = self.act(states)
        states = self.linear_fc2(states)

        return states


class Qwen3VLVisionModel(nn.Module):
    """
    Complete vision encoder for Qwen3-VL.

    Includes:
    - Patch embedding
    - Positional embeddings
    - 27 vision transformer blocks
    - DeepStack feature extraction at layers [8, 16, 24]
    - Final spatial merge
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
        intermediate_size: int = 4304,
        depth: int = 27,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        text_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_indexes: list[int] = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if deepstack_indexes is None:
            deepstack_indexes = [5, 11, 17]

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=3,
            embed_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

        # Learned position embeddings (critical for spatial awareness)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings**0.5)

        # 2D Rotary embeddings
        head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        self.rotary_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
            )
            for _ in range(depth)
        ])

        self.deepstack_indexes = deepstack_indexes

        # DeepStack mergers (with post-shuffle norm)
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(
                vision_hidden_size=hidden_size,
                text_hidden_size=text_hidden_size,
                spatial_merge_size=spatial_merge_size,
                norm_eps=norm_eps,
                use_postshuffle_norm=True,
            )
            for _ in deepstack_indexes
        ])

        # Final merger (with pre-shuffle norm)
        self.merger = Qwen3VLVisionPatchMerger(
            vision_hidden_size=hidden_size,
            text_hidden_size=text_hidden_size,
            spatial_merge_size=spatial_merge_size,
            norm_eps=norm_eps,
            use_postshuffle_norm=False,
        )

        self.spatial_merge_size = spatial_merge_size

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D rotary position embeddings for vision patches.
        Matches HuggingFace implementation exactly.

        Args:
            grid_thw: (num_images, 3) tensor of (T, H, W) dimensions

        Returns:
            embeddings: (total_tokens, head_dim//2) rotary embeddings
        """
        merge_size = self.spatial_merge_size
        device = grid_thw.device

        max_hw = int(grid_thw[:, 1:].max().item())
        inv_freq = self.rotary_emb.inv_freq.to(device=device)

        pos = torch.arange(max_hw, device=device, dtype=inv_freq.dtype)
        freq_table = torch.outer(pos, inv_freq)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for dims in grid_thw.tolist():
            num_frames, height, width = map(int, dims)
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device, dtype=torch.long)
            block_cols = torch.arange(merged_w, device=device, dtype=torch.long)
            intra_row = torch.arange(merge_size, device=device, dtype=torch.long)
            intra_col = torch.arange(merge_size, device=device, dtype=torch.long)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.size(0)
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Fast bilinear interpolation of learned position embeddings.
        Matches HuggingFace implementation.

        Args:
            grid_thw: (num_images, 3) tensor of (T, H, W) dimensions

        Returns:
            patch_pos_embeds: (total_patches, hidden_size) interpolated embeddings
        """
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Args:
            pixel_values: (batch, channels, time, height, width)

        Returns:
            vision_embeds: (batch * num_merged_patches, text_hidden_size)
            deepstack_features: List of 3 tensors, each (batch * num_merged_patches, text_hidden_size)
        """
        hidden_states, grid_thw = self.patch_embed(pixel_values)

        batch_size, num_patches, hidden_dim = hidden_states.shape

        if isinstance(grid_thw, tuple):
            grid_thw_tensor = torch.tensor([grid_thw] * batch_size, dtype=torch.long, device=hidden_states.device)
        elif isinstance(grid_thw, list):
            grid_thw_tensor = torch.tensor(grid_thw, dtype=torch.long, device=hidden_states.device)
        else:
            grid_thw_tensor = grid_thw.to(device=hidden_states.device, dtype=torch.long)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_tensor)
        hidden_states = hidden_states + pos_embeds.view(batch_size, num_patches, -1).to(hidden_states.dtype)

        hidden_states = hidden_states.reshape(-1, hidden_dim)

        rotary_pos_emb = self.rot_pos_emb(grid_thw_tensor).to(hidden_states.device, hidden_states.dtype)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        position_embeddings = (cos, sin)

        frame_lengths = torch.repeat_interleave(
            grid_thw_tensor[:, 1] * grid_thw_tensor[:, 2],
            repeats=grid_thw_tensor[:, 0],
        )
        frame_lengths = frame_lengths.to(torch.int32)
        if frame_lengths.numel() == 0:
            frame_lengths = torch.zeros(1, device=hidden_states.device, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(frame_lengths.cumsum(dim=0), (1, 0), value=0)

        deepstack_features: list[torch.Tensor] = []
        deepstack_cursor = 0

        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
            )

            if deepstack_cursor < len(self.deepstack_indexes) and layer_idx == self.deepstack_indexes[deepstack_cursor]:
                feat = self.deepstack_merger_list[deepstack_cursor](hidden_states, grid_thw_tensor)
                deepstack_features.append(feat)
                deepstack_cursor += 1

        final_grid = grid_thw_tensor.clone()
        final_grid[:, 1:] = final_grid[:, 1:] // self.spatial_merge_size

        vision_embeds = self.merger(hidden_states, grid_thw_tensor)

        return vision_embeds, deepstack_features, final_grid
