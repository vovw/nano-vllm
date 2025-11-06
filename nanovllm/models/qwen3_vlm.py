import torch
from torch import nn
import torch.distributed as dist
from transformers import PretrainedConfig

from nanovllm.models.qwen3 import Qwen3MLP, Qwen3ForCausalLM
from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionModel
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import MultiModalRotaryEmbedding
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3VLAttention(nn.Module):
    """
    Attention layer for Qwen3-VL text decoder with MRoPE support.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 5000000.0,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # Get MRoPE section from config
        mrope_section = None
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            mrope_section = rope_scaling.get('mrope_section', None)

        self.rotary_emb = MultiModalRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
            mrope_section=mrope_section,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3VLDecoderLayer(nn.Module):
    """
    Decoder layer for Qwen3-VL with DeepStack support.
    """

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3VLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 5000000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3VLModel(nn.Module):
    """
    Qwen3-VL text model with DeepStack injection.
    """

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3VLDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # DeepStack injects at the first N layers, where N = number of deepstack features
        # This is determined dynamically based on deepstack_embeds length

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        deepstack_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            positions: Position IDs - can be 1D or 3D for MRoPE
            inputs_embeds: Optional pre-computed embeddings (for vision+text)
            vision_mask: Boolean mask indicating vision token positions
            deepstack_embeds: List of DeepStack visual features

        Returns:
            hidden_states: (batch_size * seq_len, hidden_size)
        """
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Flatten for processing
        if hidden_states.ndim == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(batch_size * seq_len, hidden_size)

        vision_mask_flat = None
        if vision_mask is not None:
            vision_mask_flat = vision_mask.reshape(-1).to(hidden_states.device)

        if deepstack_embeds is not None:
            deepstack_embeds = [
                ds.to(hidden_states.device, hidden_states.dtype) for ds in deepstack_embeds
            ]

        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)

            if (
                deepstack_embeds is not None
                and vision_mask_flat is not None
                and layer_idx < len(deepstack_embeds)
            ):
                hidden_states = self._inject_deepstack(
                    hidden_states,
                    vision_mask_flat,
                    deepstack_embeds[layer_idx],
                )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _inject_deepstack(
        self,
        hidden_states: torch.Tensor,
        vision_mask: torch.Tensor,
        deepstack_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject DeepStack visual features into hidden states.

        Args:
            hidden_states: (batch_size * seq_len, hidden_size)
            vision_mask: (batch_size * seq_len,) boolean mask
            deepstack_features: (num_vision_tokens, hidden_size)

        Returns:
            hidden_states with visual features added
        """
        # Flatten vision mask if needed
        if vision_mask.ndim > 1:
            vision_mask = vision_mask.reshape(-1)

        vision_mask = vision_mask.to(dtype=torch.bool, device=hidden_states.device)

        if not vision_mask.any():
            return hidden_states

        num_vision_tokens = int(vision_mask.sum().item())
        if deepstack_features.size(0) != num_vision_tokens:
            raise ValueError(
                f"DeepStack feature count ({deepstack_features.size(0)}) does not match vision tokens ({num_vision_tokens})."
            )

        hidden_states = hidden_states.clone()
        hidden_states[vision_mask] = hidden_states[vision_mask] + deepstack_features.to(
            hidden_states.device,
            hidden_states.dtype,
        )

        return hidden_states


class Qwen3VLForConditionalGeneration(nn.Module):
    """
    Complete Qwen3-VL model for conditional generation.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig
    ) -> None:
        super().__init__()
        self.config = config

        # Handle nested config structure (HuggingFace format)
        text_config = getattr(config, 'text_config', config)
        vision_config = getattr(config, 'vision_config', None)

        # Vision encoder
        if vision_config is not None:
            self.visual = Qwen3VLVisionModel(
                hidden_size=vision_config.hidden_size,
                num_heads=vision_config.num_heads,
                intermediate_size=vision_config.intermediate_size,
                depth=vision_config.depth,
                patch_size=vision_config.patch_size,
                temporal_patch_size=vision_config.temporal_patch_size,
                spatial_merge_size=vision_config.spatial_merge_size,
                text_hidden_size=getattr(vision_config, 'out_hidden_size', vision_config.hidden_size),
                num_position_embeddings=vision_config.num_position_embeddings,
                deepstack_indexes=getattr(vision_config, 'deepstack_visual_indexes', [5, 11, 17]),
            )
        else:
            self.visual = None

        # Text model (use text_config if available)
        self.model = Qwen3VLModel(text_config)
        vocab_size = getattr(text_config, 'vocab_size', getattr(config, 'vocab_size', 151936))
        hidden_size = getattr(text_config, 'hidden_size', getattr(config, 'hidden_size', 2048))
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)

        tie_embeddings = getattr(config, 'tie_word_embeddings', False) or getattr(text_config, 'tie_word_embeddings', False)
        if tie_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

        # Special token IDs
        self.image_token_id = getattr(config, 'image_token_id', 151655)
        self.vision_start_token_id = getattr(config, 'vision_start_token_id', 151652)
        self.vision_end_token_id = getattr(config, 'vision_end_token_id', 151653)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int, int] | None]:
        """
        Forward pass for VLM.

        Args:
            input_ids: (batch_size, seq_len)
            positions: Position IDs
            pixel_values: Optional (batch_size, channels, time, height, width)

        Returns:
            hidden_states: (batch_size * seq_len, hidden_size)
            final_grid: (T, H, W) grid dimensions after merge, or None if no vision
        """
        # Process vision if pixel_values provided
        vision_embeds = None
        deepstack_embeds = None
        vision_mask = None
        final_grid = None

        if pixel_values is not None and self.visual is not None:
            # Encode vision
            vision_embeds, deepstack_embeds, final_grid = self.visual(pixel_values)

            # Get text embeddings
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Create vision mask
            vision_mask = (input_ids == self.image_token_id)

            # Scatter vision embeddings into text sequence
            if vision_mask.any():
                # Flatten batch dimension
                batch_size, seq_len = input_ids.shape
                inputs_embeds_flat = inputs_embeds.view(-1, inputs_embeds.size(-1))
                vision_mask_flat = vision_mask.flatten()

                if vision_embeds.size(0) != int(vision_mask_flat.sum().item()):
                    raise ValueError(
                        "Vision embedding count does not match number of image tokens "
                        f"({vision_embeds.size(0)} vs {vision_mask_flat.sum().item()})."
                    )

                vision_embeds = vision_embeds.to(inputs_embeds_flat.dtype)
                # Replace image tokens with vision embeddings
                inputs_embeds_flat[vision_mask_flat] = vision_embeds

                # Reshape back
                inputs_embeds = inputs_embeds_flat.view(batch_size, seq_len, -1)
            else:
                deepstack_embeds = None
        else:
            inputs_embeds = None

        # Forward through text model
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            vision_mask=vision_mask,
            deepstack_embeds=deepstack_embeds,
        )

        return hidden_states, final_grid

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """
        Calculate 3D position IDs for MRoPE following HF implementation.

        Args:
            input_ids: (batch_size, seq_len)
            image_grid_thw: List of (T, H, W) tuples for each image

        Returns:
            position_ids: (3, batch_size, seq_len) for MRoPE
        """
        spatial_merge_size = 2
        batch_size, seq_len = input_ids.shape

        # Initialize position IDs: (3, batch_size, seq_len)
        position_ids = torch.ones(3, batch_size, seq_len, dtype=torch.long, device=input_ids.device)

        for i, input_id_seq in enumerate(input_ids):
            # Find vision start tokens
            vision_start_indices = torch.where(input_id_seq == self.vision_start_token_id)[0]

            if len(vision_start_indices) == 0:
                # No vision tokens, all dimensions have SAME sequential positions
                pos = torch.arange(seq_len, device=input_ids.device)
                position_ids[0, i] = pos  # T dimension
                position_ids[1, i] = pos  # H dimension (same as T)
                position_ids[2, i] = pos  # W dimension (same as T)
                continue

            llm_pos_ids_list = []
            st = 0
            image_index = 0


            for _ in range(len(vision_start_indices)):
                # Find the next image token (after vision_start)
                vision_start_pos = vision_start_indices[image_index].item()

                # Text before vision_start (including vision_start itself)
                text_len = vision_start_pos + 1 - st
                if text_len > 0:
                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
                    # Text tokens: ALL dimensions have SAME values (matching HF implementation)
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    )

                # Get grid dimensions for vision tokens
                # Note: image_grid_thw is already AFTER spatial merge
                if image_index < len(image_grid_thw):
                    t, h, w = image_grid_thw[image_index]
                    llm_grid_t, llm_grid_h, llm_grid_w = t, h, w
                    num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w

                    st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0

                    # Add vision token positions (3D)
                    # Offset is added to ALL dimensions (matching HF implementation)
                    t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                    # Find vision_end and continue from there
                    vision_end_indices = torch.where(input_id_seq[vision_start_pos:] == self.vision_end_token_id)[0]
                    if len(vision_end_indices) > 0:
                        vision_end_pos = vision_start_pos + vision_end_indices[0].item()
                        # Add position for vision_end token itself (ALL dimensions same)
                        st_idx_after_vision = llm_pos_ids_list[-1].max().item() + 1
                        llm_pos_ids_list.append(torch.tensor([[st_idx_after_vision], [st_idx_after_vision], [st_idx_after_vision]], device=input_ids.device))
                        st = vision_end_pos + 1
                    else:
                        st = vision_start_pos + 1 + num_vision_tokens

                    image_index += 1

            # Remaining text tokens
            if st < seq_len:
                st_idx = llm_pos_ids_list[-1].max().item() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = seq_len - st
                # Text tokens: ALL dimensions have SAME values (matching HF implementation)
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1)
            position_ids[:, i, :llm_positions.size(1)] = llm_positions.to(position_ids.device)

        return position_ids
