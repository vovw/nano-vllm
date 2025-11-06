import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    model_type: str | None = None
    trust_remote_code: bool | None = None
    hf_config: AutoConfig | None = None
    text_config: AutoConfig | None = None
    is_vlm: bool = False
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        trust_remote_code = self.trust_remote_code or False
        self.hf_config = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=trust_remote_code,
        )

        if self.model_type is None:
            self.model_type = self._infer_model_type()

        self.is_vlm = self._is_vlm()

        if self.is_vlm and not trust_remote_code:
            trust_remote_code = True
            self.hf_config = AutoConfig.from_pretrained(
                self.model,
                trust_remote_code=trust_remote_code,
            )

        self.trust_remote_code = trust_remote_code
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)

        max_pos_embeddings = getattr(
            self.text_config,
            "max_position_embeddings",
            getattr(self.hf_config, "max_position_embeddings", self.max_model_len),
        )
        self.max_model_len = min(self.max_model_len, max_pos_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

    def _infer_model_type(self) -> str | None:
        if self.hf_config is None:
            return None
        model_type = getattr(self.hf_config, "model_type", None)
        if model_type is None and hasattr(self.hf_config, "text_config"):
            model_type = getattr(self.hf_config.text_config, "model_type", None)
        if model_type is None and getattr(self.hf_config, "vision_config", None) is not None:
            return "qwen3_vl"
        return model_type

    def _is_vlm(self) -> bool:
        if self.hf_config is None:
            return False
        if getattr(self.hf_config, "vision_config", None) is not None:
            return True
        model_type = self.model_type or getattr(self.hf_config, "model_type", "")
        return isinstance(model_type, str) and "vl" in model_type.lower()
