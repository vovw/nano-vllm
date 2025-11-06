import logging
import os
from collections import defaultdict
from glob import glob
from typing import Any

from safetensors import safe_open
from torch import Tensor, nn

from nanovllm.utils.weight_mapping import WeightNameMapper


logger = logging.getLogger(__name__)


def default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    param.data.copy_(loaded_weight)


def load_model(
    model: nn.Module,
    path: str,
    *,
    hf_config: Any | None = None,
    strict: bool = True,
) -> None:
    """Load model weights from a directory of safetensors files.

    Args:
        model: The instantiated nano-vllm module.
        path: Directory containing safetensors shards.
        hf_config: Optional HuggingFace config used to pick a mapping strategy.
        strict: If ``True`` (default) raise when missing or unexpected weights remain.
    """

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    mapper = WeightNameMapper.for_model(model, hf_config)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    expected = set(parameters.keys()) | set(buffers.keys())

    loaded = set()
    shard_counters: dict[str, int] = defaultdict(int)
    unexpected: dict[str, list[str]] = defaultdict(list)

    weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")

    for file in weight_files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                map_result = mapper.map(weight_name)
                if map_result.skip or map_result.name is None:
                    continue

                remapped_name = map_result.name
                tensor = f.get_tensor(weight_name)

                handled = False
                for pattern, (target, shard_id) in packed_modules_mapping.items():
                    if pattern in remapped_name:
                        param_name = remapped_name.replace(pattern, target)
                        param = parameters.get(param_name)
                        if param is None:
                            unexpected[param_name].append(weight_name)
                            break
                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is None:
                            raise AttributeError(
                                f"Parameter {param_name} missing weight_loader for packed module"
                            )
                        weight_loader(param, tensor, shard_id)
                        loaded.add(param_name)
                        shard_counters[param_name] += 1
                        handled = True
                        break

                if handled:
                    continue

                if remapped_name in parameters:
                    param = parameters[remapped_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
                    loaded.add(remapped_name)
                    continue

                if remapped_name in buffers:
                    buffer = buffers[remapped_name]
                    if buffer.shape != tensor.shape:
                        raise ValueError(
                            f"Shape mismatch for buffer {remapped_name}: "
                            f"expected {tuple(buffer.shape)}, got {tuple(tensor.shape)}"
                        )
                    buffer.copy_(tensor)
                    loaded.add(remapped_name)
                    continue

                unexpected[remapped_name].append(weight_name)

    missing = sorted(expected - loaded)
    unexpected_flat = sorted(unexpected.keys())

    if missing or unexpected_flat:
        message_lines = [
            f"Weight loading summary for {model.__class__.__name__}",
            f"  Loaded parameters: {len(loaded)}/{len(expected)}",
            f"  Missing parameters: {len(missing)}",
            f"  Unexpected parameters: {len(unexpected_flat)}",
        ]
        if missing:
            sample_missing = "\n    ".join(missing[:10])
            message_lines.append(f"    Missing (sample):\n    {sample_missing}")
        if unexpected_flat:
            sample_unexpected = "\n    ".join(
                f"{name} <- {', '.join(originals)}" for name, originals in list(unexpected.items())[:10]
            )
            message_lines.append(f"    Unexpected mappings (sample):\n    {sample_unexpected}")
        message = "\n".join(message_lines)

        if strict:
            raise RuntimeError(message)
        logger.warning(message)

    for param_name, counter in shard_counters.items():
        expected_shards = _expected_shard_count(param_name, packed_modules_mapping)
        if expected_shards is not None and counter != expected_shards:
            msg = (
                f"Packed parameter {param_name} expected {expected_shards} shards but loaded {counter}."
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)


def _expected_shard_count(param_name: str, packed_mapping: dict[str, tuple[str, Any]]) -> int | None:
    target_name = None
    for _, (target, _) in packed_mapping.items():
        if target in param_name:
            target_name = target
            break
    if target_name is None:
        return None
    shard_ids = {
        shard_id for _, (target, shard_id) in packed_mapping.items() if target == target_name
    }
    return len(shard_ids) if shard_ids else None
