"""Utilities for remapping checkpoint weight names to nano-vllm modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class WeightMapResult:
    """Result of mapping a single checkpoint weight name."""

    name: str | None
    skip: bool = False


class WeightNameMapper:
    """Maps HuggingFace checkpoint parameter names to nano-vllm modules."""

    def __init__(self, rules: Iterable[Callable[[str], WeightMapResult | None]]):
        self._rules = list(rules)

    def map(self, weight_name: str) -> WeightMapResult:
        """Return the remapped weight name (or indicate it should be skipped)."""

        for rule in self._rules:
            result = rule(weight_name)
            if result is not None:
                return result
        return WeightMapResult(name=weight_name)

    @classmethod
    def for_model(cls, model, hf_config=None) -> "WeightNameMapper":
        """Factory that selects a mapper based on the instantiated model."""

        if getattr(model, "__class__", None) is not None:
            class_name = model.__class__.__name__.lower()
        else:
            class_name = ""

        has_vision = hasattr(model, "visual") or (
            hf_config is not None and getattr(hf_config, "vision_config", None) is not None
        )

        if "vl" in class_name or has_vision:
            return cls(_build_qwen3_vl_rules())
        return cls(_build_identity_rules())


def _build_identity_rules():
    return [_identity_rule]


def _build_qwen3_vl_rules():
    def drop_outer_model(weight_name: str) -> WeightMapResult | None:
        if not weight_name.startswith("model."):
            return None

        parts = weight_name.split(".")
        # Drop outer "model"
        parts = parts[1:]
        if parts and parts[0] == "language_model":
            parts = parts[1:]
        if len(parts) >= 2 and parts[0] == "visual" and parts[1] == "model":
            parts = [parts[0]] + parts[2:]
        remapped = ".".join(parts)
        return WeightMapResult(name=remapped)

    return [drop_outer_model, _identity_rule]


def _identity_rule(weight_name: str) -> WeightMapResult:
    return WeightMapResult(name=weight_name)
