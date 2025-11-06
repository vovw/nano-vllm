"""Utilities for representing multimodal prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from nanovllm.engine.sequence import VisionInput


@dataclass(slots=True)
class Prompt:
    """Container describing input tokens and optional vision attachments."""

    token_ids: list[int]
    vision_inputs: list[VisionInput] = field(default_factory=list)

    @classmethod
    def from_iterable(
        cls,
        token_ids: Iterable[int],
        vision_inputs: Iterable[VisionInput] | None = None,
    ) -> "Prompt":
        return cls(list(token_ids), list(vision_inputs) if vision_inputs is not None else [])
