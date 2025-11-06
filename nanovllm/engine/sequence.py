from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import Iterable

import torch

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class VisionInput:
    """Container for per-image metadata associated with a sequence."""

    pixel_values: torch.Tensor
    grid_thw: tuple[int, int, int]
    token_span: tuple[int, int]

    def num_tokens(self) -> int:
        return self.token_span[1] - self.token_span[0]


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        vision_inputs: Iterable[VisionInput] | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.vision_inputs = list(vision_inputs) if vision_inputs is not None else []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        vision_state = [
            (
                vision.pixel_values,
                vision.grid_thw,
                vision.token_span,
            )
            for vision in self.vision_inputs
        ]
        payload = (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            vision_state,
            self.token_ids,
            self.last_token,
        )
        return payload

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            vision_state,
            token_ids,
            last_token,
        ) = state
        self.vision_inputs = [
            VisionInput(pixel_values=pixel_values, grid_thw=grid_thw, token_span=token_span)
            for pixel_values, grid_thw, token_span in vision_state
        ]
        self.token_ids = token_ids
        self.last_token = last_token
