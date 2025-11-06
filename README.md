# Nano-vLLM

A lightweight vLLM implementation built from scratch, now with **Qwen3-VL 2B** vision-language model support!

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 👁️ **Vision-Language Support** - Full support for Qwen3-VL 2B multimodal inference
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Manual Download

### Qwen3-VL 2B (Vision-Language Model)
```bash
huggingface-cli download --resume-download Qwen/Qwen3-VL-2B \
  --local-dir ~/huggingface/Qwen3-VL-2B/ \
  --local-dir-use-symlinks False
```

### Qwen3 0.6B (Text-Only Model)
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

### Text-Only Model (Qwen3 0.6B)
See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

### Vision-Language Model (Qwen3-VL 2B)
See `examples/example_vlm.py` for a complete example with image processing and generation. Basic usage:
```python
from nanovllm.models.qwen3_vlm import Qwen3VLForConditionalGeneration
from nanovllm.vision_preprocessing import preprocess_image
from transformers import AutoConfig, AutoTokenizer

# Load model
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration(config)
# ... load weights and preprocess image ...
# See examples/example_vlm.py for full details
```

## Benchmark

See `bench.py` for text-only model benchmarks.

### Text-Only Model (Qwen3 0.6B)
**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

### Vision-Language Model (Qwen3-VL 2B)
**Test Configuration:**
- Model: Qwen3-VL 2B
- Test: 100 token generation with image input (~254 input tokens)
- Hardware: CUDA-enabled GPU

**Performance Results:**
| Implementation | Time (s) | Throughput (tok/s) |
|---------------|----------|-------------------|
| HuggingFace   | 1.35s    | 74.11 tok/s       |
| nano-vllm     | 1.96s    | 50.99 tok/s       |

**Status:** VLM support is functional but not yet fully optimized. The current implementation uses full sequence re-computation without KV caching, which results in lower throughput. KV cache optimization for VLM is planned for future releases.

**Features:**
- Full multimodal inference support
- Dynamic image token handling
- Efficient vision encoder integration
- See `examples/example_vlm.py` for complete usage example with image input


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
