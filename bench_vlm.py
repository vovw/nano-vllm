"""
Benchmark script for comparing nano-vllm vs HuggingFace Qwen3-VL 2B inference speed.

Usage:
    python3 bench_vlm.py --impl nanovllm  # Test nano-vllm implementation
    python3 bench_vlm.py --impl hf        # Test HuggingFace implementation
    python3 bench_vlm.py --impl nanovllm --num-tokens 50  # Custom token count
"""

import os
import time
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer


def benchmark_nanovllm(model_path: str, image_path: str, num_tokens: int = 100):
    """Benchmark nano-vllm implementation."""
    from nanovllm.models.qwen3_vlm import Qwen3VLForConditionalGeneration
    from nanovllm.vision_preprocessing import preprocess_image
    from nanovllm.utils.context import set_context
    import glob
    from safetensors.torch import load_file

    print("=" * 60)
    print("Benchmarking nano-vllm implementation")
    print("=" * 60)

    # Load model
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Qwen3VLForConditionalGeneration(config)

    # Load weights
    print("Loading weights...")
    state_dict = {}
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        state_dict.update(load_file(f))

    remapped = {}
    for orig_key, v in state_dict.items():
        k = orig_key
        if k.startswith('model.language_model.'):
            k = k.replace('model.language_model.', 'model.')
        elif k.startswith('model.visual.'):
            k = k.replace('model.', '')

        # Skip k_proj, v_proj, up_proj for text layers (will be merged)
        if 'language_model' in orig_key and ('.k_proj.' in orig_key or '.v_proj.' in orig_key or '.up_proj.' in orig_key):
            continue

        # Merge QKV for text layers
        if '.q_proj.' in orig_key and 'language_model' in orig_key:
            q_weight = v
            k_key_orig = orig_key.replace('.q_proj.', '.k_proj.')
            v_key_orig = orig_key.replace('.q_proj.', '.v_proj.')
            k_weight = state_dict.get(k_key_orig)
            v_weight = state_dict.get(v_key_orig)

            if k_weight is not None and v_weight is not None:
                merged = torch.cat([q_weight, k_weight, v_weight], dim=0)
                remapped[k.replace('.q_proj.', '.qkv_proj.')] = merged
                continue

        # Merge gate/up for text layers
        if '.gate_proj.' in orig_key and 'language_model' in orig_key:
            gate_weight = v
            up_key_orig = orig_key.replace('.gate_proj.', '.up_proj.')
            up_weight = state_dict.get(up_key_orig)

            if up_weight is not None:
                merged = torch.cat([gate_weight, up_weight], dim=0)
                remapped[k.replace('.gate_proj.', '.gate_up_proj.')] = merged
                continue

        remapped[k] = v

    model.load_state_dict(remapped, strict=False)

    # Tie embeddings
    if hasattr(model.model, 'embed_tokens'):
        model.lm_head.weight = model.model.embed_tokens.weight

    model = model.to(device=device, dtype=inference_dtype).eval()

    # Preprocess image
    print("Preprocessing image...")
    pixel_values, grid_thw = preprocess_image(image_path)
    pixel_values = pixel_values.to(device=device, dtype=inference_dtype)

    spatial_merge_size = getattr(model.visual, "spatial_merge_size", 2) if model.visual is not None else 2
    final_grid_thw = (
        grid_thw[0],
        grid_thw[1] // spatial_merge_size,
        grid_thw[2] // spatial_merge_size,
    )
    image_grid_thw = torch.tensor([final_grid_thw], device=device, dtype=torch.long)

    # Build prompt
    question = "Describe this image in detail, including all visible elements, colors, textures, and spatial relationships."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    num_image_tokens = (grid_thw[0] * grid_thw[1] * grid_thw[2]) // (spatial_merge_size ** 2)
    placeholder_positions = (input_ids == model.image_token_id).nonzero(as_tuple=False)
    image_pos = placeholder_positions[0, 1].item()
    image_token_ids = torch.full(
        (num_image_tokens,),
        model.image_token_id,
        device=device,
        dtype=input_ids.dtype,
    )
    input_ids = torch.cat([
        input_ids[:, :image_pos],
        image_token_ids.unsqueeze(0),
        input_ids[:, image_pos + 1:],
    ], dim=1)

    print(f"Input length: {input_ids.size(1)} tokens")

    # NOTE: This benchmark uses full sequence re-computation for each token
    # A proper KV cache implementation would significantly improve performance

    # Warmup
    print("Warming up...")
    for _ in range(2):
        gen_ids = input_ids.clone()
        for step in range(3):
            seq_len = gen_ids.size(1)
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            set_context(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
            )
            positions = model.get_rope_index(gen_ids, image_grid_thw)
            with torch.no_grad():
                hidden, final_grid = model(
                    input_ids=gen_ids,
                    positions=positions,
                    pixel_values=pixel_values if step == 0 else None,
                )
                if step == 0 and final_grid is not None:
                    image_grid_thw = final_grid.to(device=device, dtype=torch.long)
                logits = model.compute_logits(hidden)
                next_token = torch.argmax(logits[-1], dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, next_token.view(1, 1)], dim=1)

    # Benchmark
    print(f"\nGenerating {num_tokens} tokens...")
    gen_ids = input_ids.clone()
    image_grid_thw = torch.tensor([final_grid_thw], device=device, dtype=torch.long)

    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(num_tokens):
        seq_len = gen_ids.size(1)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )
        positions = model.get_rope_index(gen_ids, image_grid_thw)

        with torch.no_grad():
            hidden, final_grid = model(
                input_ids=gen_ids,
                positions=positions,
                pixel_values=pixel_values if step == 0 else None,
            )
            if step == 0 and final_grid is not None:
                image_grid_thw = final_grid.to(device=device, dtype=torch.long)
            logits = model.compute_logits(hidden)
            next_token = torch.argmax(logits[-1], dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_token.view(1, 1)], dim=1)

    actual_tokens = num_tokens

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = actual_tokens / elapsed

    output_text = tokenizer.decode(gen_ids[0, input_ids.size(1):].tolist(), skip_special_tokens=True)

    print(f"\n{'=' * 60}")
    print(f"nano-vllm Results:")
    print(f"{'=' * 60}")
    print(f"Generated tokens: {actual_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"\nGenerated text: {output_text[:200]}...")

    return throughput, actual_tokens, elapsed


def benchmark_huggingface(model_path: str, image_path: str, num_tokens: int = 100):
    """Benchmark HuggingFace implementation."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    print("=" * 60)
    print("Benchmarking HuggingFace implementation")
    print("=" * 60)

    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Prepare input
    print("Preprocessing image...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image in detail, including all visible elements, colors, textures, and spatial relationships."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    print(f"Input length: {inputs['input_ids'].size(1)} tokens")

    # Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

    # Benchmark
    print(f"\nGenerating {num_tokens} tokens...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    actual_tokens = generated_ids.size(1) - inputs['input_ids'].size(1)
    throughput = actual_tokens / elapsed

    output_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].size(1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"\n{'=' * 60}")
    print(f"HuggingFace Results:")
    print(f"{'=' * 60}")
    print(f"Generated tokens: {actual_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print(f"\nGenerated text: {output_text[:200]}...")

    return throughput, actual_tokens, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-VL implementations")
    parser.add_argument("--impl", choices=["nanovllm", "hf"], required=True,
                        help="Implementation to benchmark")
    parser.add_argument("--model-path", default=os.path.expanduser("~/huggingface/Qwen3-VL-2B"),
                        help="Path to model weights")
    parser.add_argument("--image-path", default="examples/test_cat.jpg",
                        help="Path to test image")
    parser.add_argument("--num-tokens", type=int, default=100,
                        help="Number of tokens to generate")

    args = parser.parse_args()

    if args.impl == "nanovllm":
        benchmark_nanovllm(args.model_path, args.image_path, args.num_tokens)
    else:
        benchmark_huggingface(args.model_path, args.image_path, args.num_tokens)


if __name__ == "__main__":
    main()
