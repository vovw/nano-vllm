"""
Example usage of Qwen3-VL 2B model with nano-vllm.

This demonstrates:
- Loading Qwen3-VL 2B weights
- Processing images
- Running vision-language inference
- Generating text responses to image questions

Usage:
    python3 examples/example_vlm.py
"""

import os
import glob
import torch
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file

from nanovllm.models.qwen3_vlm import Qwen3VLForConditionalGeneration
from nanovllm.utils.context import set_context
from nanovllm.vision_preprocessing import preprocess_image


def load_weights(model, model_path):
    """Load and remap weights from HuggingFace safetensors format."""
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

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    print(f"\nWeight loading summary:")
    print(f"Total remapped keys: {len(remapped)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        vision_missing = [k for k in missing if 'visual' in k]
        text_missing = [k for k in missing if 'visual' not in k]
        print(f"\nMissing vision keys: {len(vision_missing)}")
        if vision_missing[:5]:
            print("Sample missing vision keys:", vision_missing[:5])
        print(f"Missing text keys: {len(text_missing)}")
        if text_missing[:5]:
            print("Sample missing text keys:", text_missing[:5])

    # Tie embeddings
    if 'lm_head.weight' in missing and hasattr(model.model, 'embed_tokens'):
        model.lm_head.weight = model.model.embed_tokens.weight


def generate(model, tokenizer, input_ids, pixel_values, image_grid_thw, max_tokens=128, temperature=0.0):
    """Generate text with optional greedy decoding when temperature=0."""
    device = input_ids.device
    gen_ids = input_ids.clone()

    if not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw_tensor = torch.tensor(image_grid_thw, device=device, dtype=torch.long)
    else:
        image_grid_thw_tensor = image_grid_thw.to(device=device, dtype=torch.long)

    for step in range(max_tokens):
        seq_len = gen_ids.size(1)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

        positions = model.get_rope_index(gen_ids, image_grid_thw_tensor)

        if step == 0:
            print(f"Input IDs shape: {gen_ids.shape}")
            print(f"Pixel values shape: {pixel_values.shape if pixel_values is not None else 'None'}")
            print(f"Position IDs shape: {positions.shape}")
            print(f"Initial image grid: {image_grid_thw_tensor.tolist()}")
            num_image_tokens = (gen_ids == model.image_token_id).sum().item()
            print(f"Number of image tokens in input: {num_image_tokens}")

        with torch.no_grad():
            hidden, final_grid = model(
                input_ids=gen_ids,
                positions=positions,
                pixel_values=pixel_values,
            )
            logits = model.compute_logits(hidden)

        if step == 0 and final_grid is not None:
            print(f"Final merged grid: {final_grid.tolist()}")
            image_grid_thw_tensor = final_grid.to(device=device, dtype=torch.long)

        next_logits = logits[-1, :]
        if temperature and temperature > 0.0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_text = tokenizer.decode([next_token.item()])
        print(f"Step {step}: token={next_token.item()}, text={token_text!r}", flush=True)

        if next_token.item() in [151643, 151645]:  # EOS tokens
            print(f"EOS token detected: {next_token.item()}")
            break

        gen_ids = torch.cat([gen_ids, next_token.view(1, 1)], dim=1)

    return tokenizer.decode(gen_ids[0, input_ids.size(1):].tolist(), skip_special_tokens=True)


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-VL-2B")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Using device={device}, dtype={inference_dtype}")

    model = Qwen3VLForConditionalGeneration(config)
    load_weights(model, path)
    model = model.to(device=device, dtype=inference_dtype).eval()

    # Preprocess image
    image_path = "test_cat.jpg"
    pixel_values, grid_thw = preprocess_image(image_path)
    pixel_values = pixel_values.to(device=device, dtype=inference_dtype)

    # Compute final grid dimensions after spatial merge (2x2)
    spatial_merge_size = getattr(model.visual, "spatial_merge_size", 2) if model.visual is not None else 2
    final_grid_thw = (
        grid_thw[0],
        grid_thw[1] // spatial_merge_size,
        grid_thw[2] // spatial_merge_size,
    )
    image_grid_thw = torch.tensor([final_grid_thw], device=device, dtype=torch.long)

    # Build prompt with image using chat template
    question = "how many people are in this image?"
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
    if placeholder_positions.size(0) != 1:
        raise ValueError("Expected a single image placeholder token in the prompt.")
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
        input_ids[:, image_pos + 1 :],
    ], dim=1)

    # Generate
    output = generate(
        model,
        tokenizer,
        input_ids,
        pixel_values,
        image_grid_thw,
        max_tokens=100,
        temperature=0.0,
    )

    print(f"\nImage: {image_path!r}")
    print(f"Question: {question!r}")
    print(f"Answer: {output!r}")


if __name__ == "__main__":
    main()
