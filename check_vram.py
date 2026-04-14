#!/usr/bin/env python3
"""
Single-GPU VRAM stress test.

Simulates the memory profile of both SFT and RL (GRPO) workloads:
  - SFT:  forward + backward on a single sequence
  - GRPO: forward + backward PLUS generation of N completions (the expensive part)

Reports peak VRAM at each stage so you can verify the workload fits in 24GB
before requesting 8 GPUs and waiting hours.

Usage:
    python check_vram.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --batch_size 2 \
        --seq_length 2048 \
        --num_generations 4
"""

from __future__ import annotations

import argparse
import gc
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def fmt(b: int) -> str:
    return f"{b / 1024**3:.2f} GB"


def report(label: str, device: torch.device, peak: bool = True):
    if peak:
        mem = torch.cuda.max_memory_allocated(device)
    else:
        mem = torch.cuda.memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    pct = 100 * mem / total
    print(f"  {label:30s}  {fmt(mem):>10s}  ({pct:.1f}%)")
    return mem


def main():
    parser = argparse.ArgumentParser(description="Single-GPU VRAM stress test")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size (match your training config)")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Sequence length for SFT forward/backward")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO completions per prompt (G parameter)")
    parser.add_argument("--gen_length", type=int, default=512,
                        help="Max tokens per generation (max_completion_length)")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    print(f"GPU:        {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {fmt(total_vram)}")

    use_4bit = not args.no_4bit

    # ---- Load model ----
    print(f"\n{'='*65}")
    print("PHASE 1: Model loading")
    print(f"{'='*65}")

    torch.cuda.reset_peak_memory_stats(device)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    report("After model load", device)

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total_params = 0, 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    report("After LoRA", device)
    print(f"  {'Trainable params':30s}  {trainable:>10,} / {total_params:,} "
          f"({100*trainable/total_params:.1f}%)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================================================================
    # SFT workload
    # ==================================================================
    print(f"\n{'='*65}")
    print(f"PHASE 2: SFT workload  (batch={args.batch_size}, seq={args.seq_length})")
    print(f"{'='*65}")

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    dummy_input = torch.randint(0, 1000, (args.batch_size, args.seq_length), device=device)

    # Forward
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=dummy_input, labels=dummy_input)
        loss = output.loss
    sft_fwd = report("After SFT forward", device)

    # Backward
    loss.backward()
    sft_bwd = report("After SFT backward", device)

    # Simulate optimizer step (Adam state = 2 copies of trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    optimizer.step()
    sft_opt = report("After optimizer step", device)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, output, loss, dummy_input
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # GRPO workload
    # ==================================================================
    print(f"\n{'='*65}")
    print(f"PHASE 3: GRPO workload  (batch=1, generations={args.num_generations}, "
          f"gen_length={args.gen_length})")
    print(f"{'='*65}")

    torch.cuda.reset_peak_memory_stats(device)

    # GRPO generates N completions per prompt, then scores and does policy update.
    # The generation phase is the memory bottleneck: the KV cache grows with
    # num_generations * gen_length.

    model.eval()
    model.gradient_checkpointing_disable()

    # Simulate: generate num_generations completions from a prompt
    prompt_len = 512  # typical prompt length after tool-call history
    prompt = torch.randint(0, 1000, (1, prompt_len), device=device)

    print(f"  Generating {args.num_generations} completions of up to {args.gen_length} tokens...")
    completions = []
    gen_peak = 0
    for i in range(args.num_generations):
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                prompt,
                max_new_tokens=args.gen_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions.append(out)
        gen_peak = max(gen_peak, torch.cuda.max_memory_allocated(device))

    print(f"  {'Peak during generation':30s}  {fmt(gen_peak):>10s}  "
          f"({100*gen_peak/total_vram:.1f}%)")

    # Now simulate the policy gradient step (forward+backward on the completions)
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    del completions
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Forward+backward on a batch of generated completions
    rl_seq_len = prompt_len + args.gen_length
    rl_batch = torch.randint(0, 1000, (args.num_generations, rl_seq_len), device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=rl_batch, labels=rl_batch)
        loss = output.loss
    report("After GRPO forward", device)

    loss.backward()
    rl_bwd = report("After GRPO backward", device)

    rl_peak = max(gen_peak, rl_bwd)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Total VRAM':30s}  {fmt(total_vram):>10s}")
    print(f"  {'SFT peak':30s}  {fmt(sft_opt):>10s}  "
          f"(headroom: {fmt(total_vram - sft_opt)})")
    print(f"  {'GRPO peak':30s}  {fmt(rl_peak):>10s}  "
          f"(headroom: {fmt(total_vram - rl_peak)})")
    print()

    ok = True
    for label, peak in [("SFT", sft_opt), ("GRPO", rl_peak)]:
        headroom = total_vram - peak
        if headroom > 2 * 1024**3:
            print(f"  {label:6s}  ✓  Safe ({fmt(headroom)} headroom)")
        elif headroom > 0:
            print(f"  {label:6s}  ⚠  Tight ({fmt(headroom)} headroom) — may OOM with "
                  "longer sequences")
            ok = False
        else:
            print(f"  {label:6s}  ✗  OOM — reduce batch_size, seq_length, or "
                  "num_generations")
            ok = False

    print(f"{'='*65}")
    if ok:
        print("All workloads fit. Safe to scale to 8 GPUs.")
    else:
        print("Some workloads are tight or OOM. Adjust parameters before scaling up.")


if __name__ == "__main__":
    main()