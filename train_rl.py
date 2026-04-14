#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for the geographic
cyclic-ordering task.

This takes the SFT-warmed model and optimizes it using a deterministic
reward signal: does the model's final answer match sgn(det(B-A, C-A))?

The key advantage over SFT: the model learns to be *correct*, not just
to imitate the training traces. This generalizes better to new questions.

Launch with:
    deepspeed --num_gpus N train_rl.py [args...]

Requires TRL >= 0.12.0 for GRPOTrainer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from reward import combined_reward, extract_answer, compute_ground_truth


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompts(records: list[dict], tokenizer: AutoTokenizer) -> list[dict]:
    """
    For GRPO, we provide the *prompt* (everything up to the final assistant turn)
    and let the model generate the *completion*.

    For pipeline 1: prompt = system + user + assistant(geocode call) + tool(result)
                    completion = assistant's final reasoning + answer
    For pipeline 2: prompt = system + user + ... + tool(cyclic_order result)
                    completion = assistant's final answer
    """
    results = []
    for rec in records:
        messages = rec["messages"]
        # The prompt is everything except the last assistant message
        prompt_messages = messages[:-1]
        tools = rec.get("tools", None)

        try:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,  # Add the "Assistant:" prefix
            )
        except TypeError:
            prompt_messages_copy = [dict(m) for m in prompt_messages]
            if tools and prompt_messages_copy and prompt_messages_copy[0]["role"] == "system":
                tool_desc = "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
                prompt_messages_copy[0]["content"] += tool_desc
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages_copy,
                tokenize=False,
                add_generation_prompt=True,
            )

        results.append({
            "prompt": prompt_text,
            "meta": rec["meta"],
            "expected_answer": rec["expected_answer"],
        })

    return results


def make_reward_fn(metas: list[dict]):
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    GRPOTrainer calls reward_fn(completions, prompts=...) where completions
    is a list of strings.
    """
    # Build a lookup from prompt index to meta
    meta_lookup = metas

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """
        Compute rewards for a batch of completions.

        When GRPOTrainer generates multiple completions per prompt, they
        share the same prompt index. We use the prompts kwarg to match back.
        """
        rewards = []
        prompts = kwargs.get("prompts", [])

        for i, completion in enumerate(completions):
            # Find the corresponding meta - GRPOTrainer passes prompts aligned
            # with completions (one prompt per completion, with repeats for
            # the group)
            if i < len(meta_lookup):
                meta = meta_lookup[i]
            else:
                # Wrap around if needed
                meta = meta_lookup[i % len(meta_lookup)]

            reward = combined_reward(completion, meta)
            rewards.append(reward)

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to SFT checkpoint or HuggingFace model ID")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/rl")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use num_epochs instead). Useful for quick tests.")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions to sample per prompt (G in GRPO)")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="geo-finetune",
                        help="Weights & Biases project name (set to '' to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    use_wandb = bool(args.wandb_project)

    if use_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            if args.wandb_run_name:
                os.environ["WANDB_NAME"] = args.wandb_run_name
        except ImportError:
            print("WARNING: wandb not installed, disabling tracking.")
            use_wandb = False

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",  # Left-padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    # With DeepSpeed, each process must load onto its own GPU.
    if args.deepspeed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    use_4bit = args.use_4bit and not args.no_4bit
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
        device_map=device_map,
    )

    # ---- Data ----
    records = load_jsonl(args.train_file)
    prompt_data = build_prompts(records, tokenizer)

    # The dataset for GRPOTrainer needs a "prompt" column
    dataset = Dataset.from_dict({
        "prompt": [d["prompt"] for d in prompt_data],
    })

    # Store metas separately for the reward function
    metas = [d["meta"] for d in prompt_data]

    # ---- LoRA config ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # ---- GRPO Config ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=42,
    )

    # ---- Reward function ----
    reward_fn = make_reward_fn(metas)

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )

    print(f"GRPO training on {len(dataset)} prompts")
    print(f"Model: {args.model_name}")
    print(f"Generations per prompt: {args.num_generations}")

    trainer.train()

    # ---- Save ----
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("GRPO training complete!")


if __name__ == "__main__":
    main()