#!/usr/bin/env python3
"""Turn-level GRPO training for live pipeline-2 tool orchestration."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from live_reward import live_turn_reward


DEFAULT_TURN_MIX = "cyclic_order=0.5,geocode=0.2,final=0.3"
TURN_TYPES = ("geocode", "cyclic_order", "final")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_turn_mix(spec: str) -> dict[str, float]:
    mix: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --turn_mix item {item!r}; expected name=value")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in TURN_TYPES:
            raise ValueError(f"Unknown turn type in --turn_mix: {name!r}")
        weight = float(value)
        if weight < 0:
            raise ValueError("--turn_mix weights must be non-negative")
        mix[name] = weight

    if not mix or sum(mix.values()) <= 0:
        raise ValueError("--turn_mix must contain at least one positive weight")

    total = sum(mix.values())
    return {name: weight / total for name, weight in mix.items() if weight > 0}


def target_counts(total: int, mix: dict[str, float]) -> dict[str, int]:
    counts = {name: int(total * weight) for name, weight in mix.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        ((total * weight - counts[name], name) for name, weight in mix.items()),
        reverse=True,
    )
    for _, name in remainders[:remaining]:
        counts[name] += 1
    return counts


def weighted_sample_turns(
    records: list[dict[str, Any]],
    *,
    turn_mix: str,
    max_train_examples: int,
    max_examples_per_type: int,
    oversample_long_n: float,
    long_n_min: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not records:
        return []

    mix = parse_turn_mix(turn_mix)
    total = max_train_examples if max_train_examples > 0 else len(records)
    counts = target_counts(total, mix)
    if max_examples_per_type > 0:
        counts = {
            name: min(count, max_examples_per_type)
            for name, count in counts.items()
        }

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        turn_type = record.get("turn_type")
        if turn_type in mix:
            by_type[turn_type].append(record)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for turn_type, count in counts.items():
        if count <= 0:
            continue
        group = by_type.get(turn_type, [])
        if not group:
            raise ValueError(
                f"--turn_mix requested {turn_type!r}, but no such examples exist"
            )
        weights = [
            oversample_long_n if int(record.get("n_points", 0)) >= long_n_min else 1.0
            for record in group
        ]
        sampled.extend(rng.choices(group, weights=weights, k=count))

    rng.shuffle(sampled)
    return sampled


def apply_chat_template_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tokenizer: AutoTokenizer,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        messages_copy = [dict(m) for m in messages]
        if tools and messages_copy and messages_copy[0]["role"] == "system":
            tool_desc = "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
            messages_copy[0]["content"] = (
                (messages_copy[0].get("content") or "") + tool_desc
            )
        return tokenizer.apply_chat_template(
            messages_copy,
            tokenize=False,
            add_generation_prompt=True,
        )


def render_dataset_rows(
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = apply_chat_template_with_tools(
            record["prompt_messages"],
            record.get("tools"),
            tokenizer,
        )
        rows.append(
            {
                "prompt": prompt,
                "turn_type": record["turn_type"],
                "n_points": int(record["n_points"]),
                "source_question_id": str(record.get("source_question_id", "")),
                "expected_tool_name": record.get("expected_tool_name") or "",
                "expected_arguments_json": json.dumps(
                    record.get("expected_arguments"),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "expected_answer": record.get("expected_answer") or "",
                "meta_json": json.dumps(
                    record.get("meta"),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return rows


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content")
            return content if isinstance(content, str) else json.dumps(last)
    return str(completion)


def make_reward_fn():
    def reward_fn(completions: list[Any], **kwargs) -> list[float]:
        turn_types = kwargs["turn_type"]
        expected_tool_names = kwargs["expected_tool_name"]
        expected_arguments = kwargs["expected_arguments_json"]
        expected_answers = kwargs["expected_answer"]
        meta_jsons = kwargs["meta_json"]

        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            row = {
                "turn_type": turn_types[idx],
                "expected_tool_name": expected_tool_names[idx],
                "expected_arguments_json": expected_arguments[idx],
                "expected_answer": expected_answers[idx],
                "meta_json": meta_jsons[idx],
            }
            rewards.append(live_turn_reward(completion_to_text(completion), row))
        return rewards

    return reward_fn


def load_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    tokenizer_path = args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left",
            model_max_length=args.max_seq_length,
        )
    except OSError:
        if not args.base_model:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            padding_side="left",
            model_max_length=args.max_seq_length,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer


def load_model(args: argparse.Namespace):
    if args.deepspeed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map: str | dict[str, int] = {"": local_rank}
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

    if args.base_model:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
        return PeftModel.from_pretrained(base, args.model_name, is_trainable=True)

    return AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path when --model_name points to a LoRA adapter.",
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/live-rl")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint directory to resume trainer state from.",
    )
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    parser.add_argument("--turn_mix", type=str, default=DEFAULT_TURN_MIX)
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=0,
        help="If >0, sample this many mixed turn examples for training.",
    )
    parser.add_argument(
        "--max_examples_per_type",
        type=int,
        default=0,
        help="If >0, cap sampled examples for each turn type.",
    )
    parser.add_argument(
        "--oversample_long_n",
        type=float,
        default=2.0,
        help="Sampling weight multiplier for examples with n >= --long_n_min.",
    )
    parser.add_argument("--long_n_min", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="geo-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        try:
            import wandb  # noqa: F401

            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            if args.wandb_run_name:
                os.environ["WANDB_NAME"] = args.wandb_run_name
        except ImportError:
            print("WARNING: wandb not installed, disabling tracking.")
            use_wandb = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args)
    model = load_model(args)

    records = load_jsonl(args.train_file)
    sampled_records = weighted_sample_turns(
        records,
        turn_mix=args.turn_mix,
        max_train_examples=args.max_train_examples,
        max_examples_per_type=args.max_examples_per_type,
        oversample_long_n=args.oversample_long_n,
        long_n_min=args.long_n_min,
        seed=args.seed,
    )
    dataset_rows = render_dataset_rows(sampled_records, tokenizer)
    dataset = Dataset.from_list(dataset_rows)

    peft_config = None
    if args.base_model is None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )

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
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=make_reward_fn(),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"Live GRPO training on {len(dataset)} turn prompts")
    print(f"Loaded expanded rows: {len(records)}")
    print(f"Model: {args.model_name}")
    if args.base_model:
        print(f"Base model: {args.base_model}")
    print(f"Turn mix: {args.turn_mix}")
    print(f"Generations per prompt: {args.num_generations}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Live GRPO training complete.")


if __name__ == "__main__":
    main()
