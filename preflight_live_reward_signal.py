#!/usr/bin/env python3
"""Preflight reward variation for turn-level live-RL examples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from live_reward import live_turn_reward_components


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


def sample_records(
    records: list[dict[str, Any]],
    *,
    turn_mix: str,
    max_samples: int,
    oversample_long_n: float,
    long_n_min: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(records):
        pool = list(records)
    else:
        pool = records

    mix = parse_turn_mix(turn_mix)
    total = len(pool) if max_samples <= 0 else min(max_samples, len(pool))
    counts = target_counts(total, mix)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pool:
        turn_type = record.get("turn_type")
        if turn_type in mix:
            by_type[turn_type].append(record)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for turn_type, count in counts.items():
        group = by_type.get(turn_type, [])
        if not group:
            continue
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


def reward_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_type": record["turn_type"],
        "expected_tool_name": record.get("expected_tool_name") or "",
        "expected_arguments_json": json.dumps(
            record.get("expected_arguments"),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "expected_answer": record.get("expected_answer") or "",
        "meta_json": json.dumps(record.get("meta"), ensure_ascii=False, sort_keys=True),
    }


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=not args.no_trust_remote_code,
            padding_side="left",
            model_max_length=args.max_prompt_length,
        )
    except OSError:
        if not args.base_model:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=not args.no_trust_remote_code,
            padding_side="left",
            model_max_length=args.max_prompt_length,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    if args.base_model:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            trust_remote_code=not args.no_trust_remote_code,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            trust_remote_code=not args.no_trust_remote_code,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def get_model_input_device(model) -> torch.device | str:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def empty_stats() -> dict[str, Any]:
    return {
        "prompts": 0,
        "completions": 0,
        "mean_reward": 0.0,
        "nonzero_advantage_rate": 0.0,
        "parseable_tool_call_rate": 0.0,
        "tool_name_accuracy": 0.0,
        "argument_schema_accuracy": 0.0,
        "argument_value_accuracy": 0.0,
        "single_call_rate": 0.0,
        "premature_final_answer_rate": 0.0,
        "final_answer_parse_rate": 0.0,
        "final_answer_accuracy": 0.0,
        "final_no_tool_call_rate": 0.0,
    }


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    summary = empty_stats()
    summary["per_turn_type"] = {turn_type: empty_stats() for turn_type in TURN_TYPES}
    if not groups:
        return summary

    def add_group(acc: dict[str, Any], group: dict[str, Any]) -> None:
        components = group["components"]
        rewards = [float(c["total"]) for c in components]
        acc["prompts"] += 1
        acc["completions"] += len(components)
        acc["_reward_sum"] = acc.get("_reward_sum", 0.0) + sum(rewards)
        acc["_advantage_groups"] = acc.get("_advantage_groups", 0) + int(
            max(rewards) > min(rewards)
        )
        acc["_parseable_tool_calls"] = acc.get("_parseable_tool_calls", 0) + sum(
            c.get("format_reward", 0.0) > 0 for c in components
        )
        acc["_tool_name_correct"] = acc.get("_tool_name_correct", 0) + sum(
            c.get("tool_name_reward", 0.0) > 0 for c in components
        )
        acc["_argument_schema_correct"] = acc.get("_argument_schema_correct", 0) + sum(
            c.get("argument_schema_reward", 0.0) > 0 for c in components
        )
        acc["_argument_value_correct"] = acc.get("_argument_value_correct", 0) + sum(
            c.get("argument_value_reward", 0.0) > 0 for c in components
        )
        acc["_single_call"] = acc.get("_single_call", 0) + sum(
            c.get("single_call_reward", 0.0) > 0 for c in components
        )
        acc["_premature_final"] = acc.get("_premature_final", 0) + sum(
            bool(c.get("predicted_answer")) and group["turn_type"] != "final"
            for c in components
        )
        acc["_final_parseable"] = acc.get("_final_parseable", 0) + sum(
            c.get("parseable_answer_reward", 0.0) > 0 for c in components
        )
        acc["_final_correct"] = acc.get("_final_correct", 0) + sum(
            c.get("correctness_reward", 0.0) > 0 for c in components
        )
        acc["_final_no_tool"] = acc.get("_final_no_tool", 0) + sum(
            c.get("no_tool_call_reward", 0.0) > 0 for c in components
        )

    for group in groups:
        add_group(summary, group)
        add_group(summary["per_turn_type"][group["turn_type"]], group)

    def finalize(acc: dict[str, Any]) -> None:
        completions = acc["completions"]
        prompts = acc["prompts"]
        if completions == 0:
            return
        acc["mean_reward"] = acc.get("_reward_sum", 0.0) / completions
        acc["nonzero_advantage_rate"] = acc.get("_advantage_groups", 0) / prompts
        acc["parseable_tool_call_rate"] = acc.get("_parseable_tool_calls", 0) / completions
        acc["tool_name_accuracy"] = acc.get("_tool_name_correct", 0) / completions
        acc["argument_schema_accuracy"] = (
            acc.get("_argument_schema_correct", 0) / completions
        )
        acc["argument_value_accuracy"] = (
            acc.get("_argument_value_correct", 0) / completions
        )
        acc["single_call_rate"] = acc.get("_single_call", 0) / completions
        acc["premature_final_answer_rate"] = (
            acc.get("_premature_final", 0) / completions
        )
        acc["final_answer_parse_rate"] = acc.get("_final_parseable", 0) / completions
        acc["final_answer_accuracy"] = acc.get("_final_correct", 0) / completions
        acc["final_no_tool_call_rate"] = acc.get("_final_no_tool", 0) / completions
        for key in list(acc):
            if key.startswith("_"):
                del acc[key]

    finalize(summary)
    for stats in summary["per_turn_type"].values():
        finalize(stats)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\nLive reward-signal preflight")
    print("=" * 72)
    print(f"Prompts:                       {summary['prompts']}")
    print(f"Completions:                   {summary['completions']}")
    print(f"Mean reward:                   {summary['mean_reward']:.3f}")
    print(f"Nonzero advantage rate:        {summary['nonzero_advantage_rate']:.1%}")
    print(f"Parseable tool-call rate:      {summary['parseable_tool_call_rate']:.1%}")
    print(f"Tool-name accuracy:            {summary['tool_name_accuracy']:.1%}")
    print(f"Argument-schema accuracy:      {summary['argument_schema_accuracy']:.1%}")
    print(f"Argument-value accuracy:       {summary['argument_value_accuracy']:.1%}")
    print(f"Single-call rate:              {summary['single_call_rate']:.1%}")
    print(f"Premature final-answer rate:   {summary['premature_final_answer_rate']:.1%}")
    print(f"Final-answer parse rate:       {summary['final_answer_parse_rate']:.1%}")
    print(f"Final-answer accuracy:         {summary['final_answer_accuracy']:.1%}")
    print(f"Final no-tool-call rate:       {summary['final_no_tool_call_rate']:.1%}")
    print("\nPer turn type")
    for turn_type in TURN_TYPES:
        stats = summary["per_turn_type"][turn_type]
        print(
            f"  {turn_type:12s} prompts={stats['prompts']:4d} "
            f"mean_reward={stats['mean_reward']:.3f} "
            f"nonzero_adv={stats['nonzero_advantage_rate']:.1%} "
            f"arg_value={stats['argument_value_accuracy']:.1%} "
            f"final_acc={stats['final_answer_accuracy']:.1%}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base model path when --model_name points to a LoRA adapter.",
    )
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--turn_mix", default=DEFAULT_TURN_MIX)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--oversample_long_n", type=float, default=2.0)
    parser.add_argument("--long_n_min", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--save_completions", default=None)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    records = sample_records(
        load_jsonl(args.train_file),
        turn_mix=args.turn_mix,
        max_samples=args.max_samples,
        oversample_long_n=args.oversample_long_n,
        long_n_min=args.long_n_min,
        seed=args.seed,
    )
    print(
        f"Preflighting {len(records)} live-turn prompts, "
        f"generations={args.num_generations}, turn_mix={args.turn_mix}"
    )
    print(f"Sampled turn types: {dict(Counter(r['turn_type'] for r in records))}")

    groups: list[dict[str, Any]] = []
    saved_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(records), args.batch_size), desc="Generating"):
        batch = records[start : start + args.batch_size]
        prompts = [
            apply_chat_template_with_tools(
                record["prompt_messages"],
                record.get("tools"),
                tokenizer,
            )
            for record in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(get_model_input_device(model))

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "num_return_sequences": args.num_generations,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            tokenizer.decode(output[prompt_len:], skip_special_tokens=False)
            for output in outputs
        ]

        for offset, record in enumerate(batch):
            row = reward_row(record)
            group_completions = completions[
                offset * args.num_generations : (offset + 1) * args.num_generations
            ]
            components = [
                live_turn_reward_components(completion, row)
                for completion in group_completions
            ]
            groups.append(
                {
                    "source_question_id": record.get("source_question_id"),
                    "turn_type": record["turn_type"],
                    "n_points": record.get("n_points"),
                    "components": components,
                }
            )
            if args.save_completions:
                for generation_idx, (completion, component) in enumerate(
                    zip(group_completions, components)
                ):
                    saved_rows.append(
                        {
                            "source_question_id": record.get("source_question_id"),
                            "turn_type": record["turn_type"],
                            "n_points": record.get("n_points"),
                            "generation_idx": generation_idx,
                            "reward": component["total"],
                            "components": component,
                            "completion": completion,
                        }
                    )

    summary = summarize(groups)
    print_summary(summary)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"args": vars(args), "summary": summary}, f, indent=2)
        print(f"Wrote metrics to {args.output_json}")

    if args.save_completions:
        Path(args.save_completions).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_completions, "w") as f:
            for row in saved_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote completions to {args.save_completions}")


if __name__ == "__main__":
    main()
