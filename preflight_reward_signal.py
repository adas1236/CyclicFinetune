#!/usr/bin/env python3
"""
Preflight the GRPO reward signal before spending a full training run.

This script samples multiple completions per prompt from a checkpoint, scores
them with the same answer parser used by GRPO/evaluation, and reports whether
GRPO would have useful within-group reward variation.

Important metrics:
  - correct_in_group_rate: fraction of prompts where at least one sampled
    completion is correct. If this is near zero for a class, GRPO has no
    useful signal to amplify for that class.
  - nonzero_advantage_rate: fraction of prompts whose sampled completions do
    not all receive the same reward. GRPO needs this to update from the group.
  - predicted_label_distribution: detects class collapse before training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward import compute_ground_truth, extract_answer


LABELS = ("clockwise", "counterclockwise", "neither")
PARSE_FAIL = "parse_fail"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(record: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages = record["messages"][:-1]
    tools = record.get("tools", None)
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
            messages_copy[0]["content"] += (
                "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
            )
        return tokenizer.apply_chat_template(
            messages_copy,
            tokenize=False,
            add_generation_prompt=True,
        )


def filter_and_sample(
    records: list[dict[str, Any]],
    pipeline: int | None,
    max_samples: int,
    balanced_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    if pipeline is not None:
        records = [r for r in records if r.get("pipeline") == pipeline]

    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)

    if balanced_per_class > 0:
        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_class[compute_ground_truth(record["meta"])].append(record)
        sampled: list[dict[str, Any]] = []
        for label in LABELS:
            sampled.extend(by_class[label][:balanced_per_class])
        rng.shuffle(sampled)
        return sampled

    if max_samples > 0:
        return records[:max_samples]
    return records


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
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


def empty_class_stats() -> dict[str, Any]:
    return {
        "prompts": 0,
        "completions": 0,
        "completion_parse_rate": 0.0,
        "completion_accuracy": 0.0,
        "mean_reward": 0.0,
        "correct_in_group_rate": 0.0,
        "nonzero_advantage_rate": 0.0,
        "predicted_label_distribution": {
            "clockwise": 0,
            "counterclockwise": 0,
            "neither": 0,
            "parse_fail": 0,
        },
    }


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "prompts": len(groups),
        "completions": 0,
        "completion_parse_rate": 0.0,
        "completion_accuracy": 0.0,
        "mean_reward": 0.0,
        "correct_in_group_rate": 0.0,
        "nonzero_advantage_rate": 0.0,
        "all_same_prediction_rate": 0.0,
        "predicted_label_distribution": {
            "clockwise": 0,
            "counterclockwise": 0,
            "neither": 0,
            "parse_fail": 0,
        },
        "per_class": {label: empty_class_stats() for label in LABELS},
    }
    if not groups:
        return summary

    counters = {
        "completions": 0,
        "parseable": 0,
        "correct": 0,
        "reward_sum": 0.0,
        "correct_groups": 0,
        "advantage_groups": 0,
        "same_prediction_groups": 0,
    }
    pred_dist: Counter[str] = Counter()

    per_class_raw: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        per_class_raw[label] = {
            "groups": 0,
            "completions": 0,
            "parseable": 0,
            "correct": 0,
            "reward_sum": 0.0,
            "correct_groups": 0,
            "advantage_groups": 0,
            "pred_dist": Counter(),
        }

    for group in groups:
        gt = group["ground_truth"]
        predictions = group["predictions"]
        rewards = group["rewards"]
        correct_flags = [p == gt for p in predictions]
        parseable_flags = [p is not None for p in predictions]
        labels = [p if p is not None else PARSE_FAIL for p in predictions]

        n = len(predictions)
        counters["completions"] += n
        counters["parseable"] += sum(parseable_flags)
        counters["correct"] += sum(correct_flags)
        counters["reward_sum"] += sum(rewards)
        counters["correct_groups"] += int(any(correct_flags))
        counters["advantage_groups"] += int(max(rewards) > min(rewards))
        counters["same_prediction_groups"] += int(len(set(labels)) == 1)
        pred_dist.update(labels)

        cls = per_class_raw[gt]
        cls["groups"] += 1
        cls["completions"] += n
        cls["parseable"] += sum(parseable_flags)
        cls["correct"] += sum(correct_flags)
        cls["reward_sum"] += sum(rewards)
        cls["correct_groups"] += int(any(correct_flags))
        cls["advantage_groups"] += int(max(rewards) > min(rewards))
        cls["pred_dist"].update(labels)

    completion_count = counters["completions"]
    prompt_count = len(groups)
    summary["completions"] = completion_count
    summary["completion_parse_rate"] = counters["parseable"] / completion_count
    summary["completion_accuracy"] = counters["correct"] / completion_count
    summary["mean_reward"] = counters["reward_sum"] / completion_count
    summary["correct_in_group_rate"] = counters["correct_groups"] / prompt_count
    summary["nonzero_advantage_rate"] = counters["advantage_groups"] / prompt_count
    summary["all_same_prediction_rate"] = counters["same_prediction_groups"] / prompt_count
    summary["predicted_label_distribution"] = {
        label: pred_dist[label] for label in (*LABELS, PARSE_FAIL)
    }

    for label, raw in per_class_raw.items():
        if raw["groups"] == 0:
            continue
        n_comp = raw["completions"]
        summary["per_class"][label] = {
            "prompts": raw["groups"],
            "completions": n_comp,
            "completion_parse_rate": raw["parseable"] / n_comp,
            "completion_accuracy": raw["correct"] / n_comp,
            "mean_reward": raw["reward_sum"] / n_comp,
            "correct_in_group_rate": raw["correct_groups"] / raw["groups"],
            "nonzero_advantage_rate": raw["advantage_groups"] / raw["groups"],
            "predicted_label_distribution": {
                pred: raw["pred_dist"][pred] for pred in (*LABELS, PARSE_FAIL)
            },
        }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\nReward-signal preflight")
    print("=" * 72)
    print(f"Prompts:                  {summary['prompts']}")
    print(f"Completions:              {summary['completions']}")
    print(f"Completion parse rate:    {summary['completion_parse_rate']:.1%}")
    print(f"Completion accuracy:      {summary['completion_accuracy']:.1%}")
    print(f"Mean reward:              {summary['mean_reward']:.3f}")
    print(f"Correct in group rate:    {summary['correct_in_group_rate']:.1%}")
    print(f"Nonzero advantage rate:   {summary['nonzero_advantage_rate']:.1%}")
    print(f"All-same prediction rate: {summary['all_same_prediction_rate']:.1%}")
    print(f"Predicted labels:         {summary['predicted_label_distribution']}")
    print("\nPer class")
    for label in LABELS:
        stats = summary["per_class"][label]
        print(
            f"  {label:18s} prompts={stats['prompts']:4d} "
            f"acc={stats['completion_accuracy']:6.1%} "
            f"parse={stats['completion_parse_rate']:6.1%} "
            f"correct_in_group={stats['correct_in_group_rate']:6.1%} "
            f"nonzero_adv={stats['nonzero_advantage_rate']:6.1%} "
            f"preds={stats['predicted_label_distribution']}"
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
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--pipeline", type=int, choices=[1, 2], default=None)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument(
        "--balanced_per_class",
        type=int,
        default=0,
        help="If >0, sample up to this many prompts per ground-truth class.",
    )
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--save_completions", default=None)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--no_trust_remote_code", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    records = filter_and_sample(
        load_jsonl(args.test_file),
        pipeline=args.pipeline,
        max_samples=args.max_samples,
        balanced_per_class=args.balanced_per_class,
        seed=args.seed,
    )
    print(
        f"Preflighting {len(records)} prompts, pipeline={args.pipeline or 'all'}, "
        f"generations={args.num_generations}"
    )

    groups: list[dict[str, Any]] = []
    saved_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(records), args.batch_size), desc="Generating"):
        batch = records[start : start + args.batch_size]
        prompts = [build_prompt(record, tokenizer) for record in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            for output in outputs
        ]

        for offset, record in enumerate(batch):
            gt = compute_ground_truth(record["meta"])
            group_completions = completions[
                offset * args.num_generations : (offset + 1) * args.num_generations
            ]
            predictions = [extract_answer(c) for c in group_completions]
            rewards = [
                0.0 if pred is None else 0.25 + (1.0 if pred == gt else 0.0)
                for pred in predictions
            ]
            groups.append(
                {
                    "question_id": record.get("question_id"),
                    "pipeline": record.get("pipeline"),
                    "ground_truth": gt,
                    "predictions": predictions,
                    "rewards": rewards,
                }
            )
            if args.save_completions:
                for generation_idx, (completion, pred, reward) in enumerate(
                    zip(group_completions, predictions, rewards)
                ):
                    saved_rows.append(
                        {
                            "question_id": record.get("question_id"),
                            "pipeline": record.get("pipeline"),
                            "generation_idx": generation_idx,
                            "ground_truth": gt,
                            "predicted": pred,
                            "reward": reward,
                            "completion": completion,
                        }
                    )

    summary = summarize_groups(groups)
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
                f.write(json.dumps(row) + "\n")
        print(f"Wrote completions to {args.save_completions}")


if __name__ == "__main__":
    main()
