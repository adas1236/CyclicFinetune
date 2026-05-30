#!/usr/bin/env python3
"""
Score canonical final answers without generation.

For each prompt, this script appends a short candidate completion for each
label and computes the average token log probability under the model. The
highest-scoring label is treated as the model's final-answer preference.

This is a fast diagnostic for SFT checkpoints and SFT learning-rate sweeps. It
does not test tool-call generation; it tests whether the model can choose the
right final label once the prompt already contains the tool results.
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
import torch.nn.functional as F
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward import compute_ground_truth


LABELS = ("clockwise", "counterclockwise", "neither")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def candidate_logprob(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidate: str,
    max_prompt_length: int,
) -> float:
    candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
    if not candidate_ids:
        return float("-inf")
    prompt_budget = max(1, max_prompt_length - len(candidate_ids))
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_budget,
    )["input_ids"]

    input_ids = torch.tensor(
        [prompt_ids + candidate_ids],
        dtype=torch.long,
        device=model.device,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]

    start = len(prompt_ids)
    token_logprobs = []
    for pos, token_id in enumerate(candidate_ids, start=start):
        token_logits = logits[pos - 1]
        token_logprobs.append(F.log_softmax(token_logits.float(), dim=-1)[token_id])

    return torch.stack(token_logprobs).mean().item()


def print_summary(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    correct = sum(row["predicted"] == row["ground_truth"] for row in rows)
    pred_dist = Counter(row["predicted"] for row in rows)

    print("\nOracle final-answer score")
    print("=" * 72)
    print(f"Prompts:             {total}")
    print(f"Accuracy:            {correct}/{total} = {correct / total:.1%}" if total else "Accuracy: n/a")
    print(f"Predicted labels:    {dict(pred_dist)}")
    print("Per class:")
    for label in LABELS:
        class_rows = [row for row in rows if row["ground_truth"] == label]
        class_correct = sum(row["predicted"] == label for row in class_rows)
        if class_rows:
            print(
                f"  {label:18s} {class_correct}/{len(class_rows)} = "
                f"{class_correct / len(class_rows):.1%}"
            )
        else:
            print(f"  {label:18s} 0/0")
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
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--balanced_per_class", type=int, default=0)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument(
        "--candidate_template",
        default="Final answer: {answer}",
        help="Short completion template used for each candidate label.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_jsonl", default=None)
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
    print(f"Scoring {len(records)} prompts, pipeline={args.pipeline or 'all'}")

    rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Scoring"):
        prompt = build_prompt(record, tokenizer)
        scores = {
            label: candidate_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidate=args.candidate_template.format(answer=label),
                max_prompt_length=args.max_prompt_length,
            )
            for label in LABELS
        }
        predicted = max(scores.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "question_id": record.get("question_id"),
                "pipeline": record.get("pipeline"),
                "ground_truth": compute_ground_truth(record["meta"]),
                "predicted": predicted,
                "scores": scores,
            }
        )

    print_summary(rows)

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote per-example scores to {args.output_jsonl}")


if __name__ == "__main__":
    main()
