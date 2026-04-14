#!/usr/bin/env python3
"""
Evaluate a fine-tuned model on the geographic cyclic-ordering task.

Feeds the model the prompt (everything up to the final assistant turn),
generates a completion, and checks whether the answer is correct.

Usage:
    python evaluate.py \
        --model_name ./checkpoints/rl \
        --test_file ./data/val.jsonl \
        --pipeline 1
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from reward import correctness_reward, extract_answer, compute_ground_truth


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(record: dict, tokenizer: AutoTokenizer) -> str:
    messages = record["messages"][:-1]  # Everything except final assistant turn
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
            messages_copy[0]["content"] += "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
        return tokenizer.apply_chat_template(
            messages_copy, tokenize=False, add_generation_prompt=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model if model_name is a LoRA adapter")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--pipeline", type=int, default=None,
                        help="Filter to a specific pipeline (1 or 2)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy decoding")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if args.base_model:
        # Load base + adapter
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    model.eval()

    # ---- Load data ----
    records = load_jsonl(args.test_file)
    if args.pipeline:
        records = [r for r in records if r.get("pipeline") == args.pipeline]
    print(f"Evaluating on {len(records)} examples (pipeline={args.pipeline or 'all'})")

    # ---- Evaluate ----
    correct = 0
    total = 0
    parse_failures = 0

    for i in range(0, len(records), args.batch_size):
        batch = records[i : i + args.batch_size]
        prompts = [build_prompt(r, tokenizer) for r in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature

            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode only the generated part
        for j, (output, record) in enumerate(zip(outputs, batch)):
            prompt_len = inputs["input_ids"][j].shape[0]
            completion = tokenizer.decode(
                output[prompt_len:], skip_special_tokens=True
            )

            predicted = extract_answer(completion)
            ground_truth = compute_ground_truth(record["meta"])

            if predicted is None:
                parse_failures += 1
            elif predicted == ground_truth:
                correct += 1

            total += 1

        if (i // args.batch_size + 1) % 10 == 0:
            print(f"  [{i + len(batch)}/{len(records)}] "
                  f"Accuracy: {correct}/{total} = {correct / total:.1%}")

    # ---- Results ----
    print("\n" + "=" * 50)
    print(f"Results (pipeline={args.pipeline or 'all'}):")
    print(f"  Total:          {total}")
    print(f"  Correct:        {correct}")
    print(f"  Accuracy:       {correct / total:.1%}" if total > 0 else "  Accuracy: N/A")
    print(f"  Parse failures: {parse_failures}")
    print("=" * 50)


if __name__ == "__main__":
    main()