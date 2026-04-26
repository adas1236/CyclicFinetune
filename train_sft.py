#!/usr/bin/env python3
"""
Supervised fine-tuning for the geographic cyclic-ordering task.

Uses TRL's SFTTrainer with:
  - QLoRA (4-bit quantized base + LoRA adapters) to fit in 24GB VRAM
  - DeepSpeed ZeRO-2 for multi-GPU data parallelism
  - The model's native chat template for tool-calling format

Launch with:
    deepspeed --num_gpus N train_sft.py [args...]
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import is_flash_attn_2_available
from trl import SFTConfig, SFTTrainer

from reward import extract_answer, compute_ground_truth


class ValAccuracyCallback(TrainerCallback):
    """
    At each eval step, run greedy generation on a small subset of
    the validation set and log accuracy to wandb.
    """

    def __init__(self, val_records: list[dict], tokenizer, max_samples: int = 50):
        self.val_records = val_records[:max_samples]
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.val_records or model is None:
            return

        correct = 0
        total = 0

        model.eval()
        for rec in self.val_records:
            # Build prompt from all messages except the last assistant turn
            messages = rec["messages"][:-1]
            tools = rec.get("tools", None)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                msgs = [dict(m) for m in messages]
                if tools and msgs and msgs[0]["role"] == "system":
                    msgs[0]["content"] += "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
                prompt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=512,
                    do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                )
            completion = self.tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )

            predicted = extract_answer(completion)
            gt = compute_ground_truth(rec["meta"])
            if predicted == gt:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0

        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"val/accuracy": acc, "global_step": state.global_step})
        except ImportError:
            pass

        print(f"  [ValAccuracy] step={state.global_step}  acc={correct}/{total} = {acc:.1%}")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_conversation(record: dict, tokenizer: AutoTokenizer) -> str:
    """
    Apply the model's chat template to the conversation messages.
    This handles tool-call formatting automatically for each model family.
    """
    messages = record["messages"]
    tools = record.get("tools", None)

    try:
        # Try with tools parameter (Qwen 2.5, Llama 3.1 support this)
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
    except TypeError:
        # Fallback: model's template doesn't accept 'tools' kwarg.
        # Inject tool descriptions into the system message instead.
        messages_copy = [dict(m) for m in messages]
        if tools and messages_copy and messages_copy[0]["role"] == "system":
            tool_desc = "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
            messages_copy[0]["content"] += tool_desc
        text = tokenizer.apply_chat_template(
            messages_copy,
            tokenize=False,
            add_generation_prompt=False,
        )

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use num_epochs instead). Useful for quick tests.")
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", default=False,
                        help="Disable 4-bit quantization (use full precision)")
    parser.add_argument("--wandb_project", type=str, default="geo-finetune",
                        help="Weights & Biases project name (set to '' to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by DeepSpeed")
    args = parser.parse_args()

    use_4bit = args.use_4bit and not args.no_4bit
    use_wandb = bool(args.wandb_project)

    if use_wandb:
        try:
            import wandb
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            if args.wandb_run_name:
                os.environ["WANDB_NAME"] = args.wandb_run_name
        except ImportError:
            print("WARNING: wandb not installed, disabling tracking. "
                  "Install with: uv add wandb")
            use_wandb = False

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=args.max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Quantization config ----
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ---- Model ----
    # With DeepSpeed, each process must load onto its own GPU.
    # LOCAL_RANK is set by the DeepSpeed launcher.
    if args.deepspeed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        device_map=device_map,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ---- LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # ---- Data ----
    train_records = load_jsonl(args.train_file)
    train_texts = [format_conversation(r, tokenizer) for r in train_records]
    train_dataset = Dataset.from_dict({"text": train_texts})

    eval_dataset = None
    if args.val_file:
        val_records = load_jsonl(args.val_file)
        val_texts = [format_conversation(r, tokenizer) for r in val_records]
        eval_dataset = Dataset.from_dict({"text": val_texts})

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=42,
    )

    # ---- Callbacks ----
    callbacks = []
    if args.val_file and eval_dataset is not None:
        val_records_raw = load_jsonl(args.val_file)
        callbacks.append(ValAccuracyCallback(val_records_raw, tokenizer, max_samples=50))

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        callbacks=callbacks,
    )

    # ---- Train ----
    print(f"Training on {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Evaluating on {len(eval_dataset)} examples")
    print(f"Model: {args.model_name}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"4-bit quantization: {use_4bit}")

    trainer.train()

    # ---- Save ----
    # Save the LoRA adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save a merged model for easier loading in the RL stage
    merged_dir = args.output_dir + "-merged"
    print(f"Merging LoRA adapter and saving to {merged_dir}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("SFT training complete!")


if __name__ == "__main__":
    main()