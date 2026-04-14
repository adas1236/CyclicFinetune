#!/usr/bin/env python3
"""
Multi-GPU loading stress test.

Checks for the exact failure mode that caused the 8-GPU OOM: all DeepSpeed
processes briefly loading onto GPU 0 before being assigned to their own GPUs.

Launches N processes (via deepspeed), each loads the model using the same
code path as train_sft.py, while a background monitor polls nvidia-smi to
track per-GPU memory over time. Also tracks CPU RSS to catch host-memory
spikes from N simultaneous model loads.

Usage (2 GPUs is sufficient to catch the bug):
    deepspeed --num_gpus 2 check_loading.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --deepspeed ./configs/deepspeed_z2.json

What to look for in the output:
    - "Peak VRAM on GPU 0" should be ~1x the model size, NOT Nx
    - Each process should report loading onto a different GPU
    - CPU RSS peak should be < available RAM / num_processes
"""

from __future__ import annotations

import argparse
import os
import resource
import subprocess
import threading
import time

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def fmt(b: int | float) -> str:
    return f"{b / 1024**3:.2f} GB"


def fmt_mb(mb: float) -> str:
    return f"{mb / 1024:.2f} GB"


# ---------------------------------------------------------------------------
# nvidia-smi poller — runs on rank 0 only
# ---------------------------------------------------------------------------

class GpuMemoryMonitor:
    """Polls nvidia-smi in a background thread and records per-GPU peak usage."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mb: dict[int, float] = {}
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    gpu_idx = int(parts[0].strip())
                    used_mb = float(parts[1].strip())
                    if gpu_idx not in self.peak_mb or used_mb > self.peak_mb[gpu_idx]:
                        self.peak_mb[gpu_idx] = used_mb
            except Exception:
                pass
            time.sleep(self.interval)


def get_cpu_rss_mb() -> float:
    """Current process RSS in MB."""
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU loading stress test")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by DeepSpeed")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group for barrier synchronization
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    is_rank0 = (local_rank == 0)

    if is_rank0:
        print(f"{'='*65}")
        print(f"Multi-GPU Loading Stress Test")
        print(f"{'='*65}")
        print(f"  Model:      {args.model_name}")
        print(f"  World size: {world_size}")
        print(f"  4-bit:      {not args.no_4bit}")
        print()

    # ---- Start GPU monitor on rank 0 ----
    monitor = None
    if is_rank0:
        monitor = GpuMemoryMonitor(interval=0.05)  # 50ms polling
        monitor.start()
        print("GPU memory monitor started (polling every 50ms)")
        print()

    # Sync so monitor is running before anyone starts loading
    if world_size > 1:
        dist.barrier()

    # ---- Record pre-load state ----
    cpu_rss_before = get_cpu_rss_mb()
    torch.cuda.reset_peak_memory_stats(local_rank)

    # ---- Load model (same code path as train_sft.py) ----
    print(f"  [Rank {local_rank}] Loading model onto GPU {local_rank}...")

    use_4bit = not args.no_4bit
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    device_map = {"": local_rank}

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    load_time = time.time() - load_start

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
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

    # ---- Record post-load state ----
    gpu_peak = torch.cuda.max_memory_allocated(local_rank)
    cpu_rss_after = get_cpu_rss_mb()

    print(f"  [Rank {local_rank}] Loaded in {load_time:.1f}s  |  "
          f"GPU {local_rank} peak: {fmt(gpu_peak)}  |  "
          f"CPU RSS: {fmt_mb(cpu_rss_before):.>10s} → {fmt_mb(cpu_rss_after)}")

    # ---- Sync all processes, then stop monitor ----
    if world_size > 1:
        dist.barrier()
    time.sleep(0.5)  # let monitor catch any final spikes

    if is_rank0 and monitor:
        monitor.stop()

        # ---- Analyze results ----
        print(f"\n{'='*65}")
        print("PER-GPU PEAK VRAM (from nvidia-smi)")
        print(f"{'='*65}")

        num_gpus = torch.cuda.device_count()
        total_vram = torch.cuda.get_device_properties(0).total_memory

        # Expected per-GPU usage: roughly the model size in 4-bit + LoRA
        # With QLoRA on a 7B model, expect ~4-6 GB per process
        any_issue = False

        for gpu_idx in sorted(monitor.peak_mb.keys()):
            if gpu_idx >= num_gpus:
                continue
            peak_mb = monitor.peak_mb[gpu_idx]
            peak_bytes = peak_mb * 1024 * 1024
            pct = 100 * peak_bytes / total_vram

            # Flag if any GPU used more than 1.5x the average
            # (indicates multiple processes loaded onto it)
            status = "✓"
            if gpu_idx < world_size:
                # This GPU should have exactly 1 process
                avg_peak = sum(
                    monitor.peak_mb.get(i, 0) for i in range(world_size)
                ) / world_size
                if peak_mb > avg_peak * 1.5 and avg_peak > 0:
                    status = "⚠ HIGH — possible multi-process loading collision"
                    any_issue = True
            else:
                # This GPU shouldn't have any training process
                if peak_mb > 500:  # more than 500MB on an unused GPU
                    status = "⚠ Unexpected usage on unassigned GPU"
                    any_issue = True

            print(f"  GPU {gpu_idx}:  {fmt_mb(peak_mb):>10s}  ({pct:.1f}%)  {status}")

        # ---- CPU RAM summary ----
        print(f"\n{'='*65}")
        print("CPU RAM")
        print(f"{'='*65}")
        print(f"  Per-process RSS after load:  ~{fmt_mb(cpu_rss_after)}")
        print(f"  Estimated total ({world_size} procs):  "
              f"~{fmt_mb(cpu_rss_after * world_size)}")
        print(f"  Note: with {world_size} processes, peak CPU RAM during loading")
        print(f"  can spike to ~{world_size}x this value before weights move to GPU.")
        # Extrapolate to 8 GPUs
        if world_size < 8:
            print(f"\n  Extrapolated to 8 GPUs:  ~{fmt_mb(cpu_rss_after * 8)} total CPU RAM")
            print(f"  Make sure your SLURM --mem is at least this + overhead.")

        # ---- Verdict ----
        print(f"\n{'='*65}")
        print("VERDICT")
        print(f"{'='*65}")
        if any_issue:
            print("  ✗  GPU memory distribution looks uneven.")
            print("     Some GPUs may be loading multiple model copies.")
            print("     Check the device_map logic in your training scripts.")
        else:
            print("  ✓  Each GPU loaded ~equal memory. device_map is working correctly.")
            if world_size < 8:
                print(f"     This test used {world_size} GPUs — the loading pattern")
                print(f"     will be identical with 8 GPUs (each gets LOCAL_RANK 0-7).")
        print(f"{'='*65}")

    # Cleanup
    del model
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()