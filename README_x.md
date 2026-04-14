# Geographic Cyclic Ordering — Fine-Tuning Pipeline

## Architectural Recommendations

### Why 2 models, not 3

You proposed fine-tuning 3 models (placename extractor, internal-computation answerer,
tool-calling answerer). I recommend **2 fine-tuned models** — one per pipeline — with
placename extraction folded into each model's first turn rather than split out.

Rationale:

1. **Placename extraction is trivial for a 7B+ model.** The names are already in the
   question text; the model just needs to learn to emit a `geocode(...)` tool call
   containing them. A separate NER model adds inference latency and an extra failure
   mode for no measurable gain.

2. **Context matters.** The model that will eventually answer the question benefits from
   seeing the original question when deciding *which* locations to geocode and *how* to
   phrase the answer. Splitting extraction from answering severs that context.

3. **Modern open-weight models already handle multi-step tool calling.** Qwen 2.5 and
   Llama 3.1 both ship with chat templates that support tool-call / tool-result turns
   natively. A single model can extract → call geocode → (optionally) call compute →
   answer, all within one conversation.

So the two models are:

| Model | Pipeline | What it learns |
|-------|----------|---------------|
| **A** | 1 — internal math | Extract names → geocode tool → read coordinates → compute sign(det) internally → answer |
| **B** | 2 — tool-assisted | Extract names → geocode tool → read coordinates → call `cyclic_order` tool → relay answer |

Both start from the same base checkpoint (e.g. `Qwen/Qwen2.5-7B-Instruct`).

> **Alternative:** fine-tune a *single* model on a mix of both pipeline formats. At
> inference time, you control which pipeline runs by including or excluding the
> `cyclic_order` tool in the system prompt. This is the simplest option and is what
> `prepare_data.py` supports via the `--pipeline both` flag.

### On SFT vs RL

You're right that SFT alone may not generalize well. The good news is that cyclic
ordering has a **perfectly verifiable reward signal**: given three points, compute
`sign(det(B−A, C−A))` and check whether the model's answer matches. This makes the
problem an ideal candidate for **GRPO** (Group Relative Policy Optimization) — the
same algorithm DeepSeek used for DeepSeek-R1.

The pipeline this repo implements:

```
SFT (warm-start) ──► GRPO (outcome-based RL)
```

SFT teaches the model the *format* (tool calling, answer phrasing). GRPO then
optimizes for *correctness* using the deterministic reward. This sidesteps the need
for hand-written reasoning traces.

### How tool calling works in this pipeline

Tool calling in modern instruct-tuned models is not special infrastructure — it's a
**text format convention** baked into the model's chat template. Here's what actually
happens:

1. **System prompt declares available tools** as JSON schemas (defined in `tools.py`).
   The model's tokenizer `apply_chat_template(..., tools=...)` injects these into the
   right place for that model family.

2. **The model generates a structured tool-call token sequence** instead of plain text.
   For Qwen 2.5, this looks like `<tool_call>{"name": "geocode", "arguments": ...}</tool_call>`.
   For Llama 3.1, it uses `<|python_tag|>` blocks. The format differs, but `apply_chat_template`
   handles it — we write the conversation in a model-agnostic dict format and the
   tokenizer renders it into the right tokens.

3. **During SFT training**, we provide the *complete* conversation including the
   tool-call turns and tool-result turns. The model learns to produce tool calls at
   the right moments by imitating these traces. There is no actual tool execution
   during training — the tool responses are pre-filled from your ground-truth data.

4. **During inference**, you need a thin orchestration loop:
   - Send the user question to the model
   - If the model outputs a tool-call, parse it, execute the real tool (e.g. call
     a geocoding API), and feed the result back as a tool-result message
   - Let the model continue generating
   
   Libraries like `transformers` and `vllm` have built-in support for this loop.

So in short: the training data in `prepare_data.py` contains simulated tool calls
and results. SFT teaches the model *when and how* to emit tool calls. At inference
time, you wire up the actual tools.

### How much data do you need?

For a problem this constrained (binary classification with a fixed reasoning
structure), a few thousand samples is a reasonable starting point, but the two
training stages have different data appetites:

- **SFT** mostly needs to learn the *format* — tool calling syntax, answer phrasing,
  reasoning trace structure. For that, **1,000–2,000 examples** is usually sufficient.
  The model already knows how to do tool calling; you're just teaching it the
  specific tools and domain.

- **GRPO** benefits from more diversity because it explores the policy space. Having
  **3,000–5,000+ prompts** gives the algorithm enough variety to avoid overfitting
  to a narrow distribution of geographic configurations. The RL stage can reuse the
  same prompts repeatedly (multiple epochs or re-sampling), so the raw count matters
  less than the diversity of spatial configurations.

Where generalization gets tricky is the *geometry*: if all your training points are
in one geographic region or follow similar spatial patterns (e.g. always roughly
collinear), the model may struggle with very different configurations. Make sure your
dataset has good coverage of angular separations, scales, and hemispheres.

For a concrete starting recommendation: **2,000–3,000 unique questions** should get
you to high accuracy on held-out data from the same distribution. Scale up if you want
out-of-distribution robustness.

### Model selection

With RTX A5000s (24 GB each), good candidates are:

| Model | Params | Why |
|-------|--------|-----|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Best-in-class tool calling among open 7B models |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Strong general reasoning, good community support |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Solid baseline, function-calling support |

All fit comfortably with QLoRA on a single A5000. Multi-GPU is used via DeepSpeed
ZeRO-2 to **distribute the training** (data parallelism), not shard the model — this
is more efficient for 7B-scale models.

If you want to try 14B+ models (e.g. `Qwen/Qwen2.5-14B-Instruct`), switch to
ZeRO-3 (config provided) which shards the model across GPUs.

---

## Dataset Format

Your parquet file must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | `int` | Unique identifier |
| `question` | `str` | The natural-language question |
| `location_names` | `list[str]` | Place names referenced in the question |
| `geometries` | `list[dict]` | Each dict has `"type"` (str) and `"coordinates"` (list of [x, y] pairs) |
| `answer` | `str` | `"clockwise"` or `"counterclockwise"` |
| `roles` | `dict` | **Required.** Maps role labels to indices into `location_names`. Keys: `"center"`, `"b"`, `"c"`. Values: int indices or string place names. |

Example `roles` value: `{"center": 0, "b": 1, "c": 2}` means `location_names[0]`
is the center point, `location_names[1]` is B, `location_names[2]` is C.

You can also use place name strings: `{"center": "Denver", "b": "Phoenix", "c": "Seattle"}`.

---

## Project Structure

```
geo-finetune/
├── README.md                  ← you are here
├── prepare_data.py            ← parquet → JSONL training data
├── train_sft.py               ← supervised fine-tuning with TRL
├── train_rl.py                ← GRPO reinforcement learning
├── reward.py                  ← deterministic reward function
├── tools.py                   ← tool schemas & implementations
├── evaluate.py                ← evaluation script
├── configs/
│   ├── deepspeed_z2.json      ← ZeRO Stage 2 (data parallelism)
│   └── deepspeed_z3.json      ← ZeRO Stage 3 (model sharding, for 14B+)
└── run_training.sh            ← end-to-end launcher
```

## Setup (using uv)

```bash
# 1. Create a new project and virtual environment
uv init geo-finetune && cd geo-finetune
#    (or cd into your existing project directory)

# 2. Add Python dependencies
uv add torch transformers trl peft datasets accelerate \
       deepspeed bitsandbytes pandas pyarrow

# 3. Optional: tracking & flash attention
uv add wandb                        # for loss/accuracy tracking
uv add flash-attn --no-build-isolation  # for faster attention (needs CUDA toolkit)

# 4. Copy the scripts into your project (or clone this directory)
#    Make sure prepare_data.py, train_sft.py, etc. are in the project root.
```

All subsequent commands assume you're running inside the `uv` virtualenv.
If you use `uv run` as a prefix, it activates the env automatically:

```bash
uv run python prepare_data.py --input data.parquet --output ./data --pipeline both
uv run deepspeed --num_gpus 4 train_sft.py ...
```

Or activate the venv once and run normally:

```bash
source .venv/bin/activate
python prepare_data.py ...
deepspeed --num_gpus 4 train_sft.py ...
```

### SLURM notes

On a SLURM cluster, you'll typically submit via `sbatch`. Here's a minimal job script:

```bash
#!/bin/bash
#SBATCH --job-name=geo-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

deepspeed --num_gpus $SLURM_GPUS_ON_NODE train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_file ./data/train.jsonl \
    --val_file ./data/val.jsonl \
    --output_dir ./checkpoints/sft \
    --deepspeed ./configs/deepspeed_z2.json \
    --wandb_project geo-finetune \
    --wandb_run_name sft-qwen7b
```

## Quick Start

```bash
# 1. Prepare data (adjust path to your parquet file)
python prepare_data.py \
    --input /path/to/data.parquet \
    --output ./data \
    --pipeline both \
    --val_fraction 0.1

# 2. SFT warm-start (uses all visible GPUs automatically)
deepspeed --num_gpus $(nvidia-smi -L | wc -l) train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_file ./data/train.jsonl \
    --val_file ./data/val.jsonl \
    --output_dir ./checkpoints/sft \
    --deepspeed ./configs/deepspeed_z2.json \
    --wandb_project geo-finetune

# 3. GRPO reinforcement learning
deepspeed --num_gpus $(nvidia-smi -L | wc -l) train_rl.py \
    --model_name ./checkpoints/sft-merged \
    --train_file ./data/train.jsonl \
    --output_dir ./checkpoints/rl \
    --deepspeed ./configs/deepspeed_z2.json \
    --wandb_project geo-finetune

# 4. Evaluate
python evaluate.py \
    --model_name ./checkpoints/rl \
    --test_file ./data/val.jsonl
```

Or just run:

```bash
bash run_training.sh --input /path/to/data.parquet --model Qwen/Qwen2.5-7B-Instruct --num_gpus 4
```

## Tracking with Weights & Biases

Both training scripts log to W&B by default (project name: `geo-finetune`). You'll see:

- **train/loss** — standard training loss (SFT: cross-entropy, GRPO: policy loss)
- **eval/loss** — validation loss (SFT only)
- **val/accuracy** — fraction of validation examples where the model's answer matches
  the ground truth (SFT only, computed by the `ValAccuracyCallback`)
- **reward/mean** — average reward across the generation group (GRPO only, logged
  automatically by `GRPOTrainer`)

To disable tracking, pass `--wandb_project ""`. First time setup: `wandb login`.

## Swapping Models

Change `--model_name` to any HuggingFace model ID. The pipeline uses the model's
own chat template and tokenizer, so format differences are handled automatically.
The only requirement is that the model's chat template supports tool/function calling
(most instruct-tuned models released after mid-2024 do).
