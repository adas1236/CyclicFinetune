# Geographic Cyclic Reasoning — Fine-Tuning Pipeline

A pipeline that fine-tunes an instruction-tuned LLM (via LoRA SFT followed by GRPO
reinforcement learning) to answer **geographic cyclic-ordering** questions using
**tool calls**.

## What the model learns to answer

Geographic cyclic-ordering questions take forms like:

- *With respect to a centroid in A, is moving from B to C clockwise or counterclockwise?*
- *From A, are B and C in a clockwise or counterclockwise order?*
- *If you're standing in A, are B and C arranged clockwise or counterclockwise around you?*

Each question has a **center** location and one or more **waypoints**. For `n`
locations there is 1 center + `n-1` waypoints and `n-2` consecutive arcs. Each arc
B→C around the center A is classified by the sign of the cross product
`det(B−A, C−A)` (planar) or the 3×3 determinant of the unit Earth vectors
(spherical). The overall answer is one of:

| Answer | Meaning |
|--------|---------|
| `clockwise` | every arc is clockwise |
| `counterclockwise` | every arc is counterclockwise |
| `neither` | the arcs disagree |

Because the answer is computed deterministically from coordinates, the task has a
**perfectly verifiable reward** — ideal for GRPO.

## Two inference pipelines

The model is trained to solve the task with tool calls. Two pipelines are supported,
selected at data-prep time (and filtered at eval time):

| Pipeline | Tools exposed | What the model does |
|----------|---------------|---------------------|
| **1 — internal math** | `geocode` | Calls `geocode`, reads coordinates, then computes the cross products / determinants *in its own reasoning* and answers. |
| **2 — tool-assisted** | `geocode`, `cyclic_order` | Calls `geocode`, then calls `cyclic_order` once per consecutive waypoint pair, and combines the per-arc results. |
| **both** | — | A mixed dataset containing pipeline-1 and pipeline-2 conversations (control which runs at inference by including/excluding `cyclic_order`). |

The two tools live in `tools.py`:

- **`geocode(place_names)`** → maps each place name to `{longitude, latitude}`.
  During training the result is filled from ground-truth geometries; at
  inference/interactive time it can call live OpenStreetMap **Nominatim** geocoding.
- **`cyclic_order(center, point_b, point_c)`** → `"clockwise"` / `"counterclockwise"`
  for one arc. Has both a planar backend and a spherical **Earth** backend.

> **Coordinate convention:** model-facing tool calls always use `[longitude, latitude]`.
> Raw synthetic geometries are stored as `[lon, lat]`; raw *Earth* geometries default
> to `[lat, lon]` and are normalized during data prep / eval.

---

## Setup

The project targets **Python 3.12** and uses [`uv`](https://docs.astral.sh/uv/) for
dependency management. **Do not use `pip`**, and **do not install `flash-attention`**
(the code falls back to PyTorch SDPA automatically).

```bash
# From the project root — installs everything pinned in uv.lock into .venv
uv sync
```

All dependencies (torch, transformers, trl, peft, datasets, accelerate, deepspeed,
bitsandbytes, gradio, pandas, polars, pyarrow, wandb) are declared in
`pyproject.toml`.

Run commands either by prefixing with `uv run`:

```bash
uv run python prepare_data.py --help
```

…or by activating the environment once:

```bash
source .venv/bin/activate
python prepare_data.py --help
```

GPU notes: training uses **QLoRA** (4-bit NF4 base + LoRA adapters) so a 7B model
fits on a single 24 GB GPU. Multi-GPU runs use **DeepSpeed ZeRO-2** for data
parallelism (`configs/deepspeed_z2.json`); switch to `configs/deepspeed_z3.json` to
shard a 14B+ model across GPUs.

---

## End-to-end workflow

```
generate_fake_data.py → prepare_data.py → train_sft.py → train_rl.py → evaluate.py
                                                       └→ (live tool-loop RL) → train_live_rl.py
                                                                            chat.py (interactive)
```

### 1. Generate synthetic data

Produces balanced and natural-distribution parquet files under `data/parquet/`,
drawing place names from an online world-cities list and question phrasings from
`question_formats.json`.

```bash
uv run python generate_fake_data.py
```

Outputs (default):
- `data/parquet/spatial_questions_train.parquet`
- `data/parquet/spatial_questions_val_balanced.parquet`
- `data/parquet/spatial_questions_val_natural.parquet`

Each row has: `question_id`, `question`, `location_names` (length `n`), `geometries`
(length `n`; index 0 is the center, the rest are ordered waypoints), and `answer`.
Each geometry is a `{"type": "point"|"line"|"polygon", "coordinates": [...]}` dict;
its representative point is the coordinate / midpoint / centroid.

### 2. Prepare training data (parquet → JSONL conversations)

`prepare_data.py` turns each parquet row into a multi-turn tool-calling conversation
and writes one JSONL per split into `<output>/<pipeline>/`. Run it once per pipeline
(`1`, `2`, or `both`):

```bash
uv run python prepare_data.py \
    --train_input        ./data/parquet/spatial_questions_train.parquet \
    --val_balanced_input ./data/parquet/spatial_questions_val_balanced.parquet \
    --val_natural_input  ./data/parquet/spatial_questions_val_natural.parquet \
    --output ./data/jsonl \
    --pipeline 2
```

This writes `./data/jsonl/2/train.jsonl`, `val_balanced.jsonl`, `val_natural.jsonl`.

To build a **spherical Earth** validation split (answers are regenerated with the
Earth backend), pass real geometries via `--val_earth_input` (default coordinate
order `latlon`):

```bash
uv run python prepare_data.py \
    --val_earth_input ./data/parquet/earth_cyclic_order.parquet \
    --output ./data/jsonl \
    --pipeline 2
# → ./data/jsonl/2/val_earth.jsonl
```

### 3. SFT warm-start

`train_sft.py` runs QLoRA supervised fine-tuning with TRL's `SFTTrainer`. It saves a
LoRA adapter to `--output_dir` **and** a merged model to `<output_dir>-merged` (handy
for the RL stage). A `ValAccuracyCallback` logs greedy-decode accuracy on a
validation subset to W&B.

```bash
deepspeed --num_gpus $(nvidia-smi -L | wc -l) train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_file ./data/jsonl/2/train.jsonl \
    --val_file   ./data/jsonl/2/val_balanced.jsonl \
    --output_dir ./checkpoints/sft-2 \
    --deepspeed  ./configs/deepspeed_z2.json \
    --num_epochs 3 \
    --lora_r 64 --lora_alpha 128 \
    --wandb_project geo-finetune --wandb_run_name sft-p2
```

Useful flags: `--max_steps N` (quick smoke test), `--no_4bit` (full precision),
`--per_device_batch_size`, `--gradient_accumulation_steps`, `--learning_rate`.

### 4. GRPO reinforcement learning

`train_rl.py` takes the SFT-warmed model and optimizes for *correctness* using the
deterministic reward (`reward.py`): the prompt is everything up to the final
assistant turn, the model generates the completion, and the reward is
`0.25 (parseable) + 1.0 (correct)`.

```bash
deepspeed --num_gpus $(nvidia-smi -L | wc -l) train_rl.py \
    --model_name ./checkpoints/sft-2-merged \
    --train_file ./data/jsonl/2/train.jsonl \
    --output_dir ./checkpoints/rl-2 \
    --deepspeed  ./configs/deepspeed_z2.json \
    --num_generations 4 \
    --lora_r 32 --lora_alpha 64 \
    --learning_rate 5e-6 \
    --wandb_project geo-finetune --wandb_run_name rl-p2
```

### 5. (Optional) Live tool-loop RL

For pipeline 2, `train_live_rl.py` does **turn-level** GRPO: it rewards each
individual tool turn (correct tool name, schema, argument values) and the final
answer turn separately (`live_reward.py`). First expand a prepared pipeline-2 JSONL
into per-turn examples with `prepare_live_rl_data.py`:

```bash
uv run python prepare_live_rl_data.py \
    --input  ./data/jsonl/2/train.jsonl \
    --output ./data/jsonl/2/train_live_turns.jsonl \
    --include_geocode --include_cyclic --include_final \
    --max_records_per_n 500

uv run deepspeed --num_gpus 1 train_live_rl.py \
    --model_name ./checkpoints/sft-2-merged \
    --train_file ./data/jsonl/2/train_live_turns.jsonl \
    --output_dir ./checkpoints/live-rl-2 \
    --turn_mix "cyclic_order=0.5,geocode=0.2,final=0.3" \
    --deepspeed ./configs/deepspeed_z2.json
```

Pass `--base_model <path>` when `--model_name` points to a bare LoRA adapter rather
than a merged checkpoint.

### 6. Evaluate

`evaluate.py` supports two modes:

- **Prefilled** (default): the prompt contains every message except the final
  assistant answer; the model only produces the answer.
- **Live tools** (`--live_tools`): start from system+user only, parse the model's
  tool calls, execute them in Python, feed results back, and loop.

```bash
# Prefilled evaluation
uv run python evaluate.py \
    --model_name ./checkpoints/rl-2 \
    --test_file  ./data/jsonl/2/val_natural.jsonl \
    --pipeline 2

# Live tool-loop evaluation (use --base_model when model_name is a LoRA adapter)
uv run python evaluate.py \
    --model_name ./checkpoints/rl-2/checkpoint-999 \
    --base_model ./checkpoints/sft-2-merged \
    --test_file  ./data/jsonl/2/val_balanced.jsonl \
    --pipeline 2 --live_tools --max_tool_turns 15 --live_max_new_tokens 256

# Spherical Earth evaluation (pipeline 2 only)
uv run python evaluate.py \
    --model_name ./checkpoints/rl-2 \
    --test_file  ./data/jsonl/2/val_earth.jsonl \
    --pipeline 2 --live_tools --earth
```

Other flags: `--limit N` (cap examples), `--save_predictions out.jsonl`,
`--temperature`, `--batch_size`, `--tool_error_policy`.

### 7. Interactive chat UI

`chat.py` is a Gradio app for poking at a trained model. It has a **Geometries** tab
(add/view/delete points, lines, polygons; persisted to `geometries.json`) and a
**Chat** tab where the model's tool calls are dispatched live against the local
geometry store and shown in a side panel. An **Earth backend** toggle switches
`cyclic_order` between planar and spherical at runtime.

```bash
uv run python chat.py --adapter ./checkpoints/rl-2 --pipeline 2
```

With `--geocode`, the `geocode` tool calls live OpenStreetMap Nominatim and
disambiguates candidate names by geographic proximity (the local store acts as an
override/cache):

```bash
uv run python chat.py --adapter ./checkpoints/rl-2 --pipeline 2 \
    --geocode --geocode_countrycodes us,ca
```

The UI serves on `0.0.0.0:7860` by default (`--port`, `--share`, `--no_4bit`).

---

## File reference

| File | Purpose |
|------|---------|
| `generate_fake_data.py` | Generate synthetic train/val parquet datasets. |
| `prepare_data.py` | Convert parquet → multi-turn JSONL conversations (per pipeline; planar or Earth). |
| `prepare_live_rl_data.py` | Expand pipeline-2 transcripts into per-turn live-RL examples. |
| `tools.py` | Tool schemas + implementations: `geocode` (incl. Nominatim) and `cyclic_order` (planar & spherical), geometry helpers. |
| `tool_call_parsing.py` | Shared parsing of assistant tool-call completions. |
| `train_sft.py` | QLoRA SFT stage (TRL `SFTTrainer`, DeepSpeed ZeRO-2). |
| `train_rl.py` | GRPO RL stage with deterministic outcome reward. |
| `train_live_rl.py` | Turn-level GRPO over the live pipeline-2 tool loop. |
| `reward.py` | Answer extraction, ground-truth computation, GRPO rewards. |
| `live_reward.py` | Per-turn rewards for live tool-loop RL. |
| `evaluate.py` | Prefilled and live-tool evaluation (planar or Earth). |
| `chat.py` | Gradio interactive test bench with live tool dispatch. |
| `check_loading.py` | Multi-GPU model-loading / OOM stress test. |
| `check_vram.py` | Single-GPU VRAM stress test for SFT and GRPO workloads. |
| `question_formats.json` | Natural-language question templates keyed by `n`. |
| `configs/deepspeed_z2.json` | ZeRO-2 (data parallelism) — default. |
| `configs/deepspeed_z3.json` | ZeRO-3 (model sharding) — for 14B+ models. |
| `scripts/` | SLURM `sbatch` launchers (`run_preprocess.sh`, `run_sft.sh`, `run_rl.sh`, `run_eval.sh`, `run_live_rl.sh`, …). |

### Dataset record format (JSONL)

Each line emitted by `prepare_data.py` is one conversation:

```json
{
  "question_id": 0,
  "pipeline": 2,
  "tools": [ /* geocode (+ cyclic_order for pipeline 2) JSON schemas */ ],
  "messages": [ /* system, user, assistant/tool turns */ ],
  "expected_answer": "clockwise",
  "meta": { "location_names": [...], "geometries": [...], "answer": "clockwise" }
}
```

---

## Tracking with Weights & Biases

All training scripts log to W&B (project `geo-finetune` by default). You'll see
`train/loss`, `eval/loss` and `val/accuracy` (SFT), and `reward/*` (GRPO). Disable
tracking with `--wandb_project ""`; first-time setup is `wandb login`.

## Swapping the base model

Pass any HuggingFace model ID to `--model_name` / `--base_model`. The pipeline relies
on the model's own chat template (`apply_chat_template(..., tools=...)`), so
tool-calling format differences are handled automatically. The only requirement is a
chat template that supports tool/function calling — `Qwen/Qwen2.5-7B-Instruct` is the
default and tested base.