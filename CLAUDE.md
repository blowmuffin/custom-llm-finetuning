# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Fine-tunes [LittleLamb](https://huggingface.co/MultiverseComputingCAI/LittleLamb) (a 290M-parameter Qwen3-derived model) on local PDF documents and serves it via an OpenAI-compatible API. The entire pipeline runs on CPU with no vector database.

## Pipeline (run in order)

```bash
# Step 1 — extract text + images from PDFs in documents/
python src/pdf_processor.py

# Step 2 — build training dataset (auto-selects qa mode if ANTHROPIC_API_KEY is set)
python src/dataset_builder.py
python src/dataset_builder.py --mode qa --n-questions 8   # explicit Q&A mode
python src/dataset_builder.py --mode chunks               # offline mode

# Step 3 — LoRA fine-tune (downloads model from HuggingFace on first run)
python src/trainer.py

# Step 4 — start inference server at http://localhost:8000
python src/server.py

# Step 5 — open chat UI at http://localhost:8501 (separate terminal)
streamlit run src/chat_app.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # then fill in ANTHROPIC_API_KEY if needed
```

## Configuration (`.env`)

All hyperparameters and API settings live in `.env`. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Enables image processing and Q&A dataset generation |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Used for image-to-text and Q&A generation |
| `FINETUNE_EPOCHS` | `3` | |
| `FINETUNE_BATCH_SIZE` | `2` | Reduce to `1` if OOM |
| `FINETUNE_LORA_RANK` | `16` | Higher = more capacity, slower |
| `FINETUNE_MAX_SEQ_LEN` | `512` | Also controls chunk size in dataset_builder |
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `8000` | |

## Architecture

```
documents/           ← input PDFs
outputs/
  dataset/
    raw_text/        ← per-doc JSON (pdf_processor.py output)
    raw_images/      ← extracted image files
    train.jsonl      ← training examples (dataset_builder.py output)
  model/             ← LoRA adapter weights (trainer.py output)
src/
  pdf_processor.py   ← step 1: calls image_processor.py per image
  image_processor.py ← Claude Vision helper (lazy-initialised Anthropic client)
  dataset_builder.py ← step 2: chunks mode or Q&A mode via Claude
  trainer.py         ← step 3: SFTTrainer + LoRA on LittleLamb
  server.py          ← step 4: FastAPI, merges LoRA at startup
  chat_app.py        ← step 5: Streamlit UI calling server.py
```

### Cross-file relationships worth knowing

- `pdf_processor.py` imports `image_processor.py` directly; `image_processor.is_available()` gates all Claude Vision calls.
- `dataset_builder.py` reads the JSON files written by `pdf_processor.py`. Run step 1 before step 2.
- `server.py` reads `outputs/model/adapter_config.json` to decide whether to merge LoRA adapters. If absent, it falls back to the base model (useful for smoke-testing).
- The training data format is **ChatML** (`<|im_start|>` / `<|im_end|>` tokens). Both `dataset_builder.py` and `server.py` share the same system prompt string — if you change the prompt in one, update the other.

### LoRA targeting

`trainer.py` targets Qwen3 attention and MLP projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. `lora_alpha` is always set to `2 × lora_rank`.

### Dataset modes

- **`qa` mode** (recommended): calls Claude to generate N question–answer pairs per text chunk. Requires `ANTHROPIC_API_KEY`.
- **`chunks` mode** (offline): wraps raw text chunks as "explain this topic" instructions. No API needed. Less chat-friendly but fully local.

Chunk size is derived from `FINETUNE_MAX_SEQ_LEN`: approximately `MAX_SEQ_LEN × 0.6` words with 40-word overlap.
