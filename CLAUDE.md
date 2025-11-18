# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a synthetic dataset generation and LLM fine-tuning system designed to train local language models to reliably use the **Claudesidian-MCP toolset** for Obsidian vault operations. Teacher models (Claude, ChatGPT, Copilot) generate synthetic training data, which is then used to fine-tune smaller models using KTO (Kahneman-Tversky Optimization) or LoRA methods.

## Project Structure

```
Toolset-Training/
├── Datasets/              # Synthetic training data in ChatML format (JSONL)
├── Tools/                 # Dataset validation and analysis utilities
├── Trainers/
│   ├── mistral_lora_mac/  # Apple Silicon (MLX) LoRA fine-tuning
│   └── rtx3090_kto/       # NVIDIA GPU (Unsloth) KTO fine-tuning
├── Evaluator/             # Model testing harness (Ollama/LM Studio)
└── docs/                  # Architecture specs and setup guides
```

## Common Development Commands

### Dataset Validation

```bash
# Validate a dataset file
python tools/validate_syngen.py Datasets/syngen_toolset_v1.0.0_claude.jsonl

# Analyze tool coverage
python tools/analyze_tool_coverage.py Datasets/syngen_toolset_v1.0.0_claude.jsonl
```

### Setup & Installation

#### WSL2 / Linux (Recommended)

```bash
cd Trainers/rtx3090_kto

# Full setup with verification
bash setup.sh

# Quick setup (no verification tests)
bash setup.sh --quick

# Setup with Flash Attention (optional, takes 5-10 min)
bash setup.sh --with-flash-attn
```

#### Windows PowerShell

**Not recommended** - Use WSL2 for best compatibility. Windows has known issues with multiprocessing and some dependencies.

### Training

#### WSL2 / Linux

```bash
cd Trainers/rtx3090_kto

# Recommended: 7B model training (uses default dataset)
./train.sh --model-size 7b

# With custom dataset
./train.sh --model-size 7b --local-file ../../Datasets/my_data.jsonl

# With W&B logging
./train.sh --model-size 7b --wandb --wandb-project my-project

# Direct Python invocation
python train_kto.py --model-size 7b --batch-size 4 --gradient-accumulation 6

# Dry run (setup without training)
python train_kto.py --model-size 7b --dry-run
```

**Model size options:** `3b` (fast), `7b` (recommended), `13b` (quality), `20b` (specialized)

#### Windows PowerShell

```powershell
cd Trainers\rtx3090_kto

# Interactive training script (checks everything, prompts for confirmation)
.\train.ps1

# Direct Python (manual)
python train_kto.py --local-file "..\..\Datasets\syngen_tools_11.14.25.jsonl"
```

**Note:** The PowerShell script automatically:
- Finds Python from `unsloth_env` conda environment
- Checks disk space and CUDA availability
- Uses the default dataset (`syngen_tools_11.14.25.jsonl`)
- Shows configuration summary before training

### Training (Mac / Apple Silicon)

```bash
cd Trainers/mistral_lora_mac

# Standard training
python main.py --config config/config.yaml

# Resume from checkpoint
python main.py --config config/config.yaml --resume checkpoints/checkpoint_step_500.npz

# Evaluation only
python main.py --config config/config.yaml --resume checkpoints/best_checkpoint.npz --eval-only
```

### Model Upload to HuggingFace

#### WSL2 / Linux

```bash
cd Trainers/rtx3090_kto

# Upload with merged 16-bit model (recommended)
./upload_model.sh username/model-name

# Upload with GGUF creation
./upload_model.sh username/model-name yes

# Direct Python invocation
python src/upload_to_hf.py ./kto_output_rtx3090/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf
```

**Save methods:**
- `merged_16bit` - Full quality, ~14GB (recommended)
- `merged_4bit` - Smaller size, ~3.5GB
- `lora` - LoRA adapters only, ~320MB

#### Windows PowerShell

```powershell
cd Trainers\rtx3090_kto

# Interactive upload script
# - Lists available trained models
# - Prompts for model name
# - Select format (16bit/4bit/lora)
# - Option to create GGUF versions
.\upload_model.ps1
```

### GGUF Creation (For Ollama/llama.cpp)

#### WSL2 / Linux (Recommended Method)

**IMPORTANT:** GGUF quantization on WSL2 with Windows NTFS drives (`/mnt/c`) can hang due to I/O buffering issues. Always use WSL native filesystem for quantization.

```bash
cd Trainers/rtx3090_kto

# 1. Copy base f16 GGUF to WSL native filesystem
mkdir -p ~/tmp_gguf
cp gguf_output/model-unsloth.gguf ~/tmp_gguf/

# 2. Build llama.cpp (if not already built)
cd gguf_output/llama.cpp
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j $(nproc)
cd ../..

# 3. Create quantizations (fast on WSL native filesystem)
./gguf_output/llama.cpp/build/bin/llama-quantize ~/tmp_gguf/model-unsloth.gguf ~/tmp_gguf/model-Q4_K_M.gguf Q4_K_M
./gguf_output/llama.cpp/build/bin/llama-quantize ~/tmp_gguf/model-unsloth.gguf ~/tmp_gguf/model-Q5_K_M.gguf Q5_K_M
./gguf_output/llama.cpp/build/bin/llama-quantize ~/tmp_gguf/model-unsloth.gguf ~/tmp_gguf/model-Q8_0.gguf Q8_0

# 4. Verify GGUF files are valid
./gguf_output/llama.cpp/build/bin/llama-cli --model ~/tmp_gguf/model-Q4_K_M.gguf -p "test" -n 1

# 5. Copy back to Windows filesystem (if needed)
cp ~/tmp_gguf/model-Q*.gguf gguf_output/

# 6. Upload to HuggingFace
python upload_ggufs.py  # Create script as needed
```

**Timing (WSL native filesystem):**
- Q4_K_M: ~72 seconds
- Q5_K_M: ~49 seconds
- Q8_0: ~39 seconds

**Why WSL native filesystem is required:**
- Windows NTFS drives via `/mnt/c` have slow I/O and buffering issues
- Quantization may hang indefinitely when writing to NTFS
- WSL native filesystem (`~` or `/home`) is 10-100x faster
- Always work in `~/tmp_gguf` or similar for GGUF operations

#### Windows PowerShell (Alternative)

```powershell
cd Trainers\rtx3090_kto

# Interactive GGUF creation script
# Creates Q4_K_M, Q5_K_M, Q8_0 quantizations
.\create_gguf.ps1
```

This script:
1. Checks prerequisites (disk space, Python, CUDA)
2. Reads HuggingFace token from root `.env` file
3. Creates multiple GGUF quantizations
4. Uploads to HuggingFace repository

**Note:** Requires `llama.cpp` repository cloned in `gguf_output/` directory. See `WINDOWS_GGUF_SETUP.md` for details.

### Evaluation

```bash
# Using Ollama
python -m Evaluator.cli \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/baseline.json \
  --output Evaluator/results/run_$(date +%s).json

# Using LM Studio
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model-name \
  --prompt-set Evaluator/prompts/full_coverage.json \
  --output Evaluator/results/run_lmstudio.json \
  --markdown Evaluator/results/report.md
```

**Prompt sets:**
- `baseline.json` - General scenarios
- `full_coverage.json` - One prompt per tool (47 prompts)
- `tool_combos.json` - Multi-step workflows

## Architecture Overview

### Dual-Platform Training Strategy

The codebase maintains **two parallel implementations** for different hardware:

**Mac (mistral_lora_mac):**
- Framework: MLX (Apple-optimized)
- Method: LoRA fine-tuning
- Config: YAML (`config/config.yaml`)
- Entry: `main.py`
- Model: Mistral-7B-Instruct-v0.3

**NVIDIA (rtx3090_kto):**
- Framework: Unsloth + TRL
- Method: KTO optimization (preference learning)
- Config: Python dataclasses (`configs/training_config.py`)
- Entry: `train_kto.py`
- Models: 3B, 7B, 13B, 20B (configurable)

Both consume the same datasets from `Datasets/` directory.

### Data Flow Pipeline

1. **Generation**: Teacher models create synthetic examples in ChatML format
2. **Validation**: `tools/validate_syngen.py` checks structure, context objects, tool schemas
3. **Preparation**: Trainer converts to platform-specific format (Mistral Instruct or KTO)
4. **Training**: LoRA or KTO fine-tuning with checkpointing
5. **Evaluation**: Test via `Evaluator/` with prompt sets

## Critical Patterns & Requirements

### Context Object Pattern

**Every tool call MUST include a complete context object as the FIRST parameter:**

```json
{
  "context": {
    "sessionId": "session_1731015400000_a1b2c3d4e",
    "workspaceId": "ws_1731015400000_f5g6h7i8j",
    "sessionDescription": "Brief summary of session",
    "sessionMemory": "Never empty - prior context",
    "toolContext": "Why calling this tool",
    "primaryGoal": "User's main objective",
    "subgoal": "What this call achieves"
  },
  "otherParams": "..."
}
```

**All 7 fields are required.** `sessionMemory` must never be empty.

### KTO Interleaved Dataset Requirement

**CRITICAL for rtx3090_kto trainer:**

TRL's KTOTrainer has a bug where it crashes on homogeneous batches (all True or all False labels). The workaround is to use **interleaved datasets** with strict True/False/True/False pattern.

```python
# Dataset must alternate labels:
[True, False, True, False, True, False, ...]
```

This guarantees mixed batches with sequential sampling and prevents CUDA errors.

**Reference:** `KTO_TRAINING_REFERENCE.md` for full details.

### Dataset Format (ChatML)

```jsonl
{
  "conversations": [
    {"role": "user", "content": "User request"},
    {"role": "assistant", "content": "tool_call: toolName\narguments: {...}\n\nResult: {...}\n\nResponse"}
  ],
  "label": true
}
```

- **NO system message** (starts with user role)
- `label`: `true` = desirable example, `false` = undesirable (for contrastive learning)
- Tool calls show complete execution: call → result → response
- Multi-turn examples maintain session continuity

## Configuration Entry Points

### RTX 3090 Configuration
**File:** `Trainers/rtx3090_kto/configs/training_config.py`

The config uses Python dataclasses with these main sections:

**ModelConfig:**
- `model_name`: Base model to use (default: `unsloth/mistral-7b-v0.3-bnb-4bit`)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Use 4-bit quantization (default: True)

**LoRAConfig:**
- `r`: LoRA rank (default: 32)
- `lora_alpha`: LoRA alpha scaling (default: 64)
- `lora_dropout`: Dropout for LoRA layers (default: 0.05)
- `target_modules`: Which layers to apply LoRA (q/k/v/o/gate/up/down projections)

**KTOTrainingConfig:**
- `per_device_train_batch_size`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Accumulation steps (default: 6, effective batch = 24)
- `learning_rate`: Learning rate (default: 2e-7)
- `beta`: KTO beta parameter (default: 0.3)
- `max_length`: Max sequence length (default: 2048)
- `num_train_epochs`: Number of epochs (default: 1)
- `use_kto_s`: Use KTO-S SIGN correction (default: False)
- `use_two_stage_lr`: Use two-stage LR schedule (default: False)

**DatasetConfig:**
- `dataset_name`: HuggingFace dataset name
- `local_file`: Path to local JSONL file (default: `../../Datasets/syngen_tools_11.14.25.jsonl`)
- `split_dataset`: Create train/val split (default: False)

**To modify:** Edit `configs/training_config.py` directly or override via CLI.

**Preset configs:**
- `get_3b_config()` - Fast iteration (batch_size=8)
- `get_7b_config()` - Production quality (batch_size=4) ⭐ **Recommended**
- `get_13b_config()` - Maximum quality (batch_size=2)
- `get_20b_config()` - Specialized tasks (batch_size=4)

**CLI Overrides:**
```bash
python train_kto.py \
  --model-size 7b \
  --batch-size 4 \
  --gradient-accumulation 6 \
  --learning-rate 2e-7 \
  --num-epochs 1 \
  --max-seq-length 2048
```

### Mac Configuration
**File:** `Trainers/mistral_lora_mac/config/config.yaml`

Key parameters:
- `model.max_seq_length: 2048`
- `lora.rank: 16` (memory vs capacity tradeoff)
- `training.per_device_batch_size: 2`
- `training.gradient_accumulation_steps: 4`
- `data.dataset_path: "path/to/dataset.jsonl"`

## Tool Coverage

The system supports **47+ tools** across **5 agent categories:**
- **vaultManager** - File/folder operations
- **contentManager** - CRUD operations
- **memoryManager** - Session/state/workspace management
- **vaultLibrarian** - Advanced search, batch operations
- **agentManager** - Agent lifecycle, prompt execution

**Schema source:** `Tools/tool_schemas.json` (central schema definitions)

## Platform-Specific Notes

### Windows vs WSL2

**WSL2 (Strongly Recommended):**
- Full compatibility with all features
- Multiprocessing works correctly
- Better performance
- Native bash script support
- All setup/upload scripts available

**Windows PowerShell (Limited Support):**
- Known issues with multiprocessing (hangs on data loading)
- `dataloader_num_workers` must be 0 in config
- Requires Windows compatibility patches (auto-applied in `train_kto.py`)
- Flash Attention not supported
- Limited to PowerShell scripts (`.ps1`)

**Windows Compatibility Patches** (auto-applied in `train_kto.py:26-50` and `upload_to_hf.py:10-38`):
1. Dataclass `fields()` wrapper for non-dataclasses
2. Disable `torch.compile` (not supported on Windows)
3. Pre-patch `torch._inductor.runtime.hints`

**If using Windows PowerShell:**
- Set `dataloader_num_workers: 0` in `configs/training_config.py`
- Use `.ps1` scripts instead of `.sh` scripts
- Expect slower training (no multiprocessing)

**Switching to WSL2:**
```powershell
# Install WSL2 (Windows 10/11)
wsl --install

# Clone repo in WSL2 filesystem (better performance)
cd ~
git clone <repo-url>

# Follow Linux setup instructions
```

### Mac Training (mistral_lora_mac)
Verify Metal GPU is available:
```bash
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Memory Optimization
Both trainers use:
- 4-bit quantization (RTX) or float16 (Mac)
- 8-bit optimizers (RTX)
- Gradient checkpointing (optional, 13B+ models)
- LoRA for parameter efficiency

Expected VRAM (RTX 3090):
- 3B: ~8-10 GB
- 7B: ~9-11 GB
- 13B: ~14-16 GB

## Testing & Validation

### Unit Tests
```bash
# Test dataset validator
python tools/validate_syngen.py <dataset.jsonl>
```

### Integration Tests
```bash
# Dry run (setup without training)
cd Trainers/rtx3090_kto
python train_kto.py --model-size 7b --dry-run
```

### Model Evaluation
```bash
# Serve model and run evaluation suite
python -m Evaluator.cli --model <name> --prompt-set Evaluator/prompts/full_coverage.json
```

## Troubleshooting

### CUDA Out of Memory (RTX)
```bash
# Reduce batch size
python train_kto.py --model-size 7b --batch-size 2 --gradient-accumulation 16

# Reduce sequence length
python train_kto.py --model-size 7b --max-seq-length 1024
```

### Mac Out of Memory
Edit `config/config.yaml`:
- Reduce `per_device_batch_size: 1`
- Reduce `max_seq_length: 1024`
- Reduce `lora.rank: 8`

### Dataset Validation Failures
Common issues:
- Missing context object (must be first parameter)
- Empty `sessionMemory` field (never allowed)
- Incorrect ID format (must match: `session_<13digits>_<9chars>`)
- Context not as first parameter in tool call

### KTO Training Crashes (TRL Bug)
If you see `CUDA error: invalid configuration argument`:
- Ensure dataset is **interleaved** (True/False/True/False pattern)
- Check `KTO_TRAINING_REFERENCE.md` for full workaround

### GGUF Quantization Hangs (WSL2)
If GGUF quantization hangs or file size stops growing:
- **Cause:** Windows NTFS drives (`/mnt/c`) have I/O buffering issues
- **Solution:** Always use WSL native filesystem (`~` or `/home`)
- **Steps:**
  1. Copy base GGUF to `~/tmp_gguf/`
  2. Run quantization in WSL native filesystem
  3. Copy results back to Windows if needed
- **Performance:** 10-100x faster on WSL native filesystem
- See "GGUF Creation (WSL2 / Linux)" section for full workflow

## Key Documentation

- `README.md` - Project overview and stats
- `KTO_TRAINING_REFERENCE.md` - **Critical:** TRL bug workaround for KTO training
- `finetuning-strategy.md` - Master strategy document (203KB, comprehensive)
- `Trainers/rtx3090_kto/README.md` - RTX training guide
- `Trainers/mistral_lora_mac/README.md` - Mac training guide
- `Evaluator/README.md` - Evaluation harness usage
- `docs/SCHEMA_VERIFICATION_REFERENCE.md` - Tool schema reference

## Development Workflow

1. **Validate dataset** before training
2. **Choose platform** (Mac vs RTX) based on available hardware
3. **Use appropriate config** (YAML for Mac, dataclass presets for RTX)
4. **Monitor logs** (`logs/training.jsonl` or console output)
5. **Evaluate checkpoints** with Evaluator before final deployment
6. **For KTO:** Always verify dataset is interleaved (check first)

## Environment Variables

Create a `.env` file in the repository root:

```bash
# HuggingFace (required for uploads, optional for training)
HF_TOKEN=hf_your_token_here
# or
HF_API_KEY=hf_your_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key

# Conda environment name (optional, for PowerShell scripts)
CONDA_ENV=unsloth_env

# Mac training (optional)
HF_HOME=/path/to/cache
```

**Getting tokens:**
- HuggingFace: https://huggingface.co/settings/tokens (create with "write" access)
- W&B: https://wandb.ai/authorize

**Note:**
- `.env` file is gitignored
- See `.env.example` in `Trainers/rtx3090_kto/` for template
- PowerShell scripts auto-load from root `.env` file

## Script Reference

### WSL2 / Linux Scripts

**Setup:**
- `Trainers/rtx3090_kto/setup.sh` - Full environment setup with verification

**Training:**
- `Trainers/rtx3090_kto/train.sh` - Training wrapper (uses conda, passes args to train_kto.py)
- `Trainers/rtx3090_kto/train_kto.py` - Main training script (Python)

**Upload:**
- `Trainers/rtx3090_kto/upload_model.sh` - Upload to HuggingFace
- `Trainers/rtx3090_kto/src/upload_to_hf.py` - Upload script (Python)

**Other:**
- `Trainers/rtx3090_kto/run_dry_run.sh` - Dry run wrapper
- `Trainers/rtx3090_kto/train_run8.sh` - Legacy training script

### Windows PowerShell Scripts

**Training:**
- `Trainers/rtx3090_kto/train.ps1` - Interactive training (checks prereqs, prompts confirmation)

**Upload:**
- `Trainers/rtx3090_kto/upload_model.ps1` - Interactive upload (select model, format, GGUF options)

**GGUF:**
- `Trainers/rtx3090_kto/create_gguf.ps1` - Create GGUF quantizations (Q4_K_M, Q5_K_M, Q8_0)
- `Trainers/rtx3090_kto/check_gguf_requirements.ps1` - Verify GGUF dependencies

**Setup:**
- `setup_unsloth_windows.ps1` (root) - Windows setup script

### Python Scripts (Cross-platform)

**Training:**
- `Trainers/rtx3090_kto/train_kto.py` - Main training entry point

**Data:**
- `Trainers/rtx3090_kto/src/data_loader.py` - Dataset loading and preprocessing
- `Trainers/rtx3090_kto/src/model_loader.py` - Model loading with Unsloth
- `Trainers/rtx3090_kto/src/kto_s_trainer.py` - Custom KTO trainer with SIGN correction

**Utilities:**
- `Trainers/rtx3090_kto/src/upload_to_hf.py` - HuggingFace upload
- `Trainers/rtx3090_kto/src/inference.py` - Inference utilities
- `Trainers/rtx3090_kto/src/training_callbacks.py` - Training callbacks
- `Trainers/rtx3090_kto/src/adaptive_memory.py` - Adaptive memory management
- `Trainers/rtx3090_kto/check_config.py` - Config verification
- `Trainers/rtx3090_kto/check_gpu_setup.py` - GPU verification
- `Trainers/rtx3090_kto/diagnose_batch_size.py` - Batch size tuning
- `Trainers/rtx3090_kto/test_installation.py` - Installation verification

**Validation:**
- `tools/validate_syngen.py` - Dataset validator
- `tools/analyze_tool_coverage.py` - Coverage analysis

## Important Notes

- **Never commit** trained models or checkpoints (large files in `.gitignore`)
- **Always run** `validate_syngen.py` before training
- **For KTO training:** Dataset interleaving is mandatory (not optional)
- **Context objects:** All 7 fields required, sessionMemory never empty
- **Tool schemas:** Reference `Tools/tool_schemas.json` for validation
- **Windows users:** Strongly recommend WSL2 for rtx3090_kto (better compatibility)
- **PowerShell scripts:** All auto-load HF_TOKEN from root `.env` file
- **Bash scripts:** Source conda and activate environment automatically
