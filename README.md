# **Research tool. The author is not responsible for misuse, training data, or generated content.**

> **Legal notice:** I am not responsible for anyone who uses this tool for illegal purposes. If you train this model and use it for hacking, criminal activity, or any other unlawful actions, that is entirely your own responsibility.

# ARIA Atom 3.6.0 stable

![LOGO](screenshots/ARIA-Logo.png)

**ARIA Atom 3.6.0 stable** is a GPT-style Transformer language model built entirely in Rust with CUDA/cuBLAS acceleration and custom PTX kernels. Checkpoints use the **GGUF** format -- weights, tokenizer, Adam optimizer state, and FP32 master weights are stored in a single file.

> **Note:** ARIA requires an NVIDIA GPU. AMD, Intel, and other GPUs are not supported.

| Version | Codename | Architecture | Parameters | VRAM | Status |
|---|---|---|---|---|---|
| 3.2.0 | Wotan | LSTM (1 layer) | ~44.5M | 6GB | Legacy |
| 3.3.0 | Atom | Transformer (12 layers) | ~124M | 8GB | Legacy |
| 3.4.0 | Atom | Transformer + warmup/clip | ~40M | 4GB | Legacy |
| 3.5.0 | Efkolos (light) | Transformer + LoRA | 250M | ~4GB | Legacy |
| 3.5.1 | Efkolos (optimized) | Transformer + LoRA + INT4 | 250M | ~3GB | Legacy |
| 3.5.2 | Efkolos | Transformer + GGUF + Q4_0 | 250M | ~3GB | Legacy |
| **3.6.0 stable** | **Efkolos** | **Transformer + training stability + restored LoRA/INT4/grad-checkpoint toggles** | **~223M** | **~3GB training / ~0.7GB inference** | **Stable** |

## Changelog

### v3.6.0 stable
- Restored v3.5.1 feature toggles in `TransformerModel`:
  - `int4_quantized`
  - `gradient_checkpointing`
  - `lora_backward_enabled`
- Added FP32 master weights for mixed-precision training stability:
  - Adam updates run on FP32 master copies, then copy back to FP16 working weights
  - Prevents delayed loss explosion from FP16-only Adam updates
  - Master weights are saved inside the GGUF checkpoint and freed in inference mode
- Added new CUDA kernel `adam_update_f32_from_f32` for FP32 master weight updates
- Fixed weight initialization:
  - `randn_f16` now uses true Gaussian (Box-Muller) instead of uniform noise
  - Added depth-dependent residual scaling (`1/sqrt(2*L)`) on `attn_out` and `ffn_down` projections
- Made gradient clipping NaN/inf-safe:
  - If `global_grad_norm` is not finite, gradients are zeroed and the step is skipped
- Split checkpoint loading:
  - `load_checkpoint()` -- inference mode: loads weights + tokenizer only (no optimizer state, no OOM)
  - `load_checkpoint_for_training()` -- training resume: loads weights + Adam state + FP32 master weights
- Removed obsolete separate `aria_tokenizer.json` save path; tokenizer is always embedded in GGUF
- Interactive dataset selection in `train_fresh` / `train_debug` (`all` or numbered files)
- Sequence cache now depends on the selected dataset set; stale/incomplete caches are auto-removed and rebuilt
- Fixed streaming cache header bug: count is now flushed to disk, empty/broken caches are detected and rebuilt
- Adaptive warmup: `warmup_steps = min(user_warmup, total_steps / 4).max(100)` so small tests reach full LR quickly
- Added `train_debug` binary with per-gradient NaN/inf diagnostics
- Training verified on RTX 4060:
  - `MICRO_BATCH_N = 4`
  - `PRETRAIN_BATCH_SIZE = 512`
  - `max_seq_len = 512`
  - Speed: ~170 seq/s
  - Stable loss decrease confirmed: 10.35 → 7.36 on 200k sequences / 2 epochs

### v3.5.2
- **GGUF checkpoint format** -- weights, tokenizer, and Adam state in a single .gguf file
- **Q4_0 quantization** -- export inference-only model at 4-bit (~2x smaller, ~2-5% quality loss)
- Streaming sequence cache -- dataset no longer loaded fully into RAM
- Removed JSON and ARIA v2 binary formats -- GGUF only
- export_gguf binary for Q4_0 inference export

## Architecture

### Efkolos (223M parameters)

Type: GPT-style decoder-only Transformer
Layers: 20
d_model: 896
Heads: 14 (head_dim = 64)
FFN dim: 3584 (4x d_model)
Context: 2048 tokens (training uses `max_seq_len = 512` for VRAM efficiency)
Vocabulary: ~31,500 BPE tokens (Cyrillic-aware)
Parameters: ~223M base
Precision: FP16 weights + FP16 activations + FP32 Adam state + FP32 master weights
Optimizer: Adam (beta1=0.9, beta2=0.999, eps=1e-8)
Gradient clipping: global L2 norm <= 1.0 with NaN/inf fallback

#### VRAM Usage (RTX 4060)

| Component | Training | Inference |
|---|---|---|
| Base weights FP16 | ~450 MB | ~450 MB |
| FP32 master weights | ~900 MB | -- |
| Adam optimizer (f32) | ~1.8 GB | -- |
| Activations / grads | ~600 MB | ~100 MB |
| Attention scores | ~200 MB | ~50 MB |
| Other | ~150 MB | ~50 MB |
| **Total** | **~3.1 GB** | **~650 MB** |

Leaves ~5GB headroom on RTX 4060 (8GB) during training.

## Requirements

| Dependency | Version | Link |
|---|---|---|
| Rust + Cargo | stable (2021 edition) | https://rustup.rs |
| Visual Studio Build Tools | 2017 or later | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| NVIDIA CUDA Toolkit | 12.x | https://developer.nvidia.com/cuda-downloads |
| NVIDIA drivers | latest | https://www.nvidia.com/drivers |

### Visual Studio Build Tools (Windows required)

1. Download **Build Tools for Visual Studio**.
2. Select **Desktop development with C++**.
3. Install (~3-5 min).
4. Restart your machine.

VS Code alone is **not enough** -- you need the Build Tools separately.

## Getting Started

git clone https://github.com/USER/ARIA.git
cd ARIA
cargo build --release
.\target\release\aria.exe

On first launch, ARIA will:
1. Scan data base/ for training data
2. Train a BPE tokenizer (~31,500 tokens)
3. Initialize Transformer weights
4. Train on the dataset
5. Save checkpoint to aria json/aria_checkpoint.gguf

### Train from scratch

.\target\release\train_fresh.exe "data base"

Reads JSONL files from `data base/`. At startup you will see a numbered list of datasets and a prompt:

```
Choose datasets to use:
  - enter numbers separated by spaces (e.g. "1 3 7")
  - or type "all" to use every file
>
```

- Type numbers separated by spaces to train only on selected files.
- Type `all` (or press Enter) to use every `.jsonl` file.

Saves GGUF checkpoint after each epoch.

### Continue training (resume from checkpoint)

.\target\release\train_fresh.exe "data base"

`train_fresh.exe` automatically resumes from `aria json/aria_checkpoint.gguf` when the checkpoint exists, loading optimizer state and FP32 master weights. Dataset selection works the same way.

### Supervised Fine-Tuning (SFT)

.\target\release\sft_train.exe

### Interactive training

Set-Item Env:ARIA_CONTINUE_TRAIN 1
.\target\release\aria.exe

### Inference and Testing

.\target\release\greedy_test.exe
.\target\release\sample_test.exe
.\target\release\test_suite.exe
.\target\release\inference.exe your prompt here
.\target\release\debug_logits.exe

### Debug training (gradient diagnostics)

.\target\release\train_debug.exe "data base"

Same as `train_fresh.exe`, but compiled with `#[cfg(feature = "train_debug")]` diagnostics that print the first tensor with non-finite gradients on each step. Use for tracking down NaN/inf sources.

### Export Q4_0 Inference Model

.\target\release\export_gguf.exe aria json/aria_checkpoint.gguf aria json/aria_inference.gguf

Produces a ~300MB inference-only file with no optimizer state or master weights.

## Dataset Format

Place JSONL files in data base/. Each line:

{text: User: hello\nAssistant: hi, how can I help?}

Use USER / ASSISTANT tokens for dialog fine-tuning. The tokenizer is trained from scratch on your data.

## Training Parameters

| Variable | Description | Default |
|---|---|---|
| ARIA_LR | Peak learning rate | 0.0003 |
| ARIA_WARMUP | Warmup steps (capped at total_steps/4) | 1000 |
| ARIA_MAX_SEQS | Sequences per epoch (omit to use all selected sequences) | -- |
| ARIA_EPOCHS | Number of epochs | 5 |
| ARIA_VOCAB_LINES | Lines for tokenizer training | 500,000 |
| ARIA_CONTINUE_TRAIN | Resume from checkpoint in interactive mode | -- |
| ARIA_CHECKPOINT_EVERY | Intermediate checkpoint every N batches | 80000 |

Gradient clipping is always enabled (norm=1.0) with NaN/inf fallback.

Intermediate checkpoints are saved as `aria json/aria_checkpoint.gguf.<N>_batches.gguf`.

LR schedule: linear warmup to ARIA_LR over `min(ARIA_WARMUP, total_steps/4)` steps (at least 100), then cosine decay to 0.3x ARIA_LR.

Quick run on selected datasets (e.g. only wiki + sberquad):
# at prompt type: 14 17
Set-Item Env:ARIA_EPOCHS 1
.\target\release\train_fresh.exe "data base"

Full run on all datasets (RTX 4060, batch=512, micro-batch=4, max_seq_len=512, ~170 seq/s):
# at prompt type: all
Set-Item Env:ARIA_EPOCHS 5
.\target\release\train_fresh.exe "data base"

Smoke test (used for stability validation):
# at prompt type: all
Set-Item Env:ARIA_MAX_SEQS 5000
Set-Item Env:ARIA_EPOCHS 5
Set-Item Env:ARIA_LR 0.0001
.\target\release\train_fresh.exe "data base"

## Interactive Commands

| Command | Description |
|---|---|
| stats | Print model statistics |
| settings | Show current inference settings |
| mode greedy | Greedy decoding |
| mode topk | Top-K sampling (default k=20) |
| mode topp | Nucleus sampling (default p=0.9) |
| temp 0.1-2.0 | Set temperature |
| topk n | Set K |
| topp 0.0-1.0 | Set P |
| exit | Quit |

## Files

| Path | Description |
|---|---|
| aria json/aria_checkpoint.gguf | Full checkpoint (weights + tokenizer + Adam state + FP32 master weights) |
| aria json/aria_inference.gguf | Q4_0 inference model (created by export_gguf) |
| aria json/aria_dialogs.json | Saved dialog history |
| data base/sequences_cache_*.bin | Tokenized sequence cache |
| data base/sequences_cache_*.bin.idx | Cache index |
| logs/validation_log.txt | Output from test_suite |

## Troubleshooting

**error: linker link.exe not found**
Install Visual Studio Build Tools with the C++ workload.

**GPU not detected**
Check NVIDIA drivers and CUDA Toolkit 12.x are installed. nvcc must be on your PATH.

**Old checkpoints do not load**
3.5.2 uses GGUF only. JSON and ARIA v2 binary checkpoints are not supported -- retrain with train_fresh.exe.

**Out of memory**
Lower `ARIA_MAX_SEQS` or reduce batch size in source. Use `cargo build --release` -- debug builds are 10-20x slower and use more memory.

**Old checkpoints do not load**
- 3.5.2+ uses GGUF only. JSON and ARIA v2 binary checkpoints are not supported -- retrain with `train_fresh.exe`.
- 3.6.0 stable can load 3.5.2/3.6.0 GGUF checkpoints; legacy checkpoints without FP32 master weights are automatically upgraded in training mode.

**Loss explosion during long training**
- 3.6.0 stable fixes the most common cause: FP16-only Adam updates accumulating rounding errors.
- If loss still explodes after many hours, check:
  1. Dataset quality (no extremely long sequences or corrupted JSONL lines)
  2. Learning rate is not too high for your data
  3. GPU memory is not overheating / throttling

**Bad output quality**
1. Check dataset format and size.
2. Verify dialog lines follow User: ...\nAssistant: ...
3. Run test_suite and check logs/validation_log.txt.
4. Try more epochs or a lower learning rate.