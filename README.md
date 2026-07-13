# **Research tool. The author is not responsible for misuse, training data, or generated content.**

> **Legal notice:** I am not responsible for anyone who uses this tool for illegal purposes. If you train this model and use it for hacking, criminal activity, or any other unlawful actions, that is entirely your own responsibility.

# ARIA Atom 3.5.2

![LOGO](screenshots/ARIA-Logo.png)

**ARIA Atom 3.5.2** is a GPT-style Transformer language model built entirely in Rust with CUDA/cuBLAS acceleration and custom PTX kernels. Version 3.5.2 introduces **GGUF** as the unified checkpoint format -- weights, tokenizer, and Adam optimizer state are stored in a single file.

> **Note:** ARIA requires an NVIDIA GPU. AMD, Intel, and other GPUs are not supported.

| Version | Codename | Architecture | Parameters | VRAM | Status |
|---|---|---|---|---|---|
| 3.2.0 | Wotan | LSTM (1 layer) | ~44.5M | 6GB | Legacy |
| 3.3.0 | Atom | Transformer (12 layers) | ~124M | 8GB | Legacy |
| 3.4.0 | Atom | Transformer + warmup/clip | ~40M | 4GB | Legacy |
| 3.5.0 | Efkolos (light) | Transformer + LoRA | 250M | ~4GB | Legacy |
| 3.5.1 | Efkolos (optimized) | Transformer + LoRA + INT4 | 250M | ~3GB | Legacy |
| **3.5.2** | **Efkolos** | **Transformer + GGUF + Q4_0** | **250M** | **~3GB** | **Stable** |
| 3.5.0 | Varys (heavy) | Transformer + LoRA | 1B | ~8GB | Experimental |

## Changelog

### v3.5.2 (Stable)
- **GGUF checkpoint format** -- weights, tokenizer, and Adam state in a single .gguf file
- **Q4_0 quantization** -- export inference-only model at 4-bit (~2x smaller, ~2-5% quality loss)
- Streaming sequence cache -- dataset no longer loaded fully into RAM
- Removed JSON and ARIA v2 binary formats -- GGUF only
- export_gguf binary for Q4_0 inference export

### v3.5.1
- INT4 quantization of base weights
- Gradient checkpointing
- LoRA backward pass

### v3.5.0
- LoRA (Low-Rank Adaptation)
- 250M parameters, 2048-token context

## Architecture

### Efkolos (250M parameters)

Type: GPT-style decoder-only Transformer + LoRA
Layers: 20
d_model: 896
Heads: 14 (head_dim = 64)
FFN dim: 3584 (4x d_model)
Context: 2048 tokens
Vocabulary: ~31,500 BPE tokens (Cyrillic-aware)
Parameters: 250M base + 5M LoRA adapters
Precision: FP16 weights + FP16 activations + FP32 Adam
LoRA rank: 8

#### VRAM Usage (batch=1, seq=2048, gradient checkpointing)

| Component | Size |
|---|---|
| Base weights FP16 | 250 MB |
| LoRA adapters | 60 MB |
| Adam optimizer | 2000 MB |
| Activations | 1100 MB |
| Attention scores | 110 MB |
| Other | 100 MB |
| **Total** | **~3620 MB** |

Fits on RTX 4060 (8GB) with ~4GB headroom.

### Varys (1B parameters) -- Experimental

Type: GPT-style decoder-only Transformer + LoRA
Parameters: 1B base + 10M LoRA adapters
Estimated VRAM: ~8-10 GB with INT4 + gradient checkpointing
Target GPU: RTX 4080S (20GB)

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

.\target\release\train_fresh.exe

Reads JSONL files from data base/. Saves GGUF checkpoint after each epoch.

### Supervised Fine-Tuning (SFT)

.\target\release\sft_train.exe

### Continue training

Set-Item Env:ARIA_CONTINUE_TRAIN 1
.\target\release\aria.exe

### Inference and Testing

.\target\release\greedy_test.exe
.\target\release\sample_test.exe
.\target\release\test_suite.exe
.\target\release\inference.exe your prompt here
.\target\release\debug_logits.exe

### Export Q4_0 Inference Model

.\target\release\export_gguf.exe aria json/aria_checkpoint.gguf aria json/aria_inference.gguf

Produces a ~300MB inference-only file with no optimizer state.

## Dataset Format

Place JSONL files in data base/. Each line:

{text: User: hello\nAssistant: hi, how can I help?}

Use USER / ASSISTANT tokens for dialog fine-tuning. The tokenizer is trained from scratch on your data.

## Training Parameters

| Variable | Description | Default |
|---|---|---|
| ARIA_LR | Peak learning rate | 0.0003 |
| ARIA_WARMUP | Warmup steps | 1000 |
| ARIA_MAX_SEQS | Sequences per epoch | 500,000 |
| ARIA_EPOCHS | Number of epochs | 5 |
| ARIA_VOCAB_LINES | Lines for tokenizer training | 500,000 |
| ARIA_CONTINUE_TRAIN | Resume from checkpoint | -- |

Gradient clipping is always enabled (norm=1.0).

LR schedule: linear warmup to ARIA_LR over ARIA_WARMUP steps, then cosine decay to 0.3x ARIA_LR.

Quick run:
Set-Item Env:ARIA_MAX_SEQS 200000
Set-Item Env:ARIA_EPOCHS 1
.\target\release\train_fresh.exe

Full run (~6-7 hours on RTX 4060, ~450-550 seq/s):
Set-Item Env:ARIA_MAX_SEQS 500000
Set-Item Env:ARIA_EPOCHS 5
.\target\release\train_fresh.exe

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
| aria json/aria_checkpoint.gguf | Full checkpoint (weights + tokenizer + Adam state) |
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
Lower ARIA_MAX_SEQS. Always use cargo build --release -- debug builds are 10-20x slower.

**Bad output quality**
1. Check dataset format and size.
2. Verify dialog lines follow User: ...\nAssistant: ...
3. Run test_suite and check logs/validation_log.txt.
4. Try more epochs or a lower learning rate.