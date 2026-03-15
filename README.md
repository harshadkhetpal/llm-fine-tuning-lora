# 🦙 LLM Fine-Tuning with LoRA / QLoRA

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-PEFT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/harshadkhetpal/llm-fine-tuning-lora/train.yml?style=flat-square&label=CI)](https://github.com/harshadkhetpal/llm-fine-tuning-lora/actions)

> Production-grade fine-tuning pipeline for LLaMA 3, Mistral, and Phi-3 using **LoRA / QLoRA** with 4-bit quantization. Supports multi-GPU training, WandB tracking, and one-click deployment.

## ✨ Features
- 4-bit QLoRA with bitsandbytes for training on a single A100/consumer GPU
- LoRA rank-tunable adapters (r=8, 16, 32, 64)
- Supports LLaMA-3-8B, Mistral-7B, Phi-3-Mini
- WandB + MLflow experiment tracking
- GGUF export for Ollama / llama.cpp deployment
- Docker + docker-compose for reproducibility

## 🗂 Project Structure
```
llm-fine-tuning-lora/
├── train.py              # Main training script
├── merge_adapter.py      # Merge LoRA weights into base model
├── evaluate.py           # ROUGE / BLEU / perplexity eval
├── export_gguf.py        # Export to GGUF for Ollama
├── configs/
│   ├── lora_config.yaml  # LoRA hyperparameters
│   └── training_args.yaml
├── data/
│   └── prepare_dataset.py
├── scripts/
│   ├── download_base_model.sh
│   └── push_to_hub.sh
├── notebooks/
│   └── inference_demo.ipynb
├── Dockerfile
└── docker-compose.yml
```

## 🚀 Quick Start

```bash
git clone https://github.com/harshadkhetpal/llm-fine-tuning-lora
cd llm-fine-tuning-lora
pip install -r requirements.txt

# Fine-tune LLaMA-3-8B with QLoRA
python train.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset alpaca_cleaned \
  --lora_r 16 \
  --lora_alpha 32 \
  --epochs 3 \
  --output_dir ./output
```

## 📊 Benchmark Results

| Model | Dataset | ROUGE-L | Perplexity | GPU | Time |
|---|---|---|---|---|---|
| LLaMA-3-8B (QLoRA r=16) | Alpaca | 0.42 | 7.3 | A100-40GB | 4h |
| Mistral-7B (QLoRA r=8) | OpenHermes | 0.45 | 6.8 | A100-40GB | 3.5h |
| Phi-3-Mini (LoRA r=32) | CodeAlpaca | 0.39 | 9.1 | RTX 3090 | 6h |

## 🔧 Configuration

```yaml
# configs/lora_config.yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
bias: "none"
task_type: "CAUSAL_LM"
quantization: "4bit"   # 4bit / 8bit / none
```

## 📦 Deployment

```bash
# Export and run with Ollama
python export_gguf.py --checkpoint ./output/checkpoint-final
ollama create harshad-llm -f Modelfile
ollama run harshad-llm "Explain MLOps in simple terms"
```

---
**Author**: [Harshad Khetpal](https://github.com/harshadkhetpal) · MLOps & AI/ML Engineer
