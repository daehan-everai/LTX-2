# LTX-2 Trainer

This package provides tools and scripts for training and fine-tuning
Lightricks' **LTX-2** audio-video generation model. It enables LoRA training, full
fine-tuning, and training of video-to-video transformations (IC-LoRA) on custom datasets.

---

## 📖 Documentation

All detailed guides and technical documentation are in the [docs](./docs/) directory:

- [⚡ Quick Start Guide](docs/quick-start.md)
- [🎬 Dataset Preparation](docs/dataset-preparation.md) (includes OpenRouter contact-sheet captioning)
- [🛠️ Training Modes](docs/training-modes.md)
- [⚙️ Configuration Reference](docs/configuration-reference.md)
- [🎯 Small-dataset I2V LoRA Recipe](docs/small-i2v-lora-recipe.md)
- [🚀 Training Guide](docs/training-guide.md)
- [🧪 Inference Guide](../ltx-pipelines/README.md)
- [🔧 Utility Scripts](docs/utility-scripts.md)
- [📚 LTX-Core Documentation](../ltx-core/README.md)
- [🛡️ Troubleshooting Guide](docs/troubleshooting.md)

### Video captioning

- Local Qwen2.5-Omni or Gemini Flash: [`scripts/caption_videos.py`](scripts/caption_videos.py)
- OpenRouter contact sheets (the doggy SFT/DPO recaption path): [`scripts/build_contact_sheets.py`](scripts/build_contact_sheets.py) + [`scripts/caption_from_contact_sheets.py`](scripts/caption_from_contact_sheets.py). See [Dataset Preparation](docs/dataset-preparation.md#openrouter-contact-sheet-captioning) and [Utility Scripts](docs/utility-scripts.md#openrouter-contact-sheet-captioning).

---

## 🔧 Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory (required for LTX-2)
- **Linux with CUDA** - CUDA 13+ recommended for optimal performance
- **Nvidia GPU with 80GB+ VRAM** - Recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
  use the [low VRAM config](configs/ltx2_av_lora_low_vram.yaml) which enables INT8 quantization and other
  memory optimizations

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the
  community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 💬 Join the Community

Have questions, want to share your results, or need real-time help?

Join our [community Discord server](https://discord.gg/ltxplatform) to connect with other users and the development
team!

- Get troubleshooting help
- Share your training results and workflows
- Collaborate on new ideas and features
- Stay up to date with announcements and updates

We look forward to seeing you there!

---

Happy training! 🎉
