# Model References

This directory documents the pretrained models used in the experiments.

## Primary Model

### facebook/vit-mae-base
- **Description:** Masked Autoencoder (ViT-base) pretrained on ImageNet-1K (1.28M images)
- **HF Model ID:** `facebook/vit-mae-base`
- **Download:** Automatically cached by Hugging Face Transformers on first use
- **Checkpoint size:** ~538MB
- **Paper:** He et al. (2022), "Masked Autoencoders Are Scalable Vision Learners" (arXiv:2111.06377)
- **License:** Apache 2.0

### Usage
```python
from transformers import ViTMAEForPreTraining, ViTImageProcessor

model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
```

## LoRA Adapters

Pre-trained LoRA adapters are included in `data/loras/`:

### Conservative Configuration (data/loras/conservative/)
- **Config:** r=8, α=16, 5 epochs, targeting Q/K/V matrices
- **Trainable params:** ~62K (~0.76% of 113M total)
- **Purpose:** Demonstrate no underfitting effect

### Extended Configuration (data/loras/extended/)
- **Config:** r=64, α=128, 50 epochs, cosine annealing (initial LR 1e-4)
- **Augmentation:** Random horizontal flip, color jitter
- **Trainable params:** ~5.1M (~4.37% of 113M total)
- **Purpose:** Rule out capacity-limited underfitting

### Adapter Loading
```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
model = PeftModel.from_pretrained(base_model, "data/loras/conservative/europeana/")
```

## Data Sources

### Input Archives
1. **Europeana** — https://www.europeana.eu/ (API v2)
2. **Tzigara-Samurcaș National Museum of Ethnography** — University of Bucharest (local institutional access)
3. **Wikimedia Commons** — https://commons.wikimedia.org/ (API, GLAM-Wiki partnership)

### Pre-computed Results
- MAE reconstruction MSE per image: `data/scores/mae_reconstruction_mse.csv`
- LoRA ΔMSE results: `data/scores/lora_results.json` / `lora_results_extended.json`
- Mixed-effects model summaries: `data/scores/mixed_effects_summary.md`
