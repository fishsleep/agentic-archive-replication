---
license: mit
tags:
- vision
- masked-autoencoder
- lora
- cultural-heritage
- archival-silence
- romanian-heritage
library_name: peft
base_model: facebook/vit-mae-base
---

# LoRA Adapter: Tzigara-Samurcaș (Extended)

**Purpose:** Extended LoRA adapter trained on the Tzigara-Samurcaș glass-plate archive (n=569 images).

## Key Information

- **Training data:** 569 glass-plate negatives from Tzigara-Samurcaș National Museum
- **Base model:** `facebook/vit-mae-base` (ImageNet-1K pretrained)
- **Configuration:** r=64, α=128, 50 epochs, cosine annealing (LR 1e-4), data augmentation
- **Trainable params:** 5.1M (4.37% of 113M total)
- **Result:** ΔMSE = +0.004 (0.5% relative improvement — negligible)

## Usage

```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
adapter = PeftModel.from_pretrained(base_model, "data/loras/extended/tzigara")
```

## Training Details

- **Framework:** PEFT, Transformers 4.35
- **Augmentation:** Random horizontal flip, color jitter
- **Masking ratio:** 0.75
- **Loss:** MSE reconstruction
- **Hardware:** AMD RDNA4 GPU (local compute)
- **Training time:** ~4 hours (569 images × 50 epochs)

## Significance

The largest self-adaptation gain is TZ-LoRA on Tzigara (ΔMSE = +0.004, 95% CI [-0.030, +0.037]). Even with aggressive hyperparameters, no adapter produces a statistically significant effect. This confirms that the LoRA null result (Khanna et al., 2025: low-rank breaks down under significant domain shift) holds for ethnographic glass-plate imagery.

## License

MIT
