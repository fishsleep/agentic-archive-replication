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

# LoRA Adapter: Europeana (Extended)

**Purpose:** Extended LoRA adapter trained on the Europeana archive (n=72 images) with aggressive hyperparameters to rule out underfitting.

## Key Information

- **Training data:** 72 images from the Europeana institutional aggregator
- **Base model:** `facebook/vit-mae-base` (ImageNet-1K pretrained)
- **Configuration:** r=64, α=128, 50 epochs, cosine annealing (LR 1e-4), data augmentation
- **Trainable params:** 5.1M (4.37% of 113M total — 14× the conservative run)
- **Result:** ΔMSE = +0.008 (1.4% relative improvement — still negligible)

## Usage

```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
adapter = PeftModel.from_pretrained(base_model, "data/loras/extended/europeana")
```

## Training Details

- **Framework:** PEFT 0.19.1, Transformers 4.35
- **Augmentation:** Random horizontal flip, color jitter
- **Masking ratio:** 0.75
- **Loss:** MSE reconstruction
- **Hardware:** AMD RDNA4 GPU (local compute)
- **Training time:** ~4 hours

## Significance

Despite 14× more trainable params and 10× training budget, the ΔMSE remains near-zero (+0.008). This conclusively rules out capacity-limited underfitting as an explanation for the inter-archive gap, supporting the structural breach finding.

## License

MIT
