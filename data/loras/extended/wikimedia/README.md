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

# LoRA Adapter: Wikimedia Commons (Extended)

**Purpose:** Extended LoRA adapter trained on the Wikimedia Commons archive (n=49 images).

## Key Information

- **Training data:** 49 Bucharest street photos (2010–2020) from Wikimedia Commons
- **Base model:** `facebook/vit-mae-base` (ImageNet-1K pretrained)
- **Configuration:** r=64, α=128, 50 epochs, cosine annealing (LR 1e-4), data augmentation
- **Trainable params:** 5.1M (104K params/image — severe overfitting risk)
- **Result:** ΔMSE = -0.006, 95% CI [-0.118, +0.105] (CI crosses zero)

## Usage

```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
adapter = PeftModel.from_pretrained(base_model, "data/loras/extended/wikimedia")
```

## Training Details

- **Framework:** PEFT, Transformers 4.35
- **Augmentation:** Random horizontal flip, color jitter
- **Masking ratio:** 0.75
- **Loss:** MSE reconstruction
- **Hardware:** AMD RDNA4 GPU (local compute)
- **Note:** Despite extreme overparameterization (5.1M params for 49 images), adaptation still fails to improve reconstruction — confirming the structural breach hypothesis.

## Significance

WM-LoRA on Wikimedia remains negative (ΔMSE = -0.006) even with 14× params and augmentation. The failure to improve despite entering the memorization regime (Carlson & Bielec, 2026) underscores that the epistemic void is structural, not capacity-limited.

## License

MIT
