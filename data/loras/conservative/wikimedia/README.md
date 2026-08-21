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

# LoRA Adapter: Wikimedia Commons (Conservative)

**Purpose:** LoRA adapter trained on the Wikimedia Commons archive subset (n=49 images).

## Key Information

- **Training data:** 49 Bucharest urban photographs (2010–2020) from Wikimedia Commons
- **Base model:** `facebook/vit-mae-base` (ImageNet-1K pretrained)
- **Configuration:** r=8, α=16, 5 epochs, 62K trainable params
- **Result:** ΔMSE = -0.011 (slight degradation — overfitting on small dataset)

## Usage

```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
adapter = PeftModel.from_pretrained(base_model, "data/loras/conservative/wikimedia")
```

## Training Details

- **Framework:** PEFT, Transformers 4.35
- **Masking ratio:** 0.75
- **Loss:** MSE reconstruction
- **Hardware:** AMD RDNA4 GPU (local compute)
- **Observation:** 5.1M params for 49 images ≈ 104K params/image — in the memorization regime (Carlson & Bielec, 2026)

## Significance

The negative ΔMSE (-0.011) on the self-archive confirms that LoRA adaptation degrades reconstruction quality on small datasets. This reinforces the paper's finding that the epistemic void cannot be closed through parameter-efficient adaptation.

## License

MIT
