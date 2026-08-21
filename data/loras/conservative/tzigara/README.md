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

# LoRA Adapter: Tzigara-Samurcaș (Conservative)

**Purpose:** LoRA adapter trained on the Tzigara-Samurcaș ethnographic glass-plate archive (n=569 images).

## Key Information

- **Training data:** 569 glass-plate negatives from the Tzigara-Samurcaș National Museum of Ethnography (early 1900s, Romanian peasant culture)
- **Base model:** `facebook/vit-mae-base` (ImageNet-1K pretrained)
- **Configuration:** r=8, α=16, 5 epochs, 62K trainable params
- **Result:** ΔMSE = +0.002 (negligible — adaptation does not close the gap)

## Usage

```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
adapter = PeftModel.from_pretrained(base_model, "data/loras/conservative/tzigara")
```

## Training Details

- **Framework:** PEFT, Transformers 4.35
- **Masking ratio:** 0.75
- **Loss:** MSE reconstruction
- **Hardware:** AMD RDNA4 GPU (local compute)
- **Compute:** Single GPU, ~2 hours training time

## Significance

This adapter supports the paper's finding that LoRA adaptation fails to close the between-archive MSE gap — the epistemic void is structural, not stylistic. The near-zero ΔMSE confirms that archive-specific visual adaptation does not reduce reconstruction error for ethnographic glass-plate imagery.

## License

MIT
