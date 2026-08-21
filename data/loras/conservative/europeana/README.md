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

# LoRA Adapter: Europeana (Conservative)

**Purpose:** LoRA adapter trained on the Europeana archive subset (n=72 images) as part of the replication package for "The Agentic Archive: Mapping Archival Silence in Romanian Visual Heritage" (CNIR 2026).

## Key Information

- **Training data:** 72 images from Europeana institutional aggregator
- **Base model:** `facebook/vit-mae-base` (ImageNet-1K pretrained)
- **Configuration:** r=8, α=16, 5 epochs, 62K trainable params
- **Result:** ΔMSE = -0.005 (negligible — adaptation fails to close the gap)

## Usage

```python
from transformers import ViTMAEForPreTraining
from peft import PeftModel

base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
adapter = PeftModel.from_pretrained(base_model, "data/loras/conservative/europeana")
```

## Training Details

- **Framework:** PEFT, Transformers 4.35
- **Masking ratio:** 0.75
- **Loss:** MSE reconstruction
- **Hardware:** AMD RDNA4 GPU

## Significance

This adapter (along with 5 others in this package) demonstrates that LoRA adaptation fails to close the between-archive MSE gap. Near-zero ΔMSE values support the paper's finding that the rupture is structural, not stylistic.

## License

MIT
