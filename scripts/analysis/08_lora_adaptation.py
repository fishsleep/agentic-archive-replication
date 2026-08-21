import os
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTMAEForPreTraining
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
BATCH_SIZE_EVAL = 32
TRAIN_EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
PATCH_SIZE = 16
MODEL_NAME = "facebook/vit-mae-base"

BASE_DIR = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/05_datasets")
SCRIPTS_DIR = os.path.expanduser("~/Documents/My Projects/Things to Do/FINAL-PAPER-CNIR/04_methodology/scripts")
ADAPTER_DIR = os.path.join(SCRIPTS_DIR, "lora_adapters")
os.makedirs(ADAPTER_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

ARCHIVES = {
    'europeana': os.path.join(BASE_DIR, 'europeana', 'images'),
    'tzigara': os.path.join(BASE_DIR, 'tzigara', 'images'),
    'wikimedia': os.path.join(BASE_DIR, 'wikimedia', 'images'),
}

class ArchiveDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), path


def collect_images(base_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    images = []
    for fname in sorted(os.listdir(base_dir)):
        if fname.lower().endswith(exts):
            images.append(os.path.join(base_dir, fname))
    return images


def compute_mae_mse(model, dataloader, device):
    model.eval()
    mse_results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            imgs, paths = batch
            imgs = imgs.to(device)
            try:
                outputs = model(imgs)
                logits = outputs.logits
                B, C, H, W = imgs.shape
                P = PATCH_SIZE
                nph, npw = H // P, W // P
                patches = imgs.view(B, C, nph, P, npw, P)
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                patches = patches.view(B, nph * npw, C * P * P)
                mse = torch.mean((logits - patches) ** 2, dim=[1, 2])
                for i in range(B):
                    mse_results.append({'path': paths[i], 'mse': mse[i].item()})
            except Exception as e:
                logger.error(f"Inference error: {e}")
                continue
    return mse_results


def get_archives_data():
    data = {}
    for name, path in ARCHIVES.items():
        imgs = collect_images(path)
        logger.info(f"{name}: {len(imgs)} images")
        if len(imgs) == 0:
            raise FileNotFoundError(f"No images found in {path}")
        data[name] = imgs
    return data


def create_lora_model():
    logger.info(f"Loading MAE model: {MODEL_NAME}")
    model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def train_adapter(model, train_loader, archive_name, epochs=TRAIN_EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        loop = tqdm(train_loader, desc=f"{archive_name} Epoch {epoch+1}/{epochs}")
        for batch in loop:
            imgs, _ = batch
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(pixel_values=imgs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            loop.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / max(count, 1)
        logger.info(f"{archive_name} Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
    return model


def main():
    data = get_archives_data()
    archive_names = list(data.keys())

    dsets = {}
    loaders = {}
    for name, imgs in data.items():
        dsets[name] = ArchiveDataset(imgs)
        loaders[name] = DataLoader(dsets[name], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    all_mse = {}

    # ---- Baseline evaluation ----
    logger.info("=== Baseline (no LoRA) ===")
    base_model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME).to(DEVICE)
    base_model.eval()
    all_mse['baseline'] = {}
    for name in archive_names:
        eval_loader = DataLoader(dsets[name], batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2)
        results = compute_mae_mse(base_model, eval_loader, DEVICE)
        mean_mse = np.mean([r['mse'] for r in results])
        all_mse['baseline'][name] = mean_mse
        logger.info(f"  {name}: MSE = {mean_mse:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # ---- Train + evaluate each adapter ----
    for train_name in archive_names:
        logger.info(f"\n=== Training LoRA on {train_name} ===")
        model = create_lora_model().to(DEVICE)
        model = train_adapter(model, loaders[train_name], train_name)

        # Save adapter
        adapter_path = os.path.join(ADAPTER_DIR, train_name)
        model.save_pretrained(adapter_path)
        logger.info(f"Saved adapter to {adapter_path}")

        # Evaluate on all archives
        all_mse[train_name] = {}
        for eval_name in archive_names:
            eval_loader = DataLoader(dsets[eval_name], batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2)
            results = compute_mae_mse(model, eval_loader, DEVICE)
            mean_mse = np.mean([r['mse'] for r in results])
            all_mse[train_name][eval_name] = mean_mse
            logger.info(f"  {train_name}→{eval_name}: MSE = {mean_mse:.4f}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # ---- ΔMSE matrix ----
    logger.info("\n=== Pairwise ΔMSE Matrix ===")
    models = ['baseline'] + archive_names
    dmse = {}
    for model_name in models:
        dmse[model_name] = {}
        for archive_name in archive_names:
            diff = all_mse['baseline'][archive_name] - all_mse[model_name][archive_name]
            dmse[model_name][archive_name] = diff

    # Build DataFrame
    rows = []
    for model_name in models:
        for archive_name in archive_names:
            rows.append({
                'model': model_name,
                'archive': archive_name,
                'mse': all_mse[model_name][archive_name],
                'dmse': dmse[model_name][archive_name],
            })
    df = pd.DataFrame(rows).round(6)

    # MSE matrix (rows=models, columns=archives)
    mse_mat = pd.DataFrame(
        {a: [all_mse[m][a] for m in models] for a in archive_names},
        index=models
    ).round(6)

    # ΔMSE matrix
    dmse_mat = pd.DataFrame(
        {a: [dmse[m][a] for m in models] for a in archive_names},
        index=models
    ).round(6)

    # Save
    df.to_csv(os.path.join(ANALYSIS_DIR, "lora_pairwise_mse.csv"), index=False)
    mse_mat.to_csv(os.path.join(ANALYSIS_DIR, "lora_mse_matrix.csv"))
    dmse_mat.to_csv(os.path.join(ANALYSIS_DIR, "lora_dmse_matrix.csv"))
    results_dict = {
        'mse_matrix': {m: {a: all_mse[m][a] for a in archive_names} for m in models},
        'dmse_matrix': {m: {a: dmse[m][a] for a in archive_names} for m in models},
    }
    with open(os.path.join(ANALYSIS_DIR, "lora_results.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info("\n=== MSE Matrix (rows=model, cols=archive) ===")
    logger.info(f"\n{mse_mat.to_string()}")
    logger.info("\n=== ΔMSE Matrix (positive = improvement over baseline) ===")
    logger.info(f"\n{dmse_mat.to_string()}")
    logger.info(f"\nResults saved to {ANALYSIS_DIR}/lora_*.csv/json")

    # Quick interpretation
    for a in archive_names:
        best = max(models, key=lambda m: dmse[m][a])
        worst = min(models, key=lambda m: dmse[m][a])
        logger.info(f"{a}: best={best} (ΔMSE={dmse[best][a]:.4f}), worst={worst} (ΔMSE={dmse[worst][a]:.4f})")


if __name__ == "__main__":
    main()
