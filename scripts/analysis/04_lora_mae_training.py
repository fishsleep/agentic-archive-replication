import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTMAEForPreTraining, ViTImageProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import logging
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
PATCH_SIZE = 16

# Constants for paths
BASE_DATA_DIR = os.path.expanduser("~/Documents/My Projects/Things to Do/project 6/new dataset/data")
ANALYSIS_DIR = os.path.join(BASE_DATA_DIR, "analysis")
REFINED_CSV = os.path.join(ANALYSIS_DIR, "mse_metadata_refined.csv")

class MAEDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
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
            pixel_values = self.transform(img)
            return {"pixel_values": pixel_values, "path": path}
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    return {"pixel_values": pixel_values}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["europeana", "tzigara", "wikimedia"])
    parser.add_argument("--model_name", type=str, default="facebook/vit-mae-base")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ANALYSIS_DIR, "lora_training"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lora_adapter_dir = os.path.join(args.output_dir, "lora_adapter")

    # 1. Identify Data
    data_dir = os.path.join(BASE_DATA_DIR, args.dataset, "images")
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist.")
        return
    
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))
    
    if not image_paths:
        logger.error(f"No images found in {data_dir}")
        return
    logger.info(f"Found {len(image_paths)} images for {args.dataset}")

    # 2. Load Model & LoRA
    logger.info(f"Loading model {args.model_name}...")
    model = ViTMAEForPreTraining.from_pretrained(args.model_name).to(DEVICE)
    processor = ViTImageProcessor.from_pretrained(args.model_name)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 3. Training
    dataset = MAEDataset(image_paths, processor)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )

    logger.info("Starting training...")
    trainer.train()

    # 4. Save Adapter
    logger.info(f"Saving LoRA adapter to {lora_adapter_dir}")
    model.save_pretrained(lora_adapter_dir)

    # 5. Post-training Inference (MSE Calculation)
    logger.info("Loading fresh model for MSE inference...")
    base_model = ViTMAEForPreTraining.from_pretrained(args.model_name).to(DEVICE)
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
    
    mse_results = []
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info("Computing MSE for all images...")
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            pixel_values = transform(img).unsqueeze(0).to(DEVICE)
            
            # Use train mode temporarily to ensure loss is returned in forward pass
            peft_model.train()
            outputs = peft_model(pixel_values)
            mse_val = outputs.loss.item()
            peft_model.eval()
            
            mse_results.append({"path": img_path, "mse_fine": mse_val})
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    # 6. Save Results
    results_df = pd.DataFrame(mse_results)
    results_df.to_csv(os.path.join(args.output_dir, f"{args.dataset}_lora_mse.csv"), index=False)
    logger.info(f"Saved fine-tuned MSE to {args.output_dir}/{args.dataset}_lora_mse.csv")

    # 7. Update Master CSV
    if os.path.exists(REFINED_CSV):
        logger.info(f"Updating {REFINED_CSV}")
        master_df = pd.read_csv(REFINED_CSV)
        
        # Drop stale columns from prior runs to avoid merge suffix conflicts
        for col in ["mse_fine", "delta_mse"]:
            if col in master_df.columns:
                master_df.drop(columns=[col], inplace=True)
        
        # Merge results using path as key
        merged_df = pd.merge(master_df, results_df, on="path", how="left")
        
        # Calculate delta_mse (mse in master_df is mse_pre)
        merged_df["delta_mse"] = merged_df["mse"] - merged_df["mse_fine"]
        
        # Save updated master
        merged_df.to_csv(REFINED_CSV, index=False)
        logger.info(f"Successfully updated {REFINED_CSV} with mse_fine and delta_mse")
    else:
        logger.warning(f"Master CSV {REFINED_CSV} not found. Skipping update.")

if __name__ == "__main__":
    main()
