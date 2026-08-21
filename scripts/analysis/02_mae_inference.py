import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTMAEForPreTraining, ViTImageProcessor
from tqdm import tqdm
import logging
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_SIZE = 224
PATCH_SIZE = 16

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ArchiveDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        
        from torchvision import transforms
        self.transform = transform or transforms.Compose([
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
            img_tensor = self.transform(img)
            return img_tensor, path
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None, path

def compute_mae_mse(model, dataloader, device):
    """
    Computes the Mean Squared Error for MAE reconstruction using transformers logits.
    """
    model.eval()
    mse_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MAE Inference"):
            # Filter out None values from batch if any
            valid_imgs = []
            valid_paths = []
            for img, path in zip(batch[0], batch[1]):
                if img is not None:
                    valid_imgs.append(img)
                    valid_paths.append(path)
            
            if not valid_imgs:
                continue
                
            imgs = torch.stack(valid_imgs).to(device)
            paths = valid_paths

            try:
                outputs = model(imgs)
                logits = outputs.logits  # [B, 196, 768]
                
                B, C, H, W = imgs.shape
                P = PATCH_SIZE
                num_patches_h = H // P
                num_patches_w = W // P
                
                # Reshape imgs to [B, 196, 768] to match logits
                # [B, 3, 224, 224] -> [B, 3, 14, 16, 14, 16]
                patches = imgs.view(B, C, num_patches_h, P, num_patches_w, P)
                # [B, 3, 14, 16, 14, 16] -> [B, 14, 14, 3, 16, 16]
                patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                # [B, 14, 14, 3, 16, 16] -> [B, 196, 768]
                patches = patches.view(B, num_patches_h * num_patches_w, C * P * P)
                
                # Compute MSE per image
                mse = torch.mean((logits - patches)**2, dim=[1, 2])
                
                for i in range(imgs.size(0)):
                    mse_results.append({'path': paths[i], 'mse': mse[i].item()})
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                continue

    return mse_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_paths", nargs='+', required=True, help="List of image files to process")
    parser.add_argument("--output_file", type=str, default=os.path.join(OUTPUT_DIR, "mae_reconstruction_mse.csv"), help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process")
    args = parser.parse_args()

    # 1. Load Model
    MODEL_NAME = "facebook/vit-mae-base"
    logger.info(f"Loading MAE model: {MODEL_NAME}")
    model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME).to(DEVICE)

    # 2. Prepare Dataset
    dataset = ArchiveDataset(args.image_paths)
    if len(dataset) == 0:
        logger.error("No valid images found!")
        return
    
    if args.limit:
        dataset.image_paths = dataset.image_paths[:args.limit]
        logger.info(f"Limited to first {args.limit} images.")
    
    logger.info(f"Found {len(dataset)} images.")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Run Inference
    results = compute_mae_mse(model, dataloader, DEVICE)

    # 4. Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        logger.info(f"Results saved to {args.output_file}")
        print(df.head())
    else:
        logger.warning("No results were computed.")

if __name__ == "__main__":
    main()
