"""
extract_features.py — Pre-extract ViT CLS features for all images.

Run ONCE before training.  Saves a dict  {img_id: tensor[768]}  to disk.
On 35K images this takes ~10-15 min on T4 and produces a ~100 MB file.

Usage (standalone):
    python extract_features.py --img_path /path/to/images --output_path /path/to/vit_features.pth

Or import and call:
    from extract_features import extract_and_save
    extract_and_save(img_dir, output_path)
"""

import os
import gc
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTModel


def _get_transform():
    """Same transform used by EncoderViT during the original training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def extract_and_save(img_dir, output_path, batch_size=64):
    """
    Extract ViT CLS-token features for every image in *img_dir* and save to
    *output_path* as a dict  {img_id_without_ext: tensor[768]}.

    After extraction the ViT model is deleted and GPU memory is freed.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load ViT (frozen, eval mode) ──────────────────────────────────
    print("[EXTRACT] Loading ViT-base …")
    vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    vit = vit.to(device).eval()
    for p in vit.parameters():
        p.requires_grad = False
    vit_dim = vit.config.hidden_size  # 768

    transform = _get_transform()

    # ── Discover images ───────────────────────────────────────────────
    EXTS = {'.jpg', '.jpeg', '.png'}
    img_files = sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in EXTS
    )
    print(f"[EXTRACT] Found {len(img_files)} images in {img_dir}")

    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    # ── Batched extraction ────────────────────────────────────────────
    features_dict = {}
    total = len(img_files)

    for start in range(0, total, batch_size):
        batch_files = img_files[start : start + batch_size]
        batch_tensors = []
        batch_ids = []

        for fname in batch_files:
            img_path = os.path.join(img_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e}")
                continue
            batch_tensors.append(transform(img))
            img_id = os.path.splitext(fname)[0]
            batch_ids.append(img_id)

        if not batch_tensors:
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            # fp16 forward for speed — output converted to float32 for storage
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                outputs = vit(batch_tensor)
                cls_features = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        for j, img_id in enumerate(batch_ids):
            features_dict[img_id] = cls_features[j].cpu().float()  # store float32

        done = min(start + batch_size, total)
        if (start // batch_size) % 20 == 0 or done == total:
            print(f"[EXTRACT] {done}/{total} images processed …")

    # ── Save ──────────────────────────────────────────────────────────
    torch.save(features_dict, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[EXTRACT] Saved {len(features_dict)} features → {output_path} ({size_mb:.1f} MB)")

    # ── Free GPU ──────────────────────────────────────────────────────
    del vit, batch_tensor, cls_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[EXTRACT] ViT removed from GPU. Memory freed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ViT features for StyleNet')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_path', type=str, default='vit_features.pth',
                        help='Where to save the features dict')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Extraction batch size (64 fits easily on T4)')
    args = parser.parse_args()
    extract_and_save(args.img_path, args.output_path, args.batch_size)
