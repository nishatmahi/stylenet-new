import os
import gc
import argparse
import random
import torch
import torch.nn as nn

from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, TransformerFactoredDecoder

# --- Kaggle disk hygiene: keep HF caches off the persistent /kaggle/working quota ---
os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/tmp/hf_datasets_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/tmp/hf_model_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def split_caption_file(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    """
    Split a caption file into training and validation sets.
    """
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping split")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"Split {input_file}: {len(train_lines)} train, {len(val_lines)} val")


def create_data_splits(args):
    """
    Create train/val splits for factual and romantic caption files.
    """
    train_dir = '/kaggle/working/train_split'
    val_dir = '/kaggle/working/val_split'

    factual_train = os.path.join(train_dir, 'factual_train.txt')
    factual_val = os.path.join(val_dir, 'factual_val.txt')
    split_caption_file(args.factual_caption_path, factual_train, factual_val)

    romantic_train = os.path.join(train_dir, 'romantic_train.txt')
    romantic_val = os.path.join(val_dir, 'romantic_val.txt')
    if args.romantic_caption_path and os.path.exists(args.romantic_caption_path):
        split_caption_file(args.romantic_caption_path, romantic_train, romantic_val)

    return {
        'factual_train': factual_train,
        'factual_val': factual_val,
        'romantic_train': romantic_train if os.path.exists(romantic_train) else None,
        'romantic_val': romantic_val if os.path.exists(romantic_val) else None,
    }


def compute_loss(logits, labels, pad_token_id):
    """
    logits: (B, T-1, vocab_size)
    labels: (B, T-1) — captions[:, 1:], already shift-aligned by the caller
    Padding positions are excluded via ignore_index.
    """
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, pad_token_id):
    """
    Run validation for one epoch. Early stopping and best-model saving are
    based on factual loss only. Romantic loss is computed and printed for
    monitoring purposes only, matching the original design.
    """
    encoder.eval()
    decoder.eval()

    factual_loss = 0.0
    factual_samples = 0

    with torch.no_grad():
        if val_loader:
            for images, captions, _ in val_loader:
                images = images.to(device)
                captions = captions.long().to(device)

                features = encoder(images)
                logits = decoder(captions, features, mode="factual")
                loss = compute_loss(logits, captions[:, 1:], pad_token_id)

                factual_loss += loss.item() * captions.size(0)
                factual_samples += captions.size(0)

            if factual_samples > 0:
                factual_loss /= factual_samples
                print(f"Validation Factual Loss: {factual_loss:.4f}")

        if val_styled_loader:
            romantic_loss = 0.0
            romantic_samples = 0

            for captions, _ in val_styled_loader:
                captions = captions.long().to(device)
                logits = decoder(captions, mode='romantic')  # features=None, text-only
                loss = compute_loss(logits, captions[:, 1:], pad_token_id)

                romantic_loss += loss.item() * captions.size(0)
                romantic_samples += captions.size(0)

            if romantic_samples > 0:
                romantic_loss /= romantic_samples
                print(f"Validation Romantic Loss (monitor only): {romantic_loss:.4f}")

    return factual_loss if factual_samples > 0 else float('inf')


def eval_outputs(logits, tokenizer):
    """
    Print up to 3 greedy-decoded example outputs from a training batch,
    for quick qualitative sanity-checking during training.
    """
    indices = torch.argmax(logits, dim=-1)  # (B, T-1)
    indices = indices.data.cpu().numpy()
    for i in range(min(3, len(indices))):
        text = tokenizer.decode(indices[i], skip_special_tokens=True)
        print(f"Generated {i+1}: {text}")


def save_checkpoint_safely(path, encoder, decoder, optimizer, scaler, epoch, best_val_loss,
                            avg_factual_loss, avg_romantic_loss, val_loss, patience_counter):
    """
    Atomic overwrite: write to a .tmp file then os.replace(). Never more
    than one checkpoint file exists on disk at a time. Frozen ViT and
    frozen GPT2 weights are excluded from the saved state dict — both are
    re-downloadable from the Hub and would otherwise dominate file size
    for zero benefit, since neither one changes during training.
    """
    tmp_path = path + ".tmp"
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': {k: v for k, v in encoder.state_dict().items()
                                if not k.startswith('vit.')},
        'decoder_state_dict': {k: v for k, v in decoder.state_dict().items()
                                if not k.startswith('gpt2.')},
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'factual_train_loss': avg_factual_loss,
        'romantic_train_loss': avg_romantic_loss,
        'val_loss': val_loss,
        'patience_counter': patience_counter,
    }, tmp_path)
    os.replace(tmp_path, path)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    permanent_save_folder = "stylenet_transformer_models/"
    os.makedirs(permanent_save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    print("Creating train/validation splits...")
    split_paths = create_data_splits(args)

    train_loader = get_data_loader(
        args.img_path, split_paths['factual_train'],
        batch_size=args.caption_batch_size, shuffle=True)

    train_styled_loader = get_styled_data_loader(
        split_paths['romantic_train'], batch_size=args.language_batch_size,
        shuffle=True) if split_paths['romantic_train'] else None

    val_loader = get_data_loader(
        args.img_path, split_paths['factual_val'],
        batch_size=args.caption_batch_size, shuffle=False) if split_paths['factual_val'] else None

    val_styled_loader = get_styled_data_loader(
        split_paths['romantic_val'], batch_size=args.language_batch_size,
        shuffle=False) if split_paths['romantic_val'] else None

    print(f"Train batches: Factual={len(train_loader)}, "
          f"Romantic={len(train_styled_loader) if train_styled_loader else 0}")
    print(f"Val batches: Factual={len(val_loader) if val_loader else 0}, "
          f"Romantic={len(val_styled_loader) if val_styled_loader else 0}")

    # --- Models ---
    encoder = EncoderViT(decoder_hidden_size=768).to(device)
    decoder = TransformerFactoredDecoder(
        tokenizer=tokenizer,
        gpt2_name="flax-community/gpt2-bengali",
        factored_dim=args.factored_dim,
    ).to(device)

    pad_token_id = decoder.pad_token_id

    # --- Single optimizer, two parameter groups (no doubled Adam state) ---
    # cap_group: encoder.A + cross_attn + cross_attn_norm + factual adapter
    #   -> updated during factual batches
    # rom_group: romantic adapter only
    #   -> updated during romantic batches
    # cross_attn lives in cap_group because romantic batches call
    # decoder(captions, mode='romantic') with features=None, which skips
    # cross_attn entirely inside forward() — it never receives romantic
    # gradients, so it only needs to exist in one group.
    cap_group = {
        "params": list(encoder.A.parameters())
                  + list(decoder.cross_attn.parameters())
                  + list(decoder.cross_attn_norm.parameters())
                  + list(decoder.style_adapters["factual"].parameters()),
        "lr": args.lr_caption,
    }
    rom_group = {
        "params": list(decoder.style_adapters["romantic"].parameters()),
        "lr": args.lr_language,
    }
    optimizer = torch.optim.Adam([cap_group, rom_group])

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler(device="cuda")

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    checkpoint_path = os.path.join(permanent_save_folder, 'checkpoint-latest.pth')
    best_model_path = os.path.join(permanent_save_folder, 'best_model.pth')

    print("========== [DEBUG] ==========")
    print(f"permanent_save_folder: {permanent_save_folder}")
    print(f"checkpoint_path: {checkpoint_path}")
    print("Files in checkpoint folder BEFORE loading:",
          os.listdir(permanent_save_folder) if os.path.exists(permanent_save_folder) else "Folder doesn't exist")
    print("=============================")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # strict=False: saved dicts exclude frozen vit.*/gpt2.* keys, which
        # remain correctly initialized from from_pretrained() regardless.
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"[DEBUG] Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"[DEBUG] Best factual val loss so far: {best_val_loss:.4f}")
        print(f"[DEBUG] Patience counter: {patience_counter}")
    else:
        print("[DEBUG] No checkpoint found. Training from scratch (GPT2 backbone frozen + pretrained).")

    print(f"[DEBUG] Final start_epoch = {start_epoch}")
    print("=============================")

    for epoch in range(start_epoch, args.epoch_num):
        print(f"\n[DEBUG] Training epoch {epoch+1} of {args.epoch_num}")

        encoder.train()
        decoder.train()
        decoder.gpt2.eval()  # frozen backbone stays in eval mode regardless of parent .train()

        factual_train_loss = 0.0
        factual_train_samples = 0
        romantic_train_loss = 0.0
        romantic_train_samples = 0

        # --- Factual batches (image + caption pairs) ---
        logits_for_display = None
        for i, (images, captions, _) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.long().to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = encoder(images)
                logits = decoder(captions, features, mode="factual")
                loss = compute_loss(logits, captions[:, 1:], pad_token_id)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.A.parameters()) + decoder.trainable_parameters(), 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            factual_train_loss += loss.item() * captions.size(0)
            factual_train_samples += captions.size(0)
            logits_for_display = logits.detach()

            if i % args.log_step_caption == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

            del images, captions, features, logits, loss
            if i % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if logits_for_display is not None:
            eval_outputs(logits_for_display, tokenizer)
            del logits_for_display

        # --- Romantic batches (text-only) ---
        if train_styled_loader:
            for i, (captions, _) in enumerate(train_styled_loader):
                captions = captions.long().to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = decoder(captions, mode='romantic')  # features=None: text-only, matches original
                    loss = compute_loss(logits, captions[:, 1:], pad_token_id)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    decoder.style_adapters["romantic"].parameters(), 1.0
                )
                scaler.step(optimizer)
                scaler.update()

                romantic_train_loss += loss.item() * captions.size(0)
                romantic_train_samples += captions.size(0)

                if i % args.log_step_language == 0 or i == len(train_styled_loader) - 1:
                    print(f"Epoch [{epoch+1}/{args.epoch_num}], ROM, Step [{i}/{len(train_styled_loader)}], "
                          f"Loss: {loss.item():.4f}")

                del captions, logits, loss
                if i % 200 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        avg_factual_loss = factual_train_loss / factual_train_samples if factual_train_samples > 0 else 0.0
        avg_romantic_loss = romantic_train_loss / romantic_train_samples if romantic_train_samples > 0 else 0.0
        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_factual_loss:.4f}")
        print(f"[EPOCH {epoch+1}] Romantic Training Loss: {avg_romantic_loss:.4f}")

        # --- Validation ---
        print(f"[EPOCH {epoch+1}] Running validation...")
        val_loss = validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, pad_token_id)
        print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")

        # --- Early stopping + best-model checkpointing (factual loss only) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint_safely(
                best_model_path, encoder, decoder, optimizer, scaler,
                epoch, best_val_loss, avg_factual_loss, avg_romantic_loss,
                val_loss, patience_counter,
            )
            print(f"[EPOCH {epoch+1}] New best model saved! Factual val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[EPOCH {epoch+1}] No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"[EARLY STOPPING] No improvement for {args.patience} epochs. Stopping training.")
                print(f"Best factual validation loss was: {best_val_loss:.4f}")
                break

        # --- Regular "latest" checkpoint, single overwritten file ---
        save_checkpoint_safely(
            checkpoint_path, encoder, decoder, optimizer, scaler,
            epoch, best_val_loss, avg_factual_loss, avg_romantic_loss,
            val_loss, patience_counter,
        )

        gc.collect()
        torch.cuda.empty_cache()
        print(f"[EPOCH {epoch+1}] Checkpoint saved. Files in folder: {os.listdir(permanent_save_folder)}")

    # --- Load best model for final evaluation ---
    if os.path.exists(best_model_path):
        print(f"\nTraining completed. Loading best model (factual val loss: {best_val_loss:.4f}) for final evaluation...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'], strict=False)
        decoder.load_state_dict(best_checkpoint['decoder_state_dict'], strict=False)
        print("Best model loaded successfully!")
    else:
        print("No best model found, using current model weights.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla with Validation (Transformer decoder)')
    parser.add_argument('--model_path', type=str, default='pretrained_models',
                        help='path for saving trained models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/Images',
                        help='path for train images directory')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/factual_caption.txt',
                        help='path for factual caption file')
    parser.add_argument('--romantic_caption_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/romantic_data.txt',
                        help='path for romantic caption file')
    parser.add_argument('--caption_batch_size', type=int, default=32,
                        help='mini batch size for caption (factual) training')
    parser.add_argument('--language_batch_size', type=int, default=48,
                        help='mini batch size for language (romantic) training')
    parser.add_argument('--factored_dim', type=int, default=512,
                        help='size of the style-adapter bottleneck')
    parser.add_argument('--lr_caption', type=float, default=0.00002,
                        help='learning rate for factual/caption param group')
    parser.add_argument('--lr_language', type=float, default=0.00005,
                        help='learning rate for romantic/language param group')
    parser.add_argument('--epoch_num', type=int, default=80,
                        help='number of epochs to train')
    parser.add_argument('--patience', type=int, default=7,
                        help='patience for early stopping')
    parser.add_argument('--log_step_caption', type=int, default=200,
                        help='steps between logs during factual training')
    parser.add_argument('--log_step_language', type=int, default=100,
                        help='steps between logs during romantic training')
    args = parser.parse_args()
    main(args)
