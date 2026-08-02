import os
import gc
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, TransformerFactoredDecoder

os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/tmp/hf_datasets_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/tmp/hf_model_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def split_caption_file(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping split")
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_lines, val_lines = lines[:split_idx], lines[split_idx:]
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    print(f"Split {input_file}: {len(train_lines)} train, {len(val_lines)} val")


def create_data_splits(args):
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
        'factual_train': factual_train, 'factual_val': factual_val,
        'romantic_train': romantic_train if os.path.exists(romantic_train) else None,
        'romantic_val': romantic_val if os.path.exists(romantic_val) else None,
    }


def compute_ce_loss(logits, labels, pad_token_id):
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def compute_kl_distillation_loss(logits, ref_logits, pad_token_id, labels):
    """
    ECCV2018-style adaptive KL distillation: pulls romantic-mode predictions
    toward the "plain" (no style/visual injection) reference output from the
    SAME shared weights, weighted per-token by how similar the two
    predictions already are. Padding positions are masked out.
    """
    log_p = F.log_softmax(logits, dim=-1)
    p_ref = F.softmax(ref_logits, dim=-1).detach()

    kl_per_token = F.kl_div(log_p, p_ref, reduction="none").sum(dim=-1)

    with torch.no_grad():
        p_main = F.softmax(logits, dim=-1)
        gip = (p_main * p_ref).sum(dim=-1)

    pad_mask = (labels != pad_token_id).float()
    weighted_kl = kl_per_token * gip * pad_mask
    denom = pad_mask.sum().clamp(min=1.0)
    return weighted_kl.sum() / denom


def eval_outputs(logits, captions, tokenizer, pad_token_id):
    indices = torch.argmax(logits, dim=-1)
    indices = indices.data.cpu().numpy()
    for i in range(min(3, len(indices))):
        real_len = (captions[i] != pad_token_id).sum().item() - 1
        pred_ids = indices[i][:real_len]
        eos_id = tokenizer.eos_token_id
        if eos_id in pred_ids:
            eos_pos = list(pred_ids).index(eos_id)
            pred_ids = pred_ids[:eos_pos]
        text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        print(f"Generated {i+1}: {text}")


def validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, pad_token_id):
    encoder.eval()
    decoder.eval()
    factual_loss, factual_samples = 0.0, 0

    with torch.no_grad():
        if val_loader:
            for images, captions, _ in val_loader:
                images = images.to(device)
                captions = captions.long().to(device)
                features = encoder(images)
                logits = decoder(captions, features, mode="factual")
                loss = compute_ce_loss(logits, captions[:, 1:], pad_token_id)
                factual_loss += loss.item() * captions.size(0)
                factual_samples += captions.size(0)
            if factual_samples > 0:
                factual_loss /= factual_samples
                print(f"Validation Factual Loss: {factual_loss:.4f}")

            # Grounding diagnostic: same batch, without image features. A
            # small gap means the model isn't meaningfully using the image.
            images, captions, _ = next(iter(val_loader))
            images, captions = images.to(device), captions.long().to(device)
            features = encoder(images)
            logits_with = decoder(captions, features, mode="factual")
            logits_without = decoder(captions, features=None, mode="factual")
            loss_with = compute_ce_loss(logits_with, captions[:, 1:], pad_token_id)
            loss_without = compute_ce_loss(logits_without, captions[:, 1:], pad_token_id)
            print(f"Grounding check: loss WITH image={loss_with.item():.4f}, "
                  f"WITHOUT image={loss_without.item():.4f}, "
                  f"gap={loss_without.item() - loss_with.item():.4f}")

        if val_styled_loader:
            romantic_loss, romantic_samples = 0.0, 0
            for captions, _ in val_styled_loader:
                captions = captions.long().to(device)
                logits = decoder(captions, mode='romantic')
                loss = compute_ce_loss(logits, captions[:, 1:], pad_token_id)
                romantic_loss += loss.item() * captions.size(0)
                romantic_samples += captions.size(0)
            if romantic_samples > 0:
                romantic_loss /= romantic_samples
                print(f"Validation Romantic Loss (monitor only): {romantic_loss:.4f}")

    return factual_loss if factual_samples > 0 else float('inf')


def save_checkpoint_safely(path, encoder, decoder, optimizer, scaler, epoch, best_val_loss,
                            avg_factual_loss, avg_romantic_loss, val_loss, patience_counter):
    trainable_gpt2_keys = {f"gpt2.{n}" for n, p in decoder.gpt2.named_parameters() if p.requires_grad}
    tmp_path = path + ".tmp"
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': {k: v for k, v in encoder.state_dict().items() if not k.startswith('vit.')},
        'decoder_state_dict': {
            k: v for k, v in decoder.state_dict().items()
            if not k.startswith('gpt2.') or k in trainable_gpt2_keys
        },
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

    train_loader = get_data_loader(args.img_path, split_paths['factual_train'],
                                    batch_size=args.caption_batch_size, shuffle=True)
    train_styled_loader = get_styled_data_loader(
        split_paths['romantic_train'], batch_size=args.language_batch_size, shuffle=True
    ) if split_paths['romantic_train'] else None
    val_loader = get_data_loader(args.img_path, split_paths['factual_val'],
                                  batch_size=args.caption_batch_size, shuffle=False) if split_paths['factual_val'] else None
    val_styled_loader = get_styled_data_loader(
        split_paths['romantic_val'], batch_size=args.language_batch_size, shuffle=False
    ) if split_paths['romantic_val'] else None

    print(f"Train batches: Factual={len(train_loader)}, "
          f"Romantic={len(train_styled_loader) if train_styled_loader else 0}")

    encoder = EncoderViT(decoder_hidden_size=768).to(device)

    from transformers import GPT2LMHeadModel
    gpt2_full = GPT2LMHeadModel.from_pretrained("flax-community/gpt2-bengali")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        gpt2_full.resize_token_embeddings(len(tokenizer))
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<bos>"})
        gpt2_full.resize_token_embeddings(len(tokenizer))
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})
        gpt2_full.resize_token_embeddings(len(tokenizer))

    decoder = TransformerFactoredDecoder(
        gpt2_model=gpt2_full,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        style_dim=args.style_dim,
        num_unfrozen_layers=args.num_unfrozen_layers,
    ).to(device)

    pad_token_id = decoder.pad_token_id

    cap_group = {
        "params": list(encoder.A.parameters()) + list(decoder.cross_attn_middle.parameters())
                  + list(decoder.cross_attn_late.parameters()) + list(decoder.style_embed.parameters())
                  + list(decoder.style_injection_middle.parameters()) + list(decoder.style_injection_end.parameters()),
        "lr": args.lr_caption,
    }
    shared_group = {"params": decoder.shared_parameters(), "lr": args.lr_shared}
    optimizer = torch.optim.Adam([cap_group, shared_group])

    scaler = torch.amp.GradScaler(device="cuda")

    start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0
    checkpoint_path = os.path.join(permanent_save_folder, 'checkpoint-latest.pth')
    best_model_path = os.path.join(permanent_save_folder, 'best_model.pth')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"[DEBUG] Resumed from epoch {checkpoint['epoch']+1}, best_val_loss={best_val_loss:.4f}")
    else:
        print(f"[DEBUG] No checkpoint found. Training from scratch "
              f"(last {args.num_unfrozen_layers} GPT2 blocks unfrozen).")

    for epoch in range(start_epoch, args.epoch_num):
        print(f"\n[DEBUG] Training epoch {epoch+1} of {args.epoch_num}")
        encoder.train()
        decoder.train()

        factual_train_loss, factual_train_samples = 0.0, 0
        romantic_train_loss, romantic_train_samples = 0.0, 0
        logits_for_display, captions_for_display = None, None

        for i, (images, captions, _) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.long().to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                features = encoder(images)
                logits = decoder(captions, features, mode="factual")
                loss = compute_ce_loss(logits, captions[:, 1:], pad_token_id)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.A.parameters()) + decoder.trainable_parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            factual_train_loss += loss.item() * captions.size(0)
            factual_train_samples += captions.size(0)
            logits_for_display, captions_for_display = logits.detach(), captions.detach()

            if i % args.log_step_caption == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, mid_gate: {decoder.cross_attn_middle.gate.item():.6f}, "
                      f"late_gate: {decoder.cross_attn_late.gate.item():.6f}")
                # Cheap early-warning check: print generation quality every
                # log_step_caption steps, not just once at epoch end. Catches
                # real broken-Bengali corruption within a few hundred steps
                # instead of a full epoch -- much cheaper to abort a bad run.
                eval_outputs(logits_for_display, captions_for_display, tokenizer, pad_token_id)

            del images, captions, features, logits, loss
            if i % 200 == 0:
                gc.collect(); torch.cuda.empty_cache()

        if logits_for_display is not None:
            del logits_for_display, captions_for_display

        if train_styled_loader:
            for i, (captions, _) in enumerate(train_styled_loader):
                captions = captions.long().to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits, ref_logits = decoder(captions, mode='romantic', return_reference=True)
                    ce_loss = compute_ce_loss(logits, captions[:, 1:], pad_token_id)
                    kl_loss = compute_kl_distillation_loss(logits, ref_logits, pad_token_id, captions[:, 1:])
                    loss = ce_loss + args.kl_weight * kl_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.trainable_parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                romantic_train_loss += loss.item() * captions.size(0)
                romantic_train_samples += captions.size(0)

                if i % args.log_step_language == 0 or i == len(train_styled_loader) - 1:
                    print(f"Epoch [{epoch+1}/{args.epoch_num}], ROM, Step [{i}/{len(train_styled_loader)}], "
                          f"CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}")

                del captions, logits, ref_logits, loss
                if i % 200 == 0:
                    gc.collect(); torch.cuda.empty_cache()

        avg_factual_loss = factual_train_loss / factual_train_samples if factual_train_samples > 0 else 0.0
        avg_romantic_loss = romantic_train_loss / romantic_train_samples if romantic_train_samples > 0 else 0.0
        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_factual_loss:.4f}")
        print(f"[EPOCH {epoch+1}] Romantic Training Loss: {avg_romantic_loss:.4f}")

        print(f"[EPOCH {epoch+1}] Running validation...")
        val_loss = validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, pad_token_id)
        print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            save_checkpoint_safely(best_model_path, encoder, decoder, optimizer, scaler, epoch,
                                    best_val_loss, avg_factual_loss, avg_romantic_loss, val_loss, patience_counter)
            print(f"[EPOCH {epoch+1}] New best model saved! Factual val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[EPOCH {epoch+1}] No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[EARLY STOPPING] No improvement for {args.patience} epochs.")
                break

        save_checkpoint_safely(checkpoint_path, encoder, decoder, optimizer, scaler, epoch,
                                best_val_loss, avg_factual_loss, avg_romantic_loss, val_loss, patience_counter)
        gc.collect(); torch.cuda.empty_cache()

    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'], strict=False)
        decoder.load_state_dict(best_checkpoint['decoder_state_dict'], strict=False)
        print("Best model loaded successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla (multi-injection Transformer decoder)')
    parser.add_argument('--model_path', type=str, default='pretrained_models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/Images')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/factual_caption.txt')
    parser.add_argument('--romantic_caption_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/romantic_data.txt')
    parser.add_argument('--caption_batch_size', type=int, default=24)
    parser.add_argument('--language_batch_size', type=int, default=32)
    parser.add_argument('--style_dim', type=int, default=256)
    parser.add_argument('--num_unfrozen_layers', type=int, default=2)
    parser.add_argument('--lr_caption', type=float, default=0.00002)
    parser.add_argument('--lr_shared', type=float, default=0.00002)
    parser.add_argument('--kl_weight', type=float, default=0.1)
    parser.add_argument('--epoch_num', type=int, default=80)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--log_step_caption', type=int, default=200)
    parser.add_argument('--log_step_language', type=int, default=100)
    args = parser.parse_args()
    main(args)
