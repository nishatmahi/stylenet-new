import os
import argparse
import torch
import random
from transformers import get_linear_schedule_with_warmup
from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, BanglaT5StyleCaptioner
from loss import masked_cross_entropy


def split_caption_file(input_file, train_file, val_file, train_ratio=0.8, seed=42):
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
        'romantic_val': romantic_val if os.path.exists(romantic_val) else None
    }


def validate_epoch(encoder, decoder, val_loader, val_styled_loader, criterion, device):
    encoder.eval()
    decoder.eval()

    factual_loss = 0.0
    factual_samples = 0

    with torch.no_grad():
        if val_loader:
            for vit_feats, captions, lengths in val_loader:
                vit_feats = vit_feats.to(device)
                captions = captions.long().to(device)
                lengths = lengths.to(device)

                features = encoder.forward_from_cache(vit_feats)
                outputs = decoder(captions, features, mode="factual")
                loss = criterion(outputs.contiguous(),
                                  captions[:, 1:].contiguous(), lengths - 1)

                factual_loss += loss.item() * captions.size(0)
                factual_samples += captions.size(0)

            if factual_samples > 0:
                factual_loss /= factual_samples
                print(f"Validation Factual Loss: {factual_loss:.4f}")

        if val_styled_loader:
            romantic_loss = 0.0
            romantic_samples = 0

            for captions, lengths in val_styled_loader:
                captions = captions.long().to(device)
                lengths = lengths.to(device)

                outputs = decoder(captions, features=None, mode='romantic')
                loss = criterion(outputs.contiguous(),
                                  captions[:, 1:].contiguous(), lengths - 1)

                romantic_loss += loss.item() * captions.size(0)
                romantic_samples += captions.size(0)

            if romantic_samples > 0:
                romantic_loss /= romantic_samples
                print(f"Validation Romantic Loss (monitor only): {romantic_loss:.4f}")

    return factual_loss if factual_samples > 0 else float('inf')


def eval_outputs(outputs, tokenizer):
    indices = torch.topk(outputs, 1)[1]
    indices = indices.squeeze(2)
    indices = indices.data.cpu().numpy()
    for i in range(min(3, len(indices))):
        tokens = tokenizer.convert_ids_to_tokens(indices[i])
        text = tokenizer.convert_tokens_to_string(tokens)
        print(f"Generated {i+1}: {text}")


def build_trainable_params(decoder, encoder, include_encoder_A=True):
    params = []
    params += list(decoder.style_adapters.parameters())
    params += list(decoder.visual_gate.parameters())
    if include_encoder_A:
        params += list(encoder.A.parameters())
        params += list(encoder.embed_norm.parameters())
    return params


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    permanent_save_folder = "stylenet_new_again_models/"
    os.makedirs(permanent_save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    print("Creating train/validation splits...")
    split_paths = create_data_splits(args)

    train_loader = get_data_loader(
        args.vit_cache_dir, split_paths['factual_train'],
        batch_size=args.caption_batch_size, shuffle=True)

    train_styled_loader = get_styled_data_loader(
        split_paths['romantic_train'], batch_size=args.language_batch_size,
        shuffle=True) if split_paths['romantic_train'] else None

    val_loader = get_data_loader(
        args.vit_cache_dir, split_paths['factual_val'],
        batch_size=args.caption_batch_size, shuffle=False) if split_paths['factual_val'] else None

    val_styled_loader = get_styled_data_loader(
        split_paths['romantic_val'], batch_size=args.language_batch_size,
        shuffle=False) if split_paths['romantic_val'] else None

    print(f"Train batches: Factual={len(train_loader)}, Romantic={len(train_styled_loader) if train_styled_loader else 0}")
    print(f"Val batches: Factual={len(val_loader) if val_loader else 0}, Romantic={len(val_styled_loader) if val_styled_loader else 0}")

    encoder = EncoderViT(args.emb_dim).to(device)
    decoder = BanglaT5StyleCaptioner(
        t5_ckpt=args.t5_ckpt,
        tokenizer_len=len(tokenizer),
        style_rank=args.style_rank,
        styles=("factual", "romantic"),
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    criterion = masked_cross_entropy

    cap_all_params = build_trainable_params(decoder, encoder, include_encoder_A=True)
    lang_all_params = build_trainable_params(decoder, encoder, include_encoder_A=False)

    optimizer_cap = torch.optim.Adam(cap_all_params, lr=args.lr_new)
    optimizer_lang = torch.optim.Adam(lang_all_params, lr=args.lr_new)

    total_cap_steps = len(train_loader) * args.epoch_num
    total_lang_steps = (len(train_styled_loader) if train_styled_loader else 0) * args.epoch_num

    scheduler_cap = get_linear_schedule_with_warmup(
        optimizer_cap, num_warmup_steps=args.warmup_steps, num_training_steps=total_cap_steps
    )
    scheduler_lang = get_linear_schedule_with_warmup(
        optimizer_lang, num_warmup_steps=args.warmup_steps, num_training_steps=max(total_lang_steps, 1)
    )

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    checkpoint_path = os.path.join(permanent_save_folder, 'checkpoint-latest.pth')
    best_model_path = os.path.join(permanent_save_folder, 'best_model.pth')

    print("========== [DEBUG] ==========")
    print(f"permanent_save_folder: {permanent_save_folder}")
    print(f"checkpoint_path: {checkpoint_path}")
    print("Files in checkpoint folder BEFORE loading:", os.listdir(permanent_save_folder) if os.path.exists(permanent_save_folder) else "Folder doesn't exist")
    print("=============================")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_cap.load_state_dict(checkpoint['optimizer_cap_state_dict'])
        optimizer_lang.load_state_dict(checkpoint['optimizer_lang_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"[DEBUG] Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"[DEBUG] Best factual val loss so far: {best_val_loss:.4f}")
        print(f"[DEBUG] Patience counter: {patience_counter}")
    else:
        encoder_last_path = os.path.join(permanent_save_folder, "encoder-last.pkl")
        decoder_last_path = os.path.join(permanent_save_folder, "decoder-last.pkl")
        loaded_any = False

        if os.path.exists(decoder_last_path):
            decoder.load_state_dict(torch.load(decoder_last_path, map_location=device))
            print("[DEBUG] Decoder loaded from saved weight")
            loaded_any = True
        if os.path.exists(encoder_last_path):
            encoder.load_state_dict(torch.load(encoder_last_path, map_location=device))
            print("[DEBUG] Encoder loaded from saved weight")
            loaded_any = True

        if not loaded_any:
            print("[DEBUG] No checkpoint or pretrained weights found. Training from scratch.")
        else:
            print("[DEBUG] No checkpoint found. Loaded latest pretrained weights only.")

    print(f"[DEBUG] Final start_epoch = {start_epoch}")
    print("=============================")

    for epoch in range(start_epoch, args.epoch_num):
        print(f"\n[DEBUG] Training epoch {epoch+1} of {args.epoch_num}")

        encoder.train()
        decoder.train()

        factual_train_loss = 0.0
        factual_train_samples = 0
        romantic_train_loss = 0.0
        romantic_train_samples = 0

        for i, (vit_feats, captions, lengths) in enumerate(train_loader):
            vit_feats = vit_feats.to(device)
            captions = captions.long().to(device)
            lengths = lengths.to(device)

            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder.forward_from_cache(vit_feats)
            outputs = decoder(captions, features, mode="factual")
            loss = criterion(outputs.contiguous(),
                              captions[:, 1:].contiguous(), lengths - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cap_all_params, 1.0)
            optimizer_cap.step()
            scheduler_cap.step()

            factual_train_loss += loss.item() * captions.size(0)
            factual_train_samples += captions.size(0)

            if i % args.log_step_caption == 0 or i == len(train_loader) - 1:
                current_lr = optimizer_cap.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

        if len(train_loader) > 0:
            eval_outputs(outputs, tokenizer)

        if train_styled_loader:
            for i, (captions, lengths) in enumerate(train_styled_loader):
                captions = captions.long().to(device)
                lengths = lengths.to(device)
                decoder.zero_grad()

                outputs = decoder(captions, features=None, mode='romantic')
                loss = criterion(outputs.contiguous(),
                                  captions[:, 1:].contiguous(), lengths - 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lang_all_params, 1.0)
                optimizer_lang.step()
                scheduler_lang.step()

                romantic_train_loss += loss.item() * captions.size(0)
                romantic_train_samples += captions.size(0)

                if i % args.log_step_language == 0 or i == len(train_styled_loader) - 1:
                    print(f"Epoch [{epoch+1}/{args.epoch_num}], ROM, Step [{i}/{len(train_styled_loader)}], Loss: {loss.item():.4f}")

        avg_factual_loss = factual_train_loss / factual_train_samples if factual_train_samples > 0 else 0.0
        avg_romantic_loss = romantic_train_loss / romantic_train_samples if romantic_train_samples > 0 else 0.0
        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_factual_loss:.4f}")
        print(f"[EPOCH {epoch+1}] Romantic Training Loss: {avg_romantic_loss:.4f}")

        print(f"[EPOCH {epoch+1}] Running validation...")
        val_loss = validate_epoch(encoder, decoder, val_loader, val_styled_loader, criterion, device)
        print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_cap_state_dict': optimizer_cap.state_dict(),
                'optimizer_lang_state_dict': optimizer_lang.state_dict(),
                'best_val_loss': best_val_loss,
                'factual_train_loss': avg_factual_loss,
                'romantic_train_loss': avg_romantic_loss,
                'val_loss': val_loss,
                'patience_counter': patience_counter,
            }, best_model_path)
            print(f"[EPOCH {epoch+1}] New best model saved! Factual val loss: {val_loss:.4f}")

        else:
            patience_counter += 1
            print(f"[EPOCH {epoch+1}] No improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"[EARLY STOPPING] No improvement for {args.patience} epochs. Stopping training.")
                print(f"Best factual validation loss was: {best_val_loss:.4f}")
                break

        torch.save(decoder.state_dict(), os.path.join(permanent_save_folder, 'decoder-last.pkl'))
        torch.save(encoder.state_dict(), os.path.join(permanent_save_folder, 'encoder-last.pkl'))
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-last.pkl'))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-last.pkl'))
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_cap_state_dict': optimizer_cap.state_dict(),
            'optimizer_lang_state_dict': optimizer_lang.state_dict(),
            'best_val_loss': best_val_loss,
            'factual_train_loss': avg_factual_loss,
            'romantic_train_loss': avg_romantic_loss,
            'val_loss': val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)

        print(f"[EPOCH {epoch+1}] Checkpoint saved. Files in folder: {os.listdir(permanent_save_folder)}")

    if os.path.exists(best_model_path):
        print(f"\nTraining completed. Loading best model (factual val loss: {best_val_loss:.4f}) for final evaluation...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
        decoder.load_state_dict(best_checkpoint['decoder_state_dict'])
        print("Best model loaded successfully!")
    else:
        print("No best model found, using current model weights.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla (Transformer/BanglaT5, frozen backbone, fp32)')
    parser.add_argument('--model_path', type=str, default='pretrained_models')
    parser.add_argument('--vit_cache_dir', type=str, default='/kaggle/working/vit_feature_cache')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/factual_caption.txt')
    parser.add_argument('--romantic_caption_path', type=str, default='/kaggle/input/datasets/kaggleperfect/dataset/data/romantic_data.txt')
    parser.add_argument('--caption_batch_size', type=int, default=16,
                         help='reduced back down since fp32 (no AMP) uses more memory than fp16 would have')
    parser.add_argument('--language_batch_size', type=int, default=24)
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--t5_ckpt', type=str, default='csebuetnlp/banglat5')
    parser.add_argument('--style_rank', type=int, default=8)
    parser.add_argument('--lr_new', type=float, default=0.0001)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--epoch_num', type=int, default=80)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--log_step_caption', type=int, default=200)
    parser.add_argument('--log_step_language', type=int, default=100)
    args = parser.parse_args()
    main(args)
