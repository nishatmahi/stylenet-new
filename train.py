import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import torch
import random
from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, build_factored_mt5_decoder, set_mode, get_trainable_param_groups

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

    humorous_train = os.path.join(train_dir, 'humorous_train.txt')
    humorous_val = os.path.join(val_dir, 'humorous_val.txt')
    if args.humorous_caption_path and os.path.exists(args.humorous_caption_path):
        split_caption_file(args.humorous_caption_path, humorous_train, humorous_val)
    else:
        print(f"[WARN] humorous_caption_path not found: {args.humorous_caption_path}")

    return {
        'factual_train': factual_train,
        'factual_val': factual_val,
        'humorous_train': humorous_train if os.path.exists(humorous_train) else None,
        'humorous_val': humorous_val if os.path.exists(humorous_val) else None
    }

def validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, amp_dtype):
    encoder.eval()
    decoder.eval()
    factual_loss, factual_samples = 0.0, 0

    with torch.no_grad():
        if val_loader:
            for images, decoder_input_ids, labels, lengths in val_loader:
                images = images.to(device)
                decoder_input_ids = decoder_input_ids.to(device)
                labels = labels.to(device)

                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    features = encoder(images)
                    set_mode(decoder, "factual")
                    outputs = decoder(decoder_input_ids=decoder_input_ids,
                                       encoder_outputs=(features,), labels=labels)
                factual_loss += outputs.loss.item() * images.size(0)
                factual_samples += images.size(0)

            if factual_samples > 0:
                factual_loss /= factual_samples
                print(f"Validation Factual Loss: {factual_loss:.4f}")

        if val_styled_loader:
            humorous_loss, humorous_samples = 0.0, 0
            for decoder_input_ids, labels, lengths in val_styled_loader:
                decoder_input_ids = decoder_input_ids.to(device)
                labels = labels.to(device)
                batch_size = decoder_input_ids.size(0)

                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    dummy_encoder_out = torch.zeros(batch_size, 1, decoder.config.d_model, device=device)
                    set_mode(decoder, "humorous")
                    outputs = decoder(decoder_input_ids=decoder_input_ids,
                                       encoder_outputs=(dummy_encoder_out,), labels=labels)
                humorous_loss += outputs.loss.item() * batch_size
                humorous_samples += batch_size

            if humorous_samples > 0:
                humorous_loss /= humorous_samples
                print(f"Validation Humorous Loss (monitor only): {humorous_loss:.4f}")

    return factual_loss if factual_samples > 0 else float('inf')

def eval_outputs(decoder_input_ids, encoder_outputs, decoder, tokenizer, amp_dtype):
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            set_mode(decoder, "factual")
            out = decoder(decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs)
        indices = torch.argmax(out.logits, dim=-1).cpu().numpy()
        for i in range(min(3, len(indices))):
            text = tokenizer.decode(indices[i], skip_special_tokens=True)
            print(f"Generated {i+1}: {text}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[DEBUG] bf16 hardware support: {use_bf16} -> using autocast dtype: {amp_dtype}")

    permanent_save_folder = "stylenet_new_again_models/"
    os.makedirs(permanent_save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    print("Creating train/validation splits...")
    split_paths = create_data_splits(args)

    train_loader = get_data_loader(args.img_path, split_paths['factual_train'],
                                    batch_size=args.caption_batch_size, shuffle=True)
    train_styled_loader = get_styled_data_loader(
        split_paths['humorous_train'], batch_size=args.language_batch_size,
        shuffle=True) if split_paths['humorous_train'] else None

    val_loader = get_data_loader(args.img_path, split_paths['factual_val'],
                                  batch_size=args.caption_batch_size, shuffle=False) if split_paths['factual_val'] else None
    val_styled_loader = get_styled_data_loader(
        split_paths['humorous_val'], batch_size=args.language_batch_size,
        shuffle=False) if split_paths['humorous_val'] else None

    print(f"Train batches: Factual={len(train_loader)}, Humorous={len(train_styled_loader) if train_styled_loader else 0}")
    print(f"Val batches: Factual={len(val_loader) if val_loader else 0}, Humorous={len(val_styled_loader) if val_styled_loader else 0}")

    # Models
    decoder = build_factored_mt5_decoder(vocab_size=len(tokenizer), factored_dim=args.factored_dim).to(device)
    decoder.gradient_checkpointing_enable()
    encoder = EncoderViT(decoder.config.d_model).to(device)

    cap_params, lang_params = get_trainable_param_groups(decoder)
    cap_params += list(encoder.proj.parameters())

    optimizer_cap = torch.optim.Adam(cap_params, lr=args.lr_caption)
    optimizer_lang = torch.optim.Adam(lang_params, lr=args.lr_language)

    accum_steps = args.accum_steps

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

        factual_train_loss, factual_train_samples = 0.0, 0
        humorous_train_loss, humorous_train_samples = 0.0, 0

        # --- Factual: image + caption ---
        optimizer_cap.zero_grad()
        for i, (images, decoder_input_ids, labels, lengths) in enumerate(train_loader):
            images = images.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                features = encoder(images)
                set_mode(decoder, "factual")
                outputs = decoder(decoder_input_ids=decoder_input_ids,
                                   encoder_outputs=(features,), labels=labels)
                loss = outputs.loss / accum_steps

            loss.backward()

            if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(cap_params, 1.0)
                optimizer_cap.step()
                optimizer_cap.zero_grad()

            factual_train_loss += loss.item() * accum_steps * images.size(0)
            factual_train_samples += images.size(0)

            if i % args.log_step_caption == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, Step [{i}/{len(train_loader)}], Loss: {loss.item()*accum_steps:.4f}")

        if len(train_loader) > 0:
            eval_outputs(decoder_input_ids, (features,), decoder, tokenizer, amp_dtype)

        # --- Humorous: text only, NO image ---
        if train_styled_loader:
            optimizer_lang.zero_grad()
            for i, (decoder_input_ids, labels, lengths) in enumerate(train_styled_loader):
                decoder_input_ids = decoder_input_ids.to(device)
                labels = labels.to(device)
                batch_size = decoder_input_ids.size(0)

                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    dummy_encoder_out = torch.zeros(batch_size, 1, decoder.config.d_model, device=device)
                    set_mode(decoder, "humorous")
                    outputs = decoder(decoder_input_ids=decoder_input_ids,
                                       encoder_outputs=(dummy_encoder_out,), labels=labels)
                    loss = outputs.loss / accum_steps

                loss.backward()

                if (i + 1) % accum_steps == 0 or i == len(train_styled_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(lang_params, 1.0)
                    optimizer_lang.step()
                    optimizer_lang.zero_grad()

                humorous_train_loss += loss.item() * accum_steps * batch_size
                humorous_train_samples += batch_size

                if i % args.log_step_language == 0 or i == len(train_styled_loader) - 1:
                    print(f"Epoch [{epoch+1}/{args.epoch_num}], HUM, Step [{i}/{len(train_styled_loader)}], Loss: {loss.item()*accum_steps:.4f}")

        avg_factual_loss = factual_train_loss / factual_train_samples if factual_train_samples > 0 else 0.0
        avg_humorous_loss = humorous_train_loss / humorous_train_samples if humorous_train_samples > 0 else 0.0
        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_factual_loss:.4f}")
        print(f"[EPOCH {epoch+1}] Humorous Training Loss: {avg_humorous_loss:.4f}")

        if (epoch + 1) % 2 == 0 or epoch == args.epoch_num - 1:
            print(f"[EPOCH {epoch+1}] Running validation...")
            val_loss = validate_epoch(encoder, decoder, val_loader, val_styled_loader, device, amp_dtype)
            print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")
        else:
            print(f"[EPOCH {epoch+1}] Skipping validation this epoch (runs every 2 epochs).")
            val_loss = None

        if val_loss is not None:
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
                    'humorous_train_loss': avg_humorous_loss,
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
            'humorous_train_loss': avg_humorous_loss,
            'val_loss': val_loss if val_loss is not None else best_val_loss,
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
    parser = argparse.ArgumentParser(description='StyleNet Bangla (Transformer/mT5 version) - Humorous style')
    parser.add_argument('--model_path', type=str, default='pretrained_models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/dataset/data/Images')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/dataset/data/factual_caption.txt')
    parser.add_argument('--humorous_caption_path', type=str, default='/kaggle/input/dataset/data/humorous_generated.txt')
    parser.add_argument('--caption_batch_size', type=int, default=16)
    parser.add_argument('--language_batch_size', type=int, default=24)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--factored_dim', type=int, default=512)
    parser.add_argument('--lr_caption', type=float, default=0.00002)
    parser.add_argument('--lr_language', type=float, default=0.00005)
    parser.add_argument('--epoch_num', type=int, default=80)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--train_split_ratio', type=float, default=0.8)
    parser.add_argument('--log_step_caption', type=int, default=500)
    parser.add_argument('--log_step_language', type=int, default=300)
    args = parser.parse_args()
    main(args)
