import os
import argparse
import torch
import random
from torch.cuda.amp import autocast, GradScaler
from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, FactoredGRU
from loss import masked_cross_entropy

def split_caption_file(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    """
    Split a caption file into training and validation sets
    """
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping split")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(lines)
    
    # Split
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Write split files
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"Split {input_file}: {len(train_lines)} train, {len(val_lines)} val")

def create_data_splits(args):
    """
    Create train/val splits for all data files
    """
    # Create split directories
    train_dir = '/kaggle/working/train_split'
    val_dir = '/kaggle/working/val_split'
    
    # Split factual captions
    factual_train = os.path.join(train_dir, 'factual_train.txt')
    factual_val = os.path.join(val_dir, 'factual_val.txt')
    split_caption_file(args.factual_caption_path, factual_train, factual_val)
    
    # Split romantic captions
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

def validate_epoch(encoder, decoder, val_loader, val_styled_loader, criterion, device, use_amp=False):
    """
    Run validation for one epoch.
    Early stopping and best model saving are based on factual loss only.
    Romantic loss is computed and printed for monitoring purposes only.
    """
    encoder.eval()
    decoder.eval()

    factual_loss = 0.0
    factual_samples = 0

    with torch.no_grad():
        # Validate factual captions
        if val_loader:
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.long().to(device)
                lengths = lengths.to(device)

                with autocast(enabled=use_amp):
                    features = encoder(images)
                    outputs = decoder(captions, features, mode="factual")
                    loss = criterion(outputs[:, 1:, :].contiguous(),
                                   captions[:, 1:].contiguous(), lengths - 1)

                factual_loss += loss.item() * captions.size(0)
                factual_samples += captions.size(0)

            if factual_samples > 0:
                factual_loss /= factual_samples
                print(f"Validation Factual Loss: {factual_loss:.4f}")

        # Validate romantic captions — printed for monitoring only, not used for early stopping
        if val_styled_loader:
            romantic_loss = 0.0
            romantic_samples = 0

            for captions, lengths in val_styled_loader:
                captions = captions.long().to(device)
                lengths = lengths.to(device)

                with autocast(enabled=use_amp):
                    outputs = decoder(captions, mode='romantic')
                    loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)

                romantic_loss += loss.item() * captions.size(0)
                romantic_samples += captions.size(0)

            if romantic_samples > 0:
                romantic_loss /= romantic_samples
                print(f"Validation Romantic Loss (monitor only): {romantic_loss:.4f}")

    # Only return factual loss — drives early stopping and best model saving
    return factual_loss if factual_samples > 0 else float('inf')

def eval_outputs(outputs, tokenizer):
    indices = torch.topk(outputs, 1)[1]
    indices = indices.squeeze(2)
    indices = indices.data.cpu().numpy()
    for i in range(min(3, len(indices))):  # Show max 3 examples
        tokens = tokenizer.convert_ids_to_tokens(indices[i])
        text = tokenizer.convert_tokens_to_string(tokens)
        print(f"Generated {i+1}: {text}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enable cuDNN auto-tuner for faster convolution/matmul kernels
    torch.backends.cudnn.benchmark = True

    # Mixed precision scaler — big speedup on T4/P100/V100 GPUs
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled (CPU)'}")

    permanent_save_folder = "stylenet_gru_models/"
    os.makedirs(permanent_save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    # Create data splits
    print("Creating train/validation splits...")
    split_paths = create_data_splits(args)
    
    # Training data loaders
    train_loader = get_data_loader(
        args.img_path, split_paths['factual_train'], 
        batch_size=args.caption_batch_size, shuffle=True)
    
    train_styled_loader = get_styled_data_loader(
        split_paths['romantic_train'], batch_size=args.language_batch_size, 
        shuffle=True) if split_paths['romantic_train'] else None

    # Validation data loaders
    val_loader = get_data_loader(
        args.img_path, split_paths['factual_val'], 
        batch_size=args.caption_batch_size, shuffle=False) if split_paths['factual_val'] else None
    
    val_styled_loader = get_styled_data_loader(
        split_paths['romantic_val'], batch_size=args.language_batch_size, 
        shuffle=False) if split_paths['romantic_val'] else None

    print(f"Train batches: Factual={len(train_loader)}, Romantic={len(train_styled_loader) if train_styled_loader else 0}")
    print(f"Val batches: Factual={len(val_loader) if val_loader else 0}, Romantic={len(val_styled_loader) if val_styled_loader else 0}")

    # Models
    encoder = EncoderViT(args.emb_dim).to(device)
    decoder = FactoredGRU(args.emb_dim, args.hidden_dim, args.factored_dim, len(tokenizer)).to(device)

    # Optimizer, loss
    criterion = masked_cross_entropy
    cap_params = list(decoder.parameters()) + list(encoder.A.parameters())
    lang_params = list(decoder.parameters())
    optimizer_cap = torch.optim.Adam(cap_params, lr=args.lr_caption)
    optimizer_lang = torch.optim.Adam(lang_params, lr=args.lr_language)

    # Checkpoint loading
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

    # Training loop with validation and early stopping
    for epoch in range(start_epoch, args.epoch_num):
        print(f"\n[DEBUG] Training epoch {epoch+1} of {args.epoch_num}")
        
        # Training phase
        encoder.train()
        decoder.train()
        
        factual_train_loss = 0.0
        factual_train_samples = 0
        romantic_train_loss = 0.0
        romantic_train_samples = 0
        
        # Train on factual captions (image+caption pairs)
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.long().to(device)
            lengths = lengths.to(device)

            optimizer_cap.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                features = encoder(images)
                outputs = decoder(captions, features, mode="factual")
                loss = criterion(outputs[:, 1:, :].contiguous(),
                                 captions[:, 1:].contiguous(), lengths - 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_cap)
            torch.nn.utils.clip_grad_norm_(cap_params, 1.0)
            scaler.step(optimizer_cap)
            scaler.update()
            
            factual_train_loss += loss.item() * captions.size(0)
            factual_train_samples += captions.size(0)

            if i % args.log_step_caption == 0 or i == len(train_loader)-1:
                print(f"Epoch [{epoch+1}/{args.epoch_num}], CAP, Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Show some example outputs
        if len(train_loader) > 0:
            eval_outputs(outputs, tokenizer)

        # Train on romantic captions (text-only)
        if train_styled_loader:
            for i, (captions, lengths) in enumerate(train_styled_loader):
                captions = captions.long().to(device)
                lengths = lengths.to(device)
                optimizer_lang.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    outputs = decoder(captions, mode='romantic')
                    loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_lang)
                torch.nn.utils.clip_grad_norm_(lang_params, 1.0)
                scaler.step(optimizer_lang)
                scaler.update()
                
                romantic_train_loss += loss.item() * captions.size(0)
                romantic_train_samples += captions.size(0)

                if i % args.log_step_language == 0 or i == len(train_styled_loader)-1:
                    print(f"Epoch [{epoch+1}/{args.epoch_num}], ROM, Step [{i}/{len(train_styled_loader)}], Loss: {loss.item():.4f}")

        # Print separate training losses
        avg_factual_loss = factual_train_loss / factual_train_samples if factual_train_samples > 0 else 0.0
        avg_romantic_loss = romantic_train_loss / romantic_train_samples if romantic_train_samples > 0 else 0.0
        print(f"\n[EPOCH {epoch+1}] Factual Training Loss:  {avg_factual_loss:.4f}")
        print(f"[EPOCH {epoch+1}] Romantic Training Loss: {avg_romantic_loss:.4f}")

        # Validation phase
        print(f"[EPOCH {epoch+1}] Running validation...")
        val_loss = validate_epoch(encoder, decoder, val_loader, val_styled_loader, criterion, device, use_amp=use_amp)
        print(f"[EPOCH {epoch+1}] Factual Validation Loss (early stopping): {val_loss:.4f}")

        # Early stopping logic — based on factual validation loss only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
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

        # Save regular checkpoint
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

    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        print(f"\nTraining completed. Loading best model (factual val loss: {best_val_loss:.4f}) for final evaluation...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
        decoder.load_state_dict(best_checkpoint['decoder_state_dict'])
        print("Best model loaded successfully!")
    else:
        print("No best model found, using current model weights.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla with Validation: Generating Attractive Visual Captions with Styles')
    parser.add_argument('--model_path', type=str, default='pretrained_models',
                        help='path for saving trained models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/dataset/data/Images',
                    help='path for train images directory')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/dataset/data/factual_caption.txt',
                        help='path for factual caption file')
    parser.add_argument('--humorous_caption_path', type=str, default='/kaggle/input/dataset/data/humorous_text.txt',
                        help='path for humorous caption file')
    parser.add_argument('--romantic_caption_path', type=str, default='/kaggle/input/dataset/data/romantic_data.txt',
                        help='path for romantic caption file')
    parser.add_argument('--caption_batch_size', type=int, default=64,
                        help='mini batch size for caption model training')
    parser.add_argument('--language_batch_size', type=int, default=96,
                        help='mini batch size for language model training')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding size of word, image')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden state size of factored LSTM')
    parser.add_argument('--factored_dim', type=int, default=512,
                        help='size of factored matrix')
    parser.add_argument('--lr_caption', type=float, default=0.00001,
                        help='learning rate for caption model training')
    parser.add_argument('--lr_language', type=float, default=0.00005,
                        help='learning rate for language model training')
    parser.add_argument('--epoch_num', type=int, default=80,
                        help='number of epochs to train')
    parser.add_argument('--patience', type=int, default=7,
                        help='patience for early stopping')
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                        help='ratio for train/validation split')
    parser.add_argument('--log_step_caption', type=int, default=200,
                        help='steps for print log while train caption model')
    parser.add_argument('--log_step_language', type=int, default=100,
                        help='steps for print log while train language model')
    args = parser.parse_args()
    main(args)
