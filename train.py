import os
import argparse
import torch
from data_loader import get_loader, get_styled_loader, load_img_caption_lists, load_styled_caption_list, tokenizer
from models import EncoderCNN, FactoredLSTM
import torch.nn as nn
import torch.optim as optim

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def eval_outputs(outputs, tokenizer):
    indices = torch.topk(outputs, 1)[1]
    indices = indices.squeeze(2)
    indices = indices.data.cpu().numpy()
    for i in range(len(indices)):
        tokens = tokenizer.convert_ids_to_tokens(indices[i])
        text = tokenizer.convert_tokens_to_string(tokens)
        print("Generated:", text)

def align_and_loss(outputs, targets, criterion, vocab_size):
    # ---- Shape alignment fix block ----
    if outputs.size(1) != targets.size(1):
        seq_len = min(outputs.size(1), targets.size(1))
        print(f"[Trim] outputs seq_len={outputs.size(1)}, targets seq_len={targets.size(1)} → trim to {seq_len}")
        outputs = outputs[:, :seq_len, :]
        targets = targets[:, :seq_len]
    # Now shapes must match!
    return criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

def main(args):
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    img_paths, factual_captions = load_img_caption_lists(
        args.factual_caption_path, args.img_path
    )
    humorous_captions = load_styled_caption_list(args.humorous_caption_path) if args.humorous_caption_path else []
    romantic_captions = load_styled_caption_list(args.romantic_caption_path) if args.romantic_caption_path else []

    data_loader = get_loader(
        img_paths, factual_captions, batch_size=args.caption_batch_size, shuffle=True, num_workers=2)
    styled_data_loader = get_styled_loader(
        humorous_captions, batch_size=args.language_batch_size, shuffle=True, num_workers=2) if humorous_captions else None
    styled_data_loader_romantic = get_styled_loader(
        romantic_captions, batch_size=args.language_batch_size, shuffle=True, num_workers=2) if romantic_captions else None

    encoder = EncoderCNN(args.emb_dim)
    decoder = FactoredLSTM(args.emb_dim, args.hidden_dim, args.factored_dim, tokenizer.vocab_size)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    cap_params = list(decoder.parameters()) + list(encoder.A.parameters())
    lang_params = list(decoder.parameters())
    optimizer_cap = torch.optim.Adam(cap_params, lr=args.lr_caption)
    optimizer_lang = torch.optim.Adam(lang_params, lr=args.lr_language)

    checkpoint_path = os.path.join(model_path, 'checkpoint-latest.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_cap.load_state_dict(checkpoint['optimizer_cap_state_dict'])
        optimizer_lang.load_state_dict(checkpoint['optimizer_lang_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Checkpoint] Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    else:
        print("[Checkpoint] No previous checkpoint found. Training from scratch.")

    for epoch in range(start_epoch, args.epoch_num):
        encoder.train()
        decoder.train()
        for i, (images, input_ids, attn_mask, lengths) in enumerate(data_loader):
            images = to_var(images)
            input_ids = to_var(input_ids)
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(input_ids, features, mode="factual")
            targets = input_ids[:, 1:].contiguous()
            outputs = outputs[:, :-1, :].contiguous()
            loss = align_and_loss(outputs, targets, criterion, tokenizer.vocab_size)
            loss.backward()
            optimizer_cap.step()
            # === LOGGING FIX: always log last step ===
            if (i % args.log_step_caption == 0) or (i == len(data_loader)-1):
                print("Epoch [%d/%d], CAP, Step [%d/%d], Loss: %.4f"
                      % (epoch+1, args.epoch_num, i, len(data_loader), loss.data.item()))
        eval_outputs(outputs, tokenizer)

        # Language modeling (styled captions: humorous)
        if styled_data_loader:
            for i, (input_ids, attn_mask) in enumerate(styled_data_loader):
                input_ids = to_var(input_ids)
                decoder.zero_grad()
                outputs = decoder(input_ids, features=None, mode='humorous')
                targets = input_ids[:, 1:].contiguous()
                outputs = outputs[:, :-1, :].contiguous()
                loss = align_and_loss(outputs, targets, criterion, tokenizer.vocab_size)
                loss.backward()
                optimizer_lang.step()
                # === LOGGING FIX: always log last step ===
                if (i % args.log_step_language == 0) or (i == len(styled_data_loader)-1):
                    print("Epoch [%d/%d], LANG, Step [%d/%d], Loss: %.4f"
                        % (epoch+1, args.epoch_num, i, len(styled_data_loader), loss.data.item()))
        if styled_data_loader_romantic:
            for i, (input_ids, attn_mask) in enumerate(styled_data_loader_romantic):
                input_ids = to_var(input_ids)
                decoder.zero_grad()
                outputs = decoder(input_ids, features=None, mode='romantic')
                targets = input_ids[:, 1:].contiguous()
                outputs = outputs[:, :-1, :].contiguous()
                loss = align_and_loss(outputs, targets, criterion, tokenizer.vocab_size)
                loss.backward()
                optimizer_lang.step()
                # === LOGGING FIX: always log last step ===
                if (i % args.log_step_language == 0) or (i == len(styled_data_loader_romantic)-1):
                    print("Epoch [%d/%d], ROM, Step [%d/%d], Loss: %.4f"
                        % (epoch+1, args.epoch_num, i, len(styled_data_loader_romantic), loss.data.item()))

        torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-last.pkl'))
        torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-last.pkl'))
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_cap_state_dict': optimizer_cap.state_dict(),
            'optimizer_lang_state_dict': optimizer_lang.state_dict(),
            'loss': loss.item(),
        }, os.path.join(model_path, 'checkpoint-latest.pth'))
        print(f"[Checkpoint] Saved at end of epoch {epoch+1}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla: Generating Attractive Visual Captions with Styles')
    parser.add_argument('--model_path', type=str, default='pretrained_models',
                        help='path for saving trained models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/dataset/data/Images',
                    help='path for train images directory')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/dataset/data/factual_caption.txt',
                        help='path for factual caption file')
    parser.add_argument('--humorous_caption_path', type=str, default='/kaggle/input/dataset/data/humorous_train.txt',
                        help='path for humorous caption file')
    parser.add_argument('--romantic_caption_path', type=str, default='/kaggle/input/dataset/data/romantic_train.txt',
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
    parser.add_argument('--lr_caption', type=float, default=0.0002,
                        help='learning rate for caption model training')
    parser.add_argument('--lr_language', type=float, default=0.0005,
                        help='learning rate for language model training')
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--log_step_caption', type=int, default=50,
                        help='steps for print log while train caption model')
    parser.add_argument('--log_step_language', type=int, default=10,
                        help='steps for print log while train language model')
    args = parser.parse_args()
    main(args)
