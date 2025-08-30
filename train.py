import os
import argparse
import torch
from data_loader import get_data_loader, get_styled_data_loader, tokenizer
from models import EncoderViT, FactoredLSTM
from loss import masked_cross_entropy

def eval_outputs(outputs, tokenizer):
    indices = torch.topk(outputs, 1)[1]
    indices = indices.squeeze(2)
    indices = indices.data.cpu().numpy()
    for i in range(len(indices)):
        tokens = tokenizer.convert_ids_to_tokens(indices[i])
        text = tokenizer.convert_tokens_to_string(tokens)
        print("Generated:", text)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permanent_save_folder = "stylenet_new_again_models/"
    os.makedirs(permanent_save_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    # Data loaders
    data_loader = get_data_loader(
        args.img_path, args.factual_caption_path, batch_size=args.caption_batch_size, shuffle=True)
    # styled_data_loader = get_styled_data_loader(
    #     args.humorous_caption_path, batch_size=args.language_batch_size, shuffle=True) if args.humorous_caption_path else None
    styled_data_loader_romantic = get_styled_data_loader(
        args.romantic_caption_path, batch_size=args.language_batch_size, shuffle=True) if args.romantic_caption_path else None

    # Models
    encoder = EncoderViT(args.emb_dim).to(device)
    decoder = FactoredLSTM(args.emb_dim, args.hidden_dim, args.factored_dim, len(tokenizer)).to(device)

    

    # Optimizer, loss
    criterion = masked_cross_entropy
    cap_params = list(decoder.parameters()) + list(encoder.A.parameters())
    lang_params = list(decoder.parameters())
    optimizer_cap = torch.optim.Adam(cap_params, lr=args.lr_caption)
    optimizer_lang = torch.optim.Adam(lang_params, lr=args.lr_language)

    # ======= Checkpoint Loading (NEW SECTION) =======
    start_epoch = 0
    checkpoint_path = os.path.join(permanent_save_folder, 'checkpoint-latest.pth')
    encoder_last_path = os.path.join(permanent_save_folder, "encoder-last.pkl")
    decoder_last_path = os.path.join(permanent_save_folder, "decoder-last.pkl")

    # === DEBUG: List files before loading checkpoint ===
    print("========== [DEBUG] ==========")
    print(f"permanent_save_folder: {permanent_save_folder}")
    print(f"checkpoint_path: {checkpoint_path}")
    print("Files in checkpoint folder BEFORE loading:", os.listdir(permanent_save_folder))
    print("=============================")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_cap.load_state_dict(checkpoint['optimizer_cap_state_dict'])
        optimizer_lang.load_state_dict(checkpoint['optimizer_lang_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[DEBUG] Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"[DEBUG] start_epoch = {start_epoch}")
    else:
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
            print("[DEBUG] No checkpoint or pretrained weights found. Training from scratch (random weights).")
        else:
            print("[DEBUG] No checkpoint found. Loaded latest pretrained weights only.")

    print(f"[DEBUG] Final start_epoch = {start_epoch}")
    print("=============================")

    total_cap_step = len(data_loader)
    # total_lang_step = len(styled_data_loader) if styled_data_loader else 0
    total_romantic_step = len(styled_data_loader_romantic) if styled_data_loader_romantic else 0
    epoch_num = args.epoch_num

    # ========================= Training Loop =========================
    for epoch in range(start_epoch, epoch_num):
        print(f"[DEBUG] Training epoch {epoch+1} of {epoch_num} (starting from {start_epoch+1})")

        #factual (image+caption)
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.long().to(device)
            lengths = lengths.to(device)

            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(captions, features, mode="factual")
            loss = criterion(outputs[:, 1:, :].contiguous(),
                             captions[:, 1:].contiguous(), lengths - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cap_params, 1.0)
            optimizer_cap.step()

            if i % args.log_step_caption == 0 or i == total_cap_step-1:
                print("Epoch [%d/%d], CAP, Step [%d/%d], Loss: %.4f"
                      % (epoch+1, epoch_num, i, total_cap_step, loss.item()))
        eval_outputs(outputs, tokenizer)

        #styled (humorous)
        # if styled_data_loader:
        #     for i, (captions, lengths) in enumerate(styled_data_loader):
        #         captions = captions.long().to(device)
        #         lengths = lengths.to(device)
        #         decoder.zero_grad()
        #         outputs = decoder(captions, mode='humorous')
        #         loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(lang_params, 1.0)
        #         optimizer_lang.step()
        #         if i % args.log_step_language == 0 or i == total_lang_step-1:
        #             print("Epoch [%d/%d], LANG, Step [%d/%d], Loss: %.4f"
        #                   % (epoch+1, epoch_num, i, total_lang_step, loss.item()))

         # styled (romantic)
        if styled_data_loader_romantic:
            for i, (captions, lengths) in enumerate(styled_data_loader_romantic):
                captions = captions.long().to(device)
                lengths = lengths.to(device)
                decoder.zero_grad()
                outputs = decoder(captions, mode='romantic')
                loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lang_params, 1.0)
                optimizer_lang.step()
                if i % args.log_step_language == 0 or i == total_romantic_step-1:
                    print("Epoch [%d/%d], ROM, Step [%d/%d], Loss: %.4f"
                          % (epoch+1, epoch_num, i, total_romantic_step, loss.item()))

        # ======== SAVE: After every epoch =========
        os.makedirs(permanent_save_folder, exist_ok=True)
        os.makedirs(args.model_path, exist_ok=True)
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
            'loss': loss.item(),
        }, os.path.join(permanent_save_folder, 'checkpoint-latest.pth'))
        # === DEBUG: List files after saving checkpoint ===
        print(f"[DEBUG] Saved checkpoint and models at epoch {epoch+1}")
        print(f"[DEBUG] Files in checkpoint folder AFTER saving:", os.listdir(permanent_save_folder))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StyleNet Bangla: Generating Attractive Visual Captions with Styles')
    parser.add_argument('--model_path', type=str, default='pretrained_models',
                        help='path for saving trained models')
    parser.add_argument('--img_path', type=str, default='/kaggle/input/dataset/data/Images',
                    help='path for train images directory')
    parser.add_argument('--factual_caption_path', type=str, default='/kaggle/input/dataset/data/factual_caption.txt',
                        help='path for factual caption file')
    parser.add_argument('--humorous_caption_path', type=str, default='/kaggle/input/dataset-new/data/humorous caption.txt',
                        help='path for humorous caption file')
    parser.add_argument('--romantic_caption_path', type=str, default='/kaggle/input/dataset/data/romantic_data.txt',
                        help='path for romantic caption file')
    parser.add_argument('--caption_batch_size', type=int, default=32,
                        help='mini batch size for caption model training')
    parser.add_argument('--language_batch_size', type=int, default=32,
                        help='mini batch size for language model training')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding size of word, image')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden state size of factored LSTM')
    parser.add_argument('--factored_dim', type=int, default=512,
                        help='size of factored matrix')
    parser.add_argument('--lr_caption', type=float, default=0.00002,
                        help='learning rate for caption model training')
    parser.add_argument('--lr_language', type=float, default=0.00004,
                        help='learning rate for language model training')
    parser.add_argument('--epoch_num', type=int, default=85)
    parser.add_argument('--log_step_caption', type=int, default=200,
                        help='steps for print log while train caption model')
    parser.add_argument('--log_step_language', type=int, default=100,
                        help='steps for print log while train language model')
    args = parser.parse_args()
    main(args)











































































