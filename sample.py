import os
import torch
from torchvision import transforms
from PIL import Image
from data_loader import Rescale, tokenizer
from models import EncoderCNN, FactoredLSTM

def load_sample_images(img_dir, transform, device):
    img_names = os.listdir(img_dir)
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        img_list.append(img)
    return img_names, img_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dimensions: must match training!
    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512

    encoder = EncoderCNN(emb_dim).to(device)
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, tokenizer.vocab_size).to(device)

    # Load weights (update paths as needed)
    encoder.load_state_dict(torch.load('/kaggle/working/pretrained_models/encoder-last.pkl', map_location=device))
    decoder.load_state_dict(torch.load('/kaggle/working/pretrained_models/decoder-last.pkl', map_location=device))
    encoder.eval()
    decoder.eval()

    # Prepare image(s)
    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor()
    ])
    img_dir = '/kaggle/input/sample/sample_images'  # Change as needed
    img_names, img_list = load_sample_images(img_dir, transform, device)
    idx = 1  # or whichever image index you want
    image = img_list[idx]  # already on device

    with torch.no_grad():
        features = encoder(image)
        output_token_ids = decoder.sample(
            features,
            tokenizer=tokenizer,
            beam_size=5,
            max_len=30,
            mode="factual"
        )

    # Remove BOS, stop at EOS, like StyleNet
    if output_token_ids and output_token_ids[0] == tokenizer.bos_token_id:
        output_token_ids = output_token_ids[1:]
    if tokenizer.eos_token_id in output_token_ids:
        output_token_ids = output_token_ids[:output_token_ids.index(tokenizer.eos_token_id)]

    # StyleNet-style: map index to token (like vocab.i2w[x])
    caption_tokens = [tokenizer.convert_ids_to_tokens([x])[0] for x in output_token_ids]
    print(img_names[idx])
    print(" ".join(caption_tokens))

if __name__ == '__main__':
    main()
