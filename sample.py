import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import Rescale, tokenizer
from models import EncoderCNN, FactoredLSTM

def load_sample_images(img_dir, transform, device):
    img_names = sorted(os.listdir(img_dir))
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

    # Dimensions (must match training)
    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512

    encoder = EncoderCNN(emb_dim).to(device)
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, tokenizer.vocab_size).to(device)

    # Load weights
    encoder.load_state_dict(torch.load('/kaggle/working/pretrained_models/encoder-last.pkl', map_location=device))
    decoder.load_state_dict(torch.load('/kaggle/working/pretrained_models/decoder-last.pkl', map_location=device))
    encoder.eval()
    decoder.eval()

    # Prepare image(s)
    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_dir = '/kaggle/input/sample/sample_images'  # Change as needed
    img_names, img_list = load_sample_images(img_dir, transform, device)
    idx = 4  # whichever image you want
    image = img_list[idx]

    with torch.no_grad():
        features = encoder(image)
        print("Image features shape:", features.shape)
        print("First 10 feature values:", features[0][:10])

        # ---- Feature visualization (1D plot, emb_dim=300) ----
        plt.figure(figsize=(10,3))
        plt.plot(features[0].cpu().numpy())
        plt.title("Extracted Image Features (1D plot)")
        plt.xlabel("Feature index")
        plt.ylabel("Feature value")
        plt.show()

        # ---- First token analysis ----
        h0 = torch.empty(1, decoder.hidden_dim).uniform_().to(device)
        c0 = torch.empty(1, decoder.hidden_dim).uniform_().to(device)
        first_output, _, _ = decoder.forward_step(features, h0, c0, mode="factual")
        first_output = first_output.squeeze(0)  # [vocab_size]
        top_tokens = torch.topk(first_output, 5).indices.tolist()
        print("Top 5 first tokens:", tokenizer.convert_ids_to_tokens(top_tokens))

        # ---- Caption generation ----
        output = decoder.sample(
            features,
            tokenizer=tokenizer,
            beam_size=5,
            max_len=30,
            mode="factual"
        )

    # Remove BOS if present, stop at EOS
    if output and output[0] == tokenizer.bos_token_id:
        output = output[1:]
    if tokenizer.eos_token_id in output:
        output = output[:output.index(tokenizer.eos_token_id)]

    # Convert tokens to string and print
    caption_tokens = [tokenizer.convert_ids_to_tokens([x])[0] for x in output]
    caption_text = tokenizer.convert_tokens_to_string(caption_tokens)

    print(img_names[idx])
    print("Predicted Caption:", caption_text)

if __name__ == '__main__':
    main()
