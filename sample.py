import os
import torch
from torchvision import transforms
from PIL import Image
from data_loader import tokenizer  # Ensure tokenizer is properly imported
from models import EncoderCNN, FactoredLSTM

# Add the missing Rescale transform if not in data_loader
class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return image.resize(self.output_size, Image.BILINEAR)

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

    # Load weights - use map_location for cross-device compatibility
    encoder.load_state_dict(torch.load(
        '/kaggle/working/pretrained_models/encoder-last.pkl',
        map_location=device
    ))
    decoder.load_state_dict(torch.load(
        '/kaggle/working/pretrained_models/decoder-last.pkl',
        map_location=device
    ))
    encoder.eval()
    decoder.eval()

    # Prepare image transform with normalization
    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # Essential for ResNet
    ])
    
    img_dir = '/kaggle/input/sample/sample_images'
    img_names, img_list = load_sample_images(img_dir, transform, device)
    
    # Process all images
    for i, image in enumerate(img_list):
        with torch.no_grad():
            features = encoder(image)
            output_token_ids = decoder.sample(
                features,
                tokenizer=tokenizer,
                mode="factual"
            )

        # Convert token IDs to text
        if isinstance(output_token_ids, list):
            # Remove special tokens
            tokens = [tokenizer.convert_ids_to_tokens([tid])[0] 
                      for tid in output_token_ids 
                      if tid not in [tokenizer.bos_token_id, tokenizer.eos_token_id]]
            
            caption = tokenizer.convert_tokens_to_string(tokens)
            print(f"Image: {img_names[i]}")
            print(f"Caption: {caption}\n")
        else:
            print(f"No caption generated for {img_names[i]}")

if __name__ == '__main__':
    main()
