import os
import torch
from torchvision import transforms
from PIL import Image
from data_loader import Rescale, tokenizer
from models import EncoderCNN, FactoredLSTM

def load_sample_images(img_dir, transform):
    img_names = os.listdir(img_dir)
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0)
        img_list.append(img)
    return img_names, img_list

def main():
    # build model (use the dimension you used for training!)
    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512

    encoder = EncoderCNN(emb_dim)
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, tokenizer.vocab_size)

    # Load weights (update paths as needed)
    encoder.load_state_dict(torch.load('pretrained_models/encoder-10.pkl', map_location='cpu'))
    decoder.load_state_dict(torch.load('pretrained_models/decoder-10.pkl', map_location='cpu'))

    encoder.eval()
    decoder.eval()

    # prepare images
    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor()
    ])
    img_names, img_list = load_sample_images('sample_images/', transform)
    idx = 1
    image = img_list[idx]

    # if torch.cuda.is_available():
    #     encoder = encoder.cuda()
    #     decoder = decoder.cuda()
    #     image = image.cuda()

    with torch.no_grad():
        features = encoder(image)
        output_token_ids = decoder.sample(
            features,
            tokenizer=tokenizer,      # Pass HuggingFace tokenizer!
            beam_size=5,
            max_len=30,
            mode="factual"
        )

    # BOS token skip, stop at EOS
    if output_token_ids and output_token_ids[0] == tokenizer.bos_token_id:
        output_token_ids = output_token_ids[1:]
    if tokenizer.eos_token_id in output_token_ids:
        end_idx = output_token_ids.index(tokenizer.eos_token_id)
        output_token_ids = output_token_ids[:end_idx]

    caption = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    print(img_names[idx])
    print(caption)

if __name__ == '__main__':
    main()
