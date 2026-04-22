import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
from config import config
from models import EncoderViT, FactoredGRU
from data_loader import Rescale


def load_sample_images(img_dir, transform):
    img_names = sorted(os.listdir(img_dir))
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        print("Loading image from:", img_path)
        image = Image.open(img_path).convert("RGB")
        # Apply the SAME transform pipeline as data_loader.py:
        # PIL → Rescale(224,224) → ToTensor → Normalize
        if transform:
            image = transform(image)
        img_list.append(image)
    return img_names, img_list


# ---- Setup: Load tokenizer directly (bypasses HF Hub) ----
_TOK_DIR = "/kaggle/working/stylenet/tokenizer-extended"
sys.path.insert(0, _TOK_DIR)
from tokenization_bn import BNTokenizer

_vocab_file = os.path.join(_TOK_DIR, "tokenizer.model")
_tok_config_file = os.path.join(_TOK_DIR, "tokenizer_config.json")
with open(_tok_config_file, "r", encoding="utf-8") as f:
    _tok_config = json.load(f)
tokenizer = BNTokenizer(
    vocab_file=_vocab_file,
    bos_token=_tok_config.get("bos_token", "<s>"),
    eos_token=_tok_config.get("eos_token", "</s>"),
    unk_token=_tok_config.get("unk_token", "<unk>"),
    pad_token=_tok_config.get("pad_token", "<unk>"),
)
_added_tokens_file = os.path.join(_TOK_DIR, "added_tokens.json")
if os.path.exists(_added_tokens_file):
    with open(_added_tokens_file, "r", encoding="utf-8") as f:
        _added_tokens = json.load(f)
    if _added_tokens:
        tokenizer.add_tokens(list(_added_tokens.keys()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EncoderViT(config.emb_dim).to(device)
decoder = FactoredGRU(config.emb_dim, config.hidden_dim, config.factored_dim, vocab_size=len(tokenizer)).to(device)

encoder.load_state_dict(torch.load("/kaggle/working/stylenet_gru_models/encoder-last.pkl", map_location=device))
decoder.load_state_dict(torch.load("/kaggle/working/stylenet_gru_models/decoder-last.pkl", map_location=device))

# Set eval mode right after loading weights
encoder.eval()
decoder.eval()

# FIXED: Use the EXACT same transform pipeline as data_loader.py
transform = transforms.Compose([
    Rescale((224, 224)),     # PIL resize — same as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img_names, img_list = load_sample_images(config.simg_path, transform)

# ---- No ground-truth section from here ----
with torch.no_grad():
    idx = 0  # whichever image you want
    image = img_list[idx].unsqueeze(0).to(device)
    features = encoder(image)
    print("Image features shape:", features.shape)
    print("First 10 feature values:", features[0, :10].cpu().numpy())

    # ---- First token analysis ----
    h0 = torch.empty(1, decoder.hidden_dim).uniform_().to(device)
    # GRU forward_step: no cell state, returns (output, h_t)
    first_output, _ = decoder.forward_step(features, h0, mode="factual", features=features)
    first_output = first_output.squeeze(0)  # [vocab_size]
    top_tokens = torch.topk(first_output, 5).indices.tolist()
    print("Top 5 first tokens:", tokenizer.convert_ids_to_tokens(top_tokens))

    # ---- Caption generation ----
    output = decoder.sample(
        features,
        tokenizer=tokenizer,
        beam_size=5,
        max_len=30,
        mode="romantic"
    )
    caption = tokenizer.decode(output, skip_special_tokens=True)
    print(img_names[idx], "| Predicted Caption:", caption)
