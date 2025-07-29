import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import Rescale, tokenizer
from models import EncoderCNN, FactoredLSTM

# ---- BLEU and ROUGE functions (add at the top) ----
def simple_bleu(reference, hypothesis, n=4):
    import math
    from collections import Counter
    def ngram_counts(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    weights = [0.25, 0.25, 0.25, 0.25]
    p_ns = []
    for i in range(1, n+1):
        ref_counts = ngram_counts(reference, i)
        hyp_counts = ngram_counts(hypothesis, i)
        overlap = sum((hyp_counts & ref_counts).values())
        total = max(sum(hyp_counts.values()), 1)
        p_ns.append(overlap / total)
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    bp = math.exp(1 - ref_len/hyp_len) if hyp_len < ref_len else 1
    bleu = bp * math.exp(sum(w * math.log(p) if p > 0 else 0 for w, p in zip(weights, p_ns)))
    return bleu

def simple_rouge_l(reference, hypothesis):
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if X[i] == Y[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return dp[-1][-1]
    lcs_len = lcs(reference, hypothesis)
    prec = lcs_len / len(hypothesis) if hypothesis else 0
    rec = lcs_len / len(reference) if reference else 0
    if prec + rec == 0:
        f1 = 0
    else:
        f1 = (2 * prec * rec) / (prec + rec)
    return f1

# ---- Reference captions loader ----
def load_reference_captions(reference_file):
    reference_captions = {}
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                img_name, caption = line.strip().split('\t', 1)
                reference_captions[img_name] = caption
    return reference_captions

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
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, len(tokenizer)).to(device)

    # Load weights
    encoder.load_state_dict(torch.load('stylenet_new_again_models/encoder-last.pkl', map_location=device))
    decoder.load_state_dict(torch.load('stylenet_new_again_models/decoder-last.pkl', map_location=device))

    encoder.eval()
    decoder.eval()

    # Prepare image(s)
    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_dir = '/kaggle/input/sample-data/sample/sample_images'  # Change as needed
    reference_file = '/kaggle/input/sample-data/sample/sample_images_romantic.txt'  # Add this path
    img_names, img_list = load_sample_images(img_dir, transform, device)
    reference_captions = load_reference_captions(reference_file)  # New

    # ------- BLEU/ROUGE stats -------
    all_bleu = []
    all_rouge = []

    # --------- Original logic ----------
    for idx, image in enumerate(img_list):
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
                mode="romantic"
            )
            caption = tokenizer.decode(output, skip_special_tokens=True)
            print(img_names[idx], "| Predicted Caption:", caption)

        # ------- BLEU/ROUGE calculation ---------
        img_name = img_names[idx]
        ref_caption = reference_captions.get(img_name, None)
        if ref_caption is not None:
            ref_tokens = tokenizer.tokenize(ref_caption)
            hyp_tokens = tokenizer.tokenize(caption)
            bleu_score = simple_bleu(ref_tokens, hyp_tokens)
            rouge_score = simple_rouge_l(ref_tokens, hyp_tokens)
            all_bleu.append(bleu_score)
            all_rouge.append(rouge_score)
            print(f"Reference: {ref_caption}")
            print(f"BLEU-4: {bleu_score:.4f} | ROUGE-L: {rouge_score:.4f}\n")
        else:
            print(f"Reference: NOT FOUND\n")

    # ------- Print average BLEU/ROUGE ---------
    if all_bleu and all_rouge:
        print("="*50)
        print(f"Average BLEU-4: {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"Average ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print("="*50)

if __name__ == '__main__':
    main()
