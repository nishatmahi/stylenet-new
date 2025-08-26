import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import Counter
from data_loader import Rescale, tokenizer
from models import EncoderViT, FactoredLSTM
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

# ====================================================
# ---- BLEU-4 (with smoothing)
# ====================================================
def simple_bleu(reference, hypothesis, n=4):
    def ngram_counts(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    weights = [1.0/n] * n
    p_ns = []
    for i in range(1, n+1):
        ref_counts = ngram_counts(reference, i)
        hyp_counts = ngram_counts(hypothesis, i)
        overlap = sum((hyp_counts & ref_counts).values())
        total = max(sum(hyp_counts.values()), 1)
        p = overlap / total if total > 0 else 0.0
        if p == 0:
            p = 1e-9
        p_ns.append(p)

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len/hyp_len)
    bleu = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, p_ns)))
    return bleu


# ====================================================
# ---- ROUGE-L (LCS based)
# ====================================================
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

    if not reference or not hypothesis:
        return 0.0

    lcs_len = lcs(reference, hypothesis)
    prec = lcs_len / len(hypothesis) if hypothesis else 0
    rec = lcs_len / len(reference) if reference else 0
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


# ====================================================
# ---- METEOR (Bengali-friendly with synonyms)
# ====================================================
def simple_meteor(reference, hypothesis):
    if not reference or not hypothesis:
        return 0.0

    matches = 0
    ref_set = set(reference)

    for w in hypothesis:
        if w in ref_set:
            matches += 1
        else:
            synsets = wordnet.synsets(w, lang='ben')
            for syn in synsets:
                for lemma in syn.lemma_names('ben'):
                    if lemma in ref_set:
                        matches += 1
                        break

    precision = matches / len(hypothesis) if hypothesis else 0
    recall = matches / len(reference) if reference else 0
    if precision + recall == 0:
        return 0.0
    return (10 * precision * recall) / (recall + 9 * precision)


# ====================================================
# ---- CIDEr (TF-IDF cosine similarity)
# ====================================================
def simple_cider(references, hypothesis, n=4):
    def get_ngrams(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    hyp_ngrams = Counter()
    for i in range(1, n+1):
        hyp_ngrams.update(get_ngrams(hypothesis, i))

    ref_ngrams = Counter()
    for ref in references:
        for i in range(1, n+1):
            ref_ngrams.update(get_ngrams(ref, i))

    # cosine similarity
    dot = sum(hyp_ngrams[ng] * ref_ngrams.get(ng, 0) for ng in hyp_ngrams)
    hyp_norm = math.sqrt(sum(v*v for v in hyp_ngrams.values()))
    ref_norm = math.sqrt(sum(v*v for v in ref_ngrams.values()))
    if hyp_norm * ref_norm == 0:
        return 0.0
    return dot / (hyp_norm * ref_norm)


# ====================================================
# ---- Reference captions loader ----
# ====================================================
def load_reference_captions(reference_file):
    reference_captions = {}
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if "\t" in line:
                    img_name, caption = line.strip().split("\t", 1)
                else:
                    parts = line.strip().split(None, 1)
                    if len(parts) != 2:
                        continue
                    img_name, caption = parts
                img_name = img_name.strip()
                caption = caption.strip()
                reference_captions.setdefault(img_name, []).append(caption)
    return reference_captions


# ====================================================
# ---- Image loader ----
# ====================================================
def load_sample_images(img_dir, transform, device):
    img_names = sorted(os.listdir(img_dir))
    img_list = []
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        img_list.append(img)
    return img_names, img_list


# ====================================================
# ---- Main ----
# ====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512

    encoder = EncoderViT(emb_dim).to(device)
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, len(tokenizer)).to(device)

    encoder.load_state_dict(torch.load('stylenet_new_again_models/encoder-last.pkl', map_location=device))
    decoder.load_state_dict(torch.load('stylenet_new_again_models/decoder-last.pkl', map_location=device))

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_dir = '/kaggle/input/sample-data/sample/sample_images'
    reference_file = '/kaggle/input/sample-data/sample/sample_images_factual.txt'

    img_names, img_list = load_sample_images(img_dir, transform, device)
    reference_captions = load_reference_captions(reference_file)

    all_bleu, all_rouge, all_meteor, all_cider = [], [], [], []

    for idx, image in enumerate(img_list):
        with torch.no_grad():
            features = encoder(image)

            # ---- Caption generation ----
            output = decoder.sample(
                features,
                tokenizer=tokenizer,
                beam_size=5,
                max_len=30,
                mode="factual"
            )
            caption = tokenizer.decode(output, skip_special_tokens=True)
            print(img_names[idx], "| Predicted Caption:", caption)

        img_name = img_names[idx]
        ref_list = reference_captions.get(img_name, None)

        if ref_list is not None:
            best_bleu, best_rouge, best_meteor, best_cider = 0, 0, 0, 0
            for ref_caption in ref_list:
                ref_tokens = tokenizer.tokenize(ref_caption)
                hyp_tokens = tokenizer.tokenize(caption)

                bleu_score = simple_bleu(ref_tokens, hyp_tokens)
                rouge_score = simple_rouge_l(ref_tokens, hyp_tokens)
                meteor_score = simple_meteor(ref_tokens, hyp_tokens)
                cider_score = simple_cider([ref_tokens], hyp_tokens)

                best_bleu = max(best_bleu, bleu_score)
                best_rouge = max(best_rouge, rouge_score)
                best_meteor = max(best_meteor, meteor_score)
                best_cider = max(best_cider, cider_score)

            all_bleu.append(best_bleu)
            all_rouge.append(best_rouge)
            all_meteor.append(best_meteor)
            all_cider.append(best_cider)

            print(f"Reference Captions: {ref_list}")
            print(f"BLEU-4: {best_bleu:.4f} | ROUGE-L: {best_rouge:.4f} | METEOR: {best_meteor:.4f} | CIDEr: {best_cider:.4f}\n")
        else:
            print("Reference: NOT FOUND\n")

    if all_bleu:
        print("="*50)
        print(f"Average BLEU-4: {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"Average ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print(f"Average METEOR: {sum(all_meteor)/len(all_meteor):.4f}")
        print(f"Average CIDEr: {sum(all_cider)/len(all_cider):.4f}")
        print("="*50)


if __name__ == '__main__':
    main()
