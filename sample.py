import os
import torch
import math
import re
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from collections import Counter, defaultdict

from data_loader import Rescale, tokenizer
from models import EncoderViT, FactoredLSTM

# ================= METRICS =================
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ---- BLEU-4 with NLTK ----
def nltk_bleu(references, hypothesis, max_n=4):
    if not isinstance(references[0], list):
        references = [references]
    weights = tuple([1.0/max_n] * max_n)  # uniform weights
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smooth_fn)


# ---- ROUGE-L with rouge-score ----
def rouge_l_score(references, hypothesis):
    if not isinstance(references[0], list):
        references = [references]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    best = 0.0
    hyp_str = " ".join(hypothesis)
    for ref in references:
        ref_str = " ".join(ref)
        score = scorer.score(ref_str, hyp_str)["rougeL"].fmeasure
        best = max(best, score)
    return best


# ---- BengaliWordProcessor for METEOR ----
class BengaliWordProcessor:
    def __init__(self):
        self.suffix_patterns = [
            (r'রা$', ''), (r'গুলো$', ''), (r'গুলি$', ''), (r'সব$', ''),
            (r'এর$', ''), (r'দের$', ''),
            (r'তে$', ''), (r'য়ে$', ''), (r'য়ে$', ''), (r'র$', ''), (r'কে$', ''),
            (r'ছে$', ''), (r'ছি$', ''), (r'ছেন$', ''), (r'ছিল$', ''), (r'ছিলেন$', ''), 
            (r'ছিলাম$', ''), (r'বে$', ''), (r'বো$', ''), (r'বেন$', ''), 
            (r'ল$', ''), (r'লো$', ''), (r'লেন$', ''), (r'লাম$', ''), 
            (r'টি$', ''), (r'টা$', ''), (r'খানি$', ''), (r'খানা$', ''),
        ]

        self.synonyms = {
            'মানুষ': ['ব্যক্তি', 'লোক', 'জন', 'মানব', 'আদম'],
            'ছেলে': ['শিশু', 'বালক', 'কিশোর', 'ছেলেটি'],
            'মেয়ে': ['শিশু', 'কন্যা', 'বালিকা', 'মেয়েটি'],
            'শিশু': ['বাচ্চা', 'শিশুটি', 'ছোট্ট', 'কিশোর', 'কিশোরী'],
            'মহিলা': ['নারী', 'স্ত্রী', 'বেগম', 'মেয়েলি'],
            'পুরুষ': ['ভদ্রলোক', 'ছেলে', 'মানুষ'],
            'কুকুর': ['শ্বান', 'কুত্তা'],
            'বিড়াল': ['বিড়াল', 'বেড়াল', 'মার্জার'],
            'ঘোড়া': ['অশ্ব'],
            'পাখি': ['চড়ুই', 'পাখিটি', 'পাখীগুলো'],
            'সমুদ্র': ['সাগর'],
            'সমুদ্রসৈকত': ['সৈকত', 'সমুদ্রতট', 'সমুদ্রতীরে', 'বিচ'],
            'নদী': ['স্রোত', 'খাল'],
            'রাস্তা': ['পথ', 'সড়ক'],
            'পাহাড়': ['পর্বত', 'গিরি'],
            'বন': ['অরণ্য', 'জঙ্গল'],
            'গাছ': ['বৃক্ষ', 'গাছপালা'],
            'জল': ['পানি', 'নীর'],
            'গাড়ি': ['যান', 'মোটর', 'কার'],
            'সাইকেল': ['বাইক'],
            'বাস': ['অটোবাস'],
            'বই': ['গ্রন্থ', 'পুস্তক'],
            'ব্যাগ': ['থলে', 'ঝোলা'],
            'প্লেট': ['থালা'],
            'হাঁটা': ['চলা', 'গমন'],
            'দাঁড়ানো': ['দাঁড়িয়ে', 'দাঁড়িয়ে'],
            'দৌড়ানো': ['ছোটা', 'দৌড়'],
            'খাওয়া': ['ভোজন', 'আহার', 'খাদ্যগ্রহণ'],
            'খেলা': ['ক্রীড়া', 'খেলাধুলা'],
            'হাসা': ['হাসিখুশি', 'আনন্দিত'],
            'বসা': ['আসীন হওয়া'],
            'দেখা': ['তাকানো', 'দর্শন'],
            'তোলা': ['উঁচানো', 'উত্তোলন', 'তোলা হচ্ছে'],
            'চালানো': ['ড্রাইভ', 'গাড়ি চালানো', 'বাইক চালানো'],
        }

        self.word_to_synonyms = defaultdict(set)
        for word, syns in self.synonyms.items():
            variants = [word] + syns
            for v in variants:
                self.word_to_synonyms[v].update(variants)

    def basic_stem(self, word):
        original = word
        for pattern, repl in self.suffix_patterns:
            word = re.sub(pattern, repl, word)
        return word if word else original

    def get_synonyms(self, word):
        return self.word_to_synonyms.get(word, {word})


# ---- METEOR ----
def corrected_meteor(references, hypothesis, alpha=0.9, beta=3.0, gamma=0.5):
    if not isinstance(references[0], list):
        references = [references]
    if not hypothesis or not any(references):
        return 0.0

    processor = BengaliWordProcessor()
    best_meteor = 0.0

    for reference in references:
        matches = {'exact': [], 'stem': [], 'synonym': []}
        for i, hyp_word in enumerate(hypothesis):
            for j, ref_word in enumerate(reference):
                if hyp_word == ref_word:
                    matches['exact'].append((i, j))
                elif processor.basic_stem(hyp_word) == processor.basic_stem(ref_word):
                    matches['stem'].append((i, j))
                elif ref_word in processor.get_synonyms(hyp_word):
                    matches['synonym'].append((i, j))

        used_hyp, used_ref = set(), set()
        final_matches = {'exact': [], 'stem': [], 'synonym': []}
        for t in ['exact', 'stem', 'synonym']:
            for i, j in matches[t]:
                if i not in used_hyp and j not in used_ref:
                    final_matches[t].append((i, j))
                    used_hyp.add(i)
                    used_ref.add(j)

        exact_weight, stem_weight, synonym_weight = 1.0, 0.6, 0.8
        total_matches = (len(final_matches['exact']) * exact_weight +
                         len(final_matches['stem']) * stem_weight +
                         len(final_matches['synonym']) * synonym_weight)

        if total_matches == 0:
            continue
        precision = total_matches / len(hypothesis)
        recall = total_matches / len(reference)
        if precision + recall == 0:
            continue
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        all_aligned_hyp = sorted([m[0] for v in final_matches.values() for m in v])
        if not all_aligned_hyp:
            continue
        chunks = 1
        for i in range(1, len(all_aligned_hyp)):
            if all_aligned_hyp[i] != all_aligned_hyp[i-1] + 1:
                chunks += 1
        penalty = gamma * (chunks / len(all_aligned_hyp)) ** beta
        meteor_score = f_mean * (1 - penalty)
        best_meteor = max(best_meteor, meteor_score)

    return max(0.0, best_meteor)


# ---- CIDEr ----
def corrected_cider(references, hypothesis, n_grams=4):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def compute_tf_idf_vector(ngram_counts, doc_frequencies, total_docs):
        tfidf = {}
        total_ngrams = sum(ngram_counts.values())
        for ngram, count in ngram_counts.items():
            tf = count / max(total_ngrams, 1)
            idf = math.log(total_docs / max(doc_frequencies.get(ngram, 1), 1))
            tfidf[ngram] = tf * idf
        return tfidf

    def cosine_similarity(vec1, vec2):
        common = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[n] * vec2[n] for n in common)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    if not isinstance(references[0], list):
        references = [references]
    if not hypothesis or not any(references):
        return 0.0

    all_refs = []
    for ref_group in references:
        all_refs.extend(ref_group if isinstance(ref_group[0], list) else [ref_group])
    doc_frequencies, total_docs = defaultdict(int), len(all_refs)
    for ref in all_refs:
        seen = set()
        for n in range(1, n_grams+1):
            for ngram in get_ngrams(ref, n):
                if ngram not in seen:
                    doc_frequencies[ngram] += 1
                    seen.add(ngram)

    cider_scores = []
    for n in range(1, n_grams+1):
        hyp_counts = Counter(get_ngrams(hypothesis, n))
        hyp_tfidf = compute_tf_idf_vector(hyp_counts, doc_frequencies, total_docs)
        sims = []
        for ref in all_refs:
            ref_counts = Counter(get_ngrams(ref, n))
            ref_tfidf = compute_tf_idf_vector(ref_counts, doc_frequencies, total_docs)
            sims.append(cosine_similarity(hyp_tfidf, ref_tfidf))
        cider_scores.append(sum(sims)/len(sims))
    return sum(cider_scores)/len(cider_scores)


# ================= DATA LOADING =================
def load_reference_captions(reference_file):
    reference_captions = {}
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    img, caption = parts
                    reference_captions.setdefault(img.strip(), []).append(caption.strip())
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


# ================= MAIN =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    emb_dim, hidden_dim, factored_dim = 300, 512, 512
    encoder = EncoderViT(emb_dim).to(device)
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, len(tokenizer)).to(device)
    encoder.load_state_dict(torch.load('stylenet_new_again_models/encoder-last.pkl', map_location=device))
    decoder.load_state_dict(torch.load('stylenet_new_again_models/decoder-last.pkl', map_location=device))
    encoder.eval(); decoder.eval()

    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
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
            output = decoder.sample(features, tokenizer=tokenizer, beam_size=5, max_len=30, mode="factual")
            caption = tokenizer.decode(output, skip_special_tokens=True)
            print(f"\n{img_names[idx]} | Generated Caption: {caption}")

        ref_list = reference_captions.get(img_names[idx], None)
        if ref_list:
            hyp_tokens = tokenizer.tokenize(caption)
            ref_tokens_list = [tokenizer.tokenize(r) for r in ref_list]

            bleu = nltk_bleu(ref_tokens_list, hyp_tokens)
            rouge = rouge_l_score(ref_tokens_list, hyp_tokens)
            meteor = corrected_meteor(ref_tokens_list, hyp_tokens)
            cider = corrected_cider([ref_tokens_list], hyp_tokens)

            all_bleu.append(bleu); all_rouge.append(rouge)
            all_meteor.append(meteor); all_cider.append(cider)

            print(f"References: {ref_list}")
            print(f"BLEU-4={bleu:.4f} | ROUGE-L={rouge:.4f} | METEOR={meteor:.4f} | CIDEr={cider:.4f}")
        else:
            print("⚠️ No references found")

    if all_bleu:
        print("\n===== FINAL AVERAGES =====")
        print(f"BLEU-4: {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print(f"METEOR: {sum(all_meteor)/len(all_meteor):.4f}")
        print(f"CIDEr: {sum(all_cider)/len(all_cider):.4f}")


if __name__ == '__main__':
    main()
