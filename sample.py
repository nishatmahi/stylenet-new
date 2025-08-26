import os
import torch
import math
import re
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import Rescale, tokenizer
from models import EncoderViT, FactoredLSTM
from collections import Counter, defaultdict

# ============================================================
# ---- BLEU-4 ----
# ============================================================
def corrected_bleu(references, hypothesis, max_n=4):
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    def modified_precision(ref_ngrams, hyp_ngrams):
        if not hyp_ngrams:
            return 0.0
        clipped_count = 0
        total_count = sum(hyp_ngrams.values())
        for ngram, count in hyp_ngrams.items():
            max_ref_count = max([ref_ngram.get(ngram, 0) for ref_ngram in ref_ngrams])
            clipped_count += min(count, max_ref_count)
        return clipped_count / total_count if total_count > 0 else 0.0
    
    if not isinstance(references[0], list):
        references = [references]
    ref_lengths = [len(ref) for ref in references]
    hyp_len = len(hypothesis)
    closest_ref_len = min(ref_lengths, key=lambda x: abs(x - hyp_len))
    bp = 1.0 if hyp_len > closest_ref_len else math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = get_ngrams(hypothesis, n)
        ref_ngrams = [get_ngrams(ref, n) for ref in references]
        precision = modified_precision(ref_ngrams, hyp_ngrams)
        precisions.append(precision)

    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bp * geo_mean

# ============================================================
# ---- ROUGE-L ----
# ============================================================
def corrected_rouge_l(references, hypothesis):
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    if not isinstance(references[0], list):
        references = [references]
    if not hypothesis:
        return 0.0

    best_f1 = 0.0
    for reference in references:
        if not reference:
            continue
        lcs_len = lcs_length(reference, hypothesis)
        if lcs_len == 0:
            continue
        precision = lcs_len / len(hypothesis)
        recall = lcs_len / len(reference)
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)
    return best_f1

# ============================================================
# ---- METEOR ----
# ============================================================
class BengaliWordProcessor:
    def __init__(self):
        self.suffix_patterns = [
            (r'রা$', ''), (r'েরা$', ''), (r'দের$', ''), (r'গুলো$', ''), (r'গুলি$', ''), (r'সব$', ''),
            (r'কে$', ''), (r'তে$', ''), (r'য়$', ''), (r'র$', ''), (r'এর$', ''),
            (r'ছে$', ''), (r'ছি$', ''), (r'ছেন$', ''), (r'বে$', ''), (r'বো$', ''), (r'বেন$', ''),
            (r'ল$', ''), (r'লো$', ''), (r'লেন$', ''), (r'লাম$', ''),
            (r'টি$', ''), (r'টা$', ''), (r'খানি$', ''), (r'খানা$', '')
        ]
        # expanded synonyms (include dataset terms here if needed)
        self.synonyms = {
            'মানুষ': ['ব্যক্তি', 'লোক', 'জন'],
            'শিশু': ['বাচ্চা', 'ছোট'],
            'গাড়ি': ['যান', 'মোটর', 'কার'],
            'কুকুর': ['শ্বান'],
            'বিড়াল': ['বেড়াল'],
            'বাড়ি': ['ঘর', 'গৃহ'],
            'সুন্দর': ['চমৎকার', 'ভালো'],
            'বড়': ['বিরাট', 'বিশাল'],
            'ছোট': ['ক্ষুদ্র'],
            'পানি': ['জল'],
            'খাবার': ['আহার', 'ভোজন'],
        }
        self.word_to_synonyms = defaultdict(set)
        for word, syns in self.synonyms.items():
            all_variants = [word] + syns
            for variant in all_variants:
                self.word_to_synonyms[variant].update(all_variants)

    def basic_stem(self, word):
        original_word = word
        for pattern, replacement in self.suffix_patterns:
            word = re.sub(pattern, replacement, word)
        return word if word != original_word else original_word

    def get_synonyms(self, word):
        return self.word_to_synonyms.get(word, {word})

def corrected_meteor(references, hypothesis, alpha=0.9, beta=3.0, gamma=0.5):
    if not isinstance(references[0], list):
        references = [references]
    if not hypothesis or not any(references):
        return 0.0
    processor = BengaliWordProcessor()
    best_meteor = 0.0
    for reference in references:
        if not reference:
            continue
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
        for cat in ['exact', 'stem', 'synonym']:
            for i, j in matches[cat]:
                if i not in used_hyp and j not in used_ref:
                    final_matches[cat].append((i, j))
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
        all_aligned_hyp = [i for cat in final_matches.values() for (i, _) in cat]
        if not all_aligned_hyp:
            continue
        all_aligned_hyp.sort()
        chunks = 1
        for i in range(1, len(all_aligned_hyp)):
            if all_aligned_hyp[i] != all_aligned_hyp[i-1] + 1:
                chunks += 1
        penalty = gamma * (chunks / len(all_aligned_hyp)) ** beta
        meteor_score = f_mean * (1 - penalty)
        best_meteor = max(best_meteor, meteor_score)
    return max(0.0, best_meteor)

# ============================================================
# ---- CIDEr ----
# ============================================================
def corrected_cider(references, hypothesis, n_grams=4):
    def get_ngrams(tokens, n):
        if len(tokens) < n: return []
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
        dot = sum(vec1[x] * vec2[x] for x in common)
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
    doc_frequencies = defaultdict(int)
    total_docs = len(all_refs)
    for ref in all_refs:
        seen = set()
        for n in range(1, n_grams+1):
            for ng in get_ngrams(ref, n):
                if ng not in seen:
                    doc_frequencies[ng] += 1
                    seen.add(ng)
    cider_scores = []
    for n in range(1, n_grams+1):
        hyp_counts = Counter(get_ngrams(hypothesis, n))
        if not hyp_counts: continue
        hyp_vec = compute_tf_idf_vector(hyp_counts, doc_frequencies, total_docs)
        sims = []
        for ref in all_refs:
            ref_counts = Counter(get_ngrams(ref, n))
            if not ref_counts: continue
            ref_vec = compute_tf_idf_vector(ref_counts, doc_frequencies, total_docs)
            sims.append(cosine_similarity(hyp_vec, ref_vec))
        if sims:
            cider_scores.append(sum(sims)/len(sims))
    return sum(cider_scores)/len(cider_scores) if cider_scores else 0.0

# ============================================================
# ---- Reference Loader ----
# ============================================================
def load_reference_captions(reference_file):
    reference_captions = {}
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    img_name, caption = parts
                    reference_captions.setdefault(img_name.strip(), []).append(caption.strip())
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

# ============================================================
# ---- MAIN ----
# ============================================================
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
            output = decoder.sample(features, tokenizer=tokenizer, beam_size=5, max_len=30, mode="factual")
            caption = tokenizer.decode(output, skip_special_tokens=True)
            print(f"\n{img_names[idx]} | Predicted Caption: {caption}")

        ref_list = reference_captions.get(img_names[idx], None)
        if ref_list:
            hyp_tokens = tokenizer.tokenize(caption)
            ref_tokens_list = [tokenizer.tokenize(r) for r in ref_list]

            bleu_score = corrected_bleu(ref_tokens_list, hyp_tokens)
            rouge_score = corrected_rouge_l(ref_tokens_list, hyp_tokens)
            meteor_score = corrected_meteor(ref_tokens_list, hyp_tokens)
            cider_score = corrected_cider([ref_tokens_list], hyp_tokens)

            all_bleu.append(bleu_score)
            all_rouge.append(rouge_score)
            all_meteor.append(meteor_score)
            all_cider.append(cider_score)

            print(f"   BLEU-4: {bleu_score:.4f} | ROUGE-L: {rouge_score:.4f} | METEOR: {meteor_score:.4f} | CIDEr: {cider_score:.4f}")

    if all_bleu:
        print("="*70)
        print(f"Average BLEU-4: {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"Average ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print(f"Average METEOR: {sum(all_meteor)/len(all_meteor):.4f}")
        print(f"Average CIDEr: {sum(all_cider)/len(all_cider):.4f}")
        print("="*70)

if __name__ == '__main__':
    main()
