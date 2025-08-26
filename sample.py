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

# ---- Corrected BLEU-4 implementation ----
def corrected_bleu(references, hypothesis, max_n=4):
    """
    Proper BLEU implementation following the original paper
    """
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    def modified_precision(ref_ngrams, hyp_ngrams):
        """Calculate clipped n-gram precision"""
        if not hyp_ngrams:
            return 0.0
        
        clipped_count = 0
        total_count = sum(hyp_ngrams.values())
        
        for ngram, count in hyp_ngrams.items():
            # Clip the count by the maximum reference count for this ngram
            max_ref_count = max([ref_ngram.get(ngram, 0) for ref_ngram in ref_ngrams])
            clipped_count += min(count, max_ref_count)
        
        return clipped_count / total_count if total_count > 0 else 0.0
    
    # Handle single reference case
    if not isinstance(references[0], list):
        references = [references]
    
    # Calculate reference lengths and find closest reference length
    ref_lengths = [len(ref) for ref in references]
    hyp_len = len(hypothesis)
    closest_ref_len = min(ref_lengths, key=lambda x: abs(x - hyp_len))
    
    # Calculate brevity penalty
    if hyp_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    # Calculate modified precision for each n-gram order
    precisions = []
    
    for n in range(1, max_n + 1):
        # Get n-grams for hypothesis
        hyp_ngrams = get_ngrams(hypothesis, n)
        
        # Get n-grams for all references
        ref_ngrams = [get_ngrams(ref, n) for ref in references]
        
        # Calculate modified precision
        precision = modified_precision(ref_ngrams, hyp_ngrams)
        precisions.append(precision)
    
    # If any precision is 0, BLEU is 0
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Calculate geometric mean of precisions
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    return bp * geo_mean

# ---- Corrected ROUGE-L implementation ----
def corrected_rouge_l(references, hypothesis):
    """
    Proper ROUGE-L implementation with multiple reference support
    """
    def lcs_length(X, Y):
        """Calculate longest common subsequence length"""
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
    
    # Handle single reference case
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

# ---- Corrected METEOR implementation ----
class BengaliWordProcessor:
    """Enhanced Bengali word processing for METEOR"""
    def __init__(self):
        # More comprehensive Bengali morphological patterns
        self.suffix_patterns = [
            # Plural markers
            (r'রা$', ''),      # মানুষরা -> মানুষ
            (r'েরা$', ''),     # ছেলেরা -> ছেলে
            (r'দের$', ''),     # ছেলেদের -> ছেলে
            (r'গুলো$', ''),    # বইগুলো -> বই
            (r'গুলি$', ''),    # বইগুলি -> বই
            (r'সব$', ''),      # মানুষসব -> মানুষ
            
            # Case markers
            (r'কে$', ''),      # রামকে -> রাম
            (r'তে$', ''),      # বাড়িতে -> বাড়ি
            (r'য়$', ''),       # বাড়িয় -> বাড়ি
            (r'র$', ''),       # রামের -> রাম
            (r'এর$', ''),      # বাড়ির -> বাড়ি
            
            # Verb inflections
            (r'ছে$', ''),      # করছে -> কর
            (r'ছি$', ''),      # করছি -> কর
            (r'ছেন$', ''),     # করছেন -> কর
            (r'বে$', ''),      # করবে -> কর
            (r'বো$', ''),      # করবো -> কর
            (r'বেন$', ''),     # করবেন -> কর
            (r'ল$', ''),       # করল -> কর
            (r'লো$', ''),      # করলো -> কর
            (r'লেন$', ''),     # করলেন -> কর
            (r'লাম$', ''),     # করলাম -> কর
            
            # Adjective/adverb suffixes
            (r'টি$', ''),      # লালটি -> লাল
            (r'টা$', ''),      # লালটা -> লাল
            (r'খানি$', ''),    # একখানি -> এক
            (r'খানা$', ''),    # একখানা -> এক
        ]
        
        # Extended Bengali synonyms for image captioning
        self.synonyms = {
            'মানুষ': ['ব্যক্তি', 'লোক', 'জন', 'মানব'],
            'ব্যক্তি': ['মানুষ', 'লোক', 'জন'],
            'পুরুষ': ['মরদ', 'ছেলে', 'পুরুশ'],
            'মহিলা': ['নারী', 'মেয়ে', 'স্ত্রী', 'বেগম'],
            'নারী': ['মহিলা', 'মেয়ে'],
            'শিশু': ['বাচ্চা', 'ছোট', 'বালক', 'বালিকা', 'কিশোর'],
            'বাচ্চা': ['শিশু', 'ছোট', 'বাল'],
            'গাড়ি': ['যান', 'মোটর', 'কার'],
            'কুকুর': ['কুত্তা', 'শ্বান', 'কুত্তো'],
            'বিড়াল': ['মার্জার', 'বেড়াল'],
            'বাড়ি': ['ঘর', 'গৃহ', 'আবাস', 'নিবাস'],
            'ঘর': ['বাড়ি', 'গৃহ', 'কক্ষ', 'কামরা'],
            'বড়': ['বৃহৎ', 'বিরাট', 'বিশাল', 'বড়ো'],
            'ছোট': ['ক্ষুদ্র', 'তুচ্ছ', 'সূক্ষ্ম', 'ছোটো'],
            'সুন্দর': ['মনোহর', 'চমৎকার', 'আকর্ষণীয়', 'ভালো'],
            'পুরাতন': ['পুরোনো', 'প্রাচীন', 'পুরানো'],
            'নতুন': ['নব', 'তাজা', 'আধুনিক', 'নূতন'],
            'খুশি': ['আনন্দিত', 'প্রফুল্ল', 'হাসিখুশি'],
            'দুঃখিত': ['কষ্টিত', 'বিষণ্ণ', 'দুখী'],
            'হাঁটা': ['চলা', 'গমন', 'হাঁটু'],
            'দৌড়ানো': ['ছোটা', 'দৌড়', 'ছুটে'],
            'খাওয়া': ['আহার', 'ভোজন', 'খাদ্য'],
            'পানি': ['জল', 'নীর'],
            'জল': ['পানি', 'নীর'],
            'খাবার': ['আহার', 'খাদ্য', 'ভোজন', 'খানা'],
            'লাল': ['রক্তিম', 'রাঙা', 'লাল'],
            'নীল': ['আকাশী', 'নিল'],
            'সবুজ': ['হরিৎ', 'সবুজ'],
            'কালো': ['কৃষ্ণ', 'শ্যাম', 'কাল'],
            'সাদা': ['শুভ্র', 'ধবল', 'সাদ'],
        }
        
        # Build bidirectional synonym mapping
        self.word_to_synonyms = defaultdict(set)
        for word, syns in self.synonyms.items():
            all_variants = [word] + syns
            for variant in all_variants:
                self.word_to_synonyms[variant].update(all_variants)
    
    def basic_stem(self, word):
        """Enhanced Bengali stemming"""
        original_word = word
        for pattern, replacement in self.suffix_patterns:
            word = re.sub(pattern, replacement, word)
        return word if word != original_word else original_word
    
    def get_synonyms(self, word):
        """Get all synonyms including the word itself"""
        return self.word_to_synonyms.get(word, {word})

def corrected_meteor(references, hypothesis, alpha=0.9, beta=3.0, gamma=0.5):
    """
    Corrected METEOR implementation with proper alignment and multiple references
    """
    # Handle single reference case
    if not isinstance(references[0], list):
        references = [references]
    
    if not hypothesis or not any(references):
        return 0.0
    
    processor = BengaliWordProcessor()
    best_meteor = 0.0
    
    for reference in references:
        if not reference:
            continue
            
        # Create alignment matrix for exact, stem, and synonym matches
        alignments = []
        
        # Find all possible matches
        matches = {'exact': [], 'stem': [], 'synonym': []}
        
        ref_stems = [processor.basic_stem(word) for word in reference]
        hyp_stems = [processor.basic_stem(word) for word in hypothesis]
        
        # Find matches
        for i, hyp_word in enumerate(hypothesis):
            for j, ref_word in enumerate(reference):
                if hyp_word == ref_word:
                    matches['exact'].append((i, j))
                elif processor.basic_stem(hyp_word) == processor.basic_stem(ref_word):
                    matches['stem'].append((i, j))
                elif ref_word in processor.get_synonyms(hyp_word):
                    matches['synonym'].append((i, j))
        
        # Create alignment using greedy approach (prevent many-to-one mappings)
        used_hyp = set()
        used_ref = set()
        final_matches = {'exact': [], 'stem': [], 'synonym': []}
        
        # Prioritize exact matches
        for i, j in matches['exact']:
            if i not in used_hyp and j not in used_ref:
                final_matches['exact'].append((i, j))
                used_hyp.add(i)
                used_ref.add(j)
        
        # Then stem matches
        for i, j in matches['stem']:
            if i not in used_hyp and j not in used_ref:
                final_matches['stem'].append((i, j))
                used_hyp.add(i)
                used_ref.add(j)
        
        # Finally synonym matches
        for i, j in matches['synonym']:
            if i not in used_hyp and j not in used_ref:
                final_matches['synonym'].append((i, j))
                used_hyp.add(i)
                used_ref.add(j)
        
        # Calculate weighted matches (standard METEOR weights)
        exact_weight = 1.0
        stem_weight = 0.6
        synonym_weight = 0.8
        
        total_matches = (len(final_matches['exact']) * exact_weight + 
                        len(final_matches['stem']) * stem_weight + 
                        len(final_matches['synonym']) * synonym_weight)
        
        if total_matches == 0:
            continue
        
        # Calculate precision and recall
        precision = total_matches / len(hypothesis) if hypothesis else 0
        recall = total_matches / len(reference) if reference else 0
        
        if precision + recall == 0:
            continue
        
        # F-mean calculation
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        
        # Calculate fragmentation penalty
        all_aligned_hyp = []
        for match_type in final_matches.values():
            all_aligned_hyp.extend([match[0] for match in match_type])
        
        if not all_aligned_hyp:
            continue
            
        all_aligned_hyp.sort()
        chunks = 1
        
        for i in range(1, len(all_aligned_hyp)):
            if all_aligned_hyp[i] != all_aligned_hyp[i-1] + 1:
                chunks += 1
        
        # Apply fragmentation penalty
        penalty = gamma * (chunks / len(all_aligned_hyp)) ** beta
        meteor_score = f_mean * (1 - penalty)
        
        best_meteor = max(best_meteor, meteor_score)
    
    return max(0.0, best_meteor)

# ---- Corrected CIDEr implementation ----
def corrected_cider(references, hypothesis, n_grams=4):
    """
    Corrected CIDEr implementation following the original paper
    """
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def compute_tf_idf_vector(ngram_counts, doc_frequencies, total_docs):
        """Compute TF-IDF vector for a sentence"""
        tfidf = {}
        total_ngrams = sum(ngram_counts.values())
        
        for ngram, count in ngram_counts.items():
            # Term frequency (normalized)
            tf = count / max(total_ngrams, 1)
            
            # Inverse document frequency with smoothing
            idf = math.log(total_docs / max(doc_frequencies.get(ngram, 1), 1))
            
            tfidf[ngram] = tf * idf
        
        return tfidf
    
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two TF-IDF vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        common_ngrams = set(vec1.keys()) & set(vec2.keys())
        if not common_ngrams:
            return 0.0
        
        dot_product = sum(vec1[ngram] * vec2[ngram] for ngram in common_ngrams)
        
        # Calculate norms
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    # Handle single reference case
    if not isinstance(references[0], list):
        references = [references]
    
    if not hypothesis or not any(references):
        return 0.0
    
    # Flatten all references for document frequency calculation
    all_refs = []
    for ref_group in references:
        if ref_group:  # Skip empty references
            all_refs.extend(ref_group if isinstance(ref_group[0], list) else [ref_group])
    
    if not all_refs:
        return 0.0
    
    # Calculate document frequencies for all n-grams
    doc_frequencies = defaultdict(int)
    total_docs = len(all_refs)
    
    for ref in all_refs:
        seen_ngrams = set()
        for n in range(1, n_grams + 1):
            ngrams = get_ngrams(ref, n)
            for ngram in ngrams:
                if ngram not in seen_ngrams:
                    doc_frequencies[ngram] += 1
                    seen_ngrams.add(ngram)
    
    # Calculate CIDEr score for each n-gram order
    cider_scores = []
    
    for n in range(1, n_grams + 1):
        # Get hypothesis n-grams
        hyp_ngrams = get_ngrams(hypothesis, n)
        if not hyp_ngrams:
            continue
        
        hyp_counts = Counter(hyp_ngrams)
        hyp_tfidf = compute_tf_idf_vector(hyp_counts, doc_frequencies, total_docs)
        
        # Calculate similarity with each reference
        similarities = []
        
        for ref in all_refs:
            ref_ngrams = get_ngrams(ref, n)
            if not ref_ngrams:
                similarities.append(0.0)
                continue
            
            ref_counts = Counter(ref_ngrams)
            ref_tfidf = compute_tf_idf_vector(ref_counts, doc_frequencies, total_docs)
            
            similarity = cosine_similarity(hyp_tfidf, ref_tfidf)
            similarities.append(similarity)
        
        # Average similarity for this n-gram order
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            cider_scores.append(avg_similarity)
    
    # Return average CIDEr across all n-gram orders
    return sum(cider_scores) / len(cider_scores) if cider_scores else 0.0

# ---- Reference captions loader ----
def load_reference_captions(reference_file):
    reference_captions = {}
    with open(reference_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(None, 1)  # split by any whitespace (tab or space)
                if len(parts) == 2:
                    img_name, caption = parts
                    img_name = img_name.strip()
                    caption = caption.strip()
                    if img_name not in reference_captions:
                        reference_captions[img_name] = []
                    reference_captions[img_name].append(caption)
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
    print("=" * 70)
    print("CORRECTED BENGALI IMAGE CAPTIONING EVALUATION")
    print("=" * 70)
    print("Tokenizer:", tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else 'Custom Bengali Tokenizer')

    # Dimensions (must match training)
    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512

    encoder = EncoderViT(emb_dim).to(device)
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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_dir = '/kaggle/input/sample-data/sample/sample_images'  # Change as needed
    reference_file = '/kaggle/input/sample-data/sample/sample_images_factual.txt'  # Add this path
    img_names, img_list = load_sample_images(img_dir, transform, device)
    reference_captions = load_reference_captions(reference_file)

    # ------- Evaluation metrics storage -------
    all_bleu = []
    all_rouge = []
    all_meteor = []
    all_cider = []

    print(f"Found {len(img_list)} images for evaluation")
    print("Using CORRECTED implementations:")
    print("- BLEU-4: Proper clipped precision with brevity penalty")
    print("- ROUGE-L: Correct LCS calculation with multiple references")
    print("- METEOR: Enhanced alignment with fragmentation penalty")
    print("- CIDEr: Proper TF-IDF weighting and cosine similarity")
    print("=" * 70)

    # --------- Main evaluation loop ----------
    for idx, image in enumerate(img_list):
        with torch.no_grad():
            features = encoder(image)
            print(f"\n--- Processing Image {idx+1}/{len(img_list)}: {img_names[idx]} ---")

            # ---- Caption generation ----
            output = decoder.sample(
                features,
                tokenizer=tokenizer,
                beam_size=5,
                max_len=30,
                mode="factual"  # Use factual mode for evaluation
            )
            caption = tokenizer.decode(output, skip_special_tokens=True)
            print(f"Generated Caption (Bengali): {caption}")

        # ------- Evaluation with corrected metrics ---------
        img_name = img_names[idx]
        ref_list = reference_captions.get(img_name, None)

        if ref_list is not None:
            print(f"Reference Captions ({len(ref_list)}): {ref_list}")
            
            # Tokenize hypothesis and references
            hyp_tokens = tokenizer.tokenize(caption)
            ref_tokens_list = [tokenizer.tokenize(ref_caption) for ref_caption in ref_list]
            
            print(f"Hypothesis tokens: {hyp_tokens}")
            print(f"Reference tokens (first): {ref_tokens_list[0] if ref_tokens_list else []}")
            
            # Calculate corrected metrics
            bleu_score = corrected_bleu(ref_tokens_list, hyp_tokens)
            rouge_score = corrected_rouge_l(ref_tokens_list, hyp_tokens)
            meteor_score = corrected_meteor(ref_tokens_list, hyp_tokens)
            cider_score = corrected_cider([ref_tokens_list], hyp_tokens)  # Note the extra list wrapping

            # Store scores
            all_bleu.append(bleu_score)
            all_rouge.append(rouge_score)
            all_meteor.append(meteor_score)
            all_cider.append(cider_score)
            
            # Print results
            print(f"📊 CORRECTED EVALUATION RESULTS:")
            print(f"   BLEU-4:   {bleu_score:.4f}")
            print(f"   ROUGE-L:  {rouge_score:.4f}")
            print(f"   METEOR:   {meteor_score:.4f}")
            print(f"   CIDEr:    {cider_score:.4f}")
            print("-" * 50)
        else:
            print(f"⚠️ Reference captions NOT FOUND for {img_name}")
            print("-" * 50)

    # ------- Print comprehensive evaluation results ---------
    if all_bleu and all_rouge and all_meteor and all_cider:
        print("=" * 70)
        print("📈 FINAL CORRECTED BENGALI CAPTIONING EVALUATION")
        print("=" * 70)
        print(f"Average BLEU-4:  {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"Average ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print(f"Average METEOR:  {sum(all_meteor)/len(all_meteor):.4f}")
        print(f"Average CIDEr:   {sum(all_cider)/len(all_cider):.4f}")
        print("=" * 70)
        
        # Additional statistics
        print("\n📊 DETAILED STATISTICS:")
        print("-" * 40)
        
        metrics = {
            'BLEU-4': all_bleu,
            'ROUGE-L': all_rouge,
            'METEOR': all_meteor,
            'CIDEr': all_cider
        }
        
        for metric_name, scores in metrics.items():
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            std_score = (sum((x - avg_score)**2 for x in scores) / len(scores))**0.5
            
            print(f"{metric_name:>8}: Avg={avg_score:.4f}, Max={max_score:.4f}, Min={min_score:.4f}, Std={std_score:.4f}")
        
        print("\n🔍 PERFORMANCE ANALYSIS:")
        print("-" * 40)
        avg_bleu = sum(all_bleu)/len(all_bleu)
        avg_meteor = sum(all_meteor)/len(all_meteor)
        avg_cider = sum(all_cider)/len(all_cider)
        avg_rouge = sum(all_rouge)/len(all_rouge)
        
        if avg_bleu > 0.2:
            print("✅ BLEU score indicates reasonable n-gram overlap")
        else:
            print("❌ BLEU score suggests need for improvement in word choice")
            
        if avg_meteor > 0.2:
            print("✅ METEOR score shows good semantic similarity")
        else:
            print("❌ METEOR score indicates need for better semantic alignment")
            
        if avg_cider > 0.3:
            print("✅ CIDEr score demonstrates decent consensus with references")
        else:
            print("❌ CIDEr score suggests captions need more conventional patterns")
            
        if avg_rouge > 0.2:
            print("✅ ROUGE-L score indicates good sequence overlap")
        else:
            print("❌ ROUGE-L score suggests issues with word ordering")
    else:
        print("❌ No evaluation data collected. Please check your reference file and image directory.")

if __name__ == '__main__':
    main()
