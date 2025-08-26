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

# ---- Improved BLEU-4 implementation with proper clipping and smoothing ----
def improved_bleu(reference, hypothesis, n=4, smooth=True):
    """
    Improved BLEU implementation with:
    - Proper n-gram clipping (standard BLEU requirement)
    - Smoothing to handle zero counts
    - Better edge case handling
    """
    from collections import Counter
    
    if len(hypothesis) == 0:
        return 0.0
    if len(reference) == 0:
        return 0.0
    
    def ngram_counts(tokens, n):
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    weights = [0.25, 0.25, 0.25, 0.25]
    p_ns = []
    
    for i in range(1, n+1):
        ref_counts = ngram_counts(reference, i)
        hyp_counts = ngram_counts(hypothesis, i)
        
        if sum(hyp_counts.values()) == 0:
            if smooth:
                p_ns.append(0.01)  # Small smoothing value
            else:
                p_ns.append(0.0)
            continue
        
        # Apply clipping: count each n-gram at most as many times as it appears in reference
        clipped_counts = {}
        for ngram, count in hyp_counts.items():
            clipped_counts[ngram] = min(count, ref_counts.get(ngram, 0))
        
        overlap = sum(clipped_counts.values())
        total = sum(hyp_counts.values())
        
        precision = overlap / total if total > 0 else 0
        p_ns.append(precision)
    
    # Brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    if hyp_len <= 0:
        return 0.0
    elif hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    
    # Geometric mean of precisions
    if any(p == 0 for p in p_ns):
        if smooth:
            # Add small epsilon to avoid log(0)
            log_sum = sum(w * math.log(max(p, 1e-10)) for w, p in zip(weights, p_ns))
        else:
            return 0.0
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(weights, p_ns))
    
    bleu = bp * math.exp(log_sum)
    return bleu

def simple_rouge_l(reference, hypothesis):
    """
    Simple ROUGE-L implementation based on longest common subsequence
    """
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
    
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0.0
    
    lcs_len = lcs(reference, hypothesis)
    prec = lcs_len / len(hypothesis) if hypothesis else 0
    rec = lcs_len / len(reference) if reference else 0
    if prec + rec == 0:
        f1 = 0
    else:
        f1 = (2 * prec * rec) / (prec + rec)
    return f1

# ---- Tokenizer-based METEOR implementation ----
class TokenizerBasedProcessor:
    """METEOR processing using your pretrained Bengali tokenizer"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Basic Bengali suffixes - kept minimal since tokenizer handles most morphology
        self.basic_suffixes = [
            'রা', 'দের', 'গুলো', 'গুলি', 'কে', 'তে', 'র'
        ]
        
        # Since you have a pretrained tokenizer, we can try to use its vocabulary
        # to find semantically similar tokens based on embedding proximity
        # This is a placeholder - your tokenizer might have better methods
    
    def get_token_variants(self, word):
        """
        Get token variants using tokenizer's knowledge
        This is a basic implementation - your tokenizer might have better methods
        """
        variants = {word}
        
        # Try basic suffix removal if tokenizer supports it
        for suffix in self.basic_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                stem = word[:-len(suffix)]
                variants.add(stem)
        
        # If your tokenizer has a get_similar_tokens or embedding method, 
        # you could use it here for better semantic matching
        return variants
    
    def tokenize_and_process(self, text):
        """Use your tokenizer for processing"""
        return self.tokenizer.tokenize(text)
    
    def basic_stem(self, word):
        """Simplified stemming - tokenizer should handle most cases"""
        for suffix in self.basic_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return word

def tokenizer_based_meteor(reference, hypothesis, tokenizer, alpha=0.9, beta=3.0, gamma=0.5):
    """
    METEOR implementation using your pretrained Bengali tokenizer
    """
    processor = TokenizerBasedProcessor(tokenizer)
    
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0
    
    # Use tokenizer's stemming capabilities if available
    ref_stems = [processor.basic_stem(word) for word in reference]
    hyp_stems = [processor.basic_stem(word) for word in hypothesis]
    
    # Track different types of matches
    exact_matches = set()
    stem_matches = set()
    
    # Find exact matches
    for i, hyp_word in enumerate(hypothesis):
        for j, ref_word in enumerate(reference):
            if hyp_word == ref_word:
                exact_matches.add((i, j))
    
    # Find stem matches (excluding exact matches)
    for i, hyp_stem in enumerate(hyp_stems):
        for j, ref_stem in enumerate(ref_stems):
            if (i, j) not in exact_matches and hyp_stem == ref_stem and hyp_stem != hypothesis[i]:
                stem_matches.add((i, j))
    
    # Calculate matches (simplified without synonyms for now)
    # Your tokenizer likely handles subword tokenization which is better than manual synonyms
    total_matches = len(exact_matches) + 0.8 * len(stem_matches)
    
    if total_matches == 0:
        return 0
    
    # Calculate precision and recall
    precision = total_matches / len(hypothesis)
    recall = total_matches / len(reference)
    
    if precision + recall == 0:
        return 0
    
    # F-mean with alpha parameter
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Calculate fragmentation penalty
    all_matches = exact_matches.union(stem_matches)
    if len(all_matches) == 0:
        return 0
    
    # Simple chunk calculation
    hyp_positions = sorted([match[0] for match in all_matches])
    chunks = 1
    for i in range(1, len(hyp_positions)):
        if hyp_positions[i] != hyp_positions[i-1] + 1:
            chunks += 1
    
    # Apply fragmentation penalty
    penalty = gamma * (chunks / len(all_matches)) ** beta if len(all_matches) > 0 else 1
    meteor_score = f_mean * (1 - penalty)
    
    return max(0, meteor_score)

# ---- Enhanced CIDEr implementation for Bengali ----
def enhanced_cider(references, hypothesis, n_grams=4):
    """
    CIDEr implementation adapted for Bengali tokenization
    """
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def compute_tf_idf(ngram_counts, doc_freq, total_docs):
        tf_idf = {}
        total_ngrams = sum(ngram_counts.values())
        
        for ngram, count in ngram_counts.items():
            # TF: normalized term frequency
            tf = count / max(total_ngrams, 1)
            
            # IDF: inverse document frequency with smoothing
            idf = math.log(max(total_docs, 1) / max(doc_freq.get(ngram, 0), 1))
            
            tf_idf[ngram] = tf * idf
        
        return tf_idf
    
    # Handle different reference formats
    if isinstance(references, list) and len(references) > 0:
        if not isinstance(references[0], list):
            all_refs = [references]  # Single reference case
        else:
            all_refs = references    # Multiple references case
    else:
        return 0
    
    # Flatten all reference sentences
    all_ref_sentences = []
    for ref_list in all_refs:
        if isinstance(ref_list, list):
            all_ref_sentences.extend(ref_list)
        else:
            all_ref_sentences.append(ref_list)
    
    if len(all_ref_sentences) == 0:
        return 0
    
    # Compute document frequency for n-grams
    doc_freq = defaultdict(int)
    for ref in all_ref_sentences:
        seen_ngrams = set()
        for n in range(1, n_grams + 1):
            ngrams = get_ngrams(ref, n)
            for ngram in ngrams:
                if ngram not in seen_ngrams:
                    doc_freq[ngram] += 1
                    seen_ngrams.add(ngram)
    
    total_docs = len(all_ref_sentences)
    
    # Calculate CIDEr for each n-gram order
    cider_scores = []
    
    for n in range(1, n_grams + 1):
        # Get hypothesis n-grams
        hyp_ngrams = get_ngrams(hypothesis, n)
        if not hyp_ngrams:
            continue
            
        hyp_counts = Counter(hyp_ngrams)
        hyp_tfidf = compute_tf_idf(hyp_counts, doc_freq, total_docs)
        
        # Calculate average similarity with all references
        similarities = []
        
        for ref in all_ref_sentences:
            ref_ngrams = get_ngrams(ref, n)
            if not ref_ngrams:
                continue
                
            ref_counts = Counter(ref_ngrams)
            ref_tfidf = compute_tf_idf(ref_counts, doc_freq, total_docs)
            
            # Compute cosine similarity
            common_ngrams = set(hyp_tfidf.keys()) & set(ref_tfidf.keys())
            
            if not common_ngrams:
                similarities.append(0)
                continue
            
            dot_product = sum(hyp_tfidf[ngram] * ref_tfidf[ngram] for ngram in common_ngrams)
            
            hyp_norm = math.sqrt(sum(v**2 for v in hyp_tfidf.values()))
            ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
            
            if hyp_norm > 0 and ref_norm > 0:
                similarity = dot_product / (hyp_norm * ref_norm)
                similarities.append(similarity)
            else:
                similarities.append(0)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            cider_scores.append(avg_similarity)
    
    return sum(cider_scores) / len(cider_scores) if cider_scores else 0

# ---- Reference captions loader ----
def load_reference_captions(reference_file):
    """
    Load reference captions from file
    """
    reference_captions = {}
    
    if not os.path.exists(reference_file):
        print(f"Warning: Reference file {reference_file} not found!")
        return reference_captions
        
    try:
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
    except Exception as e:
        print(f"Error loading reference captions: {e}")
        
    return reference_captions

def load_sample_images(img_dir, transform, device):
    """
    Load and transform sample images
    """
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory {img_dir} not found!")
        return [], []
        
    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    img_list = []
    
    for img_name in img_names:
        try:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            img_list.append(img)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            
    return img_names, img_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 70)
    print("BENGALI IMAGE CAPTIONING EVALUATION")
    print("=" * 70)
    print("Tokenizer:", tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else 'Custom Bengali Tokenizer')

    # Dimensions (must match training)
    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512

    # Initialize models
    try:
        encoder = EncoderViT(emb_dim).to(device)
        decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, len(tokenizer)).to(device)

        # Load weights
        encoder.load_state_dict(torch.load('stylenet_new_again_models/encoder-last.pkl', map_location=device))
        decoder.load_state_dict(torch.load('stylenet_new_again_models/decoder-last.pkl', map_location=device))

        encoder.eval()
        decoder.eval()
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Prepare image transform
    transform = transforms.Compose([
        Rescale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Load images and references
    img_dir = '/kaggle/input/sample-data/sample/sample_images'  # Change as needed
    reference_file = '/kaggle/input/sample-data/sample/sample_images_factual.txt'  # Add this path
    img_names, img_list = load_sample_images(img_dir, transform, device)
    reference_captions = load_reference_captions(reference_file)

    if not img_list:
        print("❌ No images found! Please check your image directory path.")
        return

    # ------- Evaluation metrics storage -------
    all_bleu = []
    all_rouge = []
    all_meteor = []
    all_cider = []

    print(f"Found {len(img_list)} images for evaluation")
    print("METEOR parameters: α=0.9 (recall-focused), β=3.0, γ=0.5")
    print("Note: α is a METEOR evaluation parameter, not from your training")
    print("=" * 70)

    # --------- Main evaluation loop ----------
    for idx, image in enumerate(img_list):
        try:
            with torch.no_grad():
                features = encoder(image)
                print(f"\n--- Processing Image {idx+1}/{len(img_list)}: {img_names[idx]} ---")
                print("Image features shape:", features.shape)
                print("First 10 feature values:", features[0][:10])

                # ---- Feature visualization (1D plot, emb_dim=300) ----
                plt.figure(figsize=(10,3))
                plt.plot(features[0].cpu().numpy())
                plt.title(f"Extracted Image Features - {img_names[idx]} (1D plot)")
                plt.xlabel("Feature index")
                plt.ylabel("Feature value")
                plt.show()

                # ---- First token analysis ----
                h0 = torch.empty(1, decoder.hidden_dim).uniform_().to(device)
                c0 = torch.empty(1, decoder.hidden_dim).uniform_().to(device)
                first_output, _, _ = decoder.forward_step(features, h0, c0, mode="factual", features=features)
                first_output = first_output.squeeze(0)  # [vocab_size]
                top_tokens = torch.topk(first_output, 5).indices.tolist()
                print("Top 5 first tokens:", tokenizer.convert_ids_to_tokens(top_tokens))

                # ---- Caption generation ----
                output = decoder.sample(
                    features,
                    tokenizer=tokenizer,
                    beam_size=5,
                    max_len=30,
                    mode="romantic"  # You can change this to "factual" if needed
                )
                caption = tokenizer.decode(output, skip_special_tokens=True)
                print(f"Generated Caption (Bengali): {caption}")

        except Exception as e:
            print(f"❌ Error processing image {img_names[idx]}: {e}")
            continue

        # ------- Comprehensive evaluation ---------
        img_name = img_names[idx]
        ref_list = reference_captions.get(img_name, None)

        if ref_list is not None:
            print(f"Reference Captions ({len(ref_list)}): {ref_list}")
            
            try:
                # Tokenize hypothesis and references using your Bengali tokenizer
                hyp_tokens = tokenizer.tokenize(caption)
                ref_tokens_list = [tokenizer.tokenize(ref_caption) for ref_caption in ref_list]
                
                print(f"Hypothesis tokens: {hyp_tokens}")
                print(f"Reference tokens: {ref_tokens_list[0] if ref_tokens_list else []}")
                
                # Calculate metrics
                best_bleu, best_rouge, best_meteor = 0, 0, 0
                
                # BLEU, ROUGE, METEOR: compare with each reference separately
                for ref_tokens in ref_tokens_list:
                    bleu_score = improved_bleu(ref_tokens, hyp_tokens)  # Fixed function name
                    rouge_score = simple_rouge_l(ref_tokens, hyp_tokens)
                    meteor_score = tokenizer_based_meteor(ref_tokens, hyp_tokens, tokenizer)  # Fixed function name

                    best_bleu = max(best_bleu, bleu_score)
                    best_rouge = max(best_rouge, rouge_score)
                    best_meteor = max(best_meteor, meteor_score)
                
                # CIDEr: use all references simultaneously
                cider_score = enhanced_cider(ref_tokens_list, hyp_tokens)

                # Store scores
                all_bleu.append(best_bleu)
                all_rouge.append(best_rouge)
                all_meteor.append(best_meteor)
                all_cider.append(cider_score)
                
                # Print results
                print(f"📊 EVALUATION RESULTS:")
                print(f"   BLEU-4:   {best_bleu:.4f}")
                print(f"   ROUGE-L:  {best_rouge:.4f}")
                print(f"   METEOR:   {best_meteor:.4f}")
                print(f"   CIDEr:    {cider_score:.4f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error evaluating image {img_name}: {e}")
                
        else:
            print(f"❌ Reference captions NOT FOUND for {img_name}")
            print("-" * 50)

    # ------- Print comprehensive evaluation results ---------
    if all_bleu and all_rouge and all_meteor and all_cider:
        print("=" * 70)
        print("📈 COMPREHENSIVE BENGALI CAPTIONING EVALUATION RESULTS")
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
        
        print("\n🎯 PERFORMANCE INSIGHTS:")
        print("-" * 40)
        if sum(all_bleu)/len(all_bleu) > 0.3:
            print("✅ BLEU score indicates good n-gram overlap")
        else:
            print("⚠️  BLEU score suggests room for improvement in word choice")
            
        if sum(all_meteor)/len(all_meteor) > 0.25:
            print("✅ METEOR score shows good semantic similarity")
        else:
            print("⚠️  METEOR score indicates need for better semantic alignment")
            
        if sum(all_cider)/len(all_cider) > 0.5:
            print("✅ CIDEr score demonstrates strong consensus with references")
        else:
            print("⚠️  CIDEr score suggests captions could be more conventional")
    else:
        print("❌ No evaluation data collected. Please check your reference file and image directory.")

if __name__ == '__main__':
    main()
