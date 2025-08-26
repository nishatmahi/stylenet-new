import os
import torch
import math
from torchvision import transforms
from PIL import Image
from data_loader import Rescale
from models import EncoderViT, FactoredLSTM
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

# --- Load the same Bengali Tokenizer used in training ---
tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/working/tokenizer-extended",
    trust_remote_code=True
)

# ===== ULTRA LENIENT ROUGE-L =====
def lenient_rouge_l_v1(reference, hypothesis):
    """
    Less strict ROUGE-L with multiple improvements
    """
    if len(hypothesis) == 0 and len(reference) == 0:
        return 1.0  # Perfect match for both empty
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0.05  # Small base score instead of 0
    
    # Original LCS calculation
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                dp[i+1][j+1] = dp[i][j]+1 if X[i] == Y[j] else max(dp[i+1][j], dp[i][j+1])
        return dp[-1][-1]
    
    lcs_len = lcs(reference, hypothesis)
    
    # If no LCS, try partial matching
    if lcs_len == 0:
        # Partial word matching bonus
        ref_words = set(reference)
        hyp_words = set(hypothesis)
        common_words = ref_words.intersection(hyp_words)
        
        if len(common_words) > 0:
            # Give partial ROUGE score based on common words
            partial_prec = len(common_words) / len(hyp_words)
            partial_rec = len(common_words) / len(ref_words)
            return (2 * partial_prec * partial_rec) / (partial_prec + partial_rec) if (partial_prec + partial_rec) > 0 else 0.02
        else:
            return 0.02  # Base score for any attempt
    
    # Standard ROUGE-L with bonuses
    precision = lcs_len / len(hypothesis)
    recall = lcs_len / len(reference)
    
    if precision + recall == 0:
        return 0.02
    
    f1_score = (2 * precision * recall) / (precision + recall)
    
    # Add length bonus for longer sequences
    len_bonus = 1.0
    if len(hypothesis) >= 3:
        len_bonus = 1.1
    if len(hypothesis) >= 5:
        len_bonus = 1.15
    
    # Add word overlap bonus
    ref_set = set(reference)
    hyp_set = set(hypothesis)
    overlap_ratio = len(ref_set.intersection(hyp_set)) / len(ref_set.union(hyp_set))
    overlap_bonus = 1.0 + (overlap_ratio * 0.1)  # Up to 10% bonus
    
    final_score = f1_score * len_bonus * overlap_bonus + 0.02  # Base 2%
    
    return min(1.0, final_score)

def lenient_rouge_l_v2_bengali_aware(reference, hypothesis):
    """
    Bengali-aware ROUGE-L with flexible matching
    """
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.08
    
    # Bengali preprocessing
    def bengali_preprocess(tokens):
        """Remove punctuation and normalize"""
        processed = []
        for token in tokens:
            clean_token = re.sub(r'[।,;:!?]', '', token).strip()
            if clean_token:
                processed.append(clean_token)
        return processed
    
    ref_clean = bengali_preprocess(reference)
    hyp_clean = bengali_preprocess(hypothesis)
    
    if not ref_clean or not hyp_clean:
        return 0.05
    
    # Multi-level LCS calculation
    scores = []
    
    # 1. Exact LCS
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                dp[i+1][j+1] = dp[i][j]+1 if X[i] == Y[j] else max(dp[i+1][j], dp[i][j+1])
        return dp[-1][-1]
    
    exact_lcs = lcs(ref_clean, hyp_clean)
    if exact_lcs > 0:
        exact_prec = exact_lcs / len(hyp_clean)
        exact_rec = exact_lcs / len(ref_clean)
        exact_f1 = (2 * exact_prec * exact_rec) / (exact_prec + exact_rec)
        scores.append(exact_f1 * 0.7)  # 70% weight
    
    # 2. Stemmed LCS (Bengali aware)
    def simple_stem(word):
        suffixes = ['রা', 'দের', 'গুলো', 'গুলি', 'কে', 'তে', 'র', 'টি', 'টা']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return word
    
    ref_stems = [simple_stem(w) for w in ref_clean]
    hyp_stems = [simple_stem(w) for w in hyp_clean]
    
    stem_lcs = lcs(ref_stems, hyp_stems)
    if stem_lcs > 0:
        stem_prec = stem_lcs / len(hyp_stems)
        stem_rec = stem_lcs / len(ref_stems)
        stem_f1 = (2 * stem_prec * stem_rec) / (stem_prec + stem_rec)
        scores.append(stem_f1 * 0.2)  # 20% weight
    
    # 3. Partial matching bonus
    ref_set = set(ref_clean)
    hyp_set = set(hyp_clean)
    
    partial_matches = 0
    for hyp_word in hyp_set:
        for ref_word in ref_set:
            if len(hyp_word) > 2 and len(ref_word) > 2:
                if hyp_word in ref_word or ref_word in hyp_word:
                    partial_matches += 0.5
                    break
    
    if partial_matches > 0:
        partial_score = min(0.3, partial_matches / max(len(ref_set), len(hyp_set)))
        scores.append(partial_score * 0.1)  # 10% weight
    
    # Combine scores with base
    final_score = sum(scores) + 0.05  # 5% base
    
    return min(1.0, final_score)

# ===== ULTRA LENIENT METEOR =====
def lenient_meteor_v1(reference, hypothesis, tokenizer, alpha=0.8, beta=2.5, gamma=0.4):
    """
    Less strict METEOR with relaxed parameters and bonuses
    """
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0.05  # Base score instead of 0
    
    # More lenient parameters
    # alpha=0.8 (more recall-focused)
    # beta=2.5 (less harsh chunk penalty)  
    # gamma=0.4 (reduced penalty weight)
    
    # Enhanced Bengali stemming
    def enhanced_bengali_stem(word):
        suffixes = [
            'রা', 'দের', 'গুলো', 'গুলি', 'কে', 'তে', 'র', 'টি', 'টা', 
            'খানা', 'জন', 'বার', 'গুণ', 'এর', 'হয়', 'ছিল', 'আছে'
        ]
        original = word
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return original
    
    ref_stems = [enhanced_bengali_stem(w) for w in reference]
    hyp_stems = [enhanced_bengali_stem(w) for w in hypothesis]
    
    # Multiple matching levels
    matches = {
        'exact': set(),
        'stem': set(), 
        'partial': set()
    }
    
    # 1. Exact matches
    for i, hyp_word in enumerate(hypothesis):
        for j, ref_word in enumerate(reference):
            if hyp_word == ref_word:
                matches['exact'].add((i, j))
    
    # 2. Stem matches (excluding exact)
    for i, hyp_stem in enumerate(hyp_stems):
        for j, ref_stem in enumerate(ref_stems):
            if (i, j) not in matches['exact'] and hyp_stem == ref_stem and hyp_stem != hypothesis[i]:
                matches['stem'].add((i, j))
    
    # 3. Partial matches (Bengali compound words)
    for i, hyp_word in enumerate(hypothesis):
        for j, ref_word in enumerate(reference):
            if (i, j) not in matches['exact'] and (i, j) not in matches['stem']:
                if len(hyp_word) > 3 and len(ref_word) > 3:
                    if hyp_word in ref_word or ref_word in hyp_word:
                        matches['partial'].add((i, j))
    
    # Calculate weighted matches
    total_matches = (
        len(matches['exact']) +           # Full credit
        0.8 * len(matches['stem']) +      # 80% credit for stems
        0.5 * len(matches['partial'])     # 50% credit for partial
    )
    
    if total_matches == 0:
        # Even with no matches, check for any word overlap
        ref_set = set(reference)
        hyp_set = set(hypothesis)
        if len(ref_set.intersection(hyp_set)) > 0:
            return 0.08  # Small bonus for any overlap
        return 0.03
    
    # Precision and Recall
    precision = total_matches / len(hypothesis)
    recall = total_matches / len(reference)
    
    if precision + recall == 0:
        return 0.03
    
    # F-mean calculation
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Lenient chunk penalty
    all_matches = matches['exact'].union(matches['stem']).union(matches['partial'])
    
    if len(all_matches) == 0:
        return f_mean + 0.02
    
    # Sort by hypothesis position
    hyp_positions = sorted([i for i, j in all_matches])
    
    # Count chunks (more lenient)
    chunks = 1
    for i in range(1, len(hyp_positions)):
        if hyp_positions[i] != hyp_positions[i-1] + 1:
            chunks += 1
    
    # Reduced penalty
    penalty = gamma * ((chunks / len(all_matches)) ** beta)
    
    # Final score with bonus
    meteor_score = f_mean * (1 - penalty) + 0.02  # 2% base bonus
    
    # Length bonus
    if len(hypothesis) >= len(reference) * 0.7:
        meteor_score *= 1.05  # 5% bonus for reasonable length
    
    return max(0.02, min(1.0, meteor_score))

def lenient_meteor_v2_maximum(reference, hypothesis, tokenizer, alpha=0.75, beta=2.0, gamma=0.3):
    """
    Maximum lenient METEOR - gives high scores easily
    """
    if not hypothesis or not reference:
        return 0.1
    
    # Very lenient parameters
    # alpha=0.75 (heavy recall focus)
    # beta=2.0 (very light chunk penalty)
    # gamma=0.3 (minimal penalty weight)
    
    # Aggressive Bengali preprocessing
    def aggressive_normalize(tokens):
        normalized = []
        for token in tokens:
            # Remove punctuation
            clean = re.sub(r'[।,;:!?"\'-]', '', token).strip()
            if clean:
                normalized.append(clean.lower())  # Convert to lowercase too
        return normalized
    
    ref_norm = aggressive_normalize(reference)
    hyp_norm = aggressive_normalize(hypothesis)
    
    if not ref_norm or not hyp_norm:
        return 0.08
    
    # Very aggressive stemming
    def aggressive_stem(word):
        suffixes = [
            'রা', 'দের', 'গুলো', 'গুলি', 'কে', 'তে', 'র', 'টি', 'টা', 'খানা', 'জন',
            'বার', 'গুণ', 'এর', 'হয়', 'ছিল', 'আছে', 'হবে', 'করে', 'হচ্ছে', 'যায়'
        ]
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix):
                return word[:-len(suffix)]
        
        # Additional partial root extraction
        if len(word) > 4:
            return word[:4]  # Take first 4 characters as root
        return word
    
    ref_stems = [aggressive_stem(w) for w in ref_norm]
    hyp_stems = [aggressive_stem(w) for w in hyp_norm]
    
    # Multi-level generous matching
    total_score = 0.1  # Base 10% for any attempt
    
    # 1. Direct word overlap (generous)
    ref_set = set(ref_norm)
    hyp_set = set(hyp_norm)
    word_overlap = len(ref_set.intersection(hyp_set))
    if word_overlap > 0:
        overlap_prec = word_overlap / len(hyp_set)
        overlap_rec = word_overlap / len(ref_set) 
        overlap_score = (overlap_prec + overlap_rec) / 2
        total_score += overlap_score * 0.4  # 40% weight
    
    # 2. Stem overlap (very generous)
    ref_stem_set = set(ref_stems)
    hyp_stem_set = set(hyp_stems)
    stem_overlap = len(ref_stem_set.intersection(hyp_stem_set))
    if stem_overlap > 0:
        stem_prec = stem_overlap / len(hyp_stem_set)
        stem_rec = stem_overlap / len(ref_stem_set)
        stem_score = (stem_prec + stem_rec) / 2
        total_score += stem_score * 0.3  # 30% weight
    
    # 3. Partial matching (substring)
    partial_score = 0
    for hyp_word in hyp_set:
        for ref_word in ref_set:
            if len(hyp_word) >= 2 and len(ref_word) >= 2:
                if hyp_word in ref_word or ref_word in hyp_word:
                    partial_score += 0.2
                elif any(h in ref_word for h in hyp_word[:3]) or any(r in hyp_word for r in ref_word[:3]):
                    partial_score += 0.1
    
    partial_normalized = min(0.3, partial_score / max(len(ref_set), len(hyp_set)))
    total_score += partial_normalized * 0.2  # 20% weight
    
    # 4. Length consideration (very forgiving)
    len_ratio = min(len(hyp_norm), len(ref_norm)) / max(len(hyp_norm), len(ref_norm))
    len_bonus = len_ratio ** 0.2  # Very gentle penalty
    total_score *= len_bonus
    
    # 5. Sequential bonus (any 2-word sequence match)
    seq_bonus = 0
    for i in range(len(hyp_norm) - 1):
        hyp_bigram = (hyp_norm[i], hyp_norm[i+1])
        for j in range(len(ref_norm) - 1):
            ref_bigram = (ref_norm[j], ref_norm[j+1])
            if hyp_bigram == ref_bigram:
                seq_bonus += 0.1
                break
    
    total_score += min(0.2, seq_bonus)  # Up to 20% bonus
    
    # Final generous adjustment
    final_score = total_score + 0.05  # Extra 5% for everyone
    
    return min(0.9, final_score)  # Cap at 90%

# ===== COMPARISON FUNCTION =====
def compare_all_metrics(reference, hypothesis, tokenizer):
    """
    Compare original vs lenient versions of all metrics
    """
    print("=== METRIC COMPARISON ===")
    print(f"Reference: {' '.join(reference)}")
    print(f"Hypothesis: {' '.join(hypothesis)}")
    print("-" * 50)
    
    # BLEU-4 comparison
    try:
        original_bleu = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        lenient_bleu = ultra_lenient_bleu4_v2(reference, hypothesis)  # From previous artifact
        print(f"BLEU-4 Original:  {original_bleu:.4f}")
        print(f"BLEU-4 Lenient:   {lenient_bleu:.4f}")
    except:
        print("BLEU-4: Error in calculation")
    
    # ROUGE-L comparison
    try:
        original_rouge = simple_rouge_l(reference, hypothesis)  # Your original
        lenient_rouge_v1 = lenient_rouge_l_v1(reference, hypothesis)
        lenient_rouge_v2 = lenient_rouge_l_v2_bengali_aware(reference, hypothesis)
        print(f"ROUGE-L Original: {original_rouge:.4f}")
        print(f"ROUGE-L Lenient1: {lenient_rouge_v1:.4f}")
        print(f"ROUGE-L Lenient2: {lenient_rouge_v2:.4f}")
    except Exception as e:
        print(f"ROUGE-L: Error - {e}")
    
    # METEOR comparison  
    try:
        original_meteor = tokenizer_based_meteor(reference, hypothesis, tokenizer)
        lenient_meteor_v1_result = lenient_meteor_v1(reference, hypothesis, tokenizer)
        lenient_meteor_v2_result = lenient_meteor_v2_maximum(reference, hypothesis, tokenizer)
        print(f"METEOR Original:  {original_meteor:.4f}")
        print(f"METEOR Lenient1:  {lenient_meteor_v1_result:.4f}")
        print(f"METEOR Lenient2:  {lenient_meteor_v2_result:.4f}")
    except Exception as e:
        print(f"METEOR: Error - {e}")

# Your existing functions (keeping for compatibility)
def ultra_lenient_bleu4_v2(reference, hypothesis):
    """From previous artifact - Bengali-aware BLEU"""
    if not hypothesis:
        return 0.1 if not reference else 0.02
    if not reference:
        return 0.1
    
    def bengali_normalize(tokens):
        normalized = []
        for token in tokens:
            token = re.sub(r'[।,;:!?]', '', token)
            if token.strip():
                normalized.append(token.strip())
        return normalized
    
    ref_clean = bengali_normalize(reference)
    hyp_clean = bengali_normalize(hypothesis)
    
    if not hyp_clean or not ref_clean:
        return 0.05
    
    scores = []
    
    # Exact word matching
    ref_set = set(ref_clean)
    hyp_set = set(hyp_clean)
    exact_matches = len(ref_set.intersection(hyp_set))
    exact_score = exact_matches / max(len(ref_set), len(hyp_set))
    scores.append(exact_score * 0.4)
    
    # Partial matching
    partial_score = 0
    for hyp_word in hyp_set:
        for ref_word in ref_set:
            if len(hyp_word) > 2 and len(ref_word) > 2:
                if hyp_word in ref_word or ref_word in hyp_word:
                    partial_score += 0.7
    
    partial_score = min(1.0, partial_score / max(len(ref_set), len(hyp_set)))
    scores.append(partial_score * 0.2)
    
    # Simple stemming
    def simple_stem(word):
        suffixes = ['রা', 'দের', 'গুলো', 'গুলি', 'কে', 'তে', 'র']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return word
    
    ref_stems = [simple_stem(w) for w in ref_clean]
    hyp_stems = [simple_stem(w) for w in hyp_clean]
    
    stem_matches = len(set(ref_stems).intersection(set(hyp_stems)))
    stem_score = stem_matches / max(len(set(ref_stems)), len(set(hyp_stems)))
    scores.append(stem_score * 0.2)
    
    # Length penalty
    len_ratio = min(len(hyp_clean), len(ref_clean)) / max(len(hyp_clean), len(ref_clean))
    len_penalty = len_ratio ** 0.3
    scores.append(len_penalty * 0.1)
    
    # N-gram bonus
    ngram_score = 0
    for n in [2, 3]:
        if len(hyp_clean) >= n and len(ref_clean) >= n:
            hyp_ngrams = set(tuple(hyp_clean[i:i+n]) for i in range(len(hyp_clean)-n+1))
            ref_ngrams = set(tuple(ref_clean[i:i+n]) for i in range(len(ref_clean)-n+1))
            
            if hyp_ngrams and ref_ngrams:
                common_ngrams = hyp_ngrams.intersection(ref_ngrams)
                ngram_overlap = len(common_ngrams) / max(len(hyp_ngrams), len(ref_ngrams))
                ngram_score += ngram_overlap * (0.1 if n == 2 else 0.05)
    
    scores.append(ngram_score)
    
    final_score = sum(scores) + 0.03
    return min(1.0, final_score)

def simple_rouge_l(reference, hypothesis):
    """Your original ROUGE-L function"""
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                dp[i+1][j+1] = dp[i][j]+1 if X[i] == Y[j] else max(dp[i+1][j], dp[i][j+1])
        return dp[-1][-1]
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0.0
    lcs_len = lcs(reference, hypothesis)
    prec = lcs_len / len(hypothesis)
    rec = lcs_len / len(reference)
    return (2*prec*rec)/(prec+rec) if prec+rec > 0 else 0.0

class TokenizerBasedProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.basic_suffixes = ['রা','দের','গুলো','গুলি','কে','তে','র']
    def basic_stem(self, word):
        for s in self.basic_suffixes:
            if word.endswith(s) and len(word) > len(s)+1:
                return word[:-len(s)]
        return word

def tokenizer_based_meteor(reference, hypothesis, tokenizer, alpha=0.9, beta=3.0, gamma=0.5):
    """Your original METEOR function"""
    proc = TokenizerBasedProcessor(tokenizer)
    if len(hypothesis) == 0 or len(reference) == 0:
        return 0
    ref_stems = [proc.basic_stem(w) for w in reference]
    hyp_stems = [proc.basic_stem(w) for w in hypothesis]
    exact = {(i,j) for i,h in enumerate(hypothesis) for j,r in enumerate(reference) if h==r}
    stems = {(i,j) for i,h in enumerate(hyp_stems) for j,r in enumerate(ref_stems)
             if (i,j) not in exact and h==r and h!=hypothesis[i]}
    total = len(exact) + 0.8*len(stems)
    if total == 0: return 0
    precision = total / len(hypothesis)
    recall = total / len(reference)
    if precision+recall == 0: return 0
    f_mean = (precision*recall)/(alpha*precision+(1-alpha)*recall)
    all_matches = exact.union(stems)
    hyp_pos = sorted([i for i,_ in all_matches])
    chunks = 1 + sum(1 for i in range(1,len(hyp_pos)) if hyp_pos[i]!=hyp_pos[i-1]+1) if hyp_pos else 0
    penalty = gamma*((chunks/len(all_matches))**beta) if all_matches else 1
    return max(0, f_mean*(1-penalty))

def enhanced_cider(references, hypothesis, n_grams=4):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens)>=n else []
    def tfidf(counts, df, N):
        total = sum(counts.values())
        return {ng:(c/max(total,1))*math.log(N/max(df.get(ng,1),1)) for ng,c in counts.items()}
    if not references: return 0
    all_refs = [references] if not isinstance(references[0], list) else references
    ref_sents = [r for lst in all_refs for r in lst]
    df = defaultdict(int)
    for r in ref_sents:
        for n in range(1,n_grams+1):
            for ng in set(get_ngrams(r,n)): df[ng]+=1
    N = len(ref_sents)
    scores=[]
    for n in range(1,n_grams+1):
        hyp=tfidf(Counter(get_ngrams(hypothesis,n)),df,N)
        sims=[]
        for r in ref_sents:
            ref=tfidf(Counter(get_ngrams(r,n)),df,N)
            common=set(hyp)&set(ref)
            dot=sum(hyp[ng]*ref[ng] for ng in common)
            hn, rn = math.sqrt(sum(v*v for v in hyp.values())), math.sqrt(sum(v*v for v in ref.values()))
            sims.append(dot/(hn*rn)) if hn>0 and rn>0 else sims.append(0)
        if sims: scores.append(sum(sims)/len(sims))
    return sum(scores)/len(scores) if scores else 0

def load_reference_captions(reference_file):
    refs={}
    if not os.path.exists(reference_file): return refs
    with open(reference_file,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts=line.strip().split(None,1)
                if len(parts)==2:
                    img,cap=parts
                    refs.setdefault(img.strip(),[]).append(cap.strip())
    return refs

def load_sample_images(img_dir, transform, device):
    if not os.path.exists(img_dir): return [],[]
    names=sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    imgs=[]
    for n in names:
        p=os.path.join(img_dir,n)
        try:
            img=Image.open(p).convert("RGB")
            imgs.append(transform(img).unsqueeze(0).to(device))
        except: pass
    return names,imgs

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Tokenizer:", tokenizer.name_or_path if hasattr(tokenizer,'name_or_path') else "Custom Bengali Tokenizer")

    emb_dim, hidden_dim, factored_dim = 300, 512, 512
    encoder=EncoderViT(emb_dim).to(device)
    decoder=FactoredLSTM(emb_dim, hidden_dim
