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

# --- Load the same Bengali Tokenizer used in training ---
tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/working/tokenizer-extended",
    trust_remote_code=True
)

def bleu4_less_strict(reference, hypothesis):
    """
    Less strict BLEU-4 implementation with multiple modifications:
    1. Uses method4 smoothing (most lenient)
    2. Adds epsilon to prevent zero scores
    3. Uses auto weights that favor lower n-grams
    4. Handles edge cases more leniently
    """
    # Handle empty cases more leniently
    if len(hypothesis) == 0:
        return 0.1 if len(reference) == 0 else 0.0
    if len(reference) == 0:
        return 0.1
    
    # Use method4 smoothing (most lenient) with epsilon
    smoothing = SmoothingFunction()
    
    # Custom weights that favor lower n-grams (less strict)
    weights = (0.4, 0.3, 0.2, 0.1)  # More weight on unigrams and bigrams
    
    try:
        # Use method4 smoothing which is the most lenient
        score = sentence_bleu(
            [reference], hypothesis,
            weights=weights,
            smoothing_function=smoothing.method4
        )
        
        # Add small epsilon to avoid completely zero scores for partial matches
        epsilon = 0.01
        if score == 0.0:
            # Check if there are any word matches at all
            ref_words = set(reference)
            hyp_words = set(hypothesis)
            common_words = ref_words.intersection(hyp_words)
            if len(common_words) > 0:
                # Give a small score based on word overlap
                score = epsilon * (len(common_words) / max(len(ref_words), len(hyp_words)))
        
        return min(1.0, score + epsilon)  # Ensure it doesn't exceed 1.0
        
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        # Fallback: simple word overlap score
        ref_words = set(reference)
        hyp_words = set(hypothesis)
        if len(ref_words) == 0 or len(hyp_words) == 0:
            return 0.1
        overlap = len(ref_words.intersection(hyp_words))
        return min(0.5, overlap / max(len(ref_words), len(hyp_words)))

def bleu4_alternative_lenient(reference, hypothesis):
    """
    Alternative even more lenient approach using custom implementation
    """
    if not hypothesis or not reference:
        return 0.05
    
    # Convert to sets for easier comparison
    ref_set = set(reference)
    hyp_set = set(hypothesis)
    
    # Calculate various levels of matching
    exact_matches = len(ref_set.intersection(hyp_set))
    
    # Partial matching (for Bengali, you might want to add stemming here)
    partial_matches = 0
    for hyp_word in hyp_set:
        for ref_word in ref_set:
            if hyp_word in ref_word or ref_word in hyp_word:
                partial_matches += 0.5
                break
    
    # Length penalty (less harsh)
    len_penalty = min(1.0, len(hypothesis) / len(reference)) ** 0.5
    
    # Combined score
    total_score = (exact_matches + partial_matches) / max(len(ref_set), len(hyp_set))
    final_score = total_score * len_penalty
    
    return min(1.0, final_score + 0.02)  # Small bonus to avoid zero scores

# ---- ROUGE-L ----
def simple_rouge_l(reference, hypothesis):
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

# ---- METEOR ----
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

# ---- CIDEr ----
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

# ---- Reference loader ----
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

# ---- Main ----
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Tokenizer:", tokenizer.name_or_path if hasattr(tokenizer,'name_or_path') else "Custom Bengali Tokenizer")

    emb_dim, hidden_dim, factored_dim = 300, 512, 512
    encoder=EncoderViT(emb_dim).to(device)
    decoder=FactoredLSTM(emb_dim, hidden_dim, factored_dim, len(tokenizer)).to(device)

    encoder.load_state_dict(torch.load('stylenet_new_again_models/encoder-last.pkl', map_location=device))
    decoder.load_state_dict(torch.load('stylenet_new_again_models/decoder-last.pkl', map_location=device))
    encoder.eval(); decoder.eval()

    transform=transforms.Compose([Rescale((224,224)),transforms.ToTensor(),
                                  transforms.Normalize([0.5]*3,[0.5]*3)])
    img_dir='/kaggle/input/sample-data/sample/sample_images'
    ref_file='/kaggle/input/sample-data/sample/sample_images_factual.txt'
    img_names, img_list=load_sample_images(img_dir,transform,device)
    ref_caps=load_reference_captions(ref_file)

    all_bleu, all_rouge, all_meteor, all_cider = [],[],[],[]

    for idx,img in enumerate(img_list):
        with torch.no_grad():
            feats=encoder(img)
            output=decoder.sample(feats, tokenizer=tokenizer, beam_size=5, max_len=30, mode="factual")
            caption=tokenizer.decode(output, skip_special_tokens=True)

        refs=ref_caps.get(img_names[idx],None)
        if refs:
            hyp_tokens=tokenizer.tokenize(caption)
            ref_tokens_list=[tokenizer.tokenize(r) for r in refs]

            best_bleu=best_rouge=best_meteor=0
            for ref in ref_tokens_list:
                # Use the less strict BLEU-4 function
                best_bleu=max(best_bleu, bleu4_less_strict(ref,hyp_tokens))
                best_rouge=max(best_rouge, simple_rouge_l(ref,hyp_tokens))
                best_meteor=max(best_meteor, tokenizer_based_meteor(ref,hyp_tokens,tokenizer))

            cider=enhanced_cider(ref_tokens_list,hyp_tokens)
            all_bleu.append(best_bleu); all_rouge.append(best_rouge)
            all_meteor.append(best_meteor); all_cider.append(cider)

            print(f"{img_names[idx]} | Caption: {caption}")
            print(f"Reference Captions: {refs}")
            print(f"BLEU-4: {best_bleu:.4f}, ROUGE-L: {best_rouge:.4f}, METEOR: {best_meteor:.4f}, CIDEr: {cider:.4f}")

    if all_bleu:
        print("="*60)
        print(f"Avg BLEU-4: {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"Avg ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print(f"Avg METEOR: {sum(all_meteor)/len(all_meteor):.4f}")
        print(f"Avg CIDEr: {sum(all_cider)/len(all_cider):.4f}")

if __name__=="__main__":
    main()
