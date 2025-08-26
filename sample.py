import os
import torch
import math
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import Rescale
from models import EncoderViT, FactoredLSTM
from collections import Counter, defaultdict
from transformers import AutoTokenizer

# --- Load the same Bengali Tokenizer you used in training ---
tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/working/tokenizer-extended",
    trust_remote_code=True
)

# ---- Improved BLEU-4 implementation with proper clipping and smoothing ----
def improved_bleu(reference, hypothesis, n=4, smooth=True):
    from collections import Counter
    if len(hypothesis) == 0 or len(reference) == 0:
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
            p_ns.append(0.01 if smooth else 0.0)
            continue
        clipped_counts = {ng: min(c, ref_counts.get(ng, 0)) for ng, c in hyp_counts.items()}
        overlap = sum(clipped_counts.values())
        total = sum(hyp_counts.values())
        p_ns.append(overlap / total if total > 0 else 0)
    
    ref_len, hyp_len = len(reference), len(hypothesis)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    
    if any(p == 0 for p in p_ns):
        log_sum = sum(w * math.log(max(p, 1e-10)) for w, p in zip(weights, p_ns)) if smooth else float('-inf')
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(weights, p_ns))
    
    return bp * math.exp(log_sum)

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
                best_bleu=max(best_bleu, improved_bleu(ref,hyp_tokens))
                best_rouge=max(best_rouge, simple_rouge_l(ref,hyp_tokens))
                best_meteor=max(best_meteor, tokenizer_based_meteor(ref,hyp_tokens,tokenizer))
            cider=enhanced_cider(ref_tokens_list,hyp_tokens)
            all_bleu.append(best_bleu); all_rouge.append(best_rouge)
            all_meteor.append(best_meteor); all_cider.append(cider)
            print(f"{img_names[idx]} | Caption: {caption}")
            print(f"BLEU-4: {best_bleu:.4f}, ROUGE-L: {best_rouge:.4f}, METEOR: {best_meteor:.4f}, CIDEr: {cider:.4f}")
    if all_bleu:
        print("="*60)
        print(f"Avg BLEU-4: {sum(all_bleu)/len(all_bleu):.4f}")
        print(f"Avg ROUGE-L: {sum(all_rouge)/len(all_rouge):.4f}")
        print(f"Avg METEOR: {sum(all_meteor)/len(all_meteor):.4f}")
        print(f"Avg CIDEr: {sum(all_cider)/len(all_cider):.4f}")

if __name__=="__main__":
    main()
