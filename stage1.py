import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

DEC = 'flax-community/gpt2-bengali'
STYLE_TOKENS = {'factual':'<factual>','romantic':'<romantic>','humorous':'<humorous>'}

def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(DEC)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def add_style_tokens(tok):
    tok.add_special_tokens({'additional_special_tokens': list(STYLE_TOKENS.values())})
    return {k: tok.convert_tokens_to_ids(v) for k,v in STYLE_TOKENS.items()}

def vocab_needed(tok):
    ids = [i for i in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id) if i is not None]
    return max(len(tok), *(i+1 for i in ids))

def read_lines(p): return [l.strip() for l in open(p, encoding='utf-8') if l.strip()]

def split_line(line):
    if '\t' in line: k,c = line.split('\t',1)
    elif '#' in line:
        k,c = line.split('#',1); parts = c.split(None,1); c = parts[1] if len(parts)>1 else ''
    else: return None
    k = k.split('#')[0].strip()
    for e in ('.jpg','.jpeg','.png'):
        if k.lower().endswith(e): k = k[:-len(e)]
    return k, c.strip()

def encode_with_eos(tok, text, max_len):
    ids = tok(text, max_length=max_len-1, truncation=True)['input_ids'] + [tok.eos_token_id]
    attn = [1]*len(ids); n = max_len-len(ids)
    ids = ids + [tok.pad_token_id]*n; attn = attn + [0]*n
    return torch.tensor(ids), torch.tensor(attn)

def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

class MLP(nn.Module):
    def __init__(self, in_dim, d_model, prefix_len):
        super().__init__(); out = d_model*prefix_len
        self.net = nn.Sequential(nn.Linear(in_dim, out//2), nn.Tanh(), nn.Linear(out//2, out))
        self.d_model, self.prefix_len = d_model, prefix_len
    def forward(self, x): return self.net(x).view(-1, self.prefix_len, self.d_model)

class StyleCaptioner(nn.Module):
    def __init__(self, clip_dim, tok, prefix_len=10):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(DEC)
        need = vocab_needed(tok)
        if need > self.gpt.config.vocab_size: self.gpt.resize_token_embeddings(need)
        self.d_model = self.gpt.config.n_embd; self.prefix_len = prefix_len
        self.clip_project = MLP(clip_dim, self.d_model, prefix_len)
    def embeds(self, clip_emb, style_ids, input_ids):
        wte = self.gpt.transformer.wte; w = self.clip_project.net[0].weight
        style = wte(style_ids).unsqueeze(1)
        prefix = self.clip_project(clip_emb.to(w.dtype))
        toks = wte(input_ids)
        return torch.cat([style, prefix, toks], dim=1)
    def forward(self, clip_emb, style_ids, input_ids, attn_mask):
        e = self.embeds(clip_emb, style_ids, input_ids)
        B = e.size(0); P = 1+self.prefix_len
        pre = torch.ones(B, P, device=attn_mask.device, dtype=attn_mask.dtype)
        full_mask = torch.cat([pre, attn_mask], 1)
        labels = torch.cat([torch.full((B,P), -100, device=input_ids.device, dtype=torch.long),
                            input_ids.masked_fill(attn_mask==0, -100)], 1)
        return self.gpt(inputs_embeds=e, attention_mask=full_mask, labels=labels).loss
    @torch.no_grad()
    def generate(self, clip_emb, style_ids, max_new=120, eos_id=None, rep=1.2):
        wte = self.gpt.transformer.wte; w = self.clip_project.net[0].weight
        e = torch.cat([wte(style_ids).unsqueeze(1), self.clip_project(clip_emb.to(w.dtype))], dim=1)
        B = e.size(0); dev = e.device
        ys = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev)
        past = None; inp = e
        for _ in range(max_new):
            out = self.gpt(inputs_embeds=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            for b in range(B):
                for t in set(ys[b]):
                    logits[b,t] = logits[b,t]/rep if logits[b,t]>0 else logits[b,t]*rep
            nxt = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    tk = int(nxt[b]); ys[b].append(tk)
                    if eos_id is not None and tk == eos_id: done[b] = True
            if done.all(): break
            inp = wte(nxt).unsqueeze(1)
        return ys

class FactualCapData(Dataset):
    def __init__(self, img_pt, cap_file, tok, sid, max_len=120):
        d = torch.load(img_pt, map_location='cpu')
        emb_by_id = {i: e.float() for i,e in zip(d['ids'], d['emb'])}
        self.rows = []
        for ln in read_lines(cap_file):
            p = split_line(ln)
            if not p or not p[1]: continue
            img, cap = p
            if img in emb_by_id: self.rows.append((emb_by_id[img], cap, img))
        if not self.rows: raise RuntimeError('no rows for '+cap_file)
        self.tok, self.code, self.max_len = tok, sid['factual'], max_len
        print('[data]', os.path.basename(cap_file), len(self.rows), 'rows', flush=True)
    def distinct(self, k):
        seen, embs, ids, gold = set(), [], [], {}
        for emb, cap, img in self.rows:
            if img in seen: continue
            seen.add(img); embs.append(emb); ids.append(img); gold[img] = cap
            if len(ids) == k: break
        return torch.stack(embs), ids, gold
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        emb, cap, img = self.rows[i]
        ids, attn = encode_with_eos(self.tok, cap, self.max_len)
        return unit(emb), torch.tensor(self.code), ids, attn

@torch.no_grad()
def eval_loss(model, dl, dev):
    model.eval(); s = n = 0
    for emb, code, ids, attn in dl:
        s += model(emb.to(dev), code.to(dev), ids.to(dev), attn.to(dev)).item(); n += 1
    return s / max(n,1)

def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.save_dir, exist_ok=True)
    tok = build_tokenizer(); sid = add_style_tokens(tok)
    tr_ds = FactualCapData(a.train_img, a.train_cap, tok, sid, max_len=a.max_len)
    va_ds = FactualCapData(a.val_img,   a.val_cap,   tok, sid, max_len=a.max_len)
    tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True, num_workers=2)
    va = DataLoader(va_ds, batch_size=a.batch_size, shuffle=False, num_workers=2)
    clip_dim = tr_ds.rows[0][0].numel()
    model = StyleCaptioner(clip_dim, tok, a.prefix_len).to(dev)
    print('  captioner', round(sum(p.numel() for p in model.parameters())/1e6,1),'M  clip_dim', clip_dim, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, a.warmup, a.epochs*len(tr))
    ckpt = os.path.join(a.save_dir, 'best.pth')
    start_epoch = 0; best = 1e9
    if os.path.exists(ckpt) and not a.fresh:
        r = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(r['model'])
        if 'opt' in r: opt.load_state_dict(r['opt'])
        if 'sch' in r: sch.load_state_dict(r['sch'])
        start_epoch = r.get('epoch',0); best = r.get('best',1e9)
        print('RESUMING from epoch', start_epoch, ' best val', round(best,4), flush=True)
    peek_emb, peek_ids, gold = va_ds.distinct(4); peek_emb = unit(peek_emb).to(dev)
    for ep in range(start_epoch, a.epochs):
        model.train(); tr_sum = tr_n = 0
        for i,(emb, code, ids, attn) in enumerate(tr):
            emb, code, ids, attn = emb.to(dev), code.to(dev), ids.to(dev), attn.to(dev)
            loss = model(emb, code, ids, attn); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tr_sum += loss.item(); tr_n += 1
            if i % 500 == 0: print('ep', ep+1, i,'/',len(tr),'loss',round(loss.item(),3), flush=True)
        val = eval_loss(model, va, dev); tr_loss = tr_sum/max(tr_n,1)
        print('[EPOCH', ep+1,'] train_loss', round(tr_loss,4), ' val_loss', round(val,4), flush=True)
        code = torch.full((peek_emb.size(0),), sid['factual'], device=dev)
        outs = model.generate(peek_emb, code, max_new=a.max_new, eos_id=tok.eos_token_id)
        for iid, o in zip(peek_ids, outs):
            print(' ', iid, 'gen ', tok.decode(o, skip_special_tokens=True))
            print('    gold', gold[iid])
        meta = {'model': model.state_dict(),'opt': opt.state_dict(),'sch': sch.state_dict(),
                'epoch': ep+1,'best': best,'clip_dim': clip_dim,'prefix_len': a.prefix_len}
        if val < best:
            best = val; meta['best'] = best
            torch.save(meta, ckpt); print('  NEW BEST -> saved best.pth', flush=True)
        else:
            print('  val did NOT improve (best', round(best,4),')', flush=True)
        torch.save(meta, os.path.join(a.save_dir,'last.pth'))
    print('done best', round(best,4), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_img', default='/kaggle/working/style_feats/factual_train_img.pt')
    p.add_argument('--train_cap', default='/kaggle/working/splits/factual_train.txt')
    p.add_argument('--val_img',   default='/kaggle/working/style_feats/factual_val_img.pt')
    p.add_argument('--val_cap',   default='/kaggle/working/splits/factual_val.txt')
    p.add_argument('--save_dir',  default='/kaggle/working/sc_factual')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--prefix_len', type=int, default=10)
    p.add_argument('--max_len', type=int, default=120)
    p.add_argument('--max_new', type=int, default=120)
    p.add_argument('--fresh', action='store_true')
    args, _ = p.parse_known_args(); main(args)
