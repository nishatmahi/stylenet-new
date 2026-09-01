import os, torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

IMG_DIR = '/kaggle/input/datasets/kaggleperfect/dataset/data/Images'
OUT     = '/kaggle/working/style_feats'
MODEL, PRE, LANG = 'nllb-clip-base-siglip', 'v1', 'ben_Beng'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUT, exist_ok=True)

model, preprocess = create_model_from_pretrained(MODEL, PRE, device=device)
tokenizer = get_tokenizer(MODEL); tokenizer.set_language(LANG)
model.eval()

def strip_ext(n):
    for e in ('.jpg','.jpeg','.png'):
        if n.lower().endswith(e): return n[:-len(e)]
    return n

def image_ids(path):
    seen, order = set(), []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line: continue
        key = line.split('\t',1)[0] if '\t' in line else line.split('#',1)[0]
        img = strip_ext(key.split('#')[0].strip())
        if img not in seen: seen.add(img); order.append(img)
    return order

def read_lines(p):
    return [l.strip() for l in open(p, encoding='utf-8') if l.strip()]

@torch.no_grad()
def encode_images(id_file, out):
    ids = image_ids(id_file); feats, kept, batch, bids = [], [], [], []
    def flush():
        if not batch: return
        x = torch.stack(batch).to(device)
        feats.append(model.encode_image(x).float().cpu())
        kept.extend(bids); batch.clear(); bids.clear()
    miss = 0
    for img in ids:
        path = None
        for e in ('.jpg','.jpeg','.png'):
            p = os.path.join(IMG_DIR, img+e)
            if os.path.exists(p): path = p; break
        if path is None: miss += 1; continue
        batch.append(preprocess(Image.open(path).convert('RGB'))); bids.append(img)
        if len(batch) == 64: flush()
    flush()
    emb = torch.cat(feats, 0)
    torch.save({'ids': kept, 'emb': emb}, out)
    print('wrote', out, tuple(emb.shape), 'missing', miss, flush=True)

@torch.no_grad()
def encode_text(in_file, out):
    lines = read_lines(in_file); outs = []
    for i in range(0, len(lines), 64):
        toks = tokenizer(lines[i:i+64]).to(device)
        outs.append(model.encode_text(toks).float().cpu())
    emb = torch.cat(outs, 0)
    torch.save({'lines': lines, 'emb': emb}, out)
    print('wrote', out, tuple(emb.shape), flush=True)

encode_images('/kaggle/working/splits/factual_train.txt', f'{OUT}/factual_train_img.pt')
encode_images('/kaggle/working/splits/factual_val.txt',   f'{OUT}/factual_val_img.pt')
encode_images('/kaggle/working/splits/factual_test.txt',  f'{OUT}/test_images.pt')
encode_text('/kaggle/working/splits/romantic_train.txt',  f'{OUT}/romantic_train.pt')
encode_text('/kaggle/working/splits/romantic_val.txt',    f'{OUT}/romantic_val.pt')
encode_text('/kaggle/working/splits/humorous_train.txt',  f'{OUT}/humorous_train.pt')
encode_text('/kaggle/working/splits/humorous_val.txt',    f'{OUT}/humorous_val.pt')
