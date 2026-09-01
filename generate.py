import os, json, argparse, torch
from stage1 import (StyleCaptioner, build_tokenizer, add_style_tokens,
                    read_lines, split_line, unit)

def main(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = build_tokenizer(); sid = add_style_tokens(tok)
    ck = torch.load(a.ckpt, map_location='cpu')
    model = StyleCaptioner(ck['clip_dim'], tok, ck['prefix_len']).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    offset = ck.get('offset', None)
    timg = torch.load(a.test_img, map_location='cpu')
    temb = {i: e.float() for i,e in zip(timg['ids'], timg['emb'])}
    order, refs = [], {}
    for ln in read_lines(a.test_cap):
        p = split_line(ln)
        if not p or not p[1]: continue
        img, cap = p
        if img in temb: refs.setdefault(img, []).append(cap)
        if img in temb and img not in order: order.append(img)
        if a.n_images and len(order) >= a.n_images: break
    out = {}
    for s in range(0, len(order), a.batch_size):
        ch = order[s:s+a.batch_size]
        e = unit(torch.stack([temb[i] for i in ch]))
        if offset is not None and a.style != 'factual': e = unit(e - offset)
        e = e.to(dev)
        code = torch.full((e.size(0),), sid[a.style], device=dev)
        outs = model.generate(e, code, max_new=a.max_new, eos_id=tok.eos_token_id)
        for i, o in zip(ch, outs):
            out[i] = {'image_id': i, 'references': refs[i],
                      a.style: tok.decode(o, skip_special_tokens=True)}
        print(' ', min(s+a.batch_size,len(order)),'/',len(order), flush=True)
    json.dump(list(out.values()), open(a.out_json,'w',encoding='utf-8'),
              ensure_ascii=False, indent=2)
    for r in list(out.values())[:5]:
        print('\n', r['image_id']); print('  ref ', r['references'][0]); print('  gen ', r[a.style])
    print('wrote', a.out_json, len(out), 'images', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--style', default='romantic', choices=['factual','romantic','humorous'])
    p.add_argument('--test_img', default='/kaggle/working/style_feats/test_images.pt')
    p.add_argument('--test_cap', default='/kaggle/working/splits/factual_test.txt')
    p.add_argument('--out_json', required=True)
    p.add_argument('--n_images', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_new', type=int, default=120)
    a, _ = p.parse_known_args(); main(a)
