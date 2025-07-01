from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hishab/titulm-mpt-1b-v2.0", trust_remote_code=True)

# Explicitly matching Hugging Face tokenizer's special tokens
SYMBOL_MAP = {
    '<pad>': tokenizer.pad_token_id,
    '<s>': tokenizer.cls_token_id if tokenizer.cls_token_id else tokenizer.bos_token_id,  # Handle cases explicitly
    '</s>': tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.eos_token_id, # Handle cases explicitly
    '<unk>': tokenizer.unk_token_id
}

def get_symbol_id(symbol):
    return SYMBOL_MAP[symbol]

