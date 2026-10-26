import os
import re
from pathlib import Path

import numpy as np

MODEL = "google/gemma-3-270m"
COLAB_TOKEN = Path("/content/.hf_token")


def hf_token():
    return os.environ.get("HF_TOKEN") or COLAB_TOKEN.read_text().strip()


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, token=hf_token())


def embedding():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    with safe_open(hf_hub_download(MODEL, "model.safetensors", token=hf_token()), "pt") as f:
        key = next(k for k in f.keys() if "embed_tokens" in k)
        return key, f.get_tensor(key).float().numpy()


def token_masks(tok, n_vocab):
    pieces = tok.convert_ids_to_tokens(list(range(n_vocab)))
    special_ids = {i for i, t in tok.added_tokens_decoder.items() if t.special} | set(tok.all_special_ids)
    is_unused = np.array([bool(re.fullmatch(r"<unused\d+>", p)) for p in pieces])
    is_special = np.array([i in special_ids for i in range(n_vocab)]) & ~is_unused
    return pieces, is_unused, is_special


def show(piece):
    return piece.replace("▁", "␣")


def wikitext():
    from datasets import load_dataset

    return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
